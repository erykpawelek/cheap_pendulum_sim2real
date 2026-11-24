import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from stable_baselines3 import PPO


class OnnxableFurutaPolicy(nn.Module):
    """
    Wraps VecNormalize observation scaling, the deterministic PPO Actor MLP,
    and action space clipping into a single self-contained ONNX graph.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        action_net: nn.Module,
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
        epsilon: float, #constant preventig from deviding by zero
        clip_obs: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()
        self.policy_net = policy_net
        self.action_net = action_net

        # Precompute standard deviation and register all constants as non-trainable buffers
        # so they are embedded directly as constant tensors inside the ONNX model
        obs_std = np.sqrt(obs_var + epsilon) #denominator of the normalization
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32))
        self.register_buffer("clip_obs", torch.tensor(clip_obs, dtype=torch.float32))
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        # 1. Normalize raw physical observations using frozen VecNormalize statistics
        norm_obs = (raw_obs - self.obs_mean) / self.obs_std
        norm_obs = torch.clamp(norm_obs, -self.clip_obs, self.clip_obs)

        # 2. Run deterministic forward pass through the PPO Actor (mean of Gaussian policy)
        latent_pi = self.policy_net(norm_obs)
        action = self.action_net(latent_pi)

        # 3. Clamp output velocity to match the physical action space bounds [-15.0, 15.0]
        return torch.clamp(action, self.action_low, self.action_high)


def export_model(model_path: Path, stats_path: Path, output_path: Path) -> None:
    # Load trained SB3 PPO model on CPU without needing the MuJoCo environment
    print(f"Loading PPO model from: {model_path}")
    model = PPO.load(model_path, device="cpu")
    model.policy.set_training_mode(False)

    # Load pickled VecNormalize object directly to extract running mean and variance
    print(f"Loading VecNormalize statistics from: {stats_path}")
    with open(stats_path, "rb") as f:
        vec_norm = pickle.load(f)

    obs_dim = int(model.observation_space.shape[0])

    # Build the unified exportable PyTorch module
    onnx_policy = OnnxableFurutaPolicy(
        policy_net=model.policy.mlp_extractor.policy_net,
        action_net=model.policy.action_net,
        obs_mean=vec_norm.obs_rms.mean,
        obs_var=vec_norm.obs_rms.var,
        epsilon=vec_norm.epsilon,
        clip_obs=vec_norm.clip_obs,
        action_low=model.action_space.low,
        action_high=model.action_space.high,
    )
    onnx_policy.eval()

    # Dummy input with batch size 1 and 5 observation features:
    # [sin(theta), cos(theta), pendulum_vel, motor_pos, motor_vel]
    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX model to: {output_path}")

    torch.onnx.export(
        onnx_policy,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["raw_obs"],
        output_names=["target_vel"],
        dynamic_axes={
            "raw_obs": {0: "batch_size"},
            "target_vel": {0: "batch_size"},
        },
    )

    # --- Verification: compare SB3 + VecNormalize against ONNX Runtime ---
    print("Verifying numerical equivalence with ONNX Runtime...")
    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(seed=42)
    # Generate 100 random realistic physical states
    test_obs = rng.uniform(low=-10.0, high=10.0, size=(100, obs_dim)).astype(np.float32)

    max_abs_diff = 0.0
    for i in range(test_obs.shape[0]):
        raw_sample = test_obs[i : i + 1]

        # Reference output using exact VecNormalize formula + SB3 deterministic predict
        norm_sample = np.clip(
            (raw_sample - vec_norm.obs_rms.mean) / np.sqrt(vec_norm.obs_rms.var + vec_norm.epsilon),
            -vec_norm.clip_obs,
            vec_norm.clip_obs,
        ).astype(np.float32)
        sb3_action, _ = model.predict(norm_sample, deterministic=True)

        # ONNX Runtime inference directly on raw physical observations
        ort_action = ort_session.run(None, {"raw_obs": raw_sample})[0]

        diff = float(np.max(np.abs(sb3_action - ort_action)))
        max_abs_diff = max(max_abs_diff, diff)

    print(f"Verification passed! Max absolute difference across 100 states: {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-5, f"Mismatch between SB3 and ONNX outputs: {max_abs_diff}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Stable-Baselines3 PPO policy and VecNormalize stats to ONNX."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("training/logs/ppo_furuta_pendulum.zip"),
        help="Path to the saved SB3 PPO .zip archive.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("training/logs/vec_normalize.pkl"),
        help="Path to the saved VecNormalize .pkl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/logs/furuta_policy.onnx"),
        help="Destination path for the exported .onnx file.",
    )
    args = parser.parse_args()

    export_model(args.model, args.stats, args.output)


if __name__ == "__main__":
    main()