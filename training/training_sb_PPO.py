import time
import gymnasium as gym
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import package to trigger Gymnasium environment registration
import simulation.envs


def linear_schedule(initial_value: float, final_value: float = 5e-5):
    """
    Linearly decays the learning rate from initial_value to final_value
    as progress_remaining goes from 1.0 to 0.0.
    """
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def main():

    env_id = "FurutaPendulum-v0"

    # Timestamped directory in training/logs/ to preserve historical training runs
    date = datetime.now()
    folder_name = date.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("training/logs") / folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_model_path = str(log_dir / "ppo_furuta_pendulum")
    log_stats_path = str(log_dir / "vec_normalize.pkl")

    # Root-level models/ directory (sibling to training/) storing the latest artifacts
    # under fixed filenames for convenient evaluation and ONNX export
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    latest_model_path = str(models_dir / "ppo_furuta_pendulum")
    latest_stats_path = str(models_dir / "vec_normalize.pkl")

    print("Initializing Furuta pendulum environment for compliance check...")
    raw_env = gym.make(env_id)
    check_env(raw_env, warn=True)
    raw_env.close()
    print("Environment check passed successfully!")

    # Create 4 parallel environments to collect diverse trajectories simultaneously
    vec_env = make_vec_env(env_id, n_envs=4, seed=42, vec_env_cls=DummyVecEnv)

    # Normalize observations only; keep the [0.0, 1.0] reward scale untouched
    # gamma=0.995 matches PPO's 4-second effective planning horizon at 50 Hz
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.995,
    )

    # Define custom MLP architecture: 2 hidden layers of 128 neurons each
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # Initialize PPO with KL safeguard, moderate epochs, and standard Gaussian exploration
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(initial_value=3e-4, final_value=5e-5),
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42,
        device="cpu",
    )

    print("Starting PPO training process...")
    model.learn(total_timesteps=1_000_000)
    print("Training finished successfully!")

    # Save archived copy to timestamped training/logs/ folder
    model.save(log_model_path)
    vec_env.save(log_stats_path)

    # Save latest copy to root models/ folder for immediate testing and ONNX conversion
    model.save(latest_model_path)
    vec_env.save(latest_stats_path)
    vec_env.close()

    print(f"Archived model to '{log_model_path}.zip' and stats to '{log_stats_path}'.")
    print(f"Updated latest test artifacts in '{models_dir}/'.")

    # --- Visual Evaluation ---
    print("\nRunning visual evaluation of the trained PPO policy...")
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])

    # Load observation scaling statistics from models/ and freeze them
    eval_env = VecNormalize.load(latest_stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    # Run 1000 steps at 50 Hz (20 seconds of real-time visualization)
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        time.sleep(0.02)
        if done[0]:
            obs = eval_env.reset()

    eval_env.close()
    print("Evaluation completed.")


if __name__ == "__main__":
    main()