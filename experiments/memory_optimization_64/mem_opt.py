import copy
import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import optuna
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import simulation.envs  # noqa: F401 - registers FurutaPendulum-v0

# --- Paths ---
EXP_DIR = Path(__file__).resolve().parent        # experiments/memory_optimization_N/
EXPERIMENTS_DIR = EXP_DIR.parent                 # experiments/
DB_PATH = EXPERIMENTS_DIR / "optuna_studies.db"  # shared across all experiments
CSV_PATH = EXP_DIR / "optimization_scores.csv"   # local to this experiment folder

# --- Experiment Config (update model paths per folder) ---
RUN_NAME = EXP_DIR.name                          # e.g. "memory_optimization_64"
ENV_ID = "FurutaPendulum-v0"
MODEL_PATH = "training/logs/good_model/ppo_furuta_pendulum"
STATS_PATH = "training/logs/good_model/vec_normalize.pkl"
EVAL_SEEDS = list(range(8))                      # fixed start states for all candidates
N_TRIALS = 120
N_JOBS = 4
INCLUDE_BASELINE_TRIAL = True                    # pins 0% sparsity point on Pareto plot

torch.set_num_threads(1)                         # avoid n_jobs x torch internal threads
_local = threading.local()


def get_ctx():
    """Per-thread env + base model (built lazily on first use)."""
    if not hasattr(_local, "env"):
        env = DummyVecEnv([lambda: gym.make(ENV_ID)])
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False                     # freeze obs normalization
        env.norm_reward = False                  # true unscaled reward
        _local.env = env
        _local.model = PPO.load(MODEL_PATH, env=env, device="cpu")
    return _local.env, _local.model


def get_actor_layers(policy) -> list[nn.Linear]:
    """Actor layers only (critic is discarded at deployment)."""
    modules = list(policy.mlp_extractor.policy_net.modules()) + [policy.action_net]
    return [m for m in modules if isinstance(m, nn.Linear)]


def get_arch_str(policy) -> str:
    """Fingerprint actor architecture, e.g. '4x64x64x1'."""
    layers = get_actor_layers(policy)
    dims = [layers[0].in_features] + [layer.out_features for layer in layers]
    return "x".join(map(str, dims))


def get_layer_labels(n_layers: int) -> list[str]:
    """Generate layer names dynamically for any actor depth."""
    if n_layers == 1:
        return ["Input -> Output"]
    labels = ["Input -> Hidden 1"]
    for i in range(1, n_layers - 1):
        labels.append(f"Hidden {i} -> Hidden {i + 1}")
    labels.append(f"Hidden {n_layers - 1} -> Output")
    return labels


def calc_memory_footprint(model: PPO, env=None) -> dict[str, float | int]:
    """
    Calculate actor memory footprint for deployment (baseline or pruned):
      - dense_kb:        all actor weights + biases (uncompressed baseline size)
      - active_kb:       non-zero actor weights + biases
      - norm_kb:         VecNormalize obs mean + var stored as float32
      - deploy_flash_kb: active_kb + norm_kb (total static weights/stats to store)
      - peak_act_kb:     peak SRAM for input + output buffer of widest layer
    """
    layers = get_actor_layers(model.policy)
    total_params = active_params = 0
    dense_bytes = active_bytes = 0

    for layer in layers:
        w_elem = layer.weight.element_size()
        w_total = layer.weight.numel()
        w_active = int(torch.count_nonzero(layer.weight))
        total_params += w_total
        active_params += w_active
        dense_bytes += w_total * w_elem
        active_bytes += w_active * w_elem

        if layer.bias is not None:
            b_elem = layer.bias.element_size()
            b_total = layer.bias.numel()
            b_active = int(torch.count_nonzero(layer.bias))
            total_params += b_total
            active_params += b_active
            dense_bytes += b_total * b_elem
            active_bytes += b_active * b_elem

    elem_size = layers[0].weight.element_size()
    peak_act_bytes = max(l.in_features + l.out_features for l in layers) * elem_size

    norm_bytes = 0
    if isinstance(env, VecNormalize) and env.obs_rms is not None:
        norm_bytes = 2 * int(env.obs_rms.mean.size) * 4  # mean + var as float32

    dense_kb = dense_bytes / 1024.0
    active_kb = active_bytes / 1024.0
    norm_kb = norm_bytes / 1024.0
    peak_act_kb = peak_act_bytes / 1024.0

    return {
        "total_params": total_params,
        "active_params": active_params,
        "dense_kb": dense_kb,
        "active_kb": active_kb,
        "norm_kb": norm_kb,
        "deploy_flash_kb": active_kb + norm_kb,
        "peak_act_kb": peak_act_kb,
    }


def evaluate(model: PPO, env, seeds: list[int] | None = None) -> float:
    seeds = EVAL_SEEDS if seeds is None else seeds
    total = 0.0
    for s in seeds:
        env.seed(s)                              # takes effect on the next reset
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            total += float(reward[0])
            done = bool(done_arr[0])
    return total / len(seeds)


def prune_model(model: PPO, sparsities: list[float]) -> None:
    """Prune actor layers in place and permanently commit zero weights."""
    for layer, s in zip(get_actor_layers(model.policy), sparsities):
        if s > 0.0:
            prune.l1_unstructured(layer, name="weight", amount=s)
            prune.remove(layer, "weight")


def objective(trial: optuna.Trial):
    env, base_model = get_ctx()
    candidate = copy.deepcopy(base_model)
    n_layers = len(get_actor_layers(candidate.policy))
    sparsities = [
        trial.suggest_float(f"prune_layer_{i}", 0.0, 0.95, step=0.05)
        for i in range(n_layers)
    ]
    prune_model(candidate, sparsities)
    fp = calc_memory_footprint(candidate, env)
    trial.set_user_attr("active_params", fp["active_params"])
    trial.set_user_attr("deploy_flash_kb", round(float(fp["deploy_flash_kb"]), 4))
    return evaluate(candidate, env), float(fp["active_kb"])


def inspect_pruned_weights(trial: optuna.trial.FrozenTrial) -> None:
    _, base_model = get_ctx()
    model = copy.deepcopy(base_model)
    layers = get_actor_layers(model.policy)
    labels = get_layer_labels(len(layers))
    sparsities = [trial.params[f"prune_layer_{i}"] for i in range(len(layers))]
    prune_model(model, sparsities)

    for idx, layer in enumerate(layers):
        pruned = (layer.weight == 0).nonzero(as_tuple=False).tolist()
        total = layer.weight.numel()
        print(
            f"Layer {idx} ({labels[idx]}) | Shape: {tuple(layer.weight.shape)} "
            f"| Kept: {total - len(pruned)}/{total}"
        )
        print(f"  Pruned [out, in] (first 10): {pruned[:10]}\n")


def export_study_to_csv(
    study: optuna.Study,
    run_name: str,
    model_path: str,
    arch: str,
    base_reward: float,
    base_fp: dict[str, float | int],
    csv_path: Path = CSV_PATH,
) -> None:
    """Write baseline + all completed trials for run_name to its local CSV."""
    fieldnames = [
        "timestamp",
        "run_name",
        "model_path",
        "arch",
        "n_eval_seeds",
        "trial_number",
        "is_baseline",
        "is_pareto",
        "reward",
        "reward_ratio_to_base",
        "size_kb",
        "size_ratio_to_base",
        "memory_saved_kb",
        "deploy_flash_kb",
        "peak_act_kb",
        "active_params",
        "total_params",
        "sparsity_pct",
        "layer_sparsities",
    ]

    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            existing_rows = [
                row for row in csv.DictReader(f) if row["run_name"] != run_name
            ]

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pareto_ids = {t.number for t in study.best_trials}
    base_kb = float(base_fp["dense_kb"])
    base_params = int(base_fp["total_params"])
    norm_kb = float(base_fp["norm_kb"])
    peak_act_kb = float(base_fp["peak_act_kb"])

    new_rows = [
        {
            "timestamp": ts,
            "run_name": run_name,
            "model_path": model_path,
            "arch": arch,
            "n_eval_seeds": len(EVAL_SEEDS),
            "trial_number": -1,
            "is_baseline": True,
            "is_pareto": False,
            "reward": round(base_reward, 4),
            "reward_ratio_to_base": 1.0,
            "size_kb": round(base_kb, 4),
            "size_ratio_to_base": 1.0,
            "memory_saved_kb": 0.0,
            "deploy_flash_kb": round(base_kb + norm_kb, 4),
            "peak_act_kb": round(peak_act_kb, 4),
            "active_params": base_params,
            "total_params": base_params,
            "sparsity_pct": 0.0,
            "layer_sparsities": json.dumps([]),
        }
    ]

    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        reward, size_kb = t.values
        active = int(t.user_attrs.get("active_params", round(size_kb * 1024 / 4)))
        deploy_flash_kb = float(t.user_attrs.get("deploy_flash_kb", size_kb + norm_kb))
        layer_keys = sorted(
            (k for k in t.params if k.startswith("prune_layer_")),
            key=lambda k: int(k.split("_")[-1]),
        )
        sparsities = [t.params[k] for k in layer_keys]
        new_rows.append(
            {
                "timestamp": ts,
                "run_name": run_name,
                "model_path": model_path,
                "arch": arch,
                "n_eval_seeds": len(EVAL_SEEDS),
                "trial_number": t.number,
                "is_baseline": False,
                "is_pareto": t.number in pareto_ids,
                "reward": round(reward, 4),
                "reward_ratio_to_base": round(reward / base_reward, 4) if base_reward else 0.0,
                "size_kb": round(size_kb, 4),
                "size_ratio_to_base": round(size_kb / base_kb, 4) if base_kb else 0.0,
                "memory_saved_kb": round(base_kb - size_kb, 4),
                "deploy_flash_kb": round(deploy_flash_kb, 4),
                "peak_act_kb": round(peak_act_kb, 4),
                "active_params": active,
                "total_params": base_params,
                "sparsity_pct": round((1.0 - active / base_params) * 100, 2),
                "layer_sparsities": json.dumps(sparsities),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows + new_rows)


if __name__ == "__main__":
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    env, base_model = get_ctx()
    arch = get_arch_str(base_model.policy)
    base_fp = calc_memory_footprint(base_model, env)
    base_reward = evaluate(base_model, env)

    print(
        f"BASELINE [{RUN_NAME} | {arch}]\n"
        f"  Reward:          {base_reward:.2f}\n"
        f"  Actor weights:   {base_fp['dense_kb']:.2f} KB ({base_fp['total_params']} params)\n"
        f"  VecNorm stats:   {base_fp['norm_kb']:.2f} KB\n"
        f"  Deploy Flash:    {base_fp['deploy_flash_kb']:.2f} KB (weights + norm)\n"
        f"  Peak Act. SRAM:  {base_fp['peak_act_kb']:.2f} KB (widest layer I/O)\n"
    )

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{DB_PATH}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    study = optuna.create_study(
        study_name=RUN_NAME,
        storage=storage,
        load_if_exists=True,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(
            population_size=20,
            mutation_prob=None,
            crossover_prob=0.9,
            seed=42,
        ),
    )
    study.set_metric_names(["Reward", "Memory Footprint (KB)"])
    study.set_user_attr("arch", arch)
    study.set_user_attr("model_path", MODEL_PATH)
    study.set_user_attr("base_reward", round(base_reward, 4))
    study.set_user_attr("base_params", int(base_fp["total_params"]))
    study.set_user_attr("base_size_kb", round(float(base_fp["dense_kb"]), 4))
    study.set_user_attr("base_deploy_flash_kb", round(float(base_fp["deploy_flash_kb"]), 4))
    study.set_user_attr("base_peak_act_kb", round(float(base_fp["peak_act_kb"]), 4))

    if INCLUDE_BASELINE_TRIAL and len(study.trials) == 0:
        n_layers = len(get_actor_layers(base_model.policy))
        study.enqueue_trial({f"prune_layer_{i}": 0.0 for i in range(n_layers)})

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

    export_study_to_csv(study, RUN_NAME, MODEL_PATH, arch, base_reward, base_fp)
    print(f"\nSaved study '{RUN_NAME}' -> {DB_PATH} & {CSV_PATH}")

    print("\n--- PARETO OPTIMAL PRUNED MODELS ---")
    for t in sorted(study.best_trials, key=lambda x: x.values[0], reverse=True):
        saved_kb = float(base_fp["dense_kb"]) - t.values[1]
        print(
            f"Trial #{t.number:03d} | Size: {t.values[1]:.2f} KB (saved {saved_kb:.2f} KB) "
            f"| Reward: {t.values[0]:.2f} | {t.params}"
        )

    if study.best_trials:
        inspect_pruned_weights(min(study.best_trials, key=lambda x: x.values[1]))