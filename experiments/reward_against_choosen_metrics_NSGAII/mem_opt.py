import copy
import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
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
SAMPLER = "nsga3"                                # "nsga3" (240+ trials) or "tpe" (80-160 trials)
RUN_NAME = f"{EXP_DIR.name}_4obj_{SAMPLER}"
ENV_ID = "FurutaPendulum-v0"
MODEL_PATH = "training/logs/good_model/ppo_furuta_pendulum"
STATS_PATH = "training/logs/good_model/vec_normalize.pkl"
EVAL_SEEDS = list(range(8))                      # fixed start states for every candidate
N_TRIALS = 240 if SAMPLER == "nsga3" else 120
N_JOBS = 4
INCLUDE_BASELINE_TRIAL = True                    # pins 0% sparsity point as Trial #0

# --- 4 Objectives ---
OBJECTIVES = [
    ("rmse_total_rad",    "minimize", "Whole-Episode RMSE (rad)"),
    ("control_effort",    "minimize", "Control Effort ((rad/s)^2*s)"),
    ("upright_ratio_pct", "maximize", "Time Spent Upright (%)"),
    ("active_kb",         "minimize", "Memory Footprint (KB)"),
]
MIN_UPRIGHT_PCT = 20.0  # constraint: blocks dead u=0 models from gaming control_effort

# --- Control Benchmark Thresholds (matched to evaluate.py) ---
BALANCE_ANGLE_TOL = 0.20  # ~11.5 deg from upright (pi)
BALANCE_VEL_TOL = 1.0     # rad/s max velocity to consider settled

torch.set_num_threads(1)  # avoid n_jobs x torch internal threads
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
    """Fingerprint actor architecture, e.g. '5x64x64x1'."""
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
    """Compute actor weights, VecNormalize stats, deploy flash, and peak activation SRAM."""
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

    return {
        "total_params": total_params,
        "active_params": active_params,
        "dense_kb": dense_kb,
        "active_kb": active_kb,
        "norm_kb": norm_kb,
        "deploy_flash_kb": active_kb + norm_kb,
        "peak_act_kb": peak_act_bytes / 1024.0,
    }


def evaluate(model: PPO, env, seeds: list[int] | None = None) -> dict[str, float]:
    """Run deterministic rollouts and return aggregated Furuta control metrics."""
    seeds = EVAL_SEEDS if seeds is None else seeds
    try:
        dt = float(env.envs[0].unwrapped.dt)
    except AttributeError:
        dt = 0.02

    min_settled_steps = int(1.0 / dt)
    returns, rmse_total, rmse_balanced = [], [], []
    upright_ratios, iae_totals, control_efforts = [], [], []
    stalls = successes = 0

    for s in seeds:
        env.seed(s)                              # takes effect on the next reset
        obs = env.reset()
        done = False

        ep_return = 0.0
        ep_steps = 0
        upright_steps = 0
        settled_streak = 0
        settled = False
        stalled = False

        angle_errors_sq = []
        balanced_errors_sq = []
        iae_accum = 0.0
        effort_accum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            done = bool(done_arr[0])

            # VecEnv auto-resets on done=True; grab true terminal obs before reset
            if done and "terminal_observation" in info[0]:
                raw_obs = np.asarray(
                    env.unnormalize_obs(info[0]["terminal_observation"])
                ).reshape(-1)
            else:
                raw_obs = np.asarray(env.get_original_obs()[0]).reshape(-1)

            sin_th, cos_th, pend_vel = raw_obs[0], raw_obs[1], raw_obs[2]
            angle_err = float(abs(np.arctan2(sin_th, -cos_th)))
            act_val = float(action[0][0]) if np.ndim(action) > 1 else float(action[0])

            ep_return += float(reward[0])
            ep_steps += 1
            angle_errors_sq.append(angle_err ** 2)
            iae_accum += angle_err * dt
            effort_accum += (act_val ** 2) * dt

            if angle_err < BALANCE_ANGLE_TOL:
                upright_steps += 1
                balanced_errors_sq.append(angle_err ** 2)

            if angle_err < BALANCE_ANGLE_TOL and abs(pend_vel) < BALANCE_VEL_TOL:
                settled_streak += 1
                if settled_streak >= min_settled_steps:
                    settled = True
            else:
                settled_streak = 0

            if info[0].get("stalled", False) or (
                done and not info[0].get("TimeLimit.truncated", False)
            ):
                stalled = True

        returns.append(ep_return)
        rmse_total.append(float(np.sqrt(np.mean(angle_errors_sq))))
        rmse_balanced.append(
            float(np.sqrt(np.mean(balanced_errors_sq))) if balanced_errors_sq else float(np.pi)
        )
        upright_ratios.append((upright_steps / ep_steps) * 100.0)
        iae_totals.append(iae_accum)
        control_efforts.append(effort_accum)
        if stalled:
            stalls += 1
        if settled and not stalled:
            successes += 1

    n = len(seeds)
    return {
        "return": float(np.mean(returns)),
        "upright_ratio_pct": float(np.mean(upright_ratios)),
        "rmse_total_rad": float(np.mean(rmse_total)),
        "rmse_balanced_rad": float(np.mean(rmse_balanced)),
        "iae_total_rad_s": float(np.mean(iae_totals)),
        "control_effort": float(np.mean(control_efforts)),
        "stall_rate_pct": (stalls / n) * 100.0,
        "success_rate_pct": (successes / n) * 100.0,
    }


def prune_model(model: PPO, sparsities: list[float]) -> None:
    """Prune actor layers in place and permanently commit zero weights."""
    for layer, s in zip(get_actor_layers(model.policy), sparsities):
        if s > 0.0:
            prune.l1_unstructured(layer, name="weight", amount=s)
            prune.remove(layer, "weight")


def constraints_func(trial: optuna.trial.FrozenTrial) -> list[float]:
    """<= 0 is feasible; > 0 marks collapsed controllers as infeasible."""
    return trial.user_attrs.get("constraints", [0.0])


def build_sampler(name: str) -> optuna.samplers.BaseSampler:
    if name == "nsga3":
        # 4 objectives + dividing_parameter=3 -> 20 reference points; pop=24 aligns with n_jobs=4
        return optuna.samplers.NSGAIIISampler(
            population_size=24,
            dividing_parameter=3,
            mutation_prob=None,       # auto-sets 1 / n_layers
            crossover_prob=0.9,
            constraints_func=constraints_func,
            seed=42,
        )
    if name == "tpe":
        return optuna.samplers.TPESampler(
            n_startup_trials=24,
            multivariate=True,        # captures layer-to-layer pruning interactions
            constant_liar=True,       # avoids duplicate suggestions across n_jobs=4
            constraints_func=constraints_func,
            seed=42,
        )
    raise ValueError(f"Unknown SAMPLER '{name}'. Expected 'nsga3' or 'tpe'.")


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
    metrics = evaluate(candidate, env)

    constraint_val = MIN_UPRIGHT_PCT - metrics["upright_ratio_pct"]
    trial.set_user_attr("active_params", int(fp["active_params"]))
    trial.set_user_attr("deploy_flash_kb", round(float(fp["deploy_flash_kb"]), 4))
    trial.set_user_attr("constraints", [round(constraint_val, 4)])
    trial.set_user_attr("is_feasible", constraint_val <= 0.0)
    for k, v in metrics.items():
        trial.set_user_attr(k, round(v, 4))

    scores = {**metrics, "active_kb": float(fp["active_kb"])}
    return tuple(scores[key] for key, _, _ in OBJECTIVES)


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
    base_metrics: dict[str, float],
    base_fp: dict[str, float | int],
    csv_path: Path = CSV_PATH,
) -> None:
    """Upsert baseline + all completed trials for run_name to its local CSV."""
    fieldnames = [
        "timestamp",
        "run_name",
        "sampler",
        "model_path",
        "arch",
        "n_eval_seeds",
        "trial_number",
        "is_baseline",
        "is_feasible",
        "is_pareto",
        "rmse_total_rad",
        "control_effort",
        "upright_ratio_pct",
        "rmse_balanced_rad",
        "iae_total_rad_s",
        "return",
        "stall_rate_pct",
        "success_rate_pct",
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
            reader = csv.DictReader(f)
            if reader.fieldnames == fieldnames:
                existing_rows = [r for r in reader if r["run_name"] != run_name]

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
            "sampler": SAMPLER,
            "model_path": model_path,
            "arch": arch,
            "n_eval_seeds": len(EVAL_SEEDS),
            "trial_number": -1,
            "is_baseline": True,
            "is_feasible": base_metrics["upright_ratio_pct"] >= MIN_UPRIGHT_PCT,
            "is_pareto": False,
            **{k: round(v, 4) for k, v in base_metrics.items()},
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
        size_kb = float(t.values[3])
        active = int(t.user_attrs.get("active_params", round(size_kb * 1024 / 4)))
        deploy_flash_kb = float(t.user_attrs.get("deploy_flash_kb", size_kb + norm_kb))
        layer_keys = sorted(
            (k for k in t.params if k.startswith("prune_layer_")),
            key=lambda k: int(k.split("_")[-1]),
        )
        sparsities = [t.params[k] for k in layer_keys]
        metric_cols = {k: t.user_attrs.get(k, "") for k in base_metrics}

        new_rows.append(
            {
                "timestamp": ts,
                "run_name": run_name,
                "sampler": SAMPLER,
                "model_path": model_path,
                "arch": arch,
                "n_eval_seeds": len(EVAL_SEEDS),
                "trial_number": t.number,
                "is_baseline": False,
                "is_feasible": bool(t.user_attrs.get("is_feasible", True)),
                "is_pareto": t.number in pareto_ids,
                **metric_cols,
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
    base_metrics = evaluate(base_model, env)

    print(
        f"BASELINE [{RUN_NAME} | {arch} | sampler={SAMPLER}]\n"
        f"  Time Upright (<11 deg): {base_metrics['upright_ratio_pct']:.1f} %\n"
        f"  Whole-Episode RMSE:     {base_metrics['rmse_total_rad']:.3f} rad\n"
        f"  Balanced Zone RMSE:     {base_metrics['rmse_balanced_rad']:.3f} rad\n"
        f"  IAE:                    {base_metrics['iae_total_rad_s']:.2f} rad*s\n"
        f"  Control Effort:         {base_metrics['control_effort']:.2f} (rad/s)^2*s\n"
        f"  Mean Return / Stalls:   {base_metrics['return']:.2f} | {base_metrics['stall_rate_pct']:.1f}%\n"
        f"  Actor Weights:          {base_fp['dense_kb']:.2f} KB ({base_fp['total_params']} params)\n"
        f"  VecNorm / Deploy Flash: {base_fp['norm_kb']:.2f} KB | {base_fp['deploy_flash_kb']:.2f} KB\n"
        f"  Peak Activation SRAM:   {base_fp['peak_act_kb']:.2f} KB\n"
    )
    if base_metrics["upright_ratio_pct"] < MIN_UPRIGHT_PCT:
        print(
            f"[WARN] Baseline upright ratio ({base_metrics['upright_ratio_pct']:.1f}%) "
            f"is below MIN_UPRIGHT_PCT ({MIN_UPRIGHT_PCT:.1f}%). Lower MIN_UPRIGHT_PCT.\n"
        )

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{DB_PATH}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    study = optuna.create_study(
        study_name=RUN_NAME,
        storage=storage,
        load_if_exists=True,
        directions=[d for _, d, _ in OBJECTIVES],
        sampler=build_sampler(SAMPLER),
    )
    study.set_metric_names([label for _, _, label in OBJECTIVES])
    study.set_user_attr("arch", arch)
    study.set_user_attr("sampler", SAMPLER)
    study.set_user_attr("model_path", MODEL_PATH)
    study.set_user_attr("base_params", int(base_fp["total_params"]))
    study.set_user_attr("base_size_kb", round(float(base_fp["dense_kb"]), 4))
    study.set_user_attr("base_deploy_flash_kb", round(float(base_fp["deploy_flash_kb"]), 4))
    study.set_user_attr("base_peak_act_kb", round(float(base_fp["peak_act_kb"]), 4))
    for k, v in base_metrics.items():
        study.set_user_attr(f"base_{k}", round(v, 4))

    if INCLUDE_BASELINE_TRIAL and len(study.trials) == 0:
        n_layers = len(get_actor_layers(base_model.policy))
        study.enqueue_trial({f"prune_layer_{i}": 0.0 for i in range(n_layers)})

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

    export_study_to_csv(study, RUN_NAME, MODEL_PATH, arch, base_metrics, base_fp)
    print(f"\nSaved study '{RUN_NAME}' -> {DB_PATH} & {CSV_PATH}")

    feasible_pareto = [
        t for t in study.best_trials if t.user_attrs.get("is_feasible", True)
    ]
    display_trials = feasible_pareto or study.best_trials

    print(f"\n--- PARETO OPTIMAL PRUNED MODELS ({len(display_trials)} shown) ---")
    for t in sorted(display_trials, key=lambda x: x.values[0]):
        rmse, effort, upright, size_kb = t.values
        saved_kb = float(base_fp["dense_kb"]) - size_kb
        print(
            f"Trial #{t.number:03d} | Size: {size_kb:.2f} KB (saved {saved_kb:.2f} KB) "
            f"| RMSE: {rmse:.3f} | Effort: {effort:.2f} | Upright: {upright:.1f}% "
            f"| {t.params}"
        )

    if display_trials:
        inspect_pruned_weights(min(display_trials, key=lambda x: x.values[3]))