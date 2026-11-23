import copy
import threading

import gymnasium as gym
import optuna
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import simulation.envs

ENV_ID = "FurutaPendulum-v0"
MODEL_PATH = "training/logs/good_model/ppo_furuta_pendulum"
STATS_PATH = "training/logs/good_model/vec_normalize.pkl"
EVAL_SEEDS = list(range(8))          # same start states for every candidate
LAYER_NAMES = ["Input -> Hidden 1", "Hidden 1 -> Hidden 2", "Hidden 2 -> Output"]

torch.set_num_threads(1)             # avoid n_jobs x torch internal threads
_local = threading.local()


def get_ctx():
    """Per-thread env + base model (built lazily on first use)."""
    if not hasattr(_local, "env"):
        env = DummyVecEnv([lambda: gym.make(ENV_ID)])
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False        # freeze obs normalization
        env.norm_reward = False     # true unscaled reward
        _local.env = env
        _local.model = PPO.load(MODEL_PATH, env=env, device="cpu")
    return _local.env, _local.model


def get_actor_layers(policy) -> list[nn.Linear]:
    """Actor layers only (critic is discarded at deployment)."""
    modules = list(policy.mlp_extractor.policy_net.modules()) + [policy.action_net]
    return [m for m in modules if isinstance(m, nn.Linear)]


def evaluate(model: PPO, env, seeds=[1,2,3]) -> float:
    total = 0.0
    for s in seeds:
        env.seed(s)                  # takes effect on the next reset
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            total += float(reward[0])
            done = bool(done_arr[0])
    return total / len(seeds)


def prune_model(model: PPO, sparsities) -> int:
    """Prune actor layers in place, return surviving parameter count."""
    active = 0
    for layer, s in zip(get_actor_layers(model.policy), sparsities):
        if s > 0.0:
            prune.l1_unstructured(layer, name="weight", amount=s)
            prune.remove(layer, "weight")    # commit zeros permanently
        active += int(torch.count_nonzero(layer.weight))
        if layer.bias is not None:
            active += int(torch.count_nonzero(layer.bias))
    return active


def objective(trial: optuna.Trial):
    env, base_model = get_ctx()
    candidate = copy.deepcopy(base_model)
    sparsities = [
        trial.suggest_float(f"prune_layer_{i}", 0.0, 0.95, step=0.05)
        for i in range(3)
    ]
    active = prune_model(candidate, sparsities)
    return evaluate(candidate, env), active * 4 / 1024.0


def inspect_pruned_weights(trial):
    _, base_model = get_ctx()
    model = copy.deepcopy(base_model)
    sparsities = [trial.params[f"prune_layer_{i}"] for i in range(3)]
    prune_model(model, sparsities)

    for idx, layer in enumerate(get_actor_layers(model.policy)):
        pruned = (layer.weight == 0).nonzero(as_tuple=False).tolist()
        total = layer.weight.numel()
        print(f"Layer {idx} ({LAYER_NAMES[idx]}) | Shape: {tuple(layer.weight.shape)} "
              f"| Kept: {total - len(pruned)}/{total}")
        print(f"  Pruned [out, in] (first 10): {pruned[:10]}\n")


if __name__ == "__main__":
    env, base_model = get_ctx()
    base_params = sum(p.numel() for l in get_actor_layers(base_model.policy)
                      for p in l.parameters())
    print(f"BASELINE -> Reward: {evaluate(base_model, env):.2f} "
          f"| Size: {base_params * 4 / 1024:.2f} KB ({base_params} params)\n")

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(
            population_size=5,
            mutation_prob=0.33,
            crossover_prob=0.9,
            seed=42
        ),
    )
    study.optimize(objective, n_trials=20,n_jobs=4)

    print("\n--- PARETO OPTIMAL PRUNED MODELS ---")
    for t in sorted(study.best_trials, key=lambda x: x.values[0],reverse=True):
        print(f"Trial #{t.number:03d} | Size: {t.values[1]:.2f} KB "
              f"| Reward: {t.values[0]:.2f} | {t.params}")
    print(f"BASELINE -> Reward: {evaluate(base_model, env):.2f} "
          f"| Size: {base_params * 4 / 1024:.2f} KB ({base_params} params)\n")

    inspect_pruned_weights(min(study.best_trials, key=lambda x: x.values[1]))