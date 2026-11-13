import gymnasium as gym
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Replace env = gym.make(...) with 16 parallel simulation workers:
# ==========================================
# OPTUNA OBJECTIVE (SEARCH + TRAIN + EVAL)
# ==========================================
def objective(trial: optuna.Trial) -> float:
    # 1. Search Knobs: Depth and 2^n Width
    num_layers = trial.suggest_int("num_layers", 3, 6)
    pwr_hidden = trial.suggest_int("pwr_hidden", 3, 7)  # 2^3=8 up to 2^6=64
    hidden_dim = 2 ** pwr_hidden

    # 2. Configure Dynamic Architecture for Actor & Critic
    net_arch = [hidden_dim] * num_layers
    policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))

    # 3. Instantiate Gym Environment & PPO Agent

    env = make_vec_env("Pendulum-v1", n_envs=128)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        device="cuda",
        n_steps=1024,
        batch_size=4096,
        verbose=0
    )

    # 4. Train the network so it actually learns to balance (~3 sec per trial)
    model.learn(total_timesteps=15_000)

    # 5. Measure Footprint of DEPLOYED policy (Actor only, ignore Critic)
    actor_params = (
        sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters())
        + sum(p.numel() for p in model.policy.action_net.parameters())
    )
    memory_kb = (actor_params * 4) / 1024.0  # FP32 = 4 bytes per weight
    trial.set_user_attr("memory_kb", memory_kb)
    trial.set_user_attr("actor_params", actor_params)

    # 6. Evaluate final balancing performance
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()

    return mean_reward


# ==========================================
# RUN SEARCH
# ==========================================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    print("Searching architectures and training policies...")
    study.optimize(objective, n_trials=50, n_jobs=4)  # 4 concurrent trials on GPU

    best = study.best_trial
    width = 2 ** best.params["pwr_hidden"]
    print("\n--- OPTIMAL BALANCING ARCHITECTURE ---")
    print(f"Optimal Depth:   {best.params['num_layers']} hidden layers")
    print(f"Optimal Width:   {width} neurons/layer")
    print(f"Actor Memory:    {best.user_attrs['memory_kb']:.2f} KB ({best.user_attrs['actor_params']} parameters)")
    print(f"Mean Reward:     {best.value:.2f}")