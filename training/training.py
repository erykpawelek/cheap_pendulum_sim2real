import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import package to trigger Gymnasium environment registration
import simulation.envs


def main():
    env_id = "FurutaPendulum-v0"
    model_path = "ppo_furuta_pendulum"
    stats_path = "vec_normalize.pkl"

    print("Initializing Furuta pendulum environment for compliance check...")
    raw_env = gym.make(env_id)
    check_env(raw_env, warn=True)
    raw_env.close()
    print("Environment check passed successfully!")

    # Create 4 parallel environments to collect diverse trajectories simultaneously
    vec_env = make_vec_env(env_id, n_envs=4, seed=42, vec_env_cls=DummyVecEnv)

    # Normalize observations only; keep our [0.0, 1.0] reward untouched (norm_reward=False)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )

    # Initialize PPO model with 4 * 1024 = 4096 steps per rollout update
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=42,
    )

    print("Starting PPO training process...")
    model.learn(total_timesteps=600_000)
    print("Training finished successfully!")

    # Save trained policy weights and observation normalization statistics
    model.save(model_path)
    vec_env.save(stats_path)
    vec_env.close()
    print(f"Saved model to '{model_path}.zip' and normalization stats to '{stats_path}'.")

    # --- Visual Evaluation ---
    print("\nRunning visual evaluation of the trained PPO policy...")
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])
    
    # Load observation scaling statistics from training and freeze them
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        time.sleep(0.01)
        if done[0]:
            obs = eval_env.reset()

    eval_env.close()
    print("Evaluation completed.")


if __name__ == "__main__":
    main()