import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import simulation.envs

def main():
    env_id = "FurutaPendulum-v0"
    model_path = "ppo_furuta_pendulum"
    stats_path = "vec_normalize.pkl"

    # Wrap the environment in a list to be compatible with DummyVecEnv
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])
    
    try:
        # Load normalization stats from the training phase
        eval_env = VecNormalize.load(stats_path, eval_env)
    except FileNotFoundError:
        raise FileNotFoundError(f"Normalization file '{stats_path}' not found. Train the model first.")

    # Freeze running stats so the agent does not alter its perception during evaluation
    eval_env.training = False
    eval_env.norm_reward = False

    try:
        # Load the trained model architecture and weights
        model = PPO.load(model_path, env=eval_env)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{model_path}.zip' not found.")

    # Reset environment to get initial observation and spawn the window
    obs = eval_env.reset()
    
    # Extract the internal physics timestep (dt) directly from the unwrapped MuJoCo environment
    try:
        dt = eval_env.envs[0].unwrapped.dt
    except AttributeError:
        # Fallback to standard 0.02s if the property is somehow unavailable
        dt = 0.02
        
    print(f"Physics timestep (dt) detected as: {dt} seconds.")
    print("Pumping window events to ensure smooth startup...")
    
    # Warm-up loop: Force the OS window to render without advancing the physics simulation
    for _ in range(60):
        eval_env.render()
        time.sleep(1.0 / 60.0)

    print("Starting visual evaluation. Press Ctrl+C to stop.")
    
    step_count = 0
    
    try:
        while True:
            # High-precision timer start
            step_start = time.perf_counter()
            
            # Predict best deterministic action and step the physics engine
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            # Measure elapsed execution time
            compute_time = time.perf_counter() - step_start
            
            # If the CPU was faster than the physics timestep, sleep the exact difference
            if compute_time < dt:
                time.sleep(dt - compute_time)
                
            # Measure total time taken for the step including sleep
            total_step_time = time.perf_counter() - step_start
            
            # Calculate Real Time Factor (RTF): Simulated Time / Actual Elapsed Time
            rtf = (dt / total_step_time) * 100.0
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"Step: {step_count} | RTF: {rtf:.1f}%")
            
            if done[0]:
                print("Episode finished, resetting environment...")
                obs = eval_env.reset()
                
                # Render startup frames again to prevent stutter on reset
                for _ in range(30):
                    eval_env.render()
                    time.sleep(1.0 / 60.0)
                
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        eval_env.close()

if __name__ == "__main__":
    main()