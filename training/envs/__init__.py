from gymnasium.envs.registration import register
from .pendulum_env import FurutaEnv

# Register the environment with Gymnasium so it can be instantiated via gym.make()
register(
    # The unique string identifier used to call the environment
    id='FurutaPendulum-v0',
    
    # The absolute import path to the class in the format 'module:ClassName'
    entry_point='training.envs.pendulum_env:FurutaEnv',
    
    # Automatically truncate the episode after this many steps to prevent infinite loops during training
    max_episode_steps=1000,
)