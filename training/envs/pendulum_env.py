import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class FurutaEnv(gym.Env):
    """
    Custom Gymnasium environment for the rotary inverted pendulum using MuJoCo.
    The agent controls the stepper motor velocity to swing up and balance the pendulum.
    """
    
    def __init__(self, xml_path="../simulation/pendulum.xml"):
        super().__init__()
        
        # Load the MuJoCo model and data from the XML file
        # The path should be relative to where the execution script is run
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Action space: Continuous target velocity for the stepper motor
        # Assuming maximum motor velocity is 15.0 rad/s
        self.max_velocity = 15.0
        self.action_space = spaces.Box(
            low=-self.max_velocity, 
            high=self.max_velocity, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space defines what the neural network sees
        # 0: sin(pendulum_angle) - prevents discontinuous jump from pi to -pi
        # 1: cos(pendulum_angle) - represents height of the pendulum
        # 2: pendulum_velocity - speed of the pendulum swing
        # 3: motor_position - position of the main arm
        # 4: motor_velocity - current speed of the main arm
        high = np.array([1.0, 1.0, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, 
            high=high, 
            dtype=np.float32
        )
        
        # Viewer for rendering the 3D scene (initialized on demand)
        self.viewer = None

    def step(self, action):
        """
        Applies the action to the simulation, advances one step, 
        and calculates the resulting state and reward.
        """
        # Apply the target velocity from the agent to the MuJoCo actuator (motor)
        self.data.ctrl[0] = action[0]
        
        # Advance the simulation physics by one time step
        mujoco.mj_step(self.model, self.data)
        
        # Retrieve the new state from the physics engine
        obs = self._get_obs()
        
        # Calculate the reward based on the new state and the applied action
        reward = self._compute_reward(obs, action)
        
        # Determine if the episode should terminate
        # We leave this false for continuous balancing; episode limits are usually handled by TimeLimit wrapper
        terminated = False
        truncated = False 
        
        # Additional diagnostic information
        info = {}
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo state structures to their default XML values
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random noise to break symmetry and improve training robustness
        qpos_noise = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        qvel_noise = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)
        
        # Apply noise to positions and velocities
        self.data.qpos[:] = self.model.qpos0 + qpos_noise
        self.data.qvel[:] = qvel_noise
        
        # Run forward kinematics to update derived quantities (e.g., sensor readings)
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = {}
        
        return obs, info

    def _get_obs(self):
        """
        Extracts specific sensor variables from the MuJoCo state.
        Assumes qpos[0] is motor angle, qpos[1] is pendulum angle.
        """
        motor_pos = self.data.qpos[0]
        pendulum_angle = self.data.qpos[1]
        
        motor_vel = self.data.qvel[0]
        pendulum_vel = self.data.qvel[1]
        
        return np.array([
            np.sin(pendulum_angle),
            np.cos(pendulum_angle),
            pendulum_vel,
            motor_pos,
            motor_vel
        ], dtype=np.float32)

    def _compute_reward(self, obs, action):
        """
        Calculates the fitness score for the current state.
        The goal is to keep the pendulum upright with minimal movement.
        """
        cos_theta = obs[1]
        pendulum_vel = obs[2]
        motor_pos = obs[3]
        motor_vel = obs[4]
        
        # Reward for being upright (cos(theta) == 1 when fully upright)
        reward_upright = cos_theta 
        
        # Penalties for excessive speeds, motor displacement and large action outputs
        penalty_pendulum_vel = 0.05 * abs(pendulum_vel)
        penalty_motor_vel = 0.01 * abs(motor_vel)
        penalty_motor_pos = 0.01 * abs(motor_pos)
        penalty_action = 0.01 * abs(action[0])
        
        return float(reward_upright - penalty_pendulum_vel - penalty_motor_pos - penalty_motor_vel - penalty_action)

    def close(self):
        """
        Cleans up resources.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None