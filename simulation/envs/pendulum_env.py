import os
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

import stepper_motor


def get_default_xml_path() -> str:
    """
    Dynamically resolves the absolute path to world_with_pendulum.xml 
    by searching upwards for the repository root directory.
    Works across any machine and any terminal working directory.
    """
    # Get the absolute path of the current Python script
    current_file = Path(__file__).resolve()
    
    # Define the relative path from the repository root to the target XML
    rel_xml_path = Path("simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml")
    
    # Traverse parent directories upwards to find where the 'simulation' folder lives
    for parent in current_file.parents:
        candidate = parent / rel_xml_path
        if candidate.exists():
            return str(candidate)
            
    # Fallback in case the file is not found during upward traversal
    return str(rel_xml_path.resolve())

class FurutaEnv(gym.Env):
    """
    Custom Gymnasium environment for the Furuta pendulum in MuJoCo,
    coupled with the physical NEMA 17 stepper motor model.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(
        self, 
        xml_path=None,
        frame_skip=20,
        render_mode=None
    ):
        super().__init__()
        
        # Resolve XML path dynamically if no custom path was provided
        if xml_path is None:
            xml_path = get_default_xml_path()
            
        # Verify XML existence and load MuJoCo physics structures
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Number of physics substeps (0.5 ms each) per single Gym control step
        # frame_skip=20 gives a 10 ms (100 Hz) control loop interval
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        
        # Initialize the electromechanical stepper motor model with MuJoCo's timestep
        self.stepper = stepper_motor.StepperMotor(self.model.opt.timestep)
        
        # Action space: Target angular velocity for the stepper motor [rad/s]
        self.max_velocity = stepper_motor.MAX_SPEED
        self.action_space = spaces.Box(
            low=-self.max_velocity, 
            high=self.max_velocity, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space:
        # 0: sin(pendulum_angle), 1: cos(pendulum_angle), 2: pendulum_vel,
        # 3: motor_pos, 4: motor_vel
        high = np.array([1.0, 1.0, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, 
            high=high, 
            dtype=np.float32
        )
        
        self.viewer = None

    def step(self, action):
        """
        Executes one control step (100 Hz) by running 'frame_skip' physics and 
        stepper motor substeps (2000 Hz) in MuJoCo.
        """
        # Extract scalar target velocity command from action array
        target_vel = float(np.clip(action[0], -self.max_velocity, self.max_velocity))
        
        # Run high-frequency physics and electromagnetic simulation loop
        for _ in range(self.frame_skip):
            # Calculate physical torque from the stepper motor model
            tau = self.stepper.compute_torque(
                target_vel=target_vel,
                theta=self.data.qpos[0],
                omega=self.data.qvel[0]
            )
            # Apply computed torque [N*m] to the MuJoCo actuator
            self.data.ctrl[0] = tau
            # Step MuJoCo physics forward by 0.0005 s
            mujoco.mj_step(self.model, self.data)
            
            # Stop substepping immediately if the stepper motor loses synchronization
            if self.stepper.stalled:
                break
        
        # Collect current observations
        obs = self._get_obs()
        
        # Terminate episode early if the stepper motor stalled (lost steps)
        terminated = bool(self.stepper.stalled) or abs(self.data.qpos[0]) > 2 * np.pi
        truncated = False
        
        # Compute reward (includes heavy penalty if stalled)
        reward = self._compute_reward(obs, target_vel, terminated)
        
        # Diagnostic metrics useful for debugging and Optuna logging
        info = {
            "motor_lag_deg": float(np.rad2deg(self.stepper.lag)),
            "stalled": terminated,
            "applied_torque": float(self.data.ctrl[0])
        }
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets MuJoCo physics and synchronizes the internal stepper motor state.
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo data to initial XML state
        mujoco.mj_resetData(self.model, self.data)
        
        # Add slight random perturbation to initial positions and velocities
        qpos_noise = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        qvel_noise = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)
        
        self.data.qpos[:] = self.model.qpos0 + qpos_noise
        self.data.qvel[:] = qvel_noise
        
        # Synchronize stepper motor magnetic field with the randomized starting arm angle
        self.stepper.reset(initial_theta=self.data.qpos[0])
        
        # Update MuJoCo kinematics
        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Extracts state variables from MuJoCo qpos and qvel arrays.
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

    def _compute_reward(self, obs, target_vel, stalled):
        """
        Calculates step reward: rewards upright pendulum posture (angle = pi) 
        while penalizing excessive velocities, arm drift, and stepper motor stall events.
        """
        # Heavy penalty if the controller forces the stepper to lose steps
        if stalled:
            return -50.0
            
        cos_theta = obs[1]
        pendulum_vel = obs[2]
        motor_pos = obs[3]
        motor_vel = obs[4]
        
        # Since 0 rad is hanging down (cos(0) = 1) and pi rad is balanced up (cos(pi) = -1),
        # negating cos_theta yields +1.0 at the top and -1.0 at the bottom.
        reward_upright = -cos_theta
        
        penalty_pendulum_vel = 0.05 * abs(pendulum_vel)
        penalty_motor_vel = 0.01 * abs(motor_vel)
        penalty_motor_pos = 0.01 * abs(motor_pos)
        penalty_action = 0.01 * abs(target_vel)
        
        return float(
            reward_upright 
            - penalty_pendulum_vel 
            - penalty_motor_pos 
            - penalty_motor_vel 
            - penalty_action
        )

    def render(self):
        """
        Launches and updates the interactive MuJoCo 3D viewer window.
        """
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        """
        Closes the MuJoCo viewer cleanly.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ---- Quick verification block ---------------------------------------------
if __name__ == "__main__":
    # Initialize environment with 3D visualization enabled
    env = FurutaEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Loaded XML successfully from: {get_default_xml_path()}")
    
    # Run a 5-second test (500 steps at 100 Hz) with a smooth sinusoidal velocity command
    for step_idx in range(500):
        t = step_idx * 0.01
        action = np.array([4.0 * np.sin(2.0 * np.pi * 1.0 * t)], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.01)
        
        if terminated:
            print(f"Stepper motor stalled at step {step_idx}! Resetting...")
            obs, info = env.reset()
            
    env.close()