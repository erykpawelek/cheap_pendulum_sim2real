import os
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from . import stepper_motor


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
    
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self, 
        xml_path=None,
        frame_skip=40,
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
        Executes one control step (50 Hz) by running 'frame_skip' physics and 
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
        
        # Check physical stall and arm rotation limit separately
        arm_limit_hit = bool(abs(self.data.qpos[0]) > 2 * np.pi)
        terminated = bool(self.stepper.stalled or arm_limit_hit)
        truncated = False

        # Compute reward (returns 0.0 if terminated early)
        reward = self._compute_reward(obs, target_vel, terminated)

        # Report true stepper stall separately from arm angle limit violations
        info = {
            "motor_lag_deg": float(np.rad2deg(self.stepper.lag)),
            "stalled": bool(self.stepper.stalled),
            "arm_limit_hit": arm_limit_hit,
            "applied_torque": float(self.data.ctrl[0]),
        }
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets MuJoCo physics and uses hybrid state initialization to accelerate 
        learning of both upright stabilization and full swing-up dynamics.
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo data to initial XML state
        mujoco.mj_resetData(self.model, self.data)
        
        # Small perturbation for the motor arm position and joint velocities
        qpos_noise = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        qvel_noise = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)
        
        self.data.qpos[:] = self.model.qpos0 + qpos_noise
        
        # Hybrid initialization: 50% near upright equilibrium (pi), 50% full circle
        if self.np_random.random() < 0.5:
            # Start near the top so the agent quickly discovers how to brake and balance
            self.data.qpos[1] = np.pi + self.np_random.uniform(low=-0.3, high=0.3)
        else:
            # Start anywhere to learn the full swing-up trajectory
            self.data.qpos[1] = self.np_random.uniform(low=-np.pi, high=np.pi)
            
        self.data.qvel[:] = qvel_noise
        
        # Synchronize stepper motor magnetic field with the starting arm angle
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
        Calculates a strictly non-negative reward in range [0.0, 1.0].
        Uses tight multiplicative Gaussian kernels to prevent multiple rotations 
        (windmill effect) and force crisp braking at the upright equilibrium.
        """
        # Return 0.0 immediately if the stepper motor loses synchronization
        if stalled:
            return 0.0
            
        cos_theta = obs[1]
        pendulum_vel = obs[2]
        motor_pos = obs[3]
        motor_vel = obs[4]
        
        # 1. Upright posture factor in range [0.0, 1.0]:
        # Hanging down (cos = 1) -> 0.0; Upright (cos = -1) -> 1.0.
        upright_factor = (0.5 * (1.0 - cos_theta)) ** 2
        
        # 2. Strict pendulum velocity damping in range (0.0, 1.0]:
        # Increased decay from -0.08 to -0.3 so passing the top at 3 rad/s 
        # drops the bonus to ~0.06 instead of ~0.48, forcing the agent to brake early.
        vel_damping = np.exp(-0.3 * (pendulum_vel ** 2))
        
        # 3. Smooth control and arm centering factor in range (0.0, 1.0]:
        arm_damping = np.exp(
            -0.05 * (motor_pos ** 2) 
            - 0.005 * (motor_vel ** 2) 
            - 0.002 * (target_vel ** 2)
        )
        
        # Reduced base shaping from 0.2 to 0.05 so spinning through the top gives almost 
        # no free points, while 95% of the reward requires stopping at the upright position.
        reward = 0.05 * upright_factor + 0.95 * (upright_factor * vel_damping * arm_damping)
        
        return float(reward)

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