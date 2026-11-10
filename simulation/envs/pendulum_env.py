import os
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


# ---- 17HS4401 datasheet constants ----------------------------------------
T_HOLD    = 0.40                    # Holding torque [N*m] (40 N*cm)
T_DETENT  = 0.022                   # Detent / cogging torque [N*m] (2.2 N*cm)
R_PH      = 1.5                     # Phase resistance [Ohm]
L_PH      = 2.8e-3                  # Phase inductance [H]
I_RATED   = 1.5                     # Rated current per phase [A]
FULL_STEP = np.deg2rad(1.8)         # Full step angle in radians (1.8 deg)
N_TEETH   = 50                      # Rotor teeth count (200 steps/rev = 4 * 50)
KE        = T_HOLD / (np.sqrt(2) * I_RATED)  # Back-EMF constant [V*s/rad]

# ---- Stepper driver parameters -------------------------------------------
V_SUPPLY  = 24.0                    # Driver supply voltage [V]
MICROSTEP = 16                      # Microstepping subdivision
MAX_SPEED = 15.0                    # Max commanded speed [rad/s]
MAX_ACCEL = 300.0                   # Max commanded acceleration [rad/s^2]
KD        = 0.02                    # Electrical damping coefficient [N*m*s/rad]


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
    for parent in [current_file.parent, *current_file.parents]:
        candidate = parent / rel_xml_path
        if candidate.exists():
            return str(candidate)
            
    # Fallback in case the file is not found during upward traversal
    return str(rel_xml_path.resolve())


class StepperMotor:
    """
    Electromechanical model of the Hanpose 17HS4401 (NEMA 17) stepper motor.
    Operates in velocity-command mode suitable for RL / evolutionary control.
    """

    def __init__(self, dt):
        # Simulation physics timestep (e.g., 0.0005 s)
        self.dt = dt
        # Angle of a single microstep in radians
        self.step = FULL_STEP / MICROSTEP
        
        # Internal state variables initialized via reset()
        self.cmd = 0.0
        self.cmd_v = 0.0
        self.lag = 0.0
        self.stalled = False

    def reset(self, initial_theta=0.0):
        """
        Synchronizes the internal magnetic field angle with the physical rotor 
        angle at the beginning of each Gymnasium episode.
        """
        self.cmd = float(initial_theta)
        self.cmd_v = 0.0
        self.lag = 0.0
        self.stalled = False

    def torque_limit(self, w):
        """
        Calculates available pull-out torque amplitude at rotor speed 'w' [rad/s]
        using the RL impedance and Back-EMF voltage drop model.
        """
        # Convert mechanical angular velocity to electrical angular frequency
        we = N_TEETH * abs(w)
        # Total phase impedance: Z = sqrt(R^2 + (we * L)^2)
        z = np.hypot(R_PH, we * L_PH)
        # Current available after subtracting Back-EMF voltage from supply voltage
        i_avail = max(V_SUPPLY - KE * abs(w), 0.0) / z
        # Scale holding torque proportionally to current reduction
        return T_HOLD * min(1.0, i_avail / I_RATED)

    def _advance_velocity_command(self, target_vel):
        """
        Ramps the commanded field velocity towards target_vel respecting MAX_ACCEL
        and integrates it to update the target magnetic field angle (self.cmd).
        """
        # Clamp requested velocity to safe operating range
        v_des = np.clip(target_vel, -MAX_SPEED, MAX_SPEED)
        # Limit velocity change per timestep to prevent instant step loss from inertia
        max_dv = MAX_ACCEL * self.dt
        self.cmd_v += np.clip(v_des - self.cmd_v, -max_dv, max_dv)
        # Integrate velocity to advance the commanded magnetic field position
        self.cmd += self.cmd_v * self.dt

    def compute_torque(self, target_vel, theta, omega):
        """
        Advances the stepper driver command by one physics timestep and returns 
        the resulting electromagnetic torque [N*m] acting on the rotor.
        """
        # Update internal field position and velocity
        self._advance_velocity_command(target_vel)
        
        # Quantize the continuous field angle to discrete microsteps
        cmd_q = np.round(self.cmd / self.step) * self.step
        
        # Compute tracking error between magnetic field and physical rotor
        self.lag = cmd_q - theta
        # Latch stall flag if rotor falls behind by more than 2 full steps (pull-out)
        if abs(self.lag) > 2 * FULL_STEP:
            self.stalled = True

        # Sinusoidal magnetic restoring torque + relative velocity damping
        tau = (self.torque_limit(omega) * np.sin(N_TEETH * (cmd_q - theta))
               - KD * (omega - self.cmd_v))

        # Subtract parasitic cogging (detent) torque from permanent magnets
        tau -= T_DETENT * np.sin(4 * N_TEETH * theta)
        
        # Clip output to match MuJoCo actuator limits
        return float(np.clip(tau, -0.45, 0.45))


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
        self.stepper = StepperMotor(self.model.opt.timestep)
        
        # Action space: Target angular velocity for the stepper motor [rad/s]
        self.max_velocity = MAX_SPEED
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
        terminated = bool(self.stepper.stalled)
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