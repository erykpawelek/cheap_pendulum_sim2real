import numpy as np

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
        
        # Clip output to match  actuator limits
        return float(np.clip(tau, -0.45, 0.45))