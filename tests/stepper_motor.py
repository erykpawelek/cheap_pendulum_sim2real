import numpy as np
import mujoco


#XML file
PATH = "simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml"

# ---- 17HS4401 datasheet --------------------------------------------------
T_HOLD   = 0.40        # N·m   (40 N·cm)
T_DETENT = 0.022       # N·m   (2.2 N·cm)
R_PH     = 1.5         # ohm
L_PH     = 2.8e-3      # H
I_RATED  = 1.5         # A
FULL_STEP = np.deg2rad(1.8)
N_TEETH  = 50          # rotor teeth: 200 full steps/rev = 4 * 50
KE = T_HOLD / (np.sqrt(2) * I_RATED)   # ~0.19 V·s/rad, back-EMF constant (estimate)

# ---- drive ----------------------------------------------
V_SUPPLY  = 24.0       # V
MICROSTEP = 16
MAX_SPEED = 30.0       # rad/s  commanded step-rate limit
# Must satisfy J_total * MAX_ACCEL << T_HOLD (J_total ~ 4e-4 kg·m²  ->  hard limit ~1000 rad/s²).
# Stay well under it (<~40%) or the rotor lags > 2 full steps and loses sync.
MAX_ACCEL = 300.0      # rad/s² commanded acceleration limit
KD        = 0.02       # N·m·s/rad electrical damping (tune from a step response)


class StepperMotor:
    def __init__(self, dt):
        self.dt = dt
        self.step = FULL_STEP / MICROSTEP
        self.cmd = 0.0      # commanded (pre-quantization) angle
        self.cmd_v = 0.0
        self.lag = 0.0          # commanded - actual angle [rad]
        self.stalled = False    # latched once lag exceeds 2 full steps

    def torque_limit(self, w):
        """Available torque amplitude vs speed (impedance + back-EMF model).
        Replace with the pull-out curve from the datasheet if you have it."""
        we = N_TEETH * abs(w)                       # electrical rad/s
        z = np.hypot(R_PH, we * L_PH)
        i_avail = max(V_SUPPLY - KE * abs(w), 0.0) / z
        return T_HOLD * min(1.0, i_avail / I_RATED)

    def _advance_command(self, target):
        err = target - self.cmd
        if abs(err) < self.step and abs(self.cmd_v) < MAX_ACCEL * self.dt:
            self.cmd, self.cmd_v = target, 0.0
            return
        v_stop = np.sqrt(2 * MAX_ACCEL * abs(err))
        v_des = np.sign(err) * min(MAX_SPEED, v_stop)
        self.cmd_v += np.clip(v_des - self.cmd_v, -MAX_ACCEL * self.dt, MAX_ACCEL * self.dt)
        self.cmd += self.cmd_v * self.dt

    def torque(self, target, theta, omega):
        self._advance_command(target)
        cmd_q = np.round(self.cmd / self.step) * self.step     # microstep quantization
        self.lag = cmd_q - theta
        if abs(self.lag) > 2 * FULL_STEP:                      # past pull-out: steps are lost
            self.stalled = True

        # magnetic torque: sinusoidal in electrical angle error, so the rotor
        # loses steps if it lags more than ~2 full steps (pull-out)
        # damping acts on speed RELATIVE to the commanded field (not absolute speed),
        # otherwise it brakes the rotor while it is correctly following the command
        tau = (self.torque_limit(omega) * np.sin(N_TEETH * (cmd_q - theta))
               - KD * (omega - self.cmd_v))

        # detent (cogging) torque: one period per full step = 200 per rev
        tau -= T_DETENT * np.sin(4 * N_TEETH * theta)
        return float(np.clip(tau, -0.45, 0.45))


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(PATH)
    data = mujoco.MjData(model)
    motor = StepperMotor(model.opt.timestep)

    target = np.deg2rad(90)
    n = int(2.0 / model.opt.timestep)
    for i in range(n):
        data.ctrl[0] = motor.torque(target, data.qpos[0], data.qvel[0])
        mujoco.mj_step(model, data)
        if i % 400 == 0:
            print(f"t={data.time:5.2f}s  plate={np.rad2deg(data.qpos[0]):7.2f}°  "
                  f"shaft={np.rad2deg(data.qpos[1]):7.2f}°  "
                  f"lag={np.rad2deg(motor.lag):6.2f}°  stalled={motor.stalled}")