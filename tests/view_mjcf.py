from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

from stepper_motor import StepperMotor   # keep this script next to stepper_motor.py
MODEL_PATH = Path("simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

motor = StepperMotor(model.opt.timestep)

def controller(model, data):
    
    if data.time == 0.0:                 # sim was (re)started -> reset the motor state
        motor.cmd = motor.cmd_v = motor.lag = 0.0
        motor.stalled = False
    # plate sweeps between +90 and -90 deg, switching every 3 s
    target = np.deg2rad(90) * np.sign(np.sin(2 * np.pi * data.time / 6.0))
    data.ctrl[0] = motor.torque(target, data.qpos[0], data.qvel[0])

mujoco.set_mjcb_control(controller)

print("Starting MuJoCo Interactive Viewer...")
print("Controls:")
print(" - Spacebar: Pause / Unpause physics")
print(" - Backspace: Reset (motor state resets too)")
print(" - Double Left Click on part: Select body / focal point")
print(" - Ctrl + Right Click & Drag: Apply force / torque to the body")
print(" - F1: Help menu")

mujoco.viewer.launch(model, data)