# cheap_pendulum_sim2real

Sim-to-real project for a **cheap Furuta pendulum** (a rotary inverted pendulum):

- a NEMA-17 stepper motor (`17HS4401`) drives a horizontal arm / turntable plate,
- a pendulum shaft swings freely at the arm's tip,
- the plant is modelled in **MuJoCo** together with an electromechanical model of
  the stepper motor (holding torque, back-EMF, microstepping, cogging, stall),
- a **Gymnasium** environment `FurutaPendulum-v0` wraps the simulation for RL,
- a **PPO** policy is trained with **stable-baselines3**,
- the long-term goal is to transfer the learned controller to real hardware
  (Raspberry Pi 5 ↔ ESP32 over SPI, see `embedded/`).

The Python environment is managed with **uv**.

---

## 1. Getting the repository (submodules)

This repo declares two git submodules in `.gitmodules`:

| Name       | Path        | URL                                                      |
| ---------- | ----------- | -------------------------------------------------------- |
| `host`     | `host/`     | https://github.com/erykpawelek/pendulum_embeded.git      |
| `firmware` | `embedded/` | git@github.com:firiusz123/embedded-pendulum.git (`main`) |

After cloning, initialise them with:

```bash
git submodule update --init --recursive
```

> `host/` is currently an empty placeholder in this repository.
> `embedded/` holds the hardware side: ESP32 firmware, a ROS 2 workspace and the
> shared SPI frame protocol (`shared_interface/spi_interface.h`).

---

## 2. Using `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast Python package/project manager.
This repo already has a `pyproject.toml` (dependencies) and a `uv.lock`
(pinned versions), so a single command recreates the whole environment.

### Install `uv` (once)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# or, if you already have pip
pip install uv
```

### Day-to-day commands

Run these from the **repo root**:

```bash
# Create the .venv and install all dependencies from pyproject.toml + uv.lock
uv sync

# Run any command inside the project virtual environment
# (automatically creates the env if it does not exist yet)
uv run <command>

# Add a new dependency
uv add <package>

# Show the dependency tree
uv tree

# Re-generate / update the lockfile
uv lock
```

`uv sync` creates a `.venv/` folder (already present in this repo) with the
correct Python version and packages. You do **not** need to activate it
manually — prefix commands with `uv run` instead.

> **Important:** the Python code here uses a flat namespace-package layout
> (`simulation/`, `training/`, `tests/` are plain directories, not installed via
> pip). Therefore anything that does `import simulation.envs` must be launched
> **from the repo root as a module**:
>
> ```bash
> uv run python -m training.training   # ✔ works
> uv run python training/training.py   # ✘ ModuleNotFoundError: No module named 'simulation'
> ```

---

## 3. Repository structure

```
cheap_pendulum_sim2real/
├── pyproject.toml           # project metadata + Python dependencies
├── uv.lock                  # pinned dependency versions (used by uv sync)
├── .gitignore
├── .gitmodules              # declares `host/` and `embedded/` submodules
│
├── host/                    # host-side submodule (empty placeholder)
├── embedded/                # hardware-side submodule ("firmware")
│   ├── esp_firmware/        #   ESP32 firmware (placeholder)
│   ├── ros2_ws/             #   ROS 2 workspace (placeholder)
│   └── shared_interface/
│       └── spi_interface.h  #   SPI frame protocol shared RPi5 <-> ESP32
│
├── models/                  # exported/trained model files (placeholder)
│
├── simulation/
│   ├── envs/
│   │   ├── __init__.py          # registers the `FurutaPendulum-v0` env
│   │   ├── pendulum_env.py      # FurutaEnv (Gymnasium) + reward / reset logic
│   │   └── stepper_motor.py     # StepperMotor: 17HS4401 electromechanical model
│   └── pendulum_model/
│       └── sim_mount_description/
│           ├── meshes/precise/  # STL meshes + docs/PHYSICAL_PARAMS.md (inertias)
│           └── xml/
│               └── world_with_pendulum.xml   # MJCF loaded at runtime
│
├── tests/
│   ├── evaluate.py          # load trained PPO policy + visual evaluation (RTF)
│   ├── stepper_motor.py     # headless stepper model test (also imported by viewer)
│   └── view_precise_mjcf.py # interactive MuJoCo viewer + stepper controller
│
└── training/
    ├── training.py          # PPO training (stable-baselines3 + VecNormalize)
    └── logs/                # trained artifacts (git-ignored)
        ├── ppo_furuta_pendulum.zip
        └── vec_normalize.pkl
```

Key facts:

- The model MuJoCo actually loads is
  `simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml`.
  It is a self-contained native MJCF file (no xacro/URDF step at runtime).
- The physical parameters (masses and inertias exported from Fusion 360 and
  converted to SI units) are documented in
  `simulation/pendulum_model/sim_mount_description/meshes/precise/docs/PHYSICAL_PARAMS.md`.
- `simulation/envs/stepper_motor.py` contains the electromechanical model of the
  `17HS4401` stepper (holding torque, back-EMF, microstepping, cogging and stall
  detection) used by the RL environment.

> ⚠️ `simulation/pendulum_model/utils/urdf_to_mjcf_xml.py` is a leftover helper
> that still points to an older `sim_mount_description/urdf/...` layout which no
> longer exists in the tree. The current MJCF (`xml/world_with_pendulum.xml`) is
> maintained directly.

---

## 4. The Gymnasium environment

`simulation/envs/__init__.py` registers `FurutaPendulum-v0` (entry point
`simulation.envs.pendulum_env:FurutaEnv`, `max_episode_steps=2048`).

| Item         | Value                                                          |
| ------------ | -------------------------------------------------------------- |
| Action space | `Box(1)` — target angular velocity of the arm, [-15, 15] rad/s |
| Observation  | `Box(5)` — `[sin θ, cos θ, θ̇, arm_pos, arm_vel]`               |
| Control rate | 100 Hz (`frame_skip=20` × 0.5 ms physics timestep)             |
| Physics rate | 2000 Hz (`timestep=0.0005`, RK4 integrator)                    |
| Termination  | stepper stall (lag > 2 full steps) or |arm angle| > 2π          |
| Reward       | [0, 1] shaping — upright posture + velocity/arm damping        |

Quick check from the repo root:

```bash
uv run python -c "import simulation.envs, gymnasium; e = gymnasium.make('FurutaPendulum-v0'); print(e.observation_space, e.action_space)"
```

---

## 5. Running the simulation

All commands are run from the **repo root**, using `uv run`.

### Headless stepper test (no GUI)

```bash
uv run python tests/stepper_motor.py
```

Steps the model for 2 s and prints plate/shaft angles, motor lag and the stall
flag. Good for quick checks and CI.

### Interactive MuJoCo viewer (GUI)

```bash
uv run python tests/view_precise_mjcf.py
```

Opens the MuJoCo viewer with the stepper-motor controller driving the arm.

Useful viewer controls:

| Key / mouse                    | Action                                    |
| ------------------------------ | ----------------------------------------- |
| `Spacebar`                     | Pause / resume physics                    |
| `Backspace`                    | Reset (also resets motor state)           |
| Double-left-click on a body    | Select body / set focal point             |
| `Ctrl` + right-click + drag    | Apply force/torque to a body              |
| `Ctrl` + `J`                   | Toggle joint axes                         |
| `Ctrl` + `A`                   | Toggle link coordinate frames             |
| `F1`                           | Help menu                                 |

### Quick model sanity check (Python API)

```bash
uv run python -c "import mujoco; m = mujoco.MjModel.from_xml_path('simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml'); d = mujoco.MjData(m); mujoco.mj_forward(m, d); print('bodies', m.nbody, '| joints', m.njnt, '| geoms', m.ngeom)"
```

---

## 6. Training & evaluation

### Train the PPO policy

```bash
uv run python -m training.training
```

- `check_env()` validates the environment first,
- 4 parallel environments (`DummyVecEnv`),
- `VecNormalize` normalises observations only (reward left untouched),
- PPO (`MlpPolicy`, 700 000 timesteps),
- saves the policy to `training/logs/ppo_furuta_pendulum.zip` and the
  normalisation statistics to `training/logs/vec_normalize.pkl`.

### Evaluate a trained policy

```bash
uv run python -m tests.evaluate
```

Loads the saved policy and normalisation stats, then runs a real-time visual
evaluation, printing the real-time factor (RTF) every 100 steps. Press
`Ctrl+C` to stop.

---

## 7. MuJoCo commands

There are two ways to use MuJoCo:

1. The **Python API** (what `uv` installs via the `mujoco` pip package) — used
   by the scripts above.
2. The **standalone command-line tools** (`simulate`, `compile`, `test`,
   `basic`, `record`, `benchmark`) — these are **not** shipped with the pip
   package; they come from the official MuJoCo binary release.

### 7.1 MuJoCo CLI tools (standalone binaries)

Download and unpack the official prebuilt binaries:

```bash
# Pick the latest version from: https://github.com/google-deepmind/mujoco/releases
wget https://github.com/google-deepmind/mujoco/releases/download/3.17.0/mujoco-3.17.0-linux-x86_64.tar.gz
tar -xzf mujoco-3.17.0-linux-x86_64.tar.gz
# The executables are now in ./mujoco-3.17.0/bin/
```

Run them **from the repo root** (so the relative mesh paths in the MJCF resolve):

```bash
# Interactive GUI viewer (passive model, no stepper controller)
./mujoco-3.17.0/bin/simulate simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml

# Compile the model and report any errors
./mujoco-3.17.0/bin/compile simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml

# Run MuJoCo's built-in model tests
./mujoco-3.17.0/bin/test simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml

# Headless "basic" simulation (steps the model a few times and prints info)
./mujoco-3.17.0/bin/basic simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml

# Record a video of the simulation
./mujoco-3.17.0/bin/record simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml

# Run the MuJoCo performance benchmark
./mujoco-3.17.0/bin/benchmark
```

> Tip: prefer `uv run python tests/view_precise_mjcf.py` over the standalone
> `simulate` when you want the **controlled** pendulum, because that script plugs
> in the `17HS4401` stepper-motor controller. The standalone `simulate` tool only
> runs the passive MJCF model.

### 7.2 MuJoCo Python API (installed by `uv`)

The pip `mujoco` package exposes the full API plus a viewer:

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "simulation/pendulum_model/sim_mount_description/xml/world_with_pendulum.xml"
)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
```

You can run the model headlessly step by step with `mujoco.mj_step(model, data)`
(see `tests/stepper_motor.py` for a complete example).

---

## Dependencies (from `pyproject.toml`)

- `gymnasium` — RL environments
- `mujoco` — physics engine
- `stable-baselines3` — RL algorithms (PPO)
- `torch` / `torchvision` — deep-learning backend
- `onnx` / `onnxruntime` — model export/inference
- `scipy` — scientific computing
- `trimesh` — mesh handling
- `urdf-parser-py`, `xacro` — URDF/xacro processing
