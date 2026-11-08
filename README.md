# cheap_pendulum_sim2real

Sim-to-real project for a **cheap rotary inverted pendulum**:

- a NEMA-17 stepper motor (`17HS4401`) drives a turntable plate,
- a pendulum shaft swings on top of the plate,
- simulation is done in **MuJoCo**,
- the goal is to eventually transfer learned controllers to real hardware
  (Raspberry Pi 5 ↔ ESP32 over SPI, see `embedded/`).

The Python environment is managed with **uv**.

---

## 1. Using `uv`

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

---

## 2. Repository structure (simplified)

```
cheap_pendulum_sim2real/
├── pyproject.toml        # project metadata + Python dependencies
├── uv.lock               # pinned dependency versions (used by uv sync)
├── .gitmodules           # declares `embedded/` as a git submodule
│
├── embedded/             # hardware side (git submodule)
│   ├── esp_firmware/     #   ESP32 firmware (placeholder)
│   ├── ros2_ws/          #   ROS 2 workspace (placeholder)
│   └── shared_interface/
│       └── spi_interface.h   # SPI frame protocol shared RPi5 <-> ESP32
│
├── models/               # exported model files (placeholder)
│
├── simulation/           # MuJoCo simulation side
│   └── pendulum_model/
│       ├── sim_mount_description/
│       │   ├── meshes/               # STL/OBJ meshes
│       │   │   └── precise/          #   precise CAD meshes + PHYSICAL_PARAMS.md
│       │   └── xml/
│       │       ├── sim_mount.xacro       # xacro/URDF source description
│       │       └── sim_mount_raw.xml     # compiled native MJCF (used at runtime)
│       └── utils/
│           └── urdf_to_mjcf_xml.py       # helper: xacro -> URDF -> MJCF
│
├── tests/                # runnable scripts and tests
│   ├── stepper_motor.py      # 17HS4401 stepper model + headless sim loop
│   ├── view_mjcf.py          # interactive MuJoCo viewer + stepper controller
│   ├── viewTest.py           # xacro -> MJCF compile + viewer (older layout)
│   └── urdf_parser_test.py   # xacro -> MJCF generator (older layout)
│
└── training/             # RL / sim-to-real training (placeholder)
```

Key facts:

- The actual model MuJoCo loads is
  `simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml`.
- `xml/sim_mount.xacro` is the human-editable source description.
- `tests/stepper_motor.py` contains the physics/electronics model of the
  `17HS4401` stepper (holding torque, back-EMF, microstepping, cogging, stall
  detection) and is imported by `tests/view_mjcf.py`.

> ⚠️ Note: `viewTest.py`, `urdf_parser_test.py` and
> `simulation/pendulum_model/utils/urdf_to_mjcf_xml.py` still point to an older
> `sim_mount_description/urdf/...` layout that no longer exists in the tree.
> Use `xml/sim_mount_raw.xml` directly (see commands below).

---

## 3. Running the simulation

All commands are run from the **repo root**, using `uv run`.

### Headless simulation (no GUI)

```bash
uv run python tests/stepper_motor.py
```

Steps the model for 2 s and prints plate/shaft angles, motor lag and stall
state. Good for quick checks and CI.

### Interactive MuJoCo viewer (GUI)

```bash
uv run python tests/view_mjcf.py
```

Opens the MuJoCo viewer with the stepper-motor controller driving the plate.

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
uv run python -c "import mujoco; m = mujoco.MjModel.from_xml_path('simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml'); d = mujoco.MjData(m); mujoco.mj_forward(m, d); print('bodies', m.nbody, '| joints', m.njnt, '| geoms', m.ngeom)"
```

---

## 4. MuJoCo commands

There are two ways to use MuJoCo:

1. The **Python API** (what `uv` installs via the `mujoco` pip package) — used
   by the scripts above.
2. The **standalone command-line tools** (`simulate`, `compile`, `test`,
   `basic`, `record`, `benchmark`) — these are **not** shipped with the pip
   package; they come from the official MuJoCo binary release.

### 4.1 MuJoCo CLI tools (standalone binaries)

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
./mujoco-3.17.0/bin/simulate simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml

# Compile the model and report any errors
./mujoco-3.17.0/bin/compile simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml

# Run MuJoCo's built-in model tests
./mujoco-3.17.0/bin/test simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml

# Headless "basic" simulation (steps the model a few times and prints info)
./mujoco-3.17.0/bin/basic simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml

# Record a video of the simulation
./mujoco-3.17.0/bin/record simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml

# Run the MuJoCo performance benchmark
./mujoco-3.17.0/bin/benchmark
```

> Tip: prefer `uv run python tests/view_mjcf.py` over the standalone `simulate`
> when you want the **controlled** pendulum, because that script plugs in the
> `17HS4401` stepper-motor controller. The standalone `simulate` tool only runs
> the passive MJCF model.

### 4.2 MuJoCo Python API (installed by `uv`)

The pip `mujoco` package exposes the full API plus a viewer:

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "simulation/pendulum_model/sim_mount_description/xml/sim_mount_raw.xml"
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
- `stable-baselines3` — RL algorithms (future `training/` stage)
- `torch` / `torchvision` — deep-learning backend
- `onnx` / `onnxruntime` — model export/inference
- `scipy` — scientific computing
- `trimesh` — mesh handling
- `urdf-parser-py`, `xacro` — URDF/xacro processing

The simulation scripts (`stepper_motor.py`, `view_mjcf.py`) only need `numpy`
and `mujoco`; the RL/ML packages are there for the upcoming training stage.
