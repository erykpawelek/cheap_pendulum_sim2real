import subprocess
from pathlib import Path
import mujoco

# Define paths relative to the repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
DESCRIPTION_DIR = REPO_ROOT / "simulation" / "pendulum_model" / "sim_mount_description"
XACRO_PATH = DESCRIPTION_DIR / "urdf" / "sim_mount.xacro"
COMPILED_URDF_PATH = DESCRIPTION_DIR / "urdf" / "temp_sim_mount.urdf"
OUTPUT_MJCF_PATH = DESCRIPTION_DIR / "urdf" / "sim_mount_raw.xml"

# 1. Compile Xacro to URDF in the description directory
print(f"Compiling {XACRO_PATH.name}...")
subprocess.run(
    ["xacro", str(XACRO_PATH), "-o", str(COMPILED_URDF_PATH)],
    check=True,
)

try:
    # 2. Load compiled URDF into MuJoCo compiler
    print("Loading URDF into MuJoCo...")
    model = mujoco.MjModel.from_xml_path(str(COMPILED_URDF_PATH))

    # 3. Export as native MJCF XML
    mujoco.mj_saveLastXML(str(OUTPUT_MJCF_PATH), model)
    print(f"Successfully generated native MJCF: {OUTPUT_MJCF_PATH}")

finally:
    # Clean up the intermediate URDF
    if COMPILED_URDF_PATH.exists():
        COMPILED_URDF_PATH.unlink()