import os
import re
from pathlib import Path
import mujoco
import xacro

# Resolve project root: tests/ -> root
ROOT_DIR = Path(__file__).resolve().parent.parent
XACRO_PATH = ROOT_DIR / "simulation" / "pendulum_model" / "sim_mount_description" / "urdf" / "sim_mount.xacro"
MESH_DIR = ROOT_DIR / "simulation" / "pendulum_model" / "sim_mount_description" / "meshes"


def load_model():
    """Compiles xacro to URDF, injects MuJoCo compiler tags, and returns (model, data)."""
    if not XACRO_PATH.exists():
        raise FileNotFoundError(f"Missing xacro file at: {XACRO_PATH}")

    # 1. Expand xacro
    doc = xacro.process_file(str(XACRO_PATH))
    xml = doc.toxml()

    # 2. Clean mesh relative path prefixes
    xml = re.sub(r'filename="(?:\.\./)?meshes/', 'filename="', xml)

    # 3. Add compiler configuration
    compiler = f'\n  <mujoco>\n    <compiler meshdir="{MESH_DIR}" balanceinertia="true" discardvisual="false"/>\n  </mujoco>'
    xml = re.sub(r'(<robot[^>]*>)', r'\1' + compiler, xml)

    # 4. Compile into MuJoCo model
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def test_model_compilation():
    """Automated test for pytest: verifies URDF parsing and kinematics without opening a GUI."""
    model, data = load_model()
    assert model.njnt >= 2, f"Expected at least 2 joints, found {model.njnt}"
    assert model.nbody >= 3, f"Expected at least 3 bodies, found {model.nbody}"
    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    import mujoco.viewer

    model, data = load_model()

    print("\n--- Model loaded successfully ---")
    print(f"Bodies: {model.nbody} | Joints: {model.njnt} | Geoms: {model.ngeom}")
    print("Spacebar: Pause / resume physics")
    print("Ctrl + J: Toggle joint rotation axes")
    print("Ctrl + A: Toggle link coordinate frames")
    print("Right-Click + Drag: Articulate joints\n")

    mujoco.viewer.launch(model, data)