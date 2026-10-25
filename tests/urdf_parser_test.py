import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[3]
DESCRIPTION_DIR = REPO_ROOT / "simulation" / "pendulum_model" / "sim_mount_description"
XACRO_PATH = DESCRIPTION_DIR / "urdf" / "sim_mount.xacro"
COMPILED_URDF_PATH = DESCRIPTION_DIR / "urdf" / "temp_sim_mount.urdf"
OUTPUT_MJCF_PATH = DESCRIPTION_DIR / "urdf" / "sim_mount_raw.xml"

# 1. Kompilacja xacro -> urdf
print(f"Kompilacja {XACRO_PATH.name}...")
subprocess.run(
    ["xacro", str(XACRO_PATH), "-o", str(COMPILED_URDF_PATH)],
    check=True,
)

try:
    # 2. Kompilacja do modelu MuJoCo
    print("Wczytywanie URDF do MuJoCo...")
    model = mujoco.MjModel.from_xml_path(str(COMPILED_URDF_PATH))

    # 3. Zapis do MJCF
    mujoco.mj_saveLastXML(str(OUTPUT_MJCF_PATH), model)

    # 4. Dodanie flagi balanceinertia='true' do wynikowego XML
    tree = ET.parse(OUTPUT_MJCF_PATH)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("balanceinertia", "true")
    tree.write(OUTPUT_MJCF_PATH, encoding="utf-8", xml_declaration=True)

    print(f"Pomyślnie wygenerowano poprawny MJCF: {OUTPUT_MJCF_PATH}")

finally:
    if COMPILED_URDF_PATH.exists():
        COMPILED_URDF_PATH.unlink()