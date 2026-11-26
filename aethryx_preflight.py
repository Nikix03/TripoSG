"""Diagnose the existing AETHRYX environment without installing packages."""

from importlib.util import find_spec
import json
from pathlib import Path
import shutil
import subprocess
import sys


def studio_environment():
    required = ("tkinter", "numpy", "scipy", "PIL", "matplotlib", "trimesh", "cv2", "vtk")
    optional = ("pyvista", "pymeshlab", "fast_simplification", "onnxruntime", "mediapipe", "torch")
    result = {
        "python": sys.version.split()[0],
        "required_missing": [name for name in required if find_spec(name) is None],
        "optional": {name: find_spec(name) is not None for name in optional},
        "gpu": None,
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            process = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            result["gpu"] = [row.strip() for row in process.stdout.splitlines() if row.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    root = Path(__file__).resolve().parent
    result["bundled_body"] = (root / "assets" / "genesis" / "humanoid-base.ply").is_file()
    result["launchable"] = not result["required_missing"] and result["bundled_body"]
    return result


if __name__ == "__main__":
    state = studio_environment()
    print(json.dumps(state, ensure_ascii=False, indent=2))
    raise SystemExit(0 if state["launchable"] else 1)
