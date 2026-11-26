"""Pure helpers for the desktop front end; no GPU imports."""
from pathlib import Path
from datetime import datetime
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

def generation_args(image, output, seed=42, steps=50, guidance=7.0, faces=-1):
    image, output = Path(image).resolve(), Path(output).resolve()
    if not image.is_file():
        raise ValueError("Selecciona una imagen existente.")
    if output.suffix.lower() != ".glb":
        raise ValueError("La salida del motor debe ser GLB.")
    if output.exists():
        raise ValueError("El archivo de salida ya existe.")
    if not 1 <= steps <= 200 or not 0 <= seed <= 2147483647:
        raise ValueError("Pasos o semilla fuera del intervalo.")
    if not 0 <= guidance <= 20 or (faces != -1 and faces < 100):
        raise ValueError("Configuración de geometría no válida.")
    return ["-u", "-m", "scripts.inference_triposg", "--image-input",
            str(image), "--output-path", str(output), "--seed", str(seed),
            "--num-inference-steps", str(steps), "--guidance-scale",
            str(guidance), "--faces", str(faces)]

def new_output(directory):
    directory = Path(directory).expanduser().resolve()
    folder = directory / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    return folder / "modelo.glb"
