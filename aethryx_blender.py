"""Run an optional Blender installation as an isolated export worker."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from studio_project import validate_glb
from aethryx_adaptive import build_adaptive_plan


FORMATS = {".blend", ".fbx", ".glb"}


def find_blender(configured=None):
    if configured:
        path = Path(configured)
        return path.resolve() if path.is_file() else None
    executable = shutil.which("blender")
    if executable:
        return Path(executable).resolve()
    for root in (Path(os.environ.get("ProgramFiles", "C:/Program Files")), Path("C:/Program Files")):
        candidates = sorted((root / "Blender Foundation").glob("Blender */blender.exe"), reverse=True)
        if candidates:
            return candidates[0].resolve()
    return None


def blender_version(executable):
    result = subprocess.run([str(executable), "--version"], capture_output=True, text=True,
                            timeout=15, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    version = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", result.stdout)
    if version is None or not 4 <= int(version[1]) <= 5:
        raise ValueError("El backend requiere Blender 4.x o 5.x.")
    return version[0]


def export_with_blender(executable, source, output, cancelled, *, on_process=lambda process: None, repair=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if cancelled.is_set():
        return None
    if source == output or output.suffix.lower() not in FORMATS or output.exists():
        raise ValueError("Elige una salida nueva BLEND, FBX o GLB.")
    validate_glb(source)
    version = blender_version(executable)
    plan = build_adaptive_plan(engine_id="blender")
    if cancelled.is_set():
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).resolve().parent / "scripts" / "blender_export.py"
    with TemporaryDirectory(prefix=".aethryx-blender-", dir=output.parent) as directory:
        folder = Path(directory)
        staged = folder / output.name
        request = folder / "request.json"
        response = folder / "response.json"
        request.write_text(json.dumps({"input": str(source), "output": str(staged),
                                       "response": str(response), "repair": repair is True}), encoding="utf-8")
        command = [str(executable), "--background", "--factory-startup", "--disable-autoexec",
                   "--python-exit-code", "1", "--python", str(worker), "--", str(request)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            on_process(process)
            try:
                while True:
                    if cancelled.is_set():
                        if process.poll() is None:
                            process.kill()
                        process.communicate()
                        return None
                    try:
                        log, _ = process.communicate(timeout=.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                on_process(None)
            if process.returncode or not response.is_file() or not staged.is_file():
                raise ValueError("Blender no completó la exportación.\n" + log[-3000:])
        report = json.loads(response.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or not report.get("meshes"):
            raise ValueError("El informe de Blender carece de mallas verificables.")
        if output.suffix.lower() == ".glb":
            validate_glb(staged)
        else:
            with staged.open("rb") as stream:
                magic = b"BLENDER" if output.suffix.lower() == ".blend" else b"Kaydara FBX Binary"
                if stream.read(len(magic)) != magic:
                    raise ValueError("La salida no tiene la cabecera esperada.")
        if cancelled.is_set():
            return None
        # Hard linking within the staging volume publishes the validated output without replacing a file.
        os.link(staged, output)
        return {**report, "blender": version, "bytes": output.stat().st_size, "adaptive": plan.reason}
