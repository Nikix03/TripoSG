"""Project manifests and GLB checks, independent of the GUI and GPU."""
import json
import os
from pathlib import Path
import struct
import tempfile
from uuid import uuid4
from copy import deepcopy

from character_sheet import validate_reference_regions


def validate_settings(steps, faces, seed, guidance):
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("Pasos: 1–100.")
    if type(faces) is not int or (faces != -1 and not 100 <= faces <= 2000000):
        raise ValueError("Caras: -1 (sin límite) o 100–2.000.000.")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("Semilla: 0–2147483647.")
    if type(guidance) not in (int, float) or not 0 <= guidance <= 20:
        raise ValueError("Guidance: 0–20.")

def validate_glb(path):
    """Check GLB 2 header, JSON chunk and mesh presence; not a full glTF validator."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("Cabecera GLB incompleta.")
        magic, version, length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or length != path.stat().st_size:
            raise ValueError("GLB 2 inválido o truncado.")
        chunk = stream.read(8)
        if len(chunk) != 8:
            raise ValueError("GLB sin bloque JSON.")
        size, kind = struct.unpack("<I4s", chunk)
        if kind != b"JSON" or size % 4 or size > length - 20 or size > 64 * 1024 * 1024:
            raise ValueError("Bloque JSON GLB inválido.")
        data = json.loads(stream.read(size))
        if not isinstance(data, dict) or not data.get("meshes"):
            raise ValueError("GLB sin mallas.")
        while stream.tell() < length:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError("Bloque GLB truncado.")
            size, _ = struct.unpack("<I4s", chunk)
            if size % 4 or size > length - stream.tell():
                raise ValueError("Longitud de bloque GLB inválida.")
            stream.seek(size, 1)
    return data

def _relative(value, parent):
    if not value:
        return ""
    path = Path(value).expanduser().resolve()
    try:
        return os.path.relpath(path, parent)
    except ValueError:  # Windows volumes differ
        return str(path)


def recovery_path(project):
    path = Path(project).resolve()
    if path.suffix.lower() != ".aeth" or path.stem.endswith(".recovery"):
        raise ValueError("Selecciona un proyecto principal .aeth para crear una recuperación.")
    return path.with_name(f"{path.stem}.recovery.aeth")


def unsaved_recovery_directory():
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) if local else Path.home() / ".cache") / "AETHRYXStudio" / "Recovery"


def unsaved_recovery_path():
    return unsaved_recovery_directory() / f"session-{uuid4().hex}.aeth"


def newer_recovery(project):
    try:
        snapshot = recovery_path(project)
        if snapshot.is_file() and snapshot.stat().st_mtime_ns > Path(project).stat().st_mtime_ns:
            return snapshot
    except (OSError, ValueError):
        pass
    return None


def _modular_data(hidden, anatomy, parent, *, saving):
    if (not isinstance(hidden, (list, tuple)) or any(not isinstance(name, str) or not name for name in hidden)
            or len(set(hidden)) != len(hidden)):
        raise ValueError("Los componentes ocultos deben tener nombres únicos.")
    if anatomy is None:
        return list(hidden), None
    if not isinstance(anatomy, dict) or not isinstance(anatomy.get("views"), dict):
        raise ValueError("El ajuste anatómico debe contener vistas.")
    if set(anatomy["views"]) - {"front", "side", "back"}:
        raise ValueError("El ajuste anatómico contiene una vista desconocida.")
    anatomy = deepcopy(anatomy)

    def resource(value):
        if not isinstance(value, str):
            raise ValueError("Ruta de ajuste anatómico inválida.")
        if not value:
            return ""
        return _relative(value, parent) if saving else str((parent / value).resolve())

    if anatomy.get("bundled_template"):
        anatomy["template"] = ""
    elif "template" in anatomy:
        anatomy["template"] = resource(anatomy["template"])
    for view in anatomy["views"].values():
        if not isinstance(view, dict):
            raise ValueError("Vista anatómica inválida.")
        for key in ("image", "mask", "body_mask"):
            if key in view:
                view[key] = resource(view[key])
    return list(hidden), anatomy


def save_project(
    path,
    *,
    image,
    model,
    output,
    steps,
    faces,
    seed,
    guidance,
    references=(),
    avatar_nodes=None,
    hidden_components=(),
    anatomy_setup=None,
):
    validate_settings(steps, faces, seed, guidance)
    path = Path(path).resolve()
    if path.suffix.lower() != ".aeth":
        raise ValueError("El proyecto debe tener extensión .aeth.")
    reference_regions = validate_reference_regions(references)
    for region in reference_regions:
        region["path"] = _relative(region["path"], path.parent)
    hidden, anatomy = _modular_data(hidden_components, anatomy_setup, path.parent, saving=True)
    data = {"format": "AETHRYX", "version": 3,
            "image": _relative(image, path.parent), "model": _relative(model, path.parent),
            "output": _relative(output, path.parent), "references": reference_regions,
            "avatar_nodes": avatar_nodes or [],
            "hidden_components": hidden, "anatomy_setup": anatomy,
            "settings": {"steps": steps, "faces": faces, "seed": seed, "guidance": guidance}}
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp = handle.name
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)

def load_project(path):
    path = Path(path).resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Proyecto demasiado grande.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "AETHRYX" or data.get("version") not in (1, 2, 3):
        raise ValueError("Formato de proyecto no compatible.")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Proyecto sin configuración.")
    validate_settings(*(settings.get(key) for key in ("steps", "faces", "seed", "guidance")))
    result = {"settings": settings}
    for key in ("image", "model", "output"):
        value = data.get(key, "")
        if not isinstance(value, str):
            raise ValueError("Ruta de proyecto inválida.")
        result[key] = str((path.parent / value).resolve()) if value else ""
    for key in ("image", "model"):
        if result[key] and not Path(result[key]).is_file():
            raise ValueError("Recurso no encontrado: " + result[key])
    if result["output"] and Path(result["output"]).suffix.lower() != ".glb":
        raise ValueError("La salida del proyecto debe ser GLB.")
    raw_references = data.get("references", [])
    if not isinstance(raw_references, list):
        raise ValueError("La lista de regiones de referencia no es válida.")
    resolved_references = []
    for region in raw_references:
        if not isinstance(region, dict) or not isinstance(region.get("path"), str):
            raise ValueError("Ruta de región de referencia inválida.")
        resolved_references.append(
            {
                **region,
                "path": str((path.parent / region["path"]).resolve()),
            }
        )
    result["references"] = validate_reference_regions(resolved_references)
    avatar_nodes = data.get("avatar_nodes", [])
    if not isinstance(avatar_nodes, list):
        raise ValueError("El scene graph del avatar no es válido.")
    result["avatar_nodes"] = avatar_nodes
    result["hidden_components"], result["anatomy_setup"] = _modular_data(
        data.get("hidden_components", []), data.get("anatomy_setup"), path.parent, saving=False,
    )
    return result
