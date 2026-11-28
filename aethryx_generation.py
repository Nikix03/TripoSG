"""Local engine preferences and generation worker commands."""

import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile


ENGINES = ("auto", "triposg", "triposr")
TRIPOSR_FILES = ("source", "checkpoint", "config", "dino-config")


def save_engine_settings(path, settings):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(settings, stream, ensure_ascii=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def choose_engine(preference, triposr_options, python, root):
    if preference not in ENGINES:
        raise ValueError("Motor de generación desconocido.")
    if preference == "triposg":
        return "triposg"
    options = triposr_options or {}
    source = Path(options.get("source") or ".")
    missing = [key for key in TRIPOSR_FILES[1:] if not Path(options.get(key) or ".").is_file()]
    if not (source / "tsr" / "system.py").is_file():
        missing.insert(0, "source")
    reason = "Configura los archivos locales de TripoSR: " + ", ".join(missing)
    if not missing:
        command = [python, "-m", "scripts.inference_triposr", "--probe"]
        for key in TRIPOSR_FILES:
            command.extend((f"--{key}", options[key]))
        try:
            probe = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=30,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if probe.returncode == 0:
                return "triposr"
            reason = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "Sonda TripoSR fallida."
        except (OSError, subprocess.TimeoutExpired) as exc:
            reason = str(exc)
    if preference == "auto":
        return "triposg"
    raise ValueError(reason)


def generation_command(python, engine, image, output, steps, faces, seed, guidance, options=None):
    if engine == "triposr":
        command = [python, "-m", "scripts.inference_triposr", "--image", image, "--output", output,
                   "--resolution", str(options.get("resolution", 128)), "--seed", str(seed)]
        for key in TRIPOSR_FILES:
            command.extend((f"--{key}", options[key]))
        return command
    if engine != "triposg":
        raise ValueError("Motor de generación desconocido.")
    return [python, "-m", "scripts.inference_triposg", "--image-input", image, "--output-path", output,
            "--num-inference-steps", str(steps), "--faces", str(faces), "--seed", str(seed),
            "--guidance-scale", str(guidance), "--adaptive-profile", "--skip-background-removal"]


def compare_candidates(paths, image, view):
    """Rasterize independently framed candidates against the same reviewed alpha mask."""
    import numpy as np
    import trimesh
    from aethryx_anatomy import orthographic_projection
    from aethryx_reference import compare_silhouette, reference_silhouette

    alpha = image.convert("RGBA").getchannel("A")
    if alpha.getextrema() != (0, 255):
        raise ValueError("Usa una referencia RGBA con transparencia revisada.")
    silhouette = reference_silhouette(image, alpha, allow_border=True)
    if silhouette is None:
        raise ValueError("El área de primer plano de la referencia no es válida.")
    y, x = np.nonzero(silhouette)
    center = [(float(x.min()) + float(x.max())) / 2, float(y.max())]
    results = []
    for path in paths:
        mesh = trimesh.load(path, force="scene", process=False).dump(concatenate=True)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"La malla {Path(path).name} no contiene geometría válida.")
        projection = orthographic_projection(view, center, center[1], float(y.min()), mesh.bounds)
        overlay, iou = compare_silhouette(mesh, image, projection, alpha, allow_border=True)
        results.append({"path": str(path), "overlay": overlay, "iou": iou, "faces": len(mesh.faces),
                        "closed": bool(mesh.is_volume)})
    return results


def rank_regional_candidates(candidates, references, *, body=None, guide=None):
    """Rank comparable silhouettes and keep anatomical validation explicit."""
    import numpy as np
    import trimesh
    from aethryx_modular import body_contacts

    if not references or set(references) - {"front", "side", "back"}:
        raise ValueError("Carga máscaras frontal, lateral o posterior.")
    if (body is None) != (guide is None):
        raise ValueError("La validación anatómica requiere cuerpo y guía regional.")
    paths = [item["path"] for item in candidates]
    scores = {view: compare_candidates(paths, image, view) for view, image in references.items()}
    rows = []
    for index, item in enumerate(candidates):
        mesh = trimesh.load(item["path"], force="scene", process=False).dump(concatenate=True)
        incidence = np.bincount(mesh.edges_unique_inverse)
        ious = {view: items[index]["iou"] for view, items in scores.items()}
        topology = {"closed": bool(mesh.is_volume), "boundary_edges": int((incidence == 1).sum()),
                    "nonmanifold_edges": int((incidence > 2).sum())}
        contacts = body_contacts(body, mesh) if body is not None else None
        position = None
        if guide is not None:
            position = float(np.linalg.norm(mesh.centroid - guide.centroid) / body.extents[1])
        eligible = topology["closed"] and (contacts is None or (
            contacts["contacts"] == contacts["inside"] == 0 and position <= .05
        ))
        score = .7 * float(np.mean(list(ious.values()))) + .3 * min(ious.values())
        rows.append({**item, "iou": ious, "topology": topology, "contacts": contacts, "position": position,
                     "eligible": eligible, "score": score,
                     "confidence": "three-views-and-guide" if len(ious) == 3 and guide is not None else "limited"})
    return sorted(rows, key=lambda row: (row["eligible"], row["score"]), reverse=True)
