"""Preflight of the native AETHRYX GLB before import in external editors."""

import json
from pathlib import Path
import sys

import numpy as np

from aethryx_rig import _GlbBuffers
from studio_project import validate_glb


def inspect_avatar_glb(path):
    doc = validate_glb(path)
    buffers = _GlbBuffers(Path(path).read_bytes())
    nodes = doc.get("nodes", [])
    errors = []
    skin = (doc.get("skins") or [{}])[0]
    joints = skin.get("joints", [])
    required = {"Hips", "Spine", "Chest", "Neck", "Head"}
    required.update(side + part for side in ("Left", "Right") for part in (
        "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot",
    ))
    names = {nodes[index].get("name") for index in joints if type(index) is int and 0 <= index < len(nodes)}
    missing = sorted(required - names)
    if missing:
        errors.append("Faltan huesos humanoides: " + ", ".join(missing))
    inverse = skin.get("inverseBindMatrices")
    if inverse is None or not 0 <= inverse < len(doc["accessors"]):
        errors.append("No hay matrices inversas de enlace.")
    else:
        bind = doc["accessors"][inverse]
        if bind.get("type") != "MAT4" or bind.get("count") != len(joints):
            errors.append("Las matrices inversas no corresponden al esqueleto.")
    triangles, skinned = 0, 0
    face_targets = set()
    for node in nodes:
        if "mesh" not in node:
            continue
        mesh = doc["meshes"][node["mesh"]]
        name = node.get("name", "malla")
        if node.get("skin") == 0:
            skinned += 1
        else:
            errors.append(f"Falta el skin del nodo {name}.")
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            count = doc["accessors"][attributes["POSITION"]]["count"]
            if primitive.get("mode", 4) == 4:
                triangles += doc["accessors"][primitive["indices"]]["count"] // 3 if "indices" in primitive else count // 3
            for channel, dtype, component in (("JOINTS_0", "<u2", 5123), ("WEIGHTS_0", "<f4", 5126)):
                code = attributes.get(channel)
                accessor = doc["accessors"][code] if code is not None else {}
                if (accessor.get("type") != "VEC4" or accessor.get("count") != count
                        or accessor.get("componentType") != component):
                    errors.append(f"Atributo {channel} inválido en {name}.")
                    continue
                view = doc["bufferViews"][accessor["bufferView"]]
                offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
                raw = np.frombuffer(buffers.binary, dtype=dtype, offset=offset, count=count * 4).reshape(-1, 4)
                if channel == "JOINTS_0" and np.any(raw >= len(joints)):
                    errors.append(f"Índices fuera del skin en {name}.")
                if channel == "WEIGHTS_0" and (
                    not np.isfinite(raw).all() or np.any(raw < 0) or not np.allclose(raw.sum(axis=1), 1, atol=1e-5)
                ):
                    errors.append(f"Pesos sin normalizar en {name}.")
            targets = primitive.get("targets", [])
            labels = mesh.get("extras", {}).get("targetNames", [])
            if len(targets) != len(labels) or any(
                doc["accessors"][target["POSITION"]]["count"] != count for target in targets
            ):
                errors.append(f"Índices de morph incompatibles en {name}.")
            face_targets.update(labels)
    return {
        "valid_native_glb": not errors and bool(skinned), "errors": errors,
        "joints": len(joints), "skinned_nodes": skinned, "triangles": triangles,
        "materials": len(doc.get("materials", [])), "animations": len(doc.get("animations", [])),
        "morphs": sorted(face_targets), "visemes": sum(name.startswith("vrc.v_") for name in face_targets),
        "external_import": "unverified",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python -m aethryx_bridge avatar.glb")
    try:
        report = inspect_avatar_glb(sys.argv[1])
    except (ValueError, KeyError, OSError, IndexError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid_native_glb"] else 1)
