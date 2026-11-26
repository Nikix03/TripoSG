"""Preflight of the native AETHRYX GLB before import in external editors."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from aethryx_modular import BODY
from aethryx_rig import _GlbBuffers, deform_pose
from studio_project import validate_glb


def _packed(glb, index, dtype, columns):
    doc = glb.document
    access = doc["accessors"][index]
    view = doc["bufferViews"][access["bufferView"]]
    offset = view.get("byteOffset", 0) + access.get("byteOffset", 0)
    return np.frombuffer(glb.binary, dtype=dtype, offset=offset,
                         count=access["count"] * columns).reshape(-1, columns)


def _motion_sample(glb, skin):
    doc = glb.document
    clip = (doc.get("animations") or [{}])[0]
    if not clip.get("channels"):
        return 0.0
    nodes = doc["nodes"]
    by_parent = {child: index for index, node in enumerate(nodes) for child in node.get("children", [])}
    frames = {}
    for channel in clip["channels"]:
        target = channel["target"]
        if target["path"] == "rotation":
            output = clip["samplers"][channel["sampler"]]["output"]
            rotations = _packed(glb, output, "<f4", 4)
            frames[target["node"]] = rotations[len(rotations) // 2]

    native = skin["joints"]
    world = {}
    for index in native:
        node = nodes[index]
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(frames.get(index, [0., 0., 0., 1.])).as_matrix()
        matrix[:3, 3] = node.get("translation", [0, 0, 0])
        parent = by_parent.get(index)
        if parent in native and parent not in world:
            raise ValueError("Las articulaciones nativas deben conservar su orden jerárquico.")
        world[index] = world[parent] @ matrix if parent in world else matrix
    inverse = _packed(glb, skin["inverseBindMatrices"], "<f4", 16).reshape(-1, 4, 4).transpose(0, 2, 1)
    matrices = np.asarray([world[index] for index in native]) @ inverse
    body = next((node for node in nodes if node.get("name") == BODY and "mesh" in node), None)
    if body is None:
        return 0.0
    maximum = 0.
    for primitive in doc["meshes"][body["mesh"]]["primitives"]:
        attributes = primitive["attributes"]
        vertices = glb.positions(attributes["POSITION"])
        indices = _packed(glb, attributes["JOINTS_0"], "<u2", 4)
        weights = _packed(glb, attributes["WEIGHTS_0"], "<f4", 4)
        moved = deform_pose(vertices, indices, weights, matrices)
        maximum = max(maximum, float(np.linalg.norm(moved - vertices, axis=1).max()))
    return maximum


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
                raw = _packed(buffers, code, dtype, 4)
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
    movement = _motion_sample(buffers, skin) if not errors and skinned else None
    if movement is not None and movement <= 1e-5:
        errors.append("El clip nativo de prueba no deforma el cuerpo.")
    return {
        "valid_native_glb": not errors and bool(skinned), "errors": errors,
        "joints": len(joints), "skinned_nodes": skinned, "triangles": triangles,
        "materials": len(doc.get("materials", [])), "animations": len(doc.get("animations", [])),
        "morphs": sorted(face_targets), "visemes": sum(name.startswith("vrc.v_") for name in face_targets),
        "sample_body_displacement": movement,
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
