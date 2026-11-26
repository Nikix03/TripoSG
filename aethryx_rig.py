"""Native glTF skin, animation and facial morph export for modular Genesis scenes."""

import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from aethryx_modular import BODY, EYES, ModularAvatar, head_frame


def humanoid_skeleton(body, eyes=None):
    lower, upper = body.bounds
    width, height, depth = upper - lower
    if height <= 0 or width <= 0 or depth <= 0:
        raise ValueError("El rig necesita un cuerpo tridimensional con Y vertical.")
    mid = (lower + upper) / 2
    names, parents, positions = [], [], []

    def joint(name, parent, x, y, z):
        names.append(name)
        parents.append(names.index(parent) if parent else -1)
        positions.append([mid[0] + x * width / 2, lower[1] + y * height, mid[2] + z * depth])

    joint("Hips", None, 0, .50, -.19)
    joint("Spine", "Hips", 0, .58, -.19)
    joint("Chest", "Spine", 0, .72, -.18)
    joint("Neck", "Chest", 0, .844, -.16)
    joint("Head", "Neck", 0, .918, -.15)
    for side, sign, eye_node in zip(("Left", "Right"), (1, -1), EYES):
        def side_joint(suffix, parent, x, y, z):
            joint(side + suffix, parent, sign * x, y, z)

        side_joint("Shoulder", "Chest", .28, .782, -.155)
        side_joint("UpperArm", side + "Shoulder", .395, .785, -.139)
        side_joint("LowerArm", side + "UpperArm", .715, .680, -.02)
        side_joint("Hand", side + "LowerArm", .876, .620, .224)
        side_joint("UpperLeg", "Hips", .245, .491, -.164)
        side_joint("LowerLeg", side + "UpperLeg", .354, .288, -.204)
        side_joint("Foot", side + "LowerLeg", .466, .065, -.253)
        side_joint("Toes", side + "Foot", .483, .029, -.034)
        side_joint("Eye", "Head", .094, .911, .120)
        if eyes and eye_node in eyes:
            positions[-1] = eyes[eye_node].bounds.mean(axis=0).tolist()
        for index, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Little")):
            reach = .887 + .018 * min(index, 3)
            offset = .215 + .034 * index
            side_joint(finger + "Proximal", side + "Hand", reach, .609 - .003 * index, offset)
            side_joint(finger + "Intermediate", side + finger + "Proximal",
                       reach + .031, .592 - .003 * index, offset + .042)
            side_joint(finger + "Distal", side + finger + "Intermediate",
                       reach + .054, .579 - .003 * index, offset + .072)
    return names, np.asarray(parents, dtype=np.intp), np.asarray(positions, dtype=np.float64)


def skin_weights(vertices, skeleton, influences=4):
    if type(influences) is not int or not 1 <= influences <= 4:
        raise ValueError("Usa entre una y cuatro influencias por vértice.")
    names, parents, positions = skeleton
    distances = np.empty((len(vertices), len(names)), dtype=np.float64)
    span = max(np.ptp(positions[:, 1]), 1e-6)
    for index, point in enumerate(positions):
        children = np.flatnonzero(parents == index)
        end = positions[children[0]] if len(children) else point
        segment = end - point
        delta = vertices - point
        denominator = segment @ segment
        t = np.clip(np.sum(delta * segment, axis=1) / denominator, 0, 1) if denominator > 0 else np.zeros(len(delta))
        distances[:, index] = np.sum((delta - t[:, None] * segment)**2, axis=1)
        if names[index].endswith("Eye"):
            distances[:, index] = np.inf
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :influences]
    inverse = np.take_along_axis(distances, nearest, axis=1) + (.012 * span)**2
    weights = inverse**-2
    weights /= weights.sum(axis=1, keepdims=True)
    joints, result = np.zeros((len(vertices), 4), dtype=np.uint16), np.zeros((len(vertices), 4), dtype=np.float32)
    joints[:, :influences], result[:, :influences] = nearest, weights
    return joints, result


def demo_rotations():
    return {"LeftUpperArm": (0, 0, 24), "LeftLowerArm": (0, 22, 20),
            "RightUpperArm": (0, 0, -22), "RightLowerArm": (0, -18, -16),
            "LeftUpperLeg": (-12, 0, 0), "LeftLowerLeg": (16, 0, 0), "Head": (0, 12, 0),
            "LeftHand": (-8, 10, 0)}


def pose_matrices(skeleton, rotations):
    names, parents, positions = skeleton
    global_matrices = []
    for index, (name, parent) in enumerate(zip(names, parents)):
        local = np.eye(4)
        local[:3, :3] = Rotation.from_euler("xyz", rotations.get(name, (0, 0, 0)), degrees=True).as_matrix()
        local[:3, 3] = positions[index] - positions[parent] if parent >= 0 else positions[index]
        global_matrices.append(global_matrices[parent] @ local if parent >= 0 else local)
    inverse_bind = np.tile(np.eye(4), (len(names), 1, 1))
    inverse_bind[:, :3, 3] = -positions
    return np.asarray(global_matrices) @ inverse_bind


def deform_pose(vertices, joints, weights, matrices):
    positions = np.column_stack((vertices, np.ones(len(vertices))))
    posed = np.einsum("nvij,nj->nvi", matrices[joints], positions)
    return np.sum(posed[:, :, :3] * weights[:, :, None], axis=1)


def facial_morphs(mesh):
    lower, upper = head_frame(mesh)
    width, height, depth = upper - lower
    x, y, z = ((mesh.vertices - lower) / (upper - lower)).T
    front = np.clip((z - .43) / .43, 0, 1)

    def region(cx, cy, dx, dy):
        return np.exp(-((x - cx) / dx)**2 - ((y - cy) / dy)**2) * front

    mouth = region(.5, .02, .27, .14)
    lower_lip = mouth * np.clip((.043 - y) / .11, 0, 1)
    corners = region(.5 + .19 * np.sign(x - .5), .02, .10, .10)
    result = {}
    for name, dx, dy, dz in (
        ("Smile", np.sign(x - .5) * corners * width * .031, corners * height * .077, 0),
        ("JawOpen", 0, -lower_lip * height * .19, -lower_lip * depth * .047),
        ("Sad", 0, -corners * height * .071, 0),
        ("Surprise", -(x - .5) * mouth * width * .12, -lower_lip * height * .13, mouth * depth * .052),
    ):
        result[name] = np.column_stack(np.broadcast_arrays(dx, dy, dz)).astype(np.float32)
    for name, center in (("BlinkLeft", .714), ("BlinkRight", .286)):
        zone = region(center, .39, .13, .11) * np.clip((y - .33) / .11, 0, 1)
        result[name] = np.column_stack((zone * 0, -zone * height * .092, zone * depth * .01)).astype(np.float32)
    result["Blink"] = result["BlinkLeft"] + result["BlinkRight"]
    brow = region(.5 + .22 * np.sign(x - .5), .52, .17, .088)
    result["Anger"] = np.column_stack((
        -np.sign(x - .5) * brow * width * .018,
        -brow * height * (.031 + .031 * (1 - np.clip(abs(x - .5) * 4, 0, 1))),
        brow * depth * .019,
    )).astype(np.float32)
    # Initial articulatory controls: opening, lateral spread, and rounding.
    speech = {
        "sil": (0, 0, 0), "pp": (-.019, -.031, .014), "ff": (.025, .022, -.008),
        "th": (.055, .018, .031), "dd": (.065, .021, .003), "kk": (.081, -.008, -.014),
        "ch": (.079, -.034, .042), "ss": (.029, .060, .012), "nn": (.040, .018, -.004),
        "rr": (.058, -.048, .051), "aa": (.171, .022, .010), "e": (.095, .083, -.011),
        "ih": (.050, .094, -.005), "oh": (.125, -.141, .067), "ou": (.058, -.128, .081),
    }
    for name, (opening, spread, rounding) in speech.items():
        offsets = np.column_stack((
            (x - .5) * mouth * width * spread, -lower_lip * height * opening, mouth * depth * rounding,
        ))
        if name == "sil":
            offsets[int(np.argmax(mouth)), 2] = depth * 1e-5
        result[f"vrc.v_{name}"] = offsets.astype(np.float32)
    return result


class _GlbBuffers:
    def __init__(self, content):
        if len(content) < 28 or content[:4] != b"glTF":
            raise ValueError("Entrada GLB sin datos binarios.")
        json_length = struct.unpack_from("<I", content, 12)[0]
        self.document = json.loads(content[20:20 + json_length])
        binary_start = 20 + json_length
        if binary_start + 8 > len(content) or content[binary_start + 4:binary_start + 8] != b"BIN\0":
            raise ValueError("El GLB no contiene un bloque BIN.")
        binary_length = struct.unpack_from("<I", content, binary_start)[0]
        self.binary = bytearray(content[binary_start + 8:binary_start + 8 + binary_length])

    def accessor(self, values, gltf_type, component_type=5126, *, bounds=False):
        values = np.ascontiguousarray(values, dtype="<f4" if component_type == 5126 else "<u2")
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        offset = len(self.binary)
        self.binary.extend(values.tobytes())
        views = self.document.setdefault("bufferViews", [])
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": values.nbytes})
        access = {"bufferView": len(views) - 1, "componentType": component_type,
                  "count": len(values), "type": gltf_type}
        if bounds:
            access["min"] = np.atleast_1d(values.min(axis=0)).tolist()
            access["max"] = np.atleast_1d(values.max(axis=0)).tolist()
        accessors = self.document.setdefault("accessors", [])
        accessors.append(access)
        return len(accessors) - 1

    def positions(self, index):
        access = self.document["accessors"][index]
        view = self.document["bufferViews"][access["bufferView"]]
        if access["componentType"] != 5126 or access["type"] != "VEC3" or view.get("byteStride", 12) != 12:
            raise ValueError("El GLB de entrada tiene posiciones de malla incompatibles.")
        offset = access.get("byteOffset", 0) + view.get("byteOffset", 0)
        return np.frombuffer(self.binary, "<f4", count=access["count"] * 3, offset=offset).reshape(-1, 3).copy()

    def encode(self):
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        self.document["buffers"][0]["byteLength"] = len(self.binary)
        data = json.dumps(self.document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        data += b" " * (-len(data) % 4)
        return (struct.pack("<4sII", b"glTF", 2, 28 + len(data) + len(self.binary))
                + struct.pack("<I4s", len(data), b"JSON") + data
                + struct.pack("<I4s", len(self.binary), b"BIN\0") + self.binary)


def export_rigged_avatar(scene, path, influences=4):
    avatar = ModularAvatar(scene)
    if BODY not in avatar.nodes:
        raise ValueError("La escena necesita un nodo cuerpo-base creado por Genesis.")
    parts = {node: avatar.get(node, world=True) for node in avatar.nodes}
    body = parts[BODY]
    if not body.is_volume:
        raise ValueError("La exportación articulada requiere un cuerpo cerrado y orientado.")
    skeleton = humanoid_skeleton(body, parts)
    names, parents, bind = skeleton
    body_joints, body_weights = skin_weights(np.asarray(body.vertices), skeleton, influences)
    tree = cKDTree(body.vertices)
    facial = facial_morphs(body)
    glb = _GlbBuffers(trimesh.Scene(parts).export(file_type="glb"))
    data = glb.document
    nodes = data["nodes"]
    joint_start = len(nodes)
    name_to_index = {name: index for index, name in enumerate(names)}
    for index, (name, parent) in enumerate(zip(names, parents)):
        joint = {"name": name, "translation": (bind[index] - bind[parent] if parent >= 0 else bind[index]).tolist()}
        children = (np.flatnonzero(parents == index) + joint_start).tolist()
        if children:
            joint["children"] = children
        nodes.append(joint)
    data["scenes"][data.get("scene", 0)]["nodes"].append(joint_start)
    inverse = np.tile(np.eye(4, dtype=np.float32), (len(names), 1, 1))
    inverse[:, :3, 3] = -bind
    data["skins"] = [{"name": "Genesis Humanoid", "skeleton": joint_start,
                      "joints": list(range(joint_start, len(nodes))),
                      "inverseBindMatrices": glb.accessor(inverse.transpose(0, 2, 1), "MAT4")}]
    for node in nodes[:joint_start]:
        if "mesh" not in node:
            continue
        mesh_data = data["meshes"][node["mesh"]]
        name = node["name"]
        node["skin"] = 0
        rigid = None
        if name in EYES:
            rigid = "LeftEye" if name == EYES[0] else "RightEye"
        elif name.startswith(("cabello", "boca-", "dientes-", "lengua", *EYES)):
            rigid = "Head"
        for primitive in mesh_data["primitives"]:
            attributes = primitive["attributes"]
            vertices = glb.positions(attributes["POSITION"])
            if rigid:
                joints = np.zeros((len(vertices), 4), np.uint16)
                weights = np.zeros((len(vertices), 4), np.float32)
                joints[:, 0], weights[:, 0] = name_to_index[rigid], 1.
            else:
                nearest = tree.query(vertices)[1]
                joints, weights = body_joints[nearest], body_weights[nearest]
            attributes["JOINTS_0"] = glb.accessor(joints, "VEC4", 5123)
            attributes["WEIGHTS_0"] = glb.accessor(weights, "VEC4")
            if name == BODY:
                primitive["targets"] = [
                    {"POSITION": glb.accessor(deltas[nearest], "VEC3", bounds=True)}
                    for deltas in facial.values()
                ]
                mesh_data["weights"] = [0.] * len(facial)
                mesh_data.setdefault("extras", {})["targetNames"] = list(facial)
    rest = glb.accessor(np.array([0., .75, 1.5]), "SCALAR", bounds=True)
    channels, samplers = [], []
    for joint, angle in demo_rotations().items():
        quaternions = Rotation.from_euler("xyz", [(0, 0, 0), angle, (0, 0, 0)], degrees=True).as_quat()
        channels.append({"sampler": len(samplers), "target": {"node": joint_start + name_to_index[joint], "path": "rotation"}})
        samplers.append({"input": rest, "output": glb.accessor(quaternions, "VEC4"), "interpolation": "LINEAR"})
    data["animations"] = [{"name": "Genesis · prueba de articulación", "channels": channels, "samplers": samplers}]
    path = Path(path)
    if path.suffix.lower() != ".glb":
        raise ValueError("El avatar articulado se guarda como GLB.")
    with path.open("xb") as stream:
        stream.write(glb.encode())
    return {"joints": len(names), "morphs": list(facial), "skinned_nodes": sum("skin" in n for n in nodes)}


def preview_pose(scene, influences=4):
    avatar = ModularAvatar(scene)
    if BODY not in avatar.nodes:
        raise ValueError("Crea primero el cuerpo base.")
    parts = {node: avatar.get(node, world=True) for node in avatar.nodes}
    body = parts[BODY]
    skeleton = humanoid_skeleton(body, parts)
    joints, weights = skin_weights(np.asarray(body.vertices), skeleton, influences)
    matrices = pose_matrices(skeleton, demo_rotations())
    tree = cKDTree(np.asarray(body.vertices).copy())
    for name, mesh in parts.items():
        if name in EYES or name.startswith(("cabello", "boca-", "dientes-", "lengua", *EYES)):
            head = skeleton[0].index("Head")
            mesh.apply_transform(matrices[head])
            continue
        nearest = tree.query(mesh.vertices)[1]
        mesh.vertices = deform_pose(mesh.vertices, joints[nearest], weights[nearest], matrices)
    return trimesh.Scene(parts)


def preview_expression(scene, name, strength=1.0):
    if not 0 <= strength <= 1:
        raise ValueError("La intensidad de expresión debe quedar entre 0 y 1.")
    avatar = ModularAvatar(scene)
    if BODY not in avatar.nodes:
        raise ValueError("Crea primero el cuerpo base.")
    body = avatar.get(BODY)
    shapes = facial_morphs(body)
    if name not in shapes:
        raise ValueError("Expresión desconocida. Disponibles: " + ", ".join(shapes))
    body.vertices += shapes[name] * strength
    avatar.replace(BODY, body)
    return avatar.scene
