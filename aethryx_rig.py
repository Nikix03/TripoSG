"""Native glTF skin, animation and facial morph export for modular Genesis scenes."""

import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from aethryx_modular import BODY, EYES, ModularAvatar, head_frame

EXPORT_PROFILES = {
    "Studio Original": (1., False), "VRChat PC Quality": (.8, False),
    "VRChat PC Optimized": (.47, True), "VRChat Mobile": (.30, True),
    "VRChat PC Hair Batch": (.47, True),
    "VRChat PC Preserve Shape": (1., True),
    "Studio Hair Chains": (1., False),
    "VRChat PC Hair Chains": (1., True),
    "Studio Finger Fit": (1., False),
}


def profile_parts(parts, profile):
    ratio, merge = EXPORT_PROFILES[profile]
    working = dict(parts)
    if ratio == 1. and not merge:
        return working
    from trimesh.visual.texture import TextureVisuals

    if ratio < 1.:
        import pymeshlab
    for name, mesh in (parts.items() if ratio < 1. else ()):
        if not name.startswith(("cabello-largo-", "cabello-flequillo-")):
            continue
        uv = getattr(mesh.visual, "uv", None)
        material = getattr(mesh.visual, "material", None)
        if uv is None or material is None or len(mesh.faces) < 160:
            continue
        session = pymeshlab.MeshSet()
        session.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32),
                                        w_tex_coords_matrix=uv[mesh.faces].reshape(-1, 2)))
        face_count = max(120, round(len(mesh.faces) * ratio))
        try:
            session.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=face_count, preservenormal=True, preserveboundary=True,
            )
            simplified = session.current_mesh()
            vertices, faces = simplified.vertex_matrix(), simplified.face_matrix()
            wedges = simplified.wedge_tex_coord_matrix()
        except pymeshlab.PyMeshLabException:
            picture = getattr(material, "baseColorTexture", None)
            if picture is None:
                continue
            bitmap = np.asarray(picture.convert("RGBA"))
            if not np.all(bitmap == bitmap[:, :1]):
                continue
            session = pymeshlab.MeshSet()
            session.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)))
            session.meshing_decimation_quadric_edge_collapse(
                targetfacenum=face_count, preservenormal=True, preservetopology=True,
            )
            simplified = session.current_mesh()
            vertices, faces = simplified.vertex_matrix(), simplified.face_matrix()
            distances, nearest = cKDTree(mesh.vertices).query(vertices, k=3)
            weights = 1 / np.maximum(distances, 1e-10)**2
            longitudinal = np.sum(uv[nearest, 1] * weights, axis=1) / np.sum(weights, axis=1)
            wedges = np.column_stack((np.zeros(faces.size), longitudinal[faces.ravel()]))
        if not len(faces) or len(faces) >= len(mesh.faces):
            continue
        shape = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        deviation = cKDTree(vertices).query(mesh.vertices)[0]
        if (mesh.is_volume and not shape.is_volume) or np.quantile(deviation, .95) > .07 * mesh.scale:
            continue
        indexed = np.column_stack((faces.ravel(), np.round(wedges, 8)))
        _, first, new_faces = np.unique(indexed, axis=0, return_index=True, return_inverse=True)
        source = faces.ravel()[first]
        component = trimesh.Trimesh(vertices=vertices[source], faces=new_faces.reshape(-1, 3), process=False,
                                    vertex_normals=shape.vertex_normals[source],
                                    visual=TextureVisuals(uv=wedges[first], material=material.copy()))
        working[name] = component
    if merge:
        _atlas_hair_gradients(working)
    bangs = {}
    for name, mesh in working.items():
        material = getattr(mesh.visual, "material", None)
        prefixes = (("cabello-flequillo-", "cabello-largo-")
                    if profile in {"VRChat PC Hair Batch", "VRChat PC Preserve Shape", "VRChat PC Hair Chains"}
                    else ("cabello-flequillo-",))
        if merge and name.startswith(prefixes) and material is not None:
            bangs.setdefault(hash(material), []).append(name)
    for candidates in bangs.values():
        if len(candidates) < 2:
            continue
        meshes = [working.pop(name) for name in candidates]
        # Identical materials use the same UV space, so merging does not resample textures.
        geometry = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False) for m in meshes
        ])
        geometry.visual = TextureVisuals(uv=np.vstack([m.visual.uv for m in meshes]),
                                         material=meshes[0].visual.material.copy())
        if profile == "VRChat PC Hair Chains":
            geometry.metadata["aethryx_lock_segments"] = [(name, len(mesh.vertices))
                                                          for name, mesh in zip(candidates, meshes)]
        working[candidates[0]] = geometry
    if profile in {"VRChat PC Hair Batch", "VRChat PC Preserve Shape", "VRChat PC Hair Chains"}:
        for eye in EYES:
            upper_name, lower_name = (f"{eye}-parpado-{suffix}" for suffix in ("superior", "inferior"))
            if upper_name not in working or lower_name not in working:
                continue
            upper, lower = working[upper_name], working[lower_name]
            upper_mat = getattr(upper.visual, "material", None)
            lower_mat = getattr(lower.visual, "material", None)
            if (upper_mat is None or lower_mat is None or hash(upper_mat) != hash(lower_mat)
                    or getattr(upper.visual, "uv", None) is None or getattr(lower.visual, "uv", None) is None):
                continue
            combined = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False) for part in (upper, lower)
            ])
            combined.visual = TextureVisuals(uv=np.vstack((upper.visual.uv, lower.visual.uv)),
                                             material=upper_mat.copy())
            combined.metadata["aethryx_lid_split"] = len(upper.vertices)
            del working[upper_name], working[lower_name]
            working[f"{eye}-parpado-par"] = combined
    if profile == "VRChat Mobile":
        _limit_mobile_textures(working)
    return working


def _limit_mobile_textures(parts):
    from PIL import Image
    from trimesh.visual.texture import TextureVisuals

    replacements = {}
    for name, mesh in parts.items():
        visual = mesh.visual
        material = getattr(visual, "material", None)
        uv = getattr(visual, "uv", None)
        if uv is None or material is None or not hasattr(material, "baseColorTexture"):
            continue
        fields = ("baseColorTexture", "normalTexture", "metallicRoughnessTexture", "occlusionTexture",
                  "emissiveTexture")
        oversized = [field for field in fields
                     if (image := getattr(material, field, None)) is not None and max(image.size) > 256]
        if not oversized:
            continue
        key = hash(material)
        if key not in replacements:
            reduced = material.copy()
            for field in oversized:
                image = getattr(material, field).copy()
                image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                setattr(reduced, field, image)
            replacements[key] = reduced
        copied = mesh.copy()
        copied.visual = TextureVisuals(uv=uv.copy(), material=replacements[key])
        parts[name] = copied


def _atlas_hair_gradients(parts):
    """Pack compatible opaque base-color maps without combining long locks."""
    from PIL import Image
    from trimesh.visual.texture import TextureVisuals

    groups = {}
    for name, mesh in parts.items():
        if not name.startswith(("cabello-flequillo-", "cabello-largo-")):
            continue
        material = getattr(mesh.visual, "material", None)
        image = getattr(material, "baseColorTexture", None)
        uv = getattr(mesh.visual, "uv", None)
        if image is None or uv is None or not np.all((uv >= -1e-8) & (uv <= 1 + 1e-8)):
            continue
        if (getattr(material, "alphaMode", None) not in (None, "OPAQUE") or
                any(getattr(material, field, None) is not None for field in (
                    "normalTexture", "emissiveTexture", "occlusionTexture", "metallicRoughnessTexture", "emissiveFactor",
                ))):
            continue
        key = (tuple(material.baseColorFactor) if material.baseColorFactor is not None else (),
               material.roughnessFactor, material.metallicFactor, material.doubleSided)
        groups.setdefault(key, []).append((name, mesh, image.convert("RGB")))

    for items in groups.values():
        tiles = {}
        locations = {}
        for name, _, image in items:
            key = image.size, image.tobytes()
            if key not in tiles:
                tiles[key] = (len(tiles), image)
            locations[name] = tiles[key][0]
        if len(tiles) < 2 or len(tiles) > 32:
            continue
        width, height = max(image.width for _, image in tiles.values()), max(image.height for _, image in tiles.values())
        pitch = width + 4
        bitmap = np.zeros((height + 4, len(tiles) * pitch, 3), dtype=np.uint8)
        for index, image in tiles.values():
            pixels = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
            bitmap[:, index * pitch:(index + 1) * pitch] = np.pad(pixels, ((2, 2), (2, 2), (0, 0)), mode="edge")
        material = items[0][1].visual.material.copy()
        material.baseColorTexture = Image.fromarray(bitmap)
        for name, mesh, _ in items:
            copy = mesh.copy()
            uv = np.clip(mesh.visual.uv, 0., 1.)
            uv[:, 0] = (uv[:, 0] * (width - 1) + locations[name] * pitch + 2.5) / bitmap.shape[1]
            uv[:, 1] = (uv[:, 1] * (height - 1) + 2.5) / bitmap.shape[0]
            copy.visual = TextureVisuals(uv=uv, material=material)
            parts[name] = copy


def humanoid_skeleton(body, eyes=None, offsets=None):
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
    positions = np.asarray(positions, dtype=np.float64)
    if offsets:
        if not isinstance(offsets, dict) or set(offsets) - set(names):
            raise ValueError("Los ajustes contienen una articulación desconocida.")
        for name, values in offsets.items():
            delta = np.asarray(values, dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all() or np.any(np.abs(delta) > .08):
                raise ValueError("Desplazamiento de articulación fuera del intervalo ±0,08 de la altura.")
            positions[names.index(name)] += delta * height
    return names, np.asarray(parents, dtype=np.intp), positions


def skin_weights(vertices, skeleton, influences=4, *, finger_focus=False):
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
    if finger_focus:
        digit_joint = np.array([name.endswith(("Proximal", "Intermediate", "Distal")) for name in names])
        selected = digit_joint[nearest[:, 0]]
        weights[selected] **= 3
        weights[selected] /= weights[selected].sum(axis=1, keepdims=True)
    joints, result = np.zeros((len(vertices), 4), dtype=np.uint16), np.zeros((len(vertices), 4), dtype=np.float32)
    joints[:, :influences], result[:, :influences] = nearest, weights
    return joints, result


def _clothing_binding(mesh, vertices, body, skeleton, nearest, joints, weights, influences):
    """Blend the lower skirt continuously across pelvis and thighs."""
    garment = mesh.metadata.get("aethryx_garment")
    style = garment.get("style") if isinstance(garment, dict) else None
    if style not in ("falda", "vestido"):
        return joints[nearest], weights[nearest]
    lower, upper = body.bounds
    height = upper[1] - lower[1]
    y = (vertices[:, 1] - lower[1]) / height
    x = (vertices[:, 0] - (lower[0] + upper[0]) / 2) / max(mesh.extents[0] / 2, 1e-6)
    mix = np.clip((.54 - y) / .08, 0, 1)
    mix = mix**2 * (3 - 2 * mix)
    left = .5 + .5 * np.tanh(1.5 * x)
    names = skeleton[0]
    slots = [names.index(name) for name in ("Hips", "LeftUpperLeg", "RightUpperLeg")]
    dense = np.zeros((len(vertices), len(names)), dtype=np.float64)
    np.add.at(dense, (np.arange(len(vertices))[:, None], joints[nearest]), weights[nearest] * (1 - mix[:, None]))
    dense[:, slots] += mix[:, None] * np.column_stack((np.full(len(vertices), .55), .45 * left, .45 * (1 - left)))
    ranked = np.argsort(-dense, axis=1, kind="stable")[:, :influences]
    result = np.take_along_axis(dense, ranked, axis=1)
    result /= result.sum(axis=1, keepdims=True)
    out_j = np.zeros((len(vertices), 4), np.uint16)
    out_w = np.zeros((len(vertices), 4), np.float32)
    out_j[:, :influences], out_w[:, :influences] = ranked, result
    return out_j, out_w


def _rig_attachment(name, mesh):
    if name in EYES:
        return "LeftEye" if name == EYES[0] else "RightEye"
    footwear = {"prenda-zapato-izquierdo": "LeftFoot", "prenda-zapato-derecho": "RightFoot"}
    if name in footwear and mesh.metadata.get("aethryx_footwear"):
        return footwear[name]
    if name.startswith(("cabello", "boca-", "dientes-", "lengua", "oreja-", "accesorio-hamsa-", *EYES)):
        return "Head"
    return None


def demo_rotations():
    return {"LeftUpperArm": (0, 0, 24), "LeftLowerArm": (0, 22, 20),
            "RightUpperArm": (0, 0, -22), "RightLowerArm": (0, -18, -16),
            "LeftUpperLeg": (-12, 0, 0), "LeftLowerLeg": (16, 0, 0), "Head": (0, 12, 0),
            "LeftHand": (-8, 10, 0), "LeftEye": (-4, 9, 0), "RightEye": (-4, 9, 0)}


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

    face_levels = mesh.metadata.get("aethryx_face_levels", {})
    lips = (face_levels["mouth"] - lower[1]) / height if "mouth" in face_levels else .02
    eye_level = (face_levels["eye"] - lower[1]) / height if "eye" in face_levels else .37
    mouth = region(.5, lips, .27, .14)
    lower_lip = mouth * np.clip((lips + .023 - y) / .11, 0, 1)
    corners = region(.5 + .19 * np.sign(x - .5), lips, .10, .10)
    result = {}
    for name, dx, dy, dz in (
        ("Smile", np.sign(x - .5) * corners * width * .031, corners * height * .077, 0),
        ("JawOpen", 0, -lower_lip * height * .19, -lower_lip * depth * .047),
        ("Sad", 0, -corners * height * .071, 0),
        ("Surprise", -(x - .5) * mouth * width * .12, -lower_lip * height * .13, mouth * depth * .052),
    ):
        result[name] = np.column_stack(np.broadcast_arrays(dx, dy, dz)).astype(np.float32)
    for name, center in (("BlinkLeft", .714), ("BlinkRight", .286)):
        zone = region(center, eye_level + .02, .13, .11) * np.clip((y - eye_level + .04) / .11, 0, 1)
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


def _eyelid_morphs(name, parts):
    for eye, side in zip(EYES, ("Left", "Right")):
        lash = name == f"{eye}-pestana" and parts[name].metadata.get("aethryx_fitted_lash") is True
        if (not name.startswith(f"{eye}-parpado-") and not lash) or eye not in parts:
            continue
        eyelid, globe = parts[name], parts[eye]
        upper = lash or name.endswith("-superior")
        radii = globe.extents / 2
        if np.any(radii <= 0):
            return {}
        x, y, _ = ((eyelid.vertices - globe.bounds.mean(axis=0)) / radii).T
        arc = np.sqrt(np.maximum(0, 1 - x * x))
        tilt = .09 * x
        split = eyelid.metadata.get("aethryx_lid_split")
        sign = np.where(np.arange(len(x)) < split, 1, -1) if split is not None else (1 if upper else -1)
        opening = np.where(np.asarray(sign) > 0, .49, .53)
        latitude = sign * (y - tilt) / np.maximum(arc, .05)
        influence = np.clip((.96 - latitude) / (.96 - opening), 0, 1)
        target = y + influence * (-.06 + sign * (.075 - opening)) * arc
        surface = lambda v: np.sqrt(np.maximum(0, 1 - x * x - v * v))
        displacement = np.column_stack((np.zeros(len(x)), (target - y) * radii[1],
                                        (surface(target) - surface(y)) * radii[2])).astype(np.float32)
        return {f"Blink{side}": displacement, "Blink": displacement.copy()}
    return {}


def _accessory_morphs(name, parts, facial, tree):
    if name == "boca-linea":
        nearest = tree.query(parts[name].vertices)[1]
        return {key: offset[nearest] for key, offset in facial.items() if not key.startswith("Blink")}
    return _eyelid_morphs(name, parts)


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


def hair_physics_guides(parts):
    """Keep per-lock centers in rest space before the export batches the geometry."""
    guides = []
    for name, mesh in parts.items():
        if not name.startswith("cabello-largo-"):
            continue
        uv = getattr(mesh.visual, "uv", None)
        if uv is None or len(uv) != len(mesh.vertices) or np.ptp(uv[:, 1]) < .8:
            continue
        longitudinal = uv[:, 1]
        points = [mesh.vertices[np.abs(longitudinal - level) <= np.min(abs(longitudinal - level)) + .012]
                  .mean(axis=0).tolist() for level in (1., .7, .35, 0.)]
        if np.isfinite(points).all() and np.linalg.norm(np.diff(points, axis=0), axis=1).min() > 1e-6:
            guides.append({"name": name, "attachment": "Head", "points": points})
    return guides


def hair_chain_weights(longitudinal, chain_start, influences=4):
    along = np.interp(np.clip(longitudinal, 0., 1.), [0., .35, .7, 1.], [3., 2., 1., 0.])
    low = np.minimum(along.astype(int), 2)
    fraction = along - low
    indices = np.zeros((len(along), 4), np.uint16)
    weights = np.zeros((len(along), 4), np.float32)
    if influences == 1:
        indices[:, 0] = chain_start + np.rint(along).astype(int)
        weights[:, 0] = 1
        return indices, weights
    indices[:, 0], indices[:, 1] = chain_start + low, chain_start + low + 1
    weights[:, 0], weights[:, 1] = 1 - fraction, fraction
    return indices, weights


def _batch_eye_details(glb, nodes):
    """Keep eye detail primitives intact while sharing their head-bound renderer."""
    meshes = glb.document["meshes"]
    by_name = {node.get("name"): node for node in nodes if "mesh" in node}
    for eye in EYES:
        batch = [by_name[name] for suffix in ("parpado-par", "pestana", "ceja")
                 if (name := f"{eye}-{suffix}") in by_name]
        if len(batch) < 2:
            continue
        names = list(dict.fromkeys(target for node in batch
                                  for target in meshes[node["mesh"]].get("extras", {}).get("targetNames", [])))
        primitives = []
        for node in batch:
            mesh = meshes[node.pop("mesh")]
            node.pop("skin")
            existing = mesh.get("extras", {}).get("targetNames", [])
            for source in mesh["primitives"]:
                primitive = source.copy()
                targets = dict(zip(existing, source.get("targets", [])))
                if names and set(names) - set(targets):
                    size = glb.document["accessors"][source["attributes"]["POSITION"]]["count"]
                    zero = {"POSITION": glb.accessor(np.zeros((size, 3)), "VEC3", bounds=True)}
                    primitive["targets"] = [targets.get(name, zero) for name in names]
                elif names:
                    primitive["targets"] = [targets[name] for name in names]
                primitives.append(primitive)
        batch[0]["mesh"] = len(meshes)
        batch[0]["skin"] = 0
        mesh = {"name": batch[0]["name"], "primitives": primitives}
        if names:
            mesh.update(weights=[0.] * len(names), extras={"targetNames": names})
        meshes.append(mesh)


def export_rigged_avatar(scene, path, influences=4, *, profile="Studio Original"):
    avatar = ModularAvatar(scene)
    if BODY not in avatar.nodes:
        raise ValueError("La escena necesita un nodo cuerpo-base creado por Genesis.")
    if profile not in EXPORT_PROFILES:
        raise ValueError("Perfil de exportación desconocido.")
    originals = {node: avatar.get(node, world=True) for node in avatar.nodes}
    parts = profile_parts(originals, profile)
    body = parts[BODY]
    if not body.is_volume:
        raise ValueError("La exportación articulada requiere un cuerpo cerrado y orientado.")
    skeleton = humanoid_skeleton(body, parts, scene.metadata.get("aethryx_joint_offsets"))
    names, parents, bind = skeleton
    if profile == "Studio Finger Fit":
        body_joints, body_weights = skin_weights(np.asarray(body.vertices), skeleton, influences, finger_focus=True)
    else:
        body_joints, body_weights = skin_weights(np.asarray(body.vertices), skeleton, influences)
    tree = cKDTree(body.vertices)
    facial = facial_morphs(body)
    glb = _GlbBuffers(trimesh.Scene(parts).export(file_type="glb", include_normals=True))
    data = glb.document
    guides = hair_physics_guides(originals)
    if guides:
        data.setdefault("extras", {})["aethryxHairGuides"] = guides
    hair_bones = {}
    if profile in {"Studio Hair Chains", "VRChat PC Hair Chains"}:
        names = list(names)
        parents = parents.tolist()
        bind = bind.tolist()
        for guide in guides:
            name = guide["name"]
            start = len(names)
            hair_bones[name] = start
            for index, point in enumerate(guide["points"]):
                names.append(f"{name}-junta-{index}")
                parents.append(names.index("Head") if index == 0 else start + index - 1)
                bind.append(point)
        parents, bind = np.asarray(parents), np.asarray(bind)
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
        rigid = _rig_attachment(name, parts[name])
        for primitive in mesh_data["primitives"]:
            attributes = primitive["attributes"]
            vertices = glb.positions(attributes["POSITION"])
            segments = parts[name].metadata.get("aethryx_lock_segments")
            if hair_bones and segments:
                indices = np.zeros((len(parts[name].vertices), 4), np.uint16)
                weights_by_vertex = np.zeros_like(indices, dtype=np.float32)
                offset = 0
                for lock, count in segments:
                    target = slice(offset, offset + count)
                    if lock in hair_bones:
                        indices[target], weights_by_vertex[target] = hair_chain_weights(
                            originals[lock].visual.uv[:, 1], hair_bones[lock], influences,
                        )
                    else:
                        indices[target, 0], weights_by_vertex[target, 0] = name_to_index["Head"], 1.
                    offset += count
                original_positions = parts[name].vertices
                if vertices.shape != original_positions.shape or not np.allclose(vertices, original_positions, atol=1e-7, rtol=0):
                    raise ValueError("El exportador cambió el orden de los vértices capilares agrupados.")
                joints, weights = indices, weights_by_vertex
            elif name in hair_bones:
                uv = parts[name].visual.uv
                source_indices = cKDTree(parts[name].vertices).query(vertices)[1]
                joints, weights = hair_chain_weights(uv[source_indices, 1], hair_bones[name], influences)
            elif rigid:
                joints = np.zeros((len(vertices), 4), np.uint16)
                weights = np.zeros((len(vertices), 4), np.float32)
                joints[:, 0], weights[:, 0] = name_to_index[rigid], 1.
            else:
                nearest = tree.query(vertices)[1]
                joints, weights = _clothing_binding(
                    parts[name], vertices, body, skeleton, nearest, body_joints, body_weights, influences,
                )
            attributes["JOINTS_0"] = glb.accessor(joints, "VEC4", 5123)
            attributes["WEIGHTS_0"] = glb.accessor(weights, "VEC4")
            shapes = facial if name == BODY else _accessory_morphs(name, parts, facial, tree)
            if shapes:
                mapping = nearest if name == BODY else cKDTree(parts[name].vertices).query(vertices)[1]
                primitive["targets"] = [
                    {"POSITION": glb.accessor(deltas[mapping], "VEC3", bounds=True)}
                    for deltas in shapes.values()
                ]
                mesh_data["weights"] = [0.] * len(shapes)
                mesh_data.setdefault("extras", {})["targetNames"] = list(shapes)
    if profile in {"VRChat PC Preserve Shape", "VRChat PC Hair Chains"}:
        _batch_eye_details(glb, nodes[:joint_start])
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


class PosePreview:
    def __init__(self, scene, influences=4, *, simplify=False, offsets=None, finger_focus=False):
        avatar = ModularAvatar(scene)
        if BODY not in avatar.nodes:
            raise ValueError("Crea primero el cuerpo base.")
        originals = {node: avatar.get(node, world=True) for node in avatar.nodes}
        body = originals[BODY]
        stored = scene.metadata.get("aethryx_joint_offsets") if offsets is None else offsets
        self.skeleton = humanoid_skeleton(body, originals, stored)
        if finger_focus:
            joints, weights = skin_weights(np.asarray(body.vertices), self.skeleton, influences, finger_focus=True)
        else:
            joints, weights = skin_weights(np.asarray(body.vertices), self.skeleton, influences)
        tree = cKDTree(np.asarray(body.vertices).copy())
        names = self.skeleton[0]
        self.parts, self.binding = {}, {}
        for name, mesh in originals.items():
            if simplify and len(mesh.faces) > 4000:
                try:
                    reduced = mesh.simplify_quadric_decimation(face_count=4000)
                    reduced.visual.material = getattr(mesh.visual, "material", None)
                    mesh = reduced
                except (ValueError, ImportError, RuntimeError, TypeError, AttributeError):
                    pass
            self.parts[name] = mesh
            attached = _rig_attachment(name, originals[name])
            if attached:
                self.binding[name] = names.index(attached)
            else:
                nearest = tree.query(mesh.vertices)[1]
                self.binding[name] = _clothing_binding(
                    originals[name], mesh.vertices, body, self.skeleton, nearest, joints, weights, influences,
                )
        self.morphs = facial_morphs(self.parts[BODY])
        preview_tree = cKDTree(self.parts[BODY].vertices)
        self.accessory_morphs = {name: shapes for name in self.parts
                                 if (shapes := _accessory_morphs(name, self.parts, self.morphs, preview_tree))}

    def t_pose(self):
        names, _, bind = self.skeleton
        rotations = {}
        for side, sign in (("Left", 1), ("Right", -1)):
            delta = bind[names.index(side + "Hand")] - bind[names.index(side + "UpperArm")]
            rotations[side + "UpperArm"] = (0, 0, sign * float(np.degrees(np.arctan2(-delta[1], abs(delta[0])))))
        return rotations

    def evaluate(self, rotations, expression=None, strength=0):
        if not 0 <= strength <= 1:
            raise ValueError("La intensidad de expresión requiere un valor de 0 a 1.")
        blend = expression if isinstance(expression, dict) else ({expression: strength} if expression else {})
        if set(blend) - set(self.morphs) or any(not 0 <= weight <= 1 for weight in blend.values()):
            raise ValueError("La mezcla contiene un morph o una intensidad inválidos.")
        matrices = pose_matrices(self.skeleton, rotations)
        transformed = {}
        for name, mesh in self.parts.items():
            vertices = np.asarray(mesh.vertices)
            targets = self.morphs if name == BODY else self.accessory_morphs.get(name, {})
            for key, weight in blend.items():
                if key in targets and weight:
                    vertices = vertices + targets[key] * weight
            binding = self.binding[name]
            if isinstance(binding, tuple):
                transformed[name] = deform_pose(vertices, *binding, matrices)
            else:
                matrix = matrices[binding]
                transformed[name] = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        bind = self.skeleton[2]
        points = np.einsum("nij,nj->ni", matrices[:, :3, :3], bind) + matrices[:, :3, 3]
        return transformed, points


def finger_curl_rotations(skeleton, hand, degrees):
    """Use the palm plane and each digit direction to define its flexion axis."""
    names, _, positions = skeleton
    if hand not in {"Left", "Right"} or not 0 <= degrees <= 90:
        raise ValueError("Elige una mano y una flexión de 0 a 90 grados.")
    fingers = ("Index", "Middle", "Ring", "Little")
    wrist = positions[names.index(hand + "Hand")]
    index = positions[names.index(hand + "IndexProximal")]
    little = positions[names.index(hand + "LittleProximal")]
    normal = np.cross(index - wrist, little - wrist)
    if np.linalg.norm(normal) < 1e-10:
        return {}
    normal /= np.linalg.norm(normal)
    if hand == "Right":
        normal *= -1
    rotations = {}
    for finger in fingers:
        proximal = hand + finger + "Proximal"
        middle = hand + finger + "Intermediate"
        direction = positions[names.index(middle)] - positions[names.index(proximal)]
        axis = np.cross(direction, normal)
        length = np.linalg.norm(axis)
        if length > 1e-10:
            rotations[proximal] = Rotation.from_rotvec(axis * (np.deg2rad(degrees) / length)).as_euler("xyz", degrees=True)
    return rotations


def hair_clearance_bodies(preview):
    """Express posed collision bodies in the rigid head attachment's rest space."""
    head = preview.skeleton[0].index("Head")
    bodies = {}
    for name, rotations in (("demostración", demo_rotations()), ("brazos elevados", preview.t_pose())):
        vertices, _ = preview.evaluate(rotations)
        body = preview.parts[BODY].copy()
        body.vertices = vertices[BODY]
        body.apply_transform(np.linalg.inv(pose_matrices(preview.skeleton, rotations)[head]))
        bodies[name] = body
    return bodies


def component_pose_contacts(scene, node):
    """Check a component in rest, demonstration pose and raised arms."""
    from aethryx_modular import body_contacts

    solver = PosePreview(scene)
    if node == BODY or node not in solver.parts:
        raise ValueError("Selecciona una pieza distinta del cuerpo base.")
    result = {}
    poses = {"reposo": {}, "demostración": demo_rotations(), "brazos elevados": solver.t_pose()}
    if node.startswith("prenda-guante-dedos-"):
        hand = "Left" if node.endswith("-izquierdo") else "Right"
        poses["flexión anatómica de dedos"] = finger_curl_rotations(solver.skeleton, hand, 20)
        poses["flexión de estrés (X)"] = {hand + finger + "Proximal": (20, 0, 0)
                                         for finger in ("Index", "Middle", "Ring", "Little")}
    for name, pose in poses.items():
        vertices, _ = solver.evaluate(pose)
        body, component = (solver.parts[key].copy() for key in (BODY, node))
        body.vertices, component.vertices = vertices[BODY], vertices[node]
        result[name] = body_contacts(body, component)
    return result


def preview_pose(scene, influences=4):
    preview = PosePreview(scene, influences)
    parts, _ = preview.evaluate(demo_rotations())
    for name, vertices in parts.items():
        preview.parts[name].vertices = vertices
    return trimesh.Scene(preview.parts)


def preview_expression(scene, name, strength=1.0):
    if not 0 <= strength <= 1:
        raise ValueError("La intensidad de expresión debe quedar entre 0 y 1.")
    avatar = ModularAvatar(scene)
    if BODY not in avatar.nodes:
        raise ValueError("Crea primero el cuerpo base.")
    body = avatar.get(BODY)
    shapes = facial_morphs(body)
    world_body = avatar.get(BODY, world=True)
    world_shapes = facial_morphs(world_body)
    tree = cKDTree(world_body.vertices)
    if name not in shapes:
        raise ValueError("Expresión desconocida. Disponibles: " + ", ".join(shapes))
    body.vertices += shapes[name] * strength
    avatar.replace(BODY, body)
    parts = {node: avatar.get(node, world=True) for node in avatar.nodes
             if node.startswith(EYES) or node == "boca-linea"}
    for node, mesh in parts.items():
        changes = _accessory_morphs(node, parts, world_shapes, tree)
        if name in changes:
            mesh.vertices += changes[name] * strength
            mesh.apply_transform(np.linalg.inv(avatar.scene.graph[node][0]))
            avatar.replace(node, mesh)
    return avatar.scene
