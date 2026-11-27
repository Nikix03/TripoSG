"""Procedural anime head topology and fitted cosmetic pieces."""

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from aethryx_materials import color_gradient
from aethryx_modular import EYES, _iris, anime_body, apply_material, fit_hair, fitted_eyelids


def _neck_loop(mesh):
    edges = mesh.edges_unique[np.bincount(mesh.edges_unique_inverse) == 1]
    links = {}
    for a, b in edges:
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    if not links or any(len(neighbors) != 2 for neighbors in links.values()):
        raise ValueError("El corte anatómico no produjo un anillo de cuello simple.")
    start = min(links)
    loop = [start]
    previous, current = start, links[start][0]
    while current != start and len(loop) < len(links):
        loop.append(current)
        previous, current = current, next(vertex for vertex in links[current] if vertex != previous)
    if current != start or len(loop) != len(links):
        raise ValueError("La plantilla contiene más de una abertura en el cuello.")
    return np.asarray(loop)


def sculpted_body(profile="Anime neutro"):
    """Replace the upper topology and weld the procedural head to the preserved torso."""
    original = anime_body(profile)
    lower, upper = original.bounds
    height = upper[1] - lower[1]
    neck_y = lower[1] + .833 * height
    vertices, faces, _ = trimesh.intersections.slice_faces_plane(
        original.vertices, original.faces, [0, -1, 0], [0, neck_y, 0],
    )
    trunk = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    join = _neck_loop(trunk)
    ring = np.asarray(trunk.vertices[join])
    center = ring.mean(axis=0)
    theta = np.unwrap(np.arctan2(ring[:, 0] - center[0], ring[:, 2] - center[2]))
    direction = 1 if np.median(np.diff(theta)) > 0 else -1
    uniform = theta[0] + direction * np.arange(len(theta)) * 2 * np.pi / len(theta)
    intervals = np.array([0, .10, .19, .27, .43, .58, .73, .86, .94, 1])
    rx = [0, .029, .033, .045, .063, .073, .073, .068, .053, 0]
    rz = [0, .026, .034, .042, .052, .056, .059, .057, .047, 0]
    span = height * (.199 if profile == "Anime femenino" else .194)
    layers = np.linspace(0, 1, 35)[1:-1]
    loops = [join]
    position = [np.asarray(trunk.vertices)]
    front_z = None
    for y in layers:
        relax = min(y / .25, 1)
        angle = theta * (1 - relax) + uniform * relax
        broad = .98 if profile == "Anime femenino" else 1.035 if profile == "Anime masculino" else 1
        x = np.sin(angle) * np.interp(y, intervals, rx) * height * broad
        forward = np.clip(np.cos(angle), 0, 1)
        z = np.cos(angle) * np.interp(y, intervals, rz) * height
        z += height * np.interp(y, intervals, [0, .002, .020, .021, .015, .010, .001, -.005, -.008, -.008])
        u = x / height
        nose = np.exp(-(u / .012)**2 - ((y - .44) / .095)**2) * .014
        lips = np.exp(-(u / .023)**2 - ((y - .325) / .023)**2) * .005
        cheek = np.exp(-((abs(u) - .043) / .018)**2 - ((y - .445) / .11)**2) * .008
        socket = np.exp(-((abs(u) - .030) / .025)**4 - ((y - .575) / .085)**2) * .017
        z += (nose + lips + cheek - socket) * height * forward**3
        # Retain the precise neck seam while distributing the new head's vertices.
        blend = np.clip(y / .14, 0, 1)
        blend = blend**2 * (3 - 2 * blend)
        x = ring[:, 0] * (1 - blend) + (center[0] + x) * blend
        z = ring[:, 2] * (1 - blend) + (center[2] + z) * blend
        start = len(trunk.vertices) + (len(loops) - 1) * len(join)
        loops.append(np.arange(start, start + len(join)))
        position.append(np.column_stack((x, np.full(len(join), neck_y + y * span), z)))
        if front_z is None and y >= .575:
            front_z = float(z[np.argmin((x - center[0] - .030 * height)**2 +
                                        np.where(np.cos(angle) > 0, 0, 1))])
    faces = [np.asarray(trunk.faces)]
    for row, following in zip(loops, loops[1:]):
        faces.extend((np.column_stack((row, np.roll(row, -1), following)),
                      np.column_stack((following, np.roll(row, -1), np.roll(following, -1)))))
    tip = sum(len(row) for row in position)
    row = loops[-1]
    faces.append(np.column_stack((row, np.roll(row, -1), np.full(len(row), tip))))
    position.append(np.array([[center[0], neck_y + span, center[2] - .008 * height]]))
    result = trimesh.Trimesh(vertices=np.vstack(position), faces=np.vstack(faces), process=False)
    result.fix_normals()
    if not result.is_volume or np.any(result.area_faces <= 1e-12):
        raise ValueError("La nueva cabeza no produjo un cuerpo cerrado y orientado.")
    result.metadata.update(
        aethryx_style="genesis-anime-v2",
        aethryx_profile=profile,
        aethryx_eye_guide=[float(center[0]), float(neck_y + .575 * span), front_z, float(height)],
        aethryx_face_levels={name: float(neck_y + t * span)
                             for name, t in (("jaw", .22), ("mouth", .325), ("nose", .44), ("eye", .575))},
    )
    return apply_material(result, "#efbdab", projection="cylindrical")


def _sweep(points, radii, sides=10):
    tangent = np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    reference = np.tile([0., 0., 1.], (len(points), 1))
    reference[np.abs(tangent[:, 2]) > .95] = [0, 1, 0]
    horizontal = np.cross(tangent, reference)
    horizontal /= np.linalg.norm(horizontal, axis=1, keepdims=True)
    vertical = np.cross(tangent, horizontal)
    angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = points[:, None] + radii[:, None, None] * (
        np.cos(angle)[None, :, None] * horizontal[:, None] +
        np.sin(angle)[None, :, None] * vertical[:, None]
    )
    j = np.arange(sides)
    following = np.roll(j, -1)
    faces = []
    for offset in range(0, (len(points) - 1) * sides, sides):
        faces.extend((np.column_stack((j, following, j + sides)) + offset,
                      np.column_stack((following, following + sides, j + sides)) + offset))
    last = (len(points) - 1) * sides
    faces.extend((np.column_stack((j, following, np.full(sides, len(points) * sides))),
                  np.column_stack((last + j, last + following, np.full(sides, len(points) * sides + 1)))))
    mesh = trimesh.Trimesh(vertices=np.vstack((rings.reshape(-1, 3), points[[0, -1]])),
                           faces=np.vstack(faces), process=False)
    mesh.fix_normals()
    return mesh


def sculpted_face_parts(body, colors=("#6589d4", "#ae6798")):
    guide = body.metadata.get("aethryx_eye_guide")
    if not guide or len(guide) != 4:
        raise ValueError("Crea primero un cuerpo anime v2.")
    cx, cy, cz, height = guide
    parts = {}
    t = np.linspace(-1, 1, 25)
    for name, side, color in zip(EYES, (1, -1), colors):
        center = np.array([cx + side * .030 * height, cy, cz - .003 * height])
        eye = trimesh.creation.icosphere(subdivisions=3)
        eye.vertices *= np.array([.025, .019, .016]) * height
        eye.apply_translation(center)
        parts[name] = apply_material(eye, "#ffffff", projection="cylindrical", roughness=.26, image=_iris(color))
        for lid, mesh in fitted_eyelids(eye, "#efbdab").items():
            parts[f"{name}-parpado-{lid}"] = mesh
        lash = np.column_stack((
            center[0] + t * height * .023,
            center[1] + height * (.011 * np.sqrt(np.maximum(0, 1 - t**2)) + .003 * t * side),
            center[2] + height * (.014 * np.sqrt(np.maximum(0, 1 - t**2)) + .0035),
        ))
        brow = lash.copy()
        brow[:, 1] += height * (.020 + .004 * (1 - t**2))
        brow[:, 2] -= height * .003
        parts[f"{name}-pestana"] = apply_material(_sweep(lash, height * (.0009 + .0011 * (1 - abs(t)))),
                                                  "#241b34", projection="planar")
        parts[f"{name}-ceja"] = apply_material(_sweep(brow, height * (.001 + .0017 * (1 - abs(t)))),
                                               "#3c263b", projection="planar")
    mouth_y = body.metadata["aethryx_face_levels"]["mouth"]
    mouth_x = cx + t * height * .018
    line_y = mouth_y + height * (.0031 * t**2 - .0015 * (1 - t**2))
    front = np.asarray(body.vertices)[(body.vertices[:, 2] > cz - .025 * height) &
                                      (abs(body.vertices[:, 1] - mouth_y) < .022 * height)]
    if len(front) >= 4:
        distance, index = cKDTree(front[:, :2]).query(np.column_stack((mouth_x, line_y)), k=4)
        weights = 1 / np.maximum(distance, .001 * height)**2
        z = np.sum(front[index, 2] * weights, axis=1) / weights.sum(axis=1)
        line = _sweep(np.column_stack((mouth_x, line_y, z + height * .0011)),
                      height * (.00045 + .00065 * (1 - abs(t))))
        parts["boca-linea"] = apply_material(line, "#784351", projection="planar", roughness=.65)
    return parts


def hair_locks(body, length=.45, root="#291d40", tip="#a45483"):
    if not 0 <= length <= 1:
        raise ValueError("La longitud de los mechones está entre 0 y 1.")
    lower, upper = body.bounds
    height = body.extents[1]
    head = body.vertices[body.vertices[:, 1] > upper[1] - .18 * height]
    mid = (head.min(axis=0) + head.max(axis=0)) / 2
    crown = upper[1] + .006 * height
    texture = color_gradient(root, tip)
    t = np.linspace(0, 1, 29)
    angles = np.linspace(-np.pi, np.pi, 18, endpoint=False)
    base = fit_hair(body, length=length * .53, volume=.029, color=root)
    parts = {"cabello-fondo": apply_material(base, "#ffffff", roughness=.76, image=texture)}
    for index, angle in enumerate(angles):
        front = np.clip(np.cos(angle), 0, 1)
        back = np.clip(-np.cos(angle), 0, 1)
        end_drop = (.115 - .061 * front + .068 * length * back) * height
        horizontal = (.011 + .071 * np.sin(t * np.pi * .60)) * height
        depth = (.009 + .062 * np.sin(t * np.pi * .58)) * height
        twist = angle + .16 * t * (1 if index % 2 else -1)
        points = np.column_stack((
            mid[0] + np.sin(twist) * horizontal,
            crown - end_drop * t**1.4,
            mid[2] + np.cos(twist) * depth,
        ))
        radius = height * (.011 * np.sin(np.pi * t)**.47 + .0028 * (1 - t) + .0009)
        zone = "flequillo" if front > .67 else "nuca" if back > .55 else "lateral"
        parts[f"cabello-{zone}-{index + 1:02}"] = apply_material(
            _sweep(points, radius), "#ffffff", projection="cylindrical", roughness=.72, image=texture,
        )
    return parts
