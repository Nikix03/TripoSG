"""Procedural anime head topology and fitted cosmetic pieces."""

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

from aethryx_materials import color_gradient
from aethryx_modular import (BODY, EYES, _iris, anime_body, apply_material, fit_garment_with_clearance,
                           fit_hair, fitted_eyelids, fit_footwear)


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


def sculpted_body(profile="Anime neutro", *, refined=False, facial_loops=False):
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
    if refined:
        rx = [0, .029, .035, .048, .064, .071, .072, .069, .054, 0]
        rz = [0, .026, .035, .043, .053, .058, .062, .059, .048, 0]
    span = height * (.199 if profile == "Anime femenino" else .194)
    layers = np.linspace(0, 1, 35)[1:-1]
    if facial_loops:
        # Allocate separate strips to the upper lip, lower lip, bridge and eyelid zones.
        layers = np.sort(np.r_[layers, [.311, .321, .334, .347, .416, .450, .536, .553, .568, .586, .605]])
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
        if refined:
            nose *= .80
            lips *= .75
            cheek *= .75
        if facial_loops:
            lips = .0048 * np.exp(-(u / .021)**4) * (
                np.exp(-((y - .335) / .012)**2) + .78 * np.exp(-((y - .312) / .011)**2)
                - .42 * np.exp(-((y - .324) / .006)**2))
            socket *= .84
            nose += .003 * np.exp(-(u / .007)**2 - ((y - .485) / .105)**2)
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
        aethryx_style="genesis-anime-v6" if facial_loops else "genesis-anime-v3" if refined else "genesis-anime-v2",
        aethryx_profile=profile,
        aethryx_eye_guide=[float(center[0]), float(neck_y + .575 * span), front_z, float(height)],
        aethryx_face_levels={name: float(neck_y + t * span)
                             for name, t in (("jaw", .22), ("mouth", .325), ("nose", .44), ("eye", .575))},
    )
    return apply_material(result, "#efbdab", projection="cylindrical")


def _sweep(points, radii, sides=10, *, closed=False):
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0) if closed else np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    reference = np.tile([0., 0., 1.], (len(points), 1))
    reference[np.abs(tangent[:, 2]) > .95] = [0, 1, 0]
    horizontal = np.cross(tangent, reference)
    horizontal /= np.linalg.norm(horizontal, axis=1, keepdims=True)
    vertical = np.cross(tangent, horizontal)
    angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    radii = np.broadcast_to(np.reshape(radii, (len(points), -1)), (len(points), 2))
    rings = points[:, None] + (
        radii[:, 0, None, None] * np.cos(angle)[None, :, None] * horizontal[:, None] +
        radii[:, 1, None, None] * np.sin(angle)[None, :, None] * vertical[:, None]
    )
    j = np.arange(sides)
    following = np.roll(j, -1)
    faces = []
    count = len(points) * sides
    for offset in range(0, count if closed else count - sides, sides):
        adjacent = (offset + sides) % count
        faces.extend((np.column_stack((j + offset, following + offset, j + adjacent)),
                      np.column_stack((following + offset, following + adjacent, j + adjacent))))
    vertices = rings.reshape(-1, 3)
    if not closed:
        last = count - sides
        faces.extend((np.column_stack((j, following, np.full(sides, count))),
                      np.column_stack((last + j, last + following, np.full(sides, count + 1)))))
        vertices = np.vstack((vertices, points[[0, -1]]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.vstack(faces), process=False)
    mesh.fix_normals()
    return mesh


def sculpted_face_parts(body, colors=("#6589d4", "#ae6798")):
    guide = body.metadata.get("aethryx_eye_guide")
    if not guide or len(guide) != 4:
        raise ValueError("Crea primero un cuerpo anime v2.")
    cx, cy, cz, height = guide
    style = body.metadata.get("aethryx_style")
    refined = style in {"genesis-anime-v3", "genesis-anime-v6"}
    parts = {}
    t = np.linspace(-1, 1, 25)
    for name, side, color in zip(EYES, (1, -1), colors):
        center = np.array([cx + side * .030 * height, cy, cz - .003 * height])
        eye = trimesh.creation.icosphere(subdivisions=3)
        eye.vertices *= np.array([.024, .016, .011] if refined else [.025, .019, .016]) * height
        if style == "genesis-anime-v6":
            eye.vertices[:, 1] *= 1 - .24 * (np.abs(eye.vertices[:, 0]) / (.024 * height))**2
        eye.apply_translation(center)
        if refined:
            eye.metadata["aethryx_iris"] = 3
        parts[name] = apply_material(eye, "#ffffff", projection="cylindrical", roughness=.26,
                                     image=_iris(color, refined=refined))
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
        if refined:
            lash[:, 2] -= height * .003 * (1 - t**2)
            brow[:, 1] -= height * .003
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


def nikix_hair(body, *, ribbons=False):
    """Original swept locks with separate textures and UVs following their lengths."""
    height = body.extents[1]
    cx, _, cz, _ = body.metadata["aethryx_eye_guide"]
    crown = body.bounds[1, 1]
    root = "#efb94f" if ribbons else "#e8b86c"
    parts = {"cabello-fondo": fit_hair(body, length=.58 if ribbons else .36, volume=.022, color=root)}
    t = np.linspace(0., 1., 36)
    for index, angle in enumerate(np.linspace(-np.pi, np.pi, 20, endpoint=False)):
        front = np.cos(angle) > .48
        tip = "#f19abc" if np.sin(angle) < 0 else "#8ccee9"
        radial = (.012 + .068 * np.sin(np.pi * .60 * t)) * height
        x = cx + np.sin(angle + .24 * t) * radial
        if front:
            x += height * (.037 * t**1.6)
            y = crown + height * (.007 - (.051 + .022 * abs(np.sin(angle))) * t**1.5)
            z = cz - .045 * height + np.cos(angle) * radial + .047 * height * t
        else:
            x += height * np.sin(angle) * (.035 * t**2 + .010 * np.sin(np.pi * 2 * t) * t)
            y = crown + height * (.006 - (.35 + .024 * np.sin(index * 2)) * t**1.35)
            z = cz - .044 * height + np.cos(angle) * radial - .042 * height * t
        radii = height * (.003 + .010 * np.sin(np.pi * t)**.55) * (1 - .86 * t**5)
        sides = 10
        if ribbons:
            tip = "#ff508f" if np.sin(angle) < 0 else "#32c6f5"
            wave = t**1.4 * np.sin(t * np.pi * 2 + index * .6)
            x += height * (.017 if front else .032) * wave
            z += height * (.004 if front else .019) * wave
            y += height * t**2 * .012 * np.sin(index * 1.6)
            width = height * (.003 + .020 * np.sin(np.pi * t)**.72) * (1 - .94 * t**5)
            radii = np.column_stack((width, width * (.26 if front else .20)))
            sides = 14
        mesh = _sweep(np.column_stack((x, y, z)), radii, sides=sides)
        mesh = apply_material(mesh, "#ffffff", projection="cylindrical",
                              roughness=.66, image=color_gradient(root, tip))
        v = np.r_[np.repeat(1 - t, sides), 1., 0.]
        mesh.visual.uv[:, 1] = v
        zone = "flequillo" if front else "largo"
        parts[f"cabello-{zone}-{index + 1:02}"] = mesh
    return parts


def _nikix_jewelry(body):
    height = body.extents[1]
    cx, eye_y, front, _ = body.metadata["aethryx_eye_guide"]
    gold = "#c89b4f"
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)

    def ring(center, rx, ry):
        flat = np.column_stack((rx * np.cos(theta), ry * np.sin(theta), np.zeros_like(theta)))
        return _sweep(center + height * flat, np.full(len(theta), height * .0013), closed=True)

    chest_y = body.bounds[0, 1] + .792 * height
    chest_z = np.asarray(body.vertices)[(abs(body.vertices[:, 1] - chest_y) < .009 * height) &
                                      (abs(body.vertices[:, 0] - cx) < .016 * height), 2].max()
    pendant = np.array([cx, chest_y, chest_z + .009 * height])
    pieces = [ring(pendant, .013, .016)]
    for end in (-.010, -.006, 0., .006, .010):
        t = np.linspace(0, 1, 12)
        points = pendant + height * np.column_stack((end * t**1.4, -.013 + .026 * t, .001 * t))
        pieces.append(_sweep(points, np.full(len(t), height * .00105)))
    xy = np.linspace(-1, 1, 40)
    chain = pendant + height * np.column_stack((.063 * xy, .017 + .065 * xy**2, -.043 * xy**2))
    pieces.append(_sweep(chain, np.full(len(xy), height * .00115)))
    parts = {"accesorio-collar-arbol": apply_material(trimesh.util.concatenate(pieces), gold, roughness=.33)}
    for suffix, side in (("izquierdo", 1), ("derecho", -1)):
        center = np.array([cx + side * .070 * height, eye_y - .033 * height, front - .034 * height])
        outline = np.array([[-.004, .007], [-.008, .001], [-.007, -.004], [-.004, -.002],
                            [-.005, -.009], [-.002, -.010], [-.001, -.004], [0, -.011],
                            [.003, -.010], [.004, -.003], [.006, -.009], [.008, -.006],
                            [.006, .002], [.003, .007]])
        outline = center + height * np.column_stack((outline, np.zeros(len(outline))))
        charm = _sweep(outline, np.full(len(outline), .00105 * height), closed=True)
        hook = ring(center + [0, .012 * height, 0], .004, .006)
        parts[f"accesorio-hamsa-{suffix}"] = apply_material(
            trimesh.util.concatenate((hook, charm)), gold, roughness=.33)
        ear = trimesh.creation.icosphere(subdivisions=2)
        ear.vertices *= height * np.array([.009, .017, .008])
        ear.apply_transform(trimesh.transformations.rotation_matrix(side * -.22, [0, 0, 1]))
        ear.apply_translation(center + height * np.array([-side * .001, .022, -.006]))
        parts[f"oreja-{suffix}"] = apply_material(ear, "#efbdab", projection="cylindrical")
    return parts


def nikix_preset(*, version=3):
    if version not in (3, 4, 5, 6):
        raise ValueError("Elige el preset Nikix 3, 4, 5 o 6.")
    v4 = version >= 4
    body = sculpted_body("Anime femenino", refined=True, facial_loops=version == 6)
    dress, _ = fit_garment_with_clearance(body, .016, .006, "vestido")
    dress = apply_material(dress, "#52318a" if v4 else "#655a9d", roughness=.8)
    pieces = {
        BODY: body,
        **sculpted_face_parts(body, ("#ed67a4", "#33bce3") if v4 else ("#e58caf", "#66beda")),
        **nikix_hair(body, ribbons=v4),
        "ropa-vestido": dress,
        **fit_footwear(body, color="#473060" if v4 else "#554878", rounded=v4),
        **_nikix_jewelry(body),
    }
    if v4:
        height = body.extents[1]
        vertices = np.asarray(dress.vertices)
        rim = vertices[np.isclose(vertices[:, 1], vertices[:, 1].max())]
        neckline = rim[:len(rim) // 2].copy()
        neckline[:, 1] += .001 * height
        trim = _sweep(neckline, np.full(len(neckline), .0023 * height), closed=True)
        trim = apply_material(trim, "#e4abda", projection="cylindrical", roughness=.45)
        trim.visual.material.emissiveFactor = [.17, .026, .09]
        trim.metadata["aethryx_garment"] = dress.metadata["aethryx_garment"].copy()
        pieces["ropa-ribete-emisivo"] = trim
    if version >= 5:
        _detail_nikix_materials(pieces)
    return pieces


def _detail_nikix_materials(pieces):
    body = pieces[BODY]
    _, _, front, h = body.metadata["aethryx_eye_guide"]
    u, v = np.meshgrid(np.linspace(0, 1, 1024), np.linspace(1, 0, 1024))
    mouth = (body.metadata["aethryx_face_levels"]["mouth"] - body.bounds[0, 1]) / body.extents[1]
    eye = (body.metadata["aethryx_face_levels"]["eye"] - body.bounds[0, 1]) / body.extents[1]
    depth = max(front - body.bounds.mean(axis=0)[2], .025 * h)
    offset = np.arctan2(.037 * h, depth) / (2 * np.pi)
    cheeks = sum(np.exp(-((u - center) / .035)**2 - ((v - (mouth + eye) * .5) / .014)**2)
                 for center in (.5 - offset, .5 + offset))
    lips = np.exp(-((u - .5) / .031)**4 - ((v - mouth) / .0045)**2)
    warmth = cheeks[..., None] * [-5, -26, -13] + lips[..., None] * [-7, -30, -15]
    skin = Image.fromarray(np.uint8(np.clip(np.array([239, 189, 171]) + warmth, 0, 255)))
    pieces[BODY] = apply_material(body, "#ffffff", image=skin, roughness=.62)
    cap = pieces["cabello-fondo"]
    # Uneven, narrow striations follow the cylindrical coordinates around the cap.
    x = u + .028 * np.sin(2 * np.pi * v)
    glints = (np.cos(2 * np.pi * 43 * x) + .45 * np.cos(2 * np.pi * 19 * x)) / 1.45
    blond = np.array([233, 178, 78]) + glints[..., None] * [20, 23, 19]
    pieces["cabello-fondo"] = apply_material(
        cap, "#ffffff", image=Image.fromarray(np.uint8(np.clip(blond, 0, 255))), roughness=.65,
    )
    dress = pieces["ropa-vestido"]
    warp = np.cos(2 * np.pi * 80 * u) * np.cos(2 * np.pi * 94 * v)
    thread = np.exp(-((v - .90) / .012)**2) + .65 * np.exp(-((v - .095) / .012)**2)
    fabric = np.array([82, 49, 138]) + warp[..., None] * 4 + thread[..., None] * [59, 32, 34]
    pieces["ropa-vestido"] = apply_material(
        dress, "#ffffff", image=Image.fromarray(np.uint8(np.clip(fabric, 0, 255))), roughness=.8,
    )


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
