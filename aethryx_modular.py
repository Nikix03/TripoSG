"""Static anime avatar components with independent geometry and materials."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from aethryx_anatomy import anime_proportions, load_body_template
from aethryx_materials import color_gradient
from aethryx_profiles import BODY_PROFILES, BUNDLED_BODY


BODY = "cuerpo-base"
EYES = ("ojo-izquierdo", "ojo-derecho")
FACE_CONTROLS = {
    "jaw": "Mandíbula", "cheeks": "Mejillas", "nose": "Nariz", "mouth": "Boca",
    "eye_size": "Tamaño de ojos", "eye_spacing": "Separación de ojos", "lips": "Grosor de labios",
    "chin": "Mentón", "bridge": "Puente nasal", "brow": "Arco superciliar",
}


def anime_body(profile="Anime neutro"):
    mesh = load_body_template(BUNDLED_BODY)
    proportions = {
        "head": 1.18, "neck": .92, "waist": .96, "legs": 1.08, "hands": .94,
    }
    if profile != "Anime neutro":
        proportions.update(BODY_PROFILES[profile])
        proportions["head"] += .08
    face = {"jaw": .92, "nose": .91, "eye_size": 1.10}
    mesh.vertices = anime_proportions(mesh.vertices, mesh.bounds, proportions)
    mesh.vertices = facial_vertices(mesh, face)
    mesh.metadata.update(aethryx_style="genesis-anime-v1", aethryx_profile=profile, aethryx_proportions=proportions,
                         aethryx_face_edits=[face])
    apply_material(mesh, "#edbea9", projection="cylindrical")
    return mesh


def head_frame(mesh):
    low, high = mesh.bounds
    height = high[1] - low[1]
    if height <= 0:
        raise ValueError("El cuerpo debe tener altura y usar Y vertical.")
    head = mesh.vertices[mesh.vertices[:, 1] > low[1] + .87 * height]
    bounds = np.array([head.min(axis=0), head.max(axis=0)])
    if np.any(np.ptp(bounds, axis=0) <= 1e-8 * height):
        raise ValueError("No se pudo estimar el volumen de la cabeza.")
    return bounds


def facial_vertices(mesh, factors):
    scale = {key: float(factors.get(key, 1)) - 1 for key in FACE_CONTROLS}
    if not all(-.25 <= factor <= .25 for factor in scale.values()):
        raise ValueError("Los ajustes faciales admiten factores de 0,75 a 1,25.")
    low, high = head_frame(mesh)
    width, height, depth = high - low
    x, y, z = ((mesh.vertices - low) / (high - low)).T
    side = np.sign(x - .5)
    guides = mesh.metadata.get("aethryx_face_levels", {})
    def level(name, fallback):
        return (guides[name] - low[1]) / height if name in guides else fallback

    def weight(px, py, sx, sy):
        front = np.clip((z - .40) / .46, 0, 1)
        return np.exp(-((x - px) / sx)**2 - ((y - py) / sy)**2) * front

    jaw = weight(.5, level("jaw", -.06), .48, .22)
    cheek = weight(.5 + side * .25, .29, .22, .19)
    mouth_level = level("mouth", .02)
    eye_level = level("eye", .36)
    mouth = weight(.5, mouth_level, .24, .10)
    lip = weight(.5, mouth_level + .002, .21, .055)
    nose = weight(.5, level("nose", .34), .13, .20)
    chin = weight(.5, level("jaw", -.06) - .12, .24, .10)
    bridge = weight(.5, level("nose", .34) + .11, .07, .12)
    eye_x = .5 + side * .214
    eyes = weight(eye_x, eye_level, .19, .14)
    brow = weight(eye_x, eye_level + .145, .19, .075)
    result = mesh.vertices.copy()
    result[:, 0] += width * ((x - .5) * (jaw * scale["jaw"] + mouth * scale["mouth"])
                             + (x - eye_x) * eyes * scale["eye_size"]
                             + side * .214 * eyes * scale["eye_spacing"])
    result[:, 1] += height * ((y - eye_level) * eyes * scale["eye_size"] - .20 * jaw * scale["jaw"])
    result[:, 1] += height * (y - mouth_level - .002) * lip * scale["lips"]
    result[:, 1] -= height * (.16 * chin * scale["chin"] - .09 * brow * scale["brow"])
    result[:, 2] += depth * (.20 * cheek * scale["cheeks"] + .34 * nose * scale["nose"]
                             + .10 * mouth * scale["mouth"] + .11 * lip * scale["lips"])
    result[:, 2] += depth * (.18 * chin * scale["chin"] + .21 * bridge * scale["bridge"]
                             + .12 * brow * scale["brow"])
    return result


def apply_material(mesh, color, *, projection=None, roughness=.57, image=None):
    value = color.removeprefix("#")
    if len(value) != 6:
        raise ValueError("Introduce un color RGB hexadecimal de seis dígitos.")
    rgb = [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    if not 0 <= roughness <= 1:
        raise ValueError("La rugosidad debe estar entre 0 y 1.")
    previous = mesh.visual
    uv = getattr(previous, "uv", None)
    if projection:
        vertices = np.asarray(mesh.vertices)
        center = mesh.bounds.mean(axis=0)
        extents = np.maximum(mesh.extents, 1e-9)
        vertical = (vertices[:, 1] - mesh.bounds[0, 1]) / extents[1]
        if projection == "cylindrical":
            around = np.arctan2(vertices[:, 0] - center[0], vertices[:, 2] - center[2])
            uv = np.column_stack((.5 + around / (2 * np.pi), vertical))
        elif projection == "planar":
            uv = np.column_stack(((vertices[:, 0] - mesh.bounds[0, 0]) / extents[0], vertical))
        else:
            raise ValueError("Proyección UV inválida.")
        mesh.metadata["aethryx_uv_projection"] = projection
    if image is None:
        image = getattr(getattr(previous, "material", None), "baseColorTexture", None)
    if image is not None and uv is None:
        raise ValueError("Genera coordenadas UV antes de aplicar una textura.")
    current = getattr(previous, "material", None)
    material = current.copy() if isinstance(current, PBRMaterial) else PBRMaterial(metallicFactor=0)
    material.baseColorFactor = [*rgb, 255]
    material.baseColorTexture = image
    material.roughnessFactor = roughness
    mesh.visual = TextureVisuals(uv=uv, material=material)
    return mesh


def _iris(color, *, refined=False):
    value = color.removeprefix("#")
    if len(value) != 6:
        raise ValueError("El iris necesita un color RGB hexadecimal.")
    rgb = np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)])
    y, x = np.mgrid[-1:1:192j, -1:1:192j]
    if refined:
        radius = np.hypot(x / .23, y / .57)
        angle = np.arctan2(y / .57, x / .23)
        light = np.clip(.68 + .30 * y / .57 + .075 * np.sin(25 * angle + 8 * radius), .36, 1)
        sclera = np.array([255, 249, 246]) - np.maximum(0, -.7 - y)[..., None] * [34, 53, 68]
        mix = np.clip((1 - radius) / .045, 0, 1)[..., None]
        tint = np.clip(rgb * light[..., None] + 42 * np.maximum(0, y)[..., None], 0, 255)
        tint *= np.clip((1.03 - radius) / .16, .45, 1)[..., None]
        texture = sclera * (1 - mix) + tint * mix
        texture[np.hypot(x / .084, y / .25) < 1] = [23, 19, 34]
        texture[((x + .065) / .034)**2 + ((y + .17) / .057)**2 < 1] = [255, 255, 252]
        texture[(x - .079)**2 + (y - .17)**2 < .020**2] = [248, 251, 255]
        return Image.fromarray(texture.astype(np.uint8))
    radius = np.hypot(x / .20, y / .49)
    texture = np.full((192, 192, 3), [249, 244, 242], dtype=np.uint8)
    texture[radius < 1] = rgb
    texture[(radius > .87) & (radius < 1)] = rgb * .43
    texture[radius < .38] = [20, 15, 25]
    texture[(x + .045)**2 + (y + .15)**2 < .044**2] = 255
    return Image.fromarray(texture)


def avatar_eye(body, left, color, size=1, spacing=1, *, refined=False):
    if not .75 <= size <= 1.25 or not .75 <= spacing <= 1.25:
        raise ValueError("Tamaño y separación de ojos deben quedar entre 0,75 y 1,25.")
    low, high = head_frame(body)
    width, height, depth = high - low
    side = 1 if left else -1
    center_x = (low[0] + high[0]) / 2 + side * width * .214 * spacing
    center_y = low[1] + .36 * height
    distances = ((body.vertices[:, 0] - center_x) / width)**2 + ((body.vertices[:, 1] - center_y) / height)**2
    local = body.vertices[distances < .014]
    local = local[local[:, 2] > (low[2] + high[2]) / 2]
    if not len(local):
        raise ValueError("No se encontró superficie facial en la región del ojo.")
    forward = np.quantile(local[:, 2], .63)
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.vertices *= np.array([width * .117, height * .078, depth * .104]) * size
    mesh.apply_translation([center_x, center_y, forward - depth * .077])
    if refined:
        mesh.metadata["aethryx_iris"] = 3
    return apply_material(mesh, "#ffffff", projection="cylindrical", roughness=.23,
                          image=_iris(color, refined=refined))


def recolor_iris(mesh, color, roughness=.23):
    texture = _iris(color, refined=mesh.metadata.get("aethryx_iris") == 3)
    return apply_material(mesh, "#ffffff", projection="cylindrical", roughness=roughness, image=texture)


def fitted_eyelids(eye, color="#edbea9"):
    """Build two volumetric lid bands around the fitted eye opening in its frame."""
    center = eye.bounds.mean(axis=0)
    radii = eye.extents / 2
    u = np.linspace(-.93, .93, 37)
    arc = np.sqrt(1 - u**2)
    pieces = {}
    for name, sign, opening in (("superior", 1, .49), ("inferior", -1, .53)):
        v = sign * np.linspace(opening, .96, 5)[:, None] * arc + .09 * u
        z = np.sqrt(np.maximum(0, 1 - u**2 - v**2)) + .035
        points = np.stack((np.broadcast_to(u, v.shape), v, z), axis=2).reshape(-1, 3)
        patch = _grid_patch(center + points * radii, len(u))
        if np.mean(patch.face_normals[:, 2]) < 0:
            patch.invert()
        shell = _shell(patch, np.ones(len(patch.faces), dtype=bool), radii[2] * .012, radii[2] * .16)
        if shell.volume < 0:
            shell.invert()
        pieces[name] = apply_material(shell, color, projection="planar")
    return pieces


def _grid_patch(vertices, width):
    start = np.arange(len(vertices) - width)
    start = start[start % width != width - 1]
    faces = np.vstack((np.column_stack((start, start + 1, start + width + 1)),
                       np.column_stack((start, start + width + 1, start + width))))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def mouth_components(body):
    """Create an open-front mouth bowl and separate dentition and tongue."""
    low, high = head_frame(body)
    width, height, depth = high - low
    mid = (high + low) / 2
    mouth_y = low[1] + .020 * height
    v = body.vertices
    near = (abs(v[:, 0] - mid[0]) < .24 * width) & (abs(v[:, 1] - mouth_y) < .075 * height)
    if not near.any():
        raise ValueError("La plantilla no ofrece una superficie en la zona bucal.")
    lip_z = np.quantile(v[near, 2], .90)
    origin = np.array([mid[0], mouth_y, lip_z - .10 * depth])
    radius = np.array([width * .17, height * .062, depth * .26])
    angle = np.linspace(0, 2 * np.pi, 49)
    rows = np.linspace(0, 1.40, 10)[:, None]
    x = np.cos(rows) * np.cos(angle)
    y = np.cos(rows) * np.sin(angle)
    z = -np.sin(rows) * np.ones_like(angle)
    bowl = _grid_patch(origin + np.stack((x, y, z), axis=2).reshape(-1, 3) * radius, len(angle))
    rim = np.arange(len(bowl.vertices) - len(angle), len(bowl.vertices))
    cap = np.column_stack((rim[1:], rim[:-1], np.full(len(rim) - 1, len(bowl.vertices))))
    bowl = trimesh.Trimesh(vertices=np.vstack((bowl.vertices, origin - [0, 0, radius[2]])),
                          faces=np.vstack((bowl.faces, cap)), process=False)
    bowl.merge_vertices()
    bowl.visual = TextureVisuals(material=PBRMaterial(baseColorFactor=[66, 20, 30, 255], doubleSided=True))
    parts = {"boca-cavidad": bowl}
    for label, side in (("dientes-superiores", 1), ("dientes-inferiores", -1)):
        teeth = []
        for position in np.linspace(-.78, .78, 10):
            tooth = trimesh.creation.icosphere(subdivisions=1)
            tooth.vertices *= radius * [0.081, .30, .21]
            tooth.apply_translation(origin + radius * [position, .51 * side, -.42 - .28 * position**2])
            teeth.append(tooth)
        parts[label] = apply_material(trimesh.util.concatenate(teeth), "#f5e9dd", projection="planar", roughness=.32)
    tongue = trimesh.creation.icosphere(subdivisions=2)
    tongue.vertices *= radius * [.66, .33, .46]
    tongue.apply_translation(origin + radius * [0, -.48, -.52])
    parts["lengua"] = apply_material(tongue, "#b95a6e", projection="planar", roughness=.46)
    return parts

def _shell(body, selection, clearance, thickness, adapt=None):
    triangles = body.faces[selection]
    if not len(triangles):
        raise ValueError("La selección anatómica no contiene triángulos.")
    used, remap = np.unique(triangles, return_inverse=True)
    faces = remap.reshape(-1, 3)
    positions, normals = body.vertices[used], body.vertex_normals[used]
    if adapt:
        positions = adapt(positions)
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse] == 1]
    n = len(positions)
    a, b = boundary.T
    walls = np.concatenate((np.column_stack((b, a, a + n)), np.column_stack((b, a + n, b + n))))
    shell = trimesh.Trimesh(
        vertices=np.vstack((positions + normals * (clearance + thickness), positions + normals * clearance)),
        faces=np.vstack((faces, faces[:, ::-1] + n, walls)), process=False,
    )
    if not shell.is_watertight or not shell.is_winding_consistent or np.any(shell.area_faces <= 0):
        raise ValueError("El espesor no produjo una cubierta cerrada y orientada.")
    return shell


def fit_hair(body, length=.35, volume=.055, color="#33233d"):
    if not 0 <= length <= 1 or not .02 <= volume <= .15:
        raise ValueError("Longitud de cabello entre 0 y 1 y volumen entre 0,02 y 0,15.")
    low, high = head_frame(body)
    size = high - low
    center = body.triangles_center
    z = (center[:, 2] - low[2]) / size[2]
    height = (center[:, 1] - low[1]) / size[1]
    front = np.clip((z - .30) / .58, 0, 1)
    line = .28 + .41 * front + .045 * np.cos((center[:, 0] - low[0]) / size[0] * 6 * np.pi) * front
    selection = height > line

    def extend(vertices):
        vertices = vertices.copy()
        back = np.clip((high[2] - vertices[:, 2]) / size[2], 0, 1)**2
        rim = np.clip((high[1] - vertices[:, 1]) / (size[1] * .68), 0, 1)**2
        vertices[:, 1] -= size[1] * length * back * rim
        return vertices

    shell = _shell(body, selection, size[0] * .022, size[0] * volume, extend)
    return apply_material(shell, color, projection="cylindrical", roughness=.68)


def regional_hair(body, length=.35, volume=.055, root="#261a3a", tip="#9a578c"):
    """Partition the conformal cap by angle and preserve each region as a solid."""
    hair = fit_hair(body, length, volume)
    count = len(hair.vertices) // 2
    # The first half of vertices and faces represent the outer cap.
    centers = hair.triangles_center
    low, high = head_frame(body)
    mid = (low + high) / 2
    angle = np.arctan2(centers[:, 0] - mid[0], centers[:, 2] - mid[2])
    outer = (hair.faces < count).all(axis=1)
    result = {}
    regions = (
        ("flequillo", (abs(angle) <= .81)),
        ("lateral-izquierdo", (angle > .81) & (angle <= 2.23)),
        ("lateral-derecho", (angle < -.81) & (angle >= -2.23)),
        ("posterior", abs(angle) > 2.23),
    )
    texture = color_gradient(root, tip)
    for name, mask in regions:
        selection = mask & outer
        zone = hair.submesh([selection], repair=False)[0]
        sections = zone.split(only_watertight=False, repair=False)
        piece = trimesh.util.concatenate([
            _shell(section, np.ones(len(section.faces), dtype=bool), .00045, .00165)
            for section in sections
        ])
        result[f"cabello-{name}"] = apply_material(
            piece, "#ffffff", projection="cylindrical", roughness=.66, image=texture,
        )
    return result


def body_contacts(body, component):
    """Count surface contacts and component vertices enclosed by the closed body."""
    from pyvista import PolyData
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter, vtkSelectEnclosedPoints

    def poly(mesh):
        return PolyData(mesh.vertices, np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)))

    outer, inner = poly(body), poly(component)
    collision = vtkCollisionDetectionFilter()
    collision.SetInputData(0, outer)
    collision.SetInputData(1, inner)
    collision.SetTransform(0, vtkTransform())
    collision.SetTransform(1, vtkTransform())
    collision.SetCollisionModeToAllContacts()
    collision.Update()
    inside = None
    if body.is_watertight:
        enclosed = vtkSelectEnclosedPoints()
        enclosed.SetInputData(inner)
        enclosed.SetSurfaceData(outer)
        enclosed.SetTolerance(1e-6)
        enclosed.Update()
        mask = enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
        if mask is not None:
            from vtkmodules.util.numpy_support import vtk_to_numpy
            inside = int(np.count_nonzero(vtk_to_numpy(mask)))
    return {"contacts": collision.GetNumberOfContacts(), "inside": inside}


def clear_hair_penetrations(body, piece, *, clearance=.002, maximum_shift=.035):
    """Displace a closed lock locally on a copy while fixing the textured root."""
    from pyvista import PolyData
    from scipy.sparse import coo_matrix
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

    if not body.is_volume or not 0 < clearance <= .01 or not clearance < maximum_shift <= .06:
        raise ValueError("Se requiere un cuerpo cerrado y una holgura y desplazamiento válidos.")
    before = body_contacts(body, piece)
    uv = getattr(piece.visual, "uv", None)
    if not any(before.values()) or uv is None or not piece.is_volume:
        return piece.copy(), {"before": before, "after": before, "root_contacts": 0, "changed": False}
    surface = PolyData(body.vertices, np.column_stack((np.full(len(body.faces), 3), body.faces)))
    locator = vtkImplicitPolyDataDistance()
    locator.SetInput(surface)
    vertices = np.asarray(piece.vertices).copy()
    height = body.extents[1]
    strength = np.clip((.999 - uv[:, 1]) / .10, 0., 1.)
    root_contacts = sum(locator.EvaluateFunction(point) < 0 for point in vertices[strength == 0])
    edges = piece.edges_unique
    adjacency = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(vertices), len(vertices))).tocsr()
    adjacency = adjacency + adjacency.T
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)), 1)
    candidate = piece.copy()
    best, after = piece.copy(), before
    for _ in range(6):
        moves = np.zeros_like(vertices)
        for index, point in enumerate(candidate.vertices):
            closest = [0., 0., 0.]
            signed = locator.EvaluateFunctionAndGetClosestPoint(point, closest)
            target = height * clearance
            if signed < target and abs(signed) > 1e-9:
                moves[index] = (point - np.asarray(closest)) * ((target - signed) / signed)
        offset = (candidate.vertices - vertices) + (moves + .22 * (adjacency @ moves) / degree) * strength[:, None]
        length = np.linalg.norm(offset, axis=1)
        offset *= np.minimum(1, height * maximum_shift / np.maximum(length, 1e-12))[:, None]
        candidate.vertices = vertices + offset
        current = body_contacts(body, candidate)
        if (candidate.is_volume and all(current[key] <= after[key] for key in current)
                and current != after
                and min(hair_silhouette_overlap(piece, candidate, size=128).values()) >= .78):
            best, after = candidate.copy(), current
    residual = sum(locator.EvaluateFunction(point) < 0 for point in best.vertices[strength > 0])
    return best, {"before": before, "after": after, "root_contacts": int(root_contacts),
                  "movable_inside": int(residual), "changed": after != before}


def fit_component_to_patch(body, patch, generated):
    """Align a Y-up generated component with an editable Genesis surface patch."""
    if not body.is_volume:
        raise ValueError("Se necesita un cuerpo base cerrado y orientado.")
    if any(len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all() for mesh in (patch, generated)):
        raise ValueError("El parche y la malla generada deben contener triángulos finitos.")
    target = np.quantile(patch.vertices, [.02, .98], axis=0)
    source = np.quantile(generated.vertices, [.02, .98], axis=0)
    target_size, source_size = target[1] - target[0], source[1] - source[0]
    axes = np.argsort(target_size)[-2:]
    epsilon = max(float(body.extents[1]) * 1e-5, 1e-10)
    if np.any(target_size[axes] < epsilon) or np.any(source_size[axes] < epsilon):
        raise ValueError("El ajuste requiere extensión medible en dos ejes del parche.")
    scale = float(np.exp(np.log(target_size[axes] / source_size[axes]).mean()))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("La escala del componente no es válida.")
    result = generated.copy()
    result.apply_translation(-source.mean(axis=0))
    result.apply_scale(scale)
    result.apply_translation(target.mean(axis=0))
    report = body_contacts(body, result)
    result.metadata["aethryx_component_fit"] = {
        "uniform_scale": scale, "axes": axes.tolist(), "closed_volume": bool(result.is_volume), **report,
    }
    return result, report


def refine_component_fit(body, patch, original):
    """Search proper axis orientations, then locally clear small body penetrations."""
    from scipy.spatial.transform import Rotation
    from pyvista import PolyData
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

    rest, rest_report = fit_component_to_patch(body, patch, original)
    reference = np.asarray(patch.vertices)
    guide = reference[::max(1, len(reference) // 1500)]
    tree = cKDTree(guide)
    target = np.quantile(reference, [.02, .98], axis=0)
    target_size = np.diff(target, axis=0)[0]
    axes = np.argsort(target_size)[-2:]
    ranked = []
    for matrix in Rotation.create_group("O").as_matrix():
        source = original.copy()
        transform = np.eye(4)
        transform[:3, :3] = matrix
        source.apply_transform(transform)
        bounds = np.quantile(source.vertices, [.02, .98], axis=0)
        source_size = np.diff(bounds, axis=0)[0]
        if np.any(source_size[axes] <= 1e-10):
            continue
        scale = float(np.exp(np.log(target_size[axes] / source_size[axes]).mean()))
        source.vertices = (source.vertices - bounds.mean(axis=0)) * scale + target.mean(axis=0)
        points = source.vertices[::max(1, len(source.vertices) // 1500)]
        cost = np.median(tree.query(points)[0]) + np.median(cKDTree(points).query(guide)[0])
        ranked.append((cost, source, matrix, scale))

    from scipy.sparse import coo_matrix
    surface = PolyData(body.vertices, np.column_stack((np.full(len(body.faces), 3), body.faces)))
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(surface)
    best, report = rest, rest_report
    limit = .04 * body.extents[1]
    clearance = .003 * body.extents[1]
    aligned = sorted(ranked, key=lambda row: row[0])[:3]
    identity = next(row for row in ranked if np.allclose(row[2], np.eye(3)))
    if all(row is not identity for row in aligned):
        aligned.append(identity)
    for cost, source, rotation, scale in aligned:
        for up in (.02, .04, .06):
            trial = source.copy()
            trial.apply_translation([0, up * body.extents[1], 0])
            points = trial.vertices[::max(1, len(trial.vertices) // 1500)]
            shifted = np.median(tree.query(points)[0]) + np.median(cKDTree(points).query(guide)[0])
            if shifted > cost + .035 * body.extents[1]:
                continue
            current = body_contacts(body, trial)
            if current["inside"] + current["contacts"] < report["inside"] + report["contacts"]:
                best, report = trial, current
                best.metadata["aethryx_component_fit"] = {
                    "uniform_scale": scale, "rotation": rotation.tolist(), "crown_lift": up,
                    "closed_volume": bool(trial.is_volume), **current,
                }
    for _, piece, rotation, scale in aligned:
        initial = piece.vertices.copy()
        baseline_lengths = piece.edges_unique_length.copy()
        rows, cols = piece.edges_unique.T
        edges = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(initial), len(initial))).tocsr()
        edges = edges + edges.T
        degree = np.maximum(np.asarray(edges.sum(axis=1)).ravel(), 1)
        for _ in range(3):
            moves = np.zeros_like(initial)
            for index, vertex in enumerate(piece.vertices):
                closest = [0., 0., 0.]
                depth = distance.EvaluateFunctionAndGetClosestPoint(vertex, closest)
                if depth < -1e-8:
                    moves[index] = (np.asarray(closest) - vertex) * min(2., 1 + clearance / -depth)
            offsets = (piece.vertices - initial) + moves + .35 * (edges @ moves) / degree[:, None]
            norm = np.linalg.norm(offsets, axis=1)
            piece.vertices = initial + offsets * np.minimum(1., limit / np.maximum(norm, 1e-12))[:, None]
        current = body_contacts(body, piece)
        change = np.abs(np.log(np.maximum(piece.edges_unique_length, 1e-12) / np.maximum(baseline_lengths, 1e-12)))
        strain = float(np.quantile(change, .95))
        if (strain <= .35 and (not original.is_volume or piece.is_volume)
                and (current["inside"] + current["contacts"]) < (report["inside"] + report["contacts"])):
            best, report = piece, current
            best.metadata["aethryx_component_fit"] = {
                "uniform_scale": scale, "rotation": rotation.tolist(), "edge_log_change_p95": strain,
                "closed_volume": bool(piece.is_volume), **current,
            }
    return best, report


def hair_silhouette_overlap(first, second, size=256):
    """Orthographic surface occupancy in front, profile and top views."""
    import cv2

    bounds = np.vstack((first.bounds, second.bounds))
    result = {}
    for name, axes in (("front", [0, 1]), ("side", [2, 1]), ("top", [0, 2])):
        low = bounds[:, axes].min(axis=0)
        span = max(float(np.ptp(bounds[:, axes], axis=0).max()), 1e-10)
        masks = []
        for mesh in (first, second):
            pixels = np.rint((mesh.vertices[:, axes] - low) * (size - 3) / span + 1).astype(np.int32)
            mask = np.zeros((size, size), np.uint8)
            for triangle in pixels[mesh.faces]:
                if cv2.contourArea(triangle):
                    cv2.fillConvexPoly(mask, triangle, 1)
            masks.append(mask)
        union = np.count_nonzero(masks[0] | masks[1])
        result[name] = np.count_nonzero(masks[0] & masks[1]) / union if union else 0.
    return result


def remove_nonmanifold_flaps(mesh):
    """Remove isolated triangles attached by one overloaded edge, preserving vertex attributes."""
    counts = np.bincount(mesh.edges_unique_inverse)
    incidence = counts[mesh.faces_unique_edges]
    flaps = ((incidence > 2).sum(axis=1) == 1) & ((incidence == 1).sum(axis=1) == 2)
    if not flaps.any() or flaps.sum() > max(1, .02 * len(mesh.faces)):
        return mesh, 0
    candidate = mesh.copy()
    candidate.update_faces(~flaps)
    candidate.remove_unreferenced_vertices()
    remaining = np.bincount(candidate.edges_unique_inverse)
    if (np.count_nonzero(remaining > 2) >= np.count_nonzero(counts > 2)
            or np.count_nonzero(remaining == 1) > np.count_nonzero(counts == 1)):
        return mesh, 0
    return candidate, int(flaps.sum())


def generated_hair_passes_review(review):
    """Require both collision clearance and minimum agreement with the scalp guide."""
    coverage = review["crown_coverage_after"]
    return (
        review["closed_volume"] and review["contacts"] == review["inside"] == 0
        and coverage is not None and coverage >= .50
        and min(review["guide_after"].values()) >= .30
    )


def review_generated_hair(body, guide, original):
    """Clean an aligned copy while bounding changes to the silhouette and scalp coverage."""
    fitted, initial_contacts = refine_component_fit(body, guide, original)
    candidate = fitted.copy()
    height = float(body.extents[1])
    valid = candidate.nondegenerate_faces(height=max(height * 1e-7, 1e-10)) & candidate.unique_faces()
    candidate.update_faces(valid)
    removed_faces = int(len(valid) - np.count_nonzero(valid))
    candidate.remove_unreferenced_vertices()
    if not len(candidate.faces):
        raise ValueError("La malla capilar carece de triángulos utilizables.")
    candidate, removed_flaps = remove_nonmanifold_flaps(candidate)
    split = candidate.split(only_watertight=False, repair=False)
    main = max(split, key=lambda piece: piece.area)
    guide_tree = cKDTree(guide.vertices)
    keep = [part for part in split if part is main or (
        part.area >= .001 * candidate.area and np.median(guide_tree.query(part.vertices)[0]) <= .09 * height
    )]
    candidate = trimesh.util.concatenate(keep)
    candidate.metadata.update(fitted.metadata)
    open_boundary_before = int(np.count_nonzero(np.bincount(candidate.edges_unique_inverse) == 1))
    face_count = len(candidate.faces)
    # Trimesh repairs triangular and quadrilateral holes on this copy.
    candidate.fill_holes()
    added_faces = len(candidate.faces) - face_count
    contacts = body_contacts(body, candidate)
    retained = hair_silhouette_overlap(fitted, candidate)
    guide_before = hair_silhouette_overlap(guide, fitted)
    guide_after = hair_silhouette_overlap(guide, candidate)
    crown_y = body.bounds[1, 1] - .07 * height
    crown = guide.vertices[guide.vertices[:, 1] >= crown_y]

    def coverage(mesh):
        return float(np.mean(cKDTree(mesh.vertices).query(crown)[0] <= .025 * height)) if len(crown) else None

    before, after = coverage(fitted), coverage(candidate)
    accepted = (min(retained.values()) >= .94 and
                all(guide_after[key] + .025 >= value for key, value in guide_before.items()) and
                (before is None or after + .025 >= before) and
                all(contacts[key] <= value for key, value in initial_contacts.items()) and
                (not fitted.is_volume or candidate.is_volume))
    result = candidate if accepted else fitted
    final_contacts = contacts if accepted else initial_contacts
    if not accepted:
        retained = {view: 1.0 for view in retained}
        guide_after, after = guide_before, before
    seam_repair = False
    if accepted and not result.is_volume and getattr(original.visual, "kind", None) is None:
        repaired = _repair_hair_seams(result)
        if repaired is not None:
            repaired_contacts = body_contacts(body, repaired)
            repaired_silhouette = hair_silhouette_overlap(fitted, repaired)
            repaired_guide = hair_silhouette_overlap(guide, repaired)
            repaired_coverage = coverage(repaired)
            if (min(repaired_silhouette.values()) >= .94
                    and all(repaired_guide[key] + .025 >= value for key, value in guide_before.items())
                    and (before is None or repaired_coverage + .025 >= before)
                    and all(repaired_contacts[key] <= value for key, value in final_contacts.items())):
                repaired.metadata.update(result.metadata)
                result, final_contacts = repaired, repaired_contacts
                retained, guide_after, after = repaired_silhouette, repaired_guide, repaired_coverage
                seam_repair = True
    incidence = np.bincount(result.edges_unique_inverse)
    boundary_edges = int(np.count_nonzero(incidence == 1))
    result.metadata["aethryx_hair_review"] = {
        "cleanup_kept": bool(accepted), "degenerate_or_duplicate_faces": removed_faces,
        "removed_nonmanifold_flaps": removed_flaps if accepted else 0,
        "discarded_islands": len(split) - len(keep), "added_faces": added_faces,
        "boundary_before_fill": open_boundary_before, "boundary_edges": boundary_edges,
        "nonmanifold_edges": int(np.count_nonzero(incidence > 2)),
        "silhouette_retained": retained, "guide_before": guide_before, "guide_after": guide_after,
        "crown_coverage_before": before, "crown_coverage_after": after,
        "closed_volume": bool(result.is_volume), "seam_repair_kept": seam_repair, **final_contacts,
    }
    return result, final_contacts


def _repair_hair_seams(mesh):
    """Try a local seam repair on untextured geometry and reject surface crossings."""
    import pymeshlab
    from aethryx_anatomy import _triangle_intersections

    session = pymeshlab.MeshSet()
    session.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)))
    try:
        session.meshing_repair_non_manifold_edges(method="Remove Faces")
        session.meshing_repair_non_manifold_vertices()
        session.meshing_close_holes(maxholesize=10, selfintersection=True)
    except pymeshlab.PyMeshLabException:
        return None
    repaired = session.current_mesh()
    result = trimesh.Trimesh(repaired.vertex_matrix(), repaired.face_matrix(), process=False)
    result.fix_normals(multibody=True)
    if not result.is_volume or abs(len(result.faces) - len(mesh.faces)) > max(8, len(mesh.faces) * .02):
        return None
    if _triangle_intersections(result, limit=1):
        return None
    return result


GARMENT_STYLES = {"top": (.53, .735), "falda": (.32, .55), "vestido": (.31, .735)}
BODY_PREVIEW_BANDS = {"torso": (.56, .698, .117), "cadera": (.45, .51, .18)}


def visible_body_faces(mesh):
    bands = mesh.metadata.get("aethryx_hidden_body_bands", [])
    selected = np.ones(len(mesh.faces), dtype=bool)
    if not isinstance(bands, list) or not any(name in BODY_PREVIEW_BANDS for name in bands if isinstance(name, str)):
        return selected
    lo, hi = mesh.bounds
    height = hi[1] - lo[1]
    if height <= 0:
        return selected
    centers = mesh.triangles_center
    y = (centers[:, 1] - lo[1]) / height
    x = abs(centers[:, 0] - (lo[0] + hi[0]) / 2) / height
    for region, (lower, upper, reach) in BODY_PREVIEW_BANDS.items():
        if region in bands:
            selected &= ~((y >= lower) & (y <= upper) & (x < reach))
    return selected


def fit_garment(body, clearance=.012, thickness=.006, color="#4266ac", *, style="top"):
    if not .003 <= clearance <= .06 or not .001 <= thickness <= .03:
        raise ValueError("Holgura 0,003–0,06 y espesor 0,001–0,03 de la altura.")
    if not body.is_watertight:
        raise ValueError("La prueba de ajuste requiere un cuerpo cerrado.")
    if style not in GARMENT_STYLES:
        raise ValueError("Estilo de prenda desconocido.")
    mesh = _torso_loft(body, clearance, thickness, style)
    mesh.metadata["aethryx_garment"] = {"style": style, "clearance": clearance, "thickness": thickness}
    report = body_contacts(body, mesh)
    return apply_material(mesh, color, projection="cylindrical", roughness=.8), report


def fit_garment_with_clearance(body, clearance, thickness, style):
    attempts = [clearance]
    if .003 <= clearance <= .06:
        attempts.extend(c for c in (clearance * 1.35, clearance * 1.75) if c <= .06)
    for amount in attempts:
        mesh, report = fit_garment(body, amount, thickness, style=style)
        if report["contacts"] == 0 and report["inside"] == 0:
            return mesh, amount
    raise ValueError(f"Se rechazó el ajuste. Contactos {report['contacts']}, vértices dentro {report['inside']}.")


def fit_sleeves(body, clearance=.009, thickness=.004):
    """Fit independent double-wall sleeves to the upper arm and elbow."""
    from aethryx_rig import humanoid_skeleton

    if not body.is_volume or not .003 <= clearance <= .03 or not .002 <= thickness <= .015:
        raise ValueError("Las mangas requieren un cuerpo cerrado, holgura 0,003–0,03 y espesor 0,002–0,015.")
    names, _, locations = humanoid_skeleton(body)
    centers = body.triangles_center
    height = body.extents[1]
    pieces = {}
    for suffix, side in (("izquierda", "Left"), ("derecha", "Right")):
        shoulder, elbow, wrist = [locations[names.index(side + joint)] for joint in ("UpperArm", "LowerArm", "Hand")]
        upper = elbow - shoulder
        forearm = wrist - elbow
        end = elbow + .18 * forearm
        axis = end - shoulder
        along = (centers - shoulder) @ axis / (axis @ axis)
        separation = np.linalg.norm(centers - shoulder - along[:, None] * axis, axis=1)
        sign = 1 if side == "Left" else -1
        selected = ((along > .32) & (along < 1.) & (separation < .09 * height) &
                    (sign * (centers[:, 0] - shoulder[0]) > .024 * height))
        sleeve = _shell(body, selected, clearance * height, thickness * height)
        report = body_contacts(body, sleeve)
        if any(report.values()) or not sleeve.is_volume:
            raise ValueError(f"Manga {suffix} rechazada. Contactos {report['contacts']}, vértices interiores {report['inside']}.")
        pieces[f"prenda-manga-{suffix}"] = apply_material(sleeve, "#52318a", projection="cylindrical", roughness=.83)
    return pieces


def fit_trousers(body, color="#343957", *, waist=.56, cuff=.09, clearance=.002):
    """Fit a closed garment shell from the waist to both ankles."""
    if not body.is_volume:
        raise ValueError("El pantalón requiere un cuerpo cerrado y orientado.")
    if not .51 <= waist <= .60 or not .075 <= cuff <= .20 or not .001 <= clearance <= .012:
        raise ValueError("Cintura 0,51–0,60, puño 0,075–0,20 y holgura 0,001–0,012.")
    height = body.extents[1]
    centers = body.triangles_center
    y = (centers[:, 1] - body.bounds[0, 1]) / height
    torso_width = abs(centers[:, 0] - body.bounds.mean(axis=0)[0]) < .32 * body.extents[0]
    selected = (y > cuff) & (y < waist) & torso_width
    trousers = _shell(body, selected, clearance * height, .0011 * height)
    report = body_contacts(body, trousers)
    if any(report.values()) or not trousers.is_volume:
        raise ValueError(f"Pantalón rechazado. Contactos {report['contacts']}, vértices interiores {report['inside']}.")
    return {"prenda-pantalon": apply_material(trousers, color, projection="cylindrical", roughness=.83)}


def fit_fingerless_gloves(body, color="#49445d", *, full_fingers=False):
    """Build two fitted shells over the wrist and palm, optionally covering the fingers."""
    from aethryx_rig import humanoid_skeleton

    if not body.is_volume:
        raise ValueError("Los guantes requieren un cuerpo cerrado y orientado.")
    names, _, joints = humanoid_skeleton(body)
    height = body.extents[1]
    centers = body.triangles_center
    pieces = {}
    for side, sign, label in (("Left", 1, "izquierdo"), ("Right", -1, "derecho")):
        wrist, elbow = (joints[names.index(side + joint)] for joint in ("Hand", "LowerArm"))
        forearm = wrist - elbow
        delta = centers - wrist
        along = delta @ forearm / (forearm @ forearm)
        selected = (along > -.13) & (sign * delta[:, 0] > -.045 * height)
        if full_fingers:
            clearance, thickness = .0006, .00015
        else:
            selected &= along < .27
            clearance, thickness = .002, .0011
        glove = _shell(body, selected, clearance * height, thickness * height)
        report = body_contacts(body, glove)
        if any(report.values()) or not glove.is_volume:
            raise ValueError(f"Guante {label} rechazado. Contactos {report['contacts']}, vértices interiores {report['inside']}.")
        name = "prenda-guante-dedos" if full_fingers else "prenda-guante"
        pieces[f"{name}-{label}"] = apply_material(glove, color, projection="cylindrical", roughness=.8)
    return pieces


def _radial_profile(outline, directions):
    center = (outline.min(axis=0) + outline.max(axis=0)) / 2
    first = outline[:-1] - center
    edges = np.diff(outline, axis=0)
    denominator = directions[:, :1] * edges[:, 1] - directions[:, 1:] * edges[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (first[:, 0] * edges[:, 1] - first[:, 1] * edges[:, 0]) / denominator
        fraction = (first[:, 0] * directions[:, 1:] - first[:, 1] * directions[:, :1]) / denominator
    distance[(fraction < 0) | (fraction > 1) | ~np.isfinite(distance)] = -np.inf
    radius = distance.max(axis=1)
    if not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError("El contorno no admite un ajuste radial continuo.")
    return center, np.maximum(radius, .25 * np.roll(radius, 1) + .50 * radius + .25 * np.roll(radius, -1))


def fit_footwear(body, clearance=.007, thickness=.006, color="#353044", *, rounded=False):
    """Fit independent boot shells with a floor and an open ankle collar."""
    if not body.is_watertight or not .003 <= clearance <= .03 or not .002 <= thickness <= .03:
        raise ValueError("El calzado requiere cuerpo cerrado, holgura 0,003–0,03 y espesor 0,002–0,03.")
    low, high = body.bounds
    height = high[1] - low[1]
    ankle_height = low[1] + .077 * height
    mid_x = (low[0] + high[0]) / 2
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    ring = np.arange(len(theta))
    next_ring = np.roll(ring, -1)
    n = len(ring)
    shoes = {}
    for side, sign in (("izquierdo", 1), ("derecho", -1)):
        low_points = np.asarray(body.vertices)
        selected = ((low_points[:, 0] - mid_x) * sign > 0) & (low_points[:, 1] < ankle_height + .018 * height)
        footprint = low_points[selected][:, [0, 2]]
        if len(footprint) < 4:
            raise ValueError(f"No se pudo medir el pie {side}.")
        hull = footprint[ConvexHull(footprint).vertices]
        center, radii = _radial_profile(np.vstack((hull, hull[0])), directions)
        levels = (
            (low[1] - (clearance + thickness) * height, clearance + thickness),
            (ankle_height, clearance + thickness),
            (ankle_height, clearance),
            (low[1] - clearance * height, clearance),
        )
        if rounded:
            # Lower the front of the opening over the toes and round the sole edge.
            toe = .5 + .5 * np.sin(theta)
            opening = ankle_height - .038 * height * toe**2
            levels = (
                (levels[0][0], clearance + .55 * thickness),
                (levels[0][0] + .005 * height, clearance + thickness),
                (opening - .009 * height, clearance + thickness),
                (opening, clearance + thickness),
                (opening, clearance),
                (opening - .009 * height, clearance),
                (levels[-1][0], clearance),
            )
        geometry = []
        for level, margin in levels:
            xz = center + directions * (radii + margin * height)[:, None]
            geometry.append(np.column_stack((xz[:, 0], np.broadcast_to(level, n), xz[:, 1])))
        walls = np.vstack([np.column_stack((ring + step * n, next_ring + step * n,
                                             next_ring + (step + 1) * n, ring + (step + 1) * n))
                           for step in range(len(geometry) - 1)])
        last_ring = (len(geometry) - 1) * n
        floor_index = len(geometry) * n
        caps = np.vstack((
            np.column_stack((ring, next_ring, np.full(n, floor_index))),
            np.column_stack((next_ring + last_ring, ring + last_ring, np.full(n, floor_index + 1))),
        ))
        floor_centers = [[center[0], geometry[index][0, 1], center[1]] for index in (0, -1)]
        footwear = trimesh.Trimesh(
            vertices=np.vstack((*geometry, floor_centers)),
            faces=np.vstack((walls[:, [0, 1, 2]], walls[:, [0, 2, 3]], caps)), process=False,
        )
        footwear.fix_normals()
        report = body_contacts(body, footwear)
        if not footwear.is_volume or report != {"contacts": 0, "inside": 0}:
            raise ValueError(f"El calzado {side} produjo un defecto de volumen o contacto. {report}")
        footwear.metadata["aethryx_footwear"] = {"clearance": clearance, "thickness": thickness}
        shoes[f"prenda-zapato-{side}"] = apply_material(footwear, color, projection="cylindrical", roughness=.76)
    return shoes


def _torso_loft(body, clearance, thickness, style="top"):
    """Loft a closed two-layer top from torso sections, with clean planar rims."""
    low, high = body.bounds
    height = high[1] - low[1]
    start_level, end_level = GARMENT_STYLES[style]
    if body.metadata.get("aethryx_style") in {"genesis-anime-v3", "genesis-anime-v6"} and style != "falda":
        end_level = .695
    levels = np.linspace(start_level, end_level, 36 if style != "top" else 32)
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    inside, outside = [], []
    for fraction_y in levels:
        level = low[1] + fraction_y * height
        section = body.section(plane_origin=[0, level, 0], plane_normal=[0, 1, 0])
        if section is None or not section.discrete:
            raise ValueError("El torso no forma un contorno cerrado a la altura de la prenda.")
        loops = [poly[:, [0, 2]] for poly in section.discrete]
        if style != "top" and fraction_y < .51:
            points = np.vstack(loops)
            hull = points[ConvexHull(points).vertices]
            outline = np.vstack((hull, hull[0]))
        else:
            outline = max(loops, key=lambda poly: abs(
                np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1])
            ))
        center, radius = _radial_profile(outline, directions)
        if style != "top":
            radius += height * .065 * np.clip((.52 - fraction_y) / .21, 0, 1)**2
        for rings, expansion in ((inside, clearance), (outside, clearance + thickness)):
            xz = center + directions * (radius + expansion * height)[:, None]
            rings.append(np.column_stack((xz[:, 0], np.full(len(theta), level), xz[:, 1])))
    ring_size = len(theta)
    ring = np.arange(ring_size)
    next_point = np.roll(ring, -1)
    vertex_count = len(levels) * ring_size
    quads = []
    for start in range(0, vertex_count - ring_size, ring_size):
        outer = np.column_stack((ring, next_point, next_point + ring_size, ring + ring_size)) + start
        quads.extend((outer, outer[:, ::-1] + vertex_count))
    for start in (0, vertex_count - ring_size):
        quads.append(np.column_stack((ring, next_point, next_point + vertex_count, ring + vertex_count)) + start)
    quads = np.vstack(quads)
    shell = trimesh.Trimesh(
        vertices=np.vstack((*outside, *inside)),
        faces=np.vstack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]])), process=False,
    )
    shell.fix_normals()
    if not shell.is_volume:
        raise ValueError("La prenda no forma un volumen cerrado.")
    return shell


class ModularAvatar:
    """Edit scene instances without changing shared geometry or the source scene."""

    def __init__(self, scene, hidden=()):
        self.scene = scene.copy() if scene is not None else trimesh.Scene()
        self.hidden = set(hidden)

    @property
    def nodes(self):
        return [node for node in self.scene.graph.nodes_geometry
                if isinstance(self.scene.geometry[self.scene.graph[node][1]], trimesh.Trimesh)]

    def get(self, node, world=False):
        transform, name = self.scene.graph[node]
        mesh = self.scene.geometry[name].copy()
        if world:
            mesh.apply_transform(transform)
        return mesh

    def add(self, node, mesh):
        if node in self.nodes:
            raise ValueError(f"Ya existe el componente {node}.")
        self.scene.add_geometry(mesh, node_name=node, geom_name=node)

    def remove(self, node):
        if node == BODY:
            raise ValueError("El cuerpo base se conserva. Puedes ocultarlo.")
        _, geometry = self.scene.graph[node]
        if any(self.scene.graph.transforms.parents.get(name) == node for name in self.scene.graph.nodes):
            raise ValueError("El nodo tiene hijos. Guarda un GLB de una sola pieza para editarlo.")
        self.scene.graph.transforms.remove_node(node)
        if geometry not in {self.scene.graph[n][1] for n in self.nodes}:
            self.scene.delete_geometry(geometry)
        self.hidden.discard(node)

    def replace(self, node, mesh):
        world, previous = self.scene.graph[node]
        parent = self.scene.graph.transforms.parents.get(node)
        if parent is None:
            raise ValueError("El nodo de geometría debe tener un padre para reemplazarlo.")
        local = self.scene.graph.transforms.edge_data[(parent, node)]["matrix"]
        geometry = self.scene.add_geometry(mesh, geom_name=f"{node}-geometry", transform=world)
        _, new_name = self.scene.graph[geometry]
        self.scene.graph.transforms.remove_node(geometry)
        self.scene.graph.update(frame_from=parent, frame_to=node, matrix=local, geometry=new_name)
        if previous not in {self.scene.graph[n][1] for n in self.nodes}:
            self.scene.delete_geometry(previous)

    def upsert(self, node, mesh):
        if node in self.nodes:
            self.replace(node, mesh)
        else:
            self.add(node, mesh)

    def import_file(self, path, node=None):
        mesh = load_body_template(Path(path))
        if node == BODY:
            raise ValueError("El cuerpo base se conserva. Abre otro proyecto para cambiarlo.")
        if node:
            self.replace(node, mesh)
        else:
            self.add(Path(path).stem, mesh)

    def export_piece(self, node, path):
        write_new_glb(trimesh.Scene({node: self.get(node, world=True)}), path)

    def export_outfit(self, path, *, visible_only=False):
        pieces = {node: self.get(node, world=True) for node in self.nodes
                  if node.startswith("prenda-") and (not visible_only or node not in self.hidden)}
        if not pieces:
            raise ValueError("Genera al menos una prenda para guardar el conjunto.")
        write_new_glb(trimesh.Scene(pieces), path)

    def apply_outfit(self, path, *, switch=False):
        incoming = ModularAvatar(trimesh.load(path, force="scene", process=False))
        if not incoming.nodes or not all(node.startswith("prenda-") for node in incoming.nodes):
            raise ValueError("El GLB del conjunto debe contener solamente nodos prenda-.")
        working = ModularAvatar(self.scene, self.hidden)
        for node in incoming.nodes:
            mesh = incoming.get(node, world=True)
            if node in working.nodes:
                mesh.apply_transform(np.linalg.inv(working.scene.graph[node][0]))
            working.upsert(node, mesh)
        if switch:
            incoming_nodes = set(incoming.nodes)
            working.hidden.update(node for node in working.nodes
                                  if node.startswith("prenda-") and node not in incoming_nodes)
            working.hidden.difference_update(incoming_nodes)
        self.scene = working.scene
        self.hidden = working.hidden


def write_new_glb(scene, path):
    path = Path(path)
    if path.suffix.lower() != ".glb":
        raise ValueError("La salida debe tener extensión GLB.")
    content = scene.export(file_type="glb")
    with path.open("xb") as stream:
        stream.write(content)
