"""Static anime avatar components with independent geometry and materials."""

from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from aethryx_anatomy import anime_proportions, load_body_template
from aethryx_profiles import BODY_PROFILES, BUNDLED_BODY


BODY = "cuerpo-base"
EYES = ("ojo-izquierdo", "ojo-derecho")
FACE_CONTROLS = {
    "jaw": "Mandíbula", "cheeks": "Mejillas", "nose": "Nariz", "mouth": "Boca",
    "eye_size": "Tamaño de ojos", "eye_spacing": "Separación de ojos", "lips": "Grosor de labios",
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

    def weight(px, py, sx, sy):
        front = np.clip((z - .40) / .46, 0, 1)
        return np.exp(-((x - px) / sx)**2 - ((y - py) / sy)**2) * front

    jaw = weight(.5, -.06, .48, .22)
    cheek = weight(.5 + side * .25, .29, .22, .19)
    mouth = weight(.5, .02, .24, .10)
    lip = weight(.5, .022, .21, .055)
    nose = weight(.5, .34, .13, .20)
    eye_x = .5 + side * .214
    eyes = weight(eye_x, .36, .19, .14)
    result = mesh.vertices.copy()
    result[:, 0] += width * ((x - .5) * (jaw * scale["jaw"] + mouth * scale["mouth"])
                             + (x - eye_x) * eyes * scale["eye_size"]
                             + side * .214 * eyes * scale["eye_spacing"])
    result[:, 1] += height * ((y - .36) * eyes * scale["eye_size"] - .20 * jaw * scale["jaw"])
    result[:, 1] += height * (y - .022) * lip * scale["lips"]
    result[:, 2] += depth * (.20 * cheek * scale["cheeks"] + .34 * nose * scale["nose"]
                             + .10 * mouth * scale["mouth"] + .11 * lip * scale["lips"])
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


def _iris(color):
    value = color.removeprefix("#")
    if len(value) != 6:
        raise ValueError("El iris necesita un color RGB hexadecimal.")
    rgb = np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)])
    y, x = np.mgrid[-1:1:192j, -1:1:192j]
    radius = np.hypot(x / .20, y / .49)
    texture = np.full((192, 192, 3), [249, 244, 242], dtype=np.uint8)
    texture[radius < 1] = rgb
    texture[(radius > .87) & (radius < 1)] = rgb * .43
    texture[radius < .38] = [20, 15, 25]
    texture[(x + .045)**2 + (y + .15)**2 < .044**2] = 255
    return Image.fromarray(texture)


def avatar_eye(body, left, color, size=1, spacing=1):
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
    return apply_material(mesh, "#ffffff", projection="cylindrical", roughness=.23, image=_iris(color))


def recolor_iris(mesh, color, roughness=.23):
    return apply_material(mesh, "#ffffff", projection="cylindrical", roughness=roughness, image=_iris(color))


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
    strip = np.tile(np.linspace(1, 0, 256)[:, None, None], (1, 4, 1))
    rgb = [np.array(Image.new("RGB", (1, 1), color))[0, 0] for color in (root, tip)]
    texture = Image.fromarray(np.rint(strip * rgb[0] + (1 - strip) * rgb[1]).astype(np.uint8))
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


GARMENT_STYLES = {"top": (.53, .735), "falda": (.32, .55), "vestido": (.31, .735)}


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


def _torso_loft(body, clearance, thickness, style="top"):
    """Loft a closed two-layer top from torso sections, with clean planar rims."""
    low, high = body.bounds
    height = high[1] - low[1]
    levels = np.linspace(*GARMENT_STYLES[style], 36 if style != "top" else 32)
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
            from scipy.spatial import ConvexHull
            points = np.vstack(loops)
            hull = points[ConvexHull(points).vertices]
            outline = np.vstack((hull, hull[0]))
        else:
            outline = max(loops, key=lambda poly: abs(
                np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1])
            ))
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
            raise ValueError("El torso no admite un ajuste radial continuo con esta prenda.")
        radius = np.maximum(radius, .25 * np.roll(radius, 1) + .50 * radius + .25 * np.roll(radius, -1))
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

    def export_outfit(self, path):
        pieces = {node: self.get(node, world=True) for node in self.nodes if node.startswith("prenda-")}
        if not pieces:
            raise ValueError("Genera al menos una prenda para guardar el conjunto.")
        write_new_glb(trimesh.Scene(pieces), path)

    def apply_outfit(self, path):
        incoming = ModularAvatar(trimesh.load(path, force="scene", process=False))
        if not incoming.nodes or not all(node.startswith("prenda-") for node in incoming.nodes):
            raise ValueError("El GLB del conjunto debe contener solamente nodos prenda-.")
        working = ModularAvatar(self.scene, self.hidden)
        for node in incoming.nodes:
            mesh = incoming.get(node, world=True)
            if node in working.nodes:
                mesh.apply_transform(np.linalg.inv(working.scene.graph[node][0]))
            working.upsert(node, mesh)
        self.scene = working.scene


def write_new_glb(scene, path):
    path = Path(path)
    if path.suffix.lower() != ".glb":
        raise ValueError("La salida debe tener extensión GLB.")
    content = scene.export(file_type="glb")
    with path.open("xb") as stream:
        stream.write(content)
