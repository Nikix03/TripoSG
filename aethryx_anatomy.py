"""Fit a supplied humanoid template to calibrated orthographic references."""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp
import shutil

import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import trimesh

VIEWS = ("front", "side", "back")
PROPORTIONS = {
    "height": "Altura", "torso": "Largo de torso", "waist": "Ancho de cintura",
    "hips": "Ancho de caderas", "shoulders": "Ancho de hombros", "arms": "Largo de brazos",
    "legs": "Largo de piernas", "neck": "Ancho de cuello", "head": "Tamaño de cabeza",
    "hands": "Tamaño de manos", "feet": "Tamaño de pies",
}


def orthographic_projection(view, center, ground, crown, bounds, side=1):
    """Map a shared body height and ground-center to each image's pixel frame."""
    lower, upper = np.asarray(bounds)
    height = upper[1] - lower[1]
    if not np.isfinite([*center, ground, crown, height]).all() or height <= 0 or ground <= crown:
        raise ValueError("Marca primero el suelo y después la coronilla por encima del suelo.")
    scale = (ground - crown) / height
    axis = np.asarray({"front": (1, 0, 0), "side": (0, 0, -side), "back": (-1, 0, 0)}[view])
    mid = (lower + upper) / 2
    return np.array([
        [*(axis * scale), center[0] - scale * (axis @ mid)],
        [0, -scale, 0, ground + scale * lower[1]],
    ])


def reconstruct(manifest_path):
    """Return a fitted mesh, editable surface patches and reprojection errors."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return reconstruct_data(data, manifest_path.parent)


def load_body_template(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("La plantilla debe contener una malla triangular válida.")
    return mesh


def reconstruct_data(data, root=Path("."), *, deformation_report=None):
    if data.get("bundled_template"):
        from aethryx_profiles import BUNDLED_BODY
        template = BUNDLED_BODY
    else:
        template = root / data["template"]
    mesh = load_body_template(template)
    original = mesh.copy() if deformation_report is not None else None
    ids = np.asarray(data["vertex_indices"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 4
            or len(np.unique(ids)) != len(ids) or ids.min() < 0
            or ids.max() >= len(mesh.vertices)):
        raise ValueError("Se requieren al menos cuatro índices de vértice distintos.")
    anchors = mesh.vertices[ids]
    if len(np.unique(anchors, axis=0)) != len(anchors):
        raise ValueError("Los puntos de control deben ocupar posiciones distintas.")

    matrices, offsets, observations, masks, visibility, reference_colors = [], [], [], [], [], []
    if set(data["views"]) != set(VIEWS):
        raise ValueError("Define exactamente las vistas front, side y back.")
    for name in VIEWS:
        view = data["views"][name]
        projection = np.asarray(view["projection"], dtype=float)
        points = np.asarray(view["landmarks"], dtype=float)
        with Image.open(root / view["image"]) as image:
            width, height = image.size
            reference_colors.append(np.asarray(image.convert("RGB")) if view.get("mask") else None)
        if projection.shape != (2, 4) or points.shape != (len(ids), 2):
            raise ValueError(f"Calibración o puntos inválidos en {name}.")
        visible = np.isfinite(points).all(axis=1)
        if (not np.isfinite(projection).all() or not visible.any()
                or not np.isnan(points[~visible]).all()
                or np.any(points[visible] < 0) or np.any(points[visible] >= [width, height])):
            raise ValueError(f"Calibración o puntos inválidos en {name}.")
        matrix = projection[:, :3]
        if np.linalg.matrix_rank(matrix) != 2:
            raise ValueError(f"La proyección de {name} debe tener rango 2.")
        matrices.append(matrix)
        offsets.append(projection[:, 3])
        observations.append(points)
        visibility.append(visible)
        mask = None
        if view.get("mask"):
            with Image.open(root / view["mask"]) as image:
                if image.size != (width, height):
                    raise ValueError(f"La máscara de {name} debe coincidir con su imagen.")
                mask = np.asarray(image.convert("RGB"))
        masks.append(mask)

    system = np.vstack(matrices)
    if np.linalg.matrix_rank(system) != 3:
        raise ValueError("Las cámaras no permiten recuperar profundidad.")
    targets = []
    measured = np.asarray(visibility).T.repeat(2, axis=1)
    values = np.hstack([points - offset for points, offset in zip(observations, offsets)])
    for index, rows in enumerate(measured):
        target, _, rank, _ = np.linalg.lstsq(system[rows], values[index, rows], rcond=None)
        if rank != 3:
            raise ValueError(f"Las observaciones no resuelven la profundidad del vértice {ids[index]}.")
        targets.append(target)
    smoothing = float(data.get("smoothing", 0))
    if not 0 <= smoothing <= .1:
        raise ValueError("El suavizado del ajuste debe estar entre 0 y 0,1.")
    deform = RBFInterpolator(anchors, targets - anchors, kernel="linear", degree=0,
                             smoothing=smoothing * mesh.extents[1])
    mesh.vertices = mesh.vertices + deform(mesh.vertices)
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 0):
        raise ValueError("El ajuste produjo geometría inválida o caras colapsadas.")
    errors = {
        name: float(np.sqrt(np.mean(np.sum(
            (mesh.vertices[ids[visible]] @ matrix.T + offset - points[visible]) ** 2, axis=1,
        ))))
        for name, matrix, offset, points, visible in zip(
            VIEWS, matrices, offsets, observations, visibility,
        )
    }
    patches = partition_surfaces(mesh, matrices, offsets, masks, data.get("palette", {}),
                                 reference_colors=reference_colors)
    proportions = data.get("proportions", {})
    if proportions:
        bounds = mesh.bounds.copy()
        # A scene without masks may hold the same mesh object.
        for piece in {id(piece): piece for piece in (mesh, *patches.geometry.values())}.values():
            piece.vertices = anime_proportions(piece.vertices, bounds, proportions)
        if not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 0):
            raise ValueError("Las proporciones produjeron caras inválidas.")
    if deformation_report is not None:
        deformation_report.update(validate_deformation(original, mesh))
    return mesh, patches, errors


def anime_proportions(vertices, bounds, factors):
    values = {key: float(factors.get(key, 1)) - 1 for key in PROPORTIONS}
    if not all(0.75 <= factor + 1 <= 1.35 for factor in values.values()):
        raise ValueError("Las proporciones deben estar entre 0,75 y 1,35.")
    lower, upper = bounds
    if upper[1] <= lower[1]:
        raise ValueError("El cuerpo ajustado perdió su altura.")
    center = (lower + upper) / 2
    height = upper[1] - lower[1]
    y = (vertices[:, 1] - lower[1]) / height
    result = vertices.copy()
    x = vertices[:, 0] - center[0]
    z = vertices[:, 2] - center[2]
    reach = np.abs(x) / max((upper[0] - lower[0]) / 2, 1e-9)

    def smooth(value):
        clipped = np.clip(value, 0, 1)
        return clipped**2 * (3 - 2 * clipped)

    def band(mid, radius):
        return smooth(1 - np.abs(y - mid) / radius)

    head = smooth((y - 0.86) / 0.10) * values["head"]
    torso_width = (band(.55, .10) * values["waist"] + band(.48, .12) * values["hips"]) * (
        1 - smooth((reach - .38) / .23)
    )
    neck = band(.855, .055) * values["neck"]
    shoulder = band(.76, .17) * values["shoulders"]
    result[:, 0] += x * (head + torso_width + neck + shoulder)
    result[:, 2] += z * (head + torso_width * .5 + neck)
    arm_weight = smooth((reach - .39) / .24) * smooth((y - .4) / .17) * values["arms"]
    result[:, 0] += (x - np.sign(x) * (upper[0] - lower[0]) * .19) * arm_weight
    result[:, 1] += (y - .78) * height * arm_weight
    hand = smooth((reach - .79) / .16) * smooth((y - .36) / .2) * values["hands"]
    result[:, 0] += (x - np.sign(x) * (upper[0] - lower[0]) * .39) * hand
    result[:, 1] += (y - .62) * height * hand
    result[:, 2] += (z - (upper[2] - lower[2]) * .2) * hand
    foot = (1 - smooth(y / .12)) * values["feet"]
    result[:, 2] += z * foot
    result[:, 0] += (x - np.sign(x) * (upper[0] - lower[0]) * .22) * foot * .5
    result[:, 1] += height * (
        np.minimum(y - .48, 0) * values["legs"] + np.clip(y - .48, 0, .32) * values["torso"]
        + (y - .86) * head
    )
    ground = lower[1] - .48 * height * values["legs"]
    result[:, 1] += (result[:, 1] - ground) * values["height"]
    return result


def _triangle_intersections(mesh, limit=100):
    from pyvista import PolyData
    from vtkmodules.vtkCommonCore import vtkIdList
    from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator, vtkTriangle

    faces, vertices = np.asarray(mesh.faces), np.asarray(mesh.vertices)
    triangles = vertices[faces]
    locator = vtkStaticCellLocator()
    locator.SetDataSet(PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces))))
    locator.BuildLocator()
    candidates = vtkIdList()
    mins, maxs = triangles.min(axis=1), triangles.max(axis=1)
    bounds = np.stack((mins, maxs), axis=2).reshape(-1, 6)
    found = 0
    for index, box in enumerate(bounds):
        candidates.Reset()
        locator.FindCellsWithinBounds(box, candidates)
        count = candidates.GetNumberOfIds()
        others = np.fromiter((candidates.GetId(i) for i in range(count)), dtype=np.intp, count=count)
        others = others[(others > index) & (maxs[index] >= mins[others]).all(axis=1)
                        & (maxs[others] >= mins[index]).all(axis=1)]
        adjacent = (faces[others, :, None] == faces[index]).any(axis=(1, 2))
        for other in others[~adjacent]:
            if vtkTriangle.TrianglesIntersect(*triangles[index], *triangles[other]):
                found += 1
                if found == limit:
                    return found
    return found


def validate_deformation(original, mesh):
    if not np.array_equal(original.faces, mesh.faces):
        raise ValueError("La topología cambió antes de validar la deformación.")
    height = float(mesh.extents[1])
    if height <= 0 or not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 0):
        raise ValueError("El cuerpo ajustado contiene geometría inválida.")
    center_x = float(np.mean(original.bounds[:, 0]))
    mirror = mesh.vertices.copy()
    mirror[:, 0] = 2 * center_x - mirror[:, 0]
    distances = cKDTree(mesh.vertices).query(mirror)[0] / height
    before = original.area_faces
    valid = before > 0
    ratios = mesh.area_faces[valid] / before[valid]
    return {
        "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "positive_volume": bool(mesh.volume > 0),
        "extreme_area_faces": int(np.count_nonzero((ratios < 0.2) | (ratios > 5))),
        "reflected_normal_faces": int(np.count_nonzero(
            np.einsum("ij,ij->i", original.face_normals, mesh.face_normals) < 0,
        )),
        "intersection_pairs_up_to_100": _triangle_intersections(mesh),
        "symmetry_mean_height": float(distances.mean()),
        "symmetry_p95_height": float(np.quantile(distances, 0.95)),
    }


def blocking_geometry_issues(report):
    issues = []
    if not report["watertight"]:
        issues.append("El cuerpo tiene bordes abiertos o no manifold.")
    if not report["winding_consistent"]:
        issues.append("La orientación de las caras es incoherente.")
    elif report["watertight"] and not report["positive_volume"]:
        issues.append("El volumen orientado del cuerpo no es positivo.")
    if report["intersection_pairs_up_to_100"]:
        issues.append("Se detectaron autointersecciones.")
    if report["extreme_area_faces"] > .01 * report["faces"]:
        issues.append("Más del 1 % de las caras tiene un cambio de área extremo.")
    if report["reflected_normal_faces"] > .01 * report["faces"]:
        issues.append("Más del 1 % de las normales giró más de 90°.")
    return issues


def export_studio_reconstruction(folder, data, mesh, patches, report, errors, comparisons=None):
    """Keep the complete body and optional reference patches in separate files."""
    issues = blocking_geometry_issues(report)
    if issues:
        raise ValueError("\n".join(issues))
    output = Path(mkdtemp(prefix="genesis7-", dir=folder))
    try:
        body = output / "cuerpo-base.glb"
        trimesh.Scene({"cuerpo-base": mesh}).export(body, file_type="glb")
        modular = None
        components = [{"name": "cuerpo-base", "role": "body", "file": body.name}]
        if set(patches.geometry) - {"sin-asignar"}:
            modular = output / "parches.glb"
            patches.export(modular, file_type="glb")
            for name, piece in patches.geometry.items():
                if name == "sin-asignar":
                    continue
                path = output / f"componente-{len(components):02d}.glb"
                trimesh.Scene({name: piece}).export(path, file_type="glb")
                components.append({"name": name, "role": "surface-patch", "file": path.name})
        (output / "componentes.json").write_text(
            json.dumps(components, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        for name, comparison in (comparisons or {}).items():
            comparison.save(output / f"silueta-{name}.png")
        (output / "referencias.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "validacion.json").write_text(json.dumps({
            **report, "landmark_rmse_px_before_proportions": errors,
        }, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(output)
        raise
    return body, modular


def _unoccluded_faces(locator, centers, indices, direction, span, tolerance):
    from vtkmodules.vtkCommonCore import reference

    distance, coordinate, parametric = reference(0.0), [0.0] * 3, [0.0] * 3
    sub_id, cell_id = reference(0), reference(0)
    visible = []
    for index in indices:
        center = centers[index]
        hit = locator.IntersectWithLine(
            center + direction * span, center - direction * tolerance,
            tolerance, distance, coordinate, parametric, sub_id, cell_id,
        )
        if hit and cell_id.get() == index:
            visible.append(index)
    return np.asarray(visible, dtype=int)


def partition_surfaces(mesh, matrices, offsets, masks, palette, *, reference_colors=None):
    """Partition triangles using agreeing samples visible in the labeled views."""
    centers = mesh.triangles_center
    labels = np.zeros(len(centers), dtype=int)
    ambiguous = np.zeros(len(centers), dtype=bool)
    names = list(palette)
    if "sin-asignar" in names:
        raise ValueError("La etiqueta sin-asignar está reservada.")
    colors = np.asarray(list(palette.values()))
    if names and (colors.shape != (len(names), 3) or colors.dtype.kind not in "iu"
                  or np.any(colors < 0) or np.any(colors > 255)
                  or np.any(~colors.any(axis=1)) or len(np.unique(colors, axis=0)) != len(names)):
        raise ValueError("La paleta requiere colores RGB únicos y distintos del negro.")
    if not names or all(mask is None for mask in masks):
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="sin-asignar", geom_name="sin-asignar")
        return scene

    from pyvista import PolyData
    from vtkmodules.vtkCommonDataModel import vtkCellLocator

    triangles = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces))
    locator = vtkCellLocator()
    locator.SetDataSet(PolyData(mesh.vertices, triangles))
    locator.BuildLocator()
    span = max(float(mesh.scale) * 2, 1e-9)
    tolerance = span * 1e-8
    color_sum = np.zeros((len(centers), 3))
    color_weight = np.zeros(len(centers))
    if reference_colors is None:
        reference_colors = [None] * len(masks)

    for matrix, offset, mask, reference in zip(matrices, offsets, masks, reference_colors):
        if mask is None:
            continue
        pixels = np.floor(centers @ matrix.T + offset).astype(int)
        x, y = pixels.T
        # Pixel Y points down. The negative cross product points toward the camera.
        direction = -np.cross(*matrix)
        direction = direction / np.linalg.norm(direction)
        facing = mesh.face_normals @ direction
        front = facing > 1e-8
        visible = front & (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
        indices = np.flatnonzero(visible)
        sampled = mask[y[indices], x[indices]]
        for label, color in enumerate(colors, start=1):
            candidates = indices[np.all(sampled == color, axis=1)]
            matched = _unoccluded_faces(locator, centers, candidates, direction, span, tolerance)
            ambiguous[matched] |= (labels[matched] != 0) & (labels[matched] != label)
            labels[matched] = label
            if reference is not None:
                confidence = facing[matched] ** 2
                color_sum[matched] += reference[y[matched], x[matched]] * confidence[:, None]
                color_weight[matched] += confidence
    labels[ambiguous] = 0
    scene = trimesh.Scene()
    for label, name in enumerate(["sin-asignar", *names]):
        faces = np.flatnonzero(labels == label)
        if len(faces):
            piece = mesh.submesh([faces], repair=False)[0]
            if label and np.any(color_weight[faces]):
                projected = np.asarray(piece.visual.face_colors).copy()
                known = color_weight[faces] > 0
                selected = faces[known]
                projected[known, :3] = np.rint(color_sum[selected] / color_weight[selected, None]).astype(np.uint8)
                piece.visual.face_colors = projected
            scene.add_geometry(piece, node_name=name, geom_name=name)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        mesh, scene, errors = reconstruct(args.manifest)
        if args.output.suffix.lower() != ".glb":
            raise ValueError("La salida debe ser GLB.")
        glb = scene.export(file_type="glb")
        with args.output.open("xb") as file:
            file.write(glb)
    except (ValueError, OSError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(json.dumps({"faces": len(mesh.faces), "pieces": len(scene.geometry), "landmark_rmse_px": errors}))


if __name__ == "__main__":
    main()
