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
    mesh = load_body_template(root / data["template"])
    original = mesh.copy() if deformation_report is not None else None
    ids = np.asarray(data["vertex_indices"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 4
            or len(np.unique(ids)) != len(ids) or ids.min() < 0
            or ids.max() >= len(mesh.vertices)):
        raise ValueError("Se requieren al menos cuatro índices de vértice distintos.")
    anchors = mesh.vertices[ids]
    if len(np.unique(anchors, axis=0)) != len(anchors):
        raise ValueError("Los puntos de control deben ocupar posiciones distintas.")

    matrices, offsets, observations, masks, visibility = [], [], [], [], []
    if set(data["views"]) != set(VIEWS):
        raise ValueError("Define exactamente las vistas front, side y back.")
    for name in VIEWS:
        view = data["views"][name]
        projection = np.asarray(view["projection"], dtype=float)
        points = np.asarray(view["landmarks"], dtype=float)
        with Image.open(root / view["image"]) as image:
            width, height = image.size
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
    deform = RBFInterpolator(anchors, targets - anchors, kernel="linear", degree=0)
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
    patches = partition_surfaces(mesh, matrices, offsets, masks, data.get("palette", {}))
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
    head, shoulders, legs = (float(factors.get(key, 1)) for key in ("head", "shoulders", "legs"))
    if not all(0.75 <= factor <= 1.35 for factor in (head, shoulders, legs)):
        raise ValueError("Las proporciones deben estar entre 0,75 y 1,35.")
    lower, upper = bounds
    if upper[1] <= lower[1]:
        raise ValueError("El cuerpo ajustado perdió su altura.")
    center = (lower + upper) / 2
    y = (vertices[:, 1] - lower[1]) / (upper[1] - lower[1])
    result = vertices.copy()
    head_weight = np.clip((y - 0.82) / 0.12, 0, 1)
    head_weight = head_weight**2 * (3 - 2 * head_weight)
    shoulder_weight = np.maximum(0, 1 - np.abs(y - 0.73) / 0.13)**2
    result[:, 0] += (vertices[:, 0] - center[0]) * (
        (head - 1) * head_weight + (shoulders - 1) * shoulder_weight
    )
    result[:, 2] += (vertices[:, 2] - center[2]) * (head - 1) * head_weight
    waist = lower[1] + 0.48 * (upper[1] - lower[1])
    result[:, 1] += np.minimum(vertices[:, 1] - waist, 0) * (legs - 1)
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
        a, b, c = faces[index]
        for offset in range(candidates.GetNumberOfIds()):
            other = candidates.GetId(offset)
            if other <= index or np.any(maxs[index] < mins[other]) or np.any(maxs[other] < mins[index]):
                continue
            x, y, z = faces[other]
            if a in (x, y, z) or b in (x, y, z) or c in (x, y, z):
                continue
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
        "extreme_area_faces": int(np.count_nonzero((ratios < 0.2) | (ratios > 5))),
        "reflected_normal_faces": int(np.count_nonzero(
            np.einsum("ij,ij->i", original.face_normals, mesh.face_normals) < 0,
        )),
        "intersection_pairs_up_to_100": _triangle_intersections(mesh),
        "symmetry_mean_height": float(distances.mean()),
        "symmetry_p95_height": float(np.quantile(distances, 0.95)),
    }


def export_studio_reconstruction(folder, data, mesh, patches, report, errors):
    """Keep the complete body and optional reference patches in separate files."""
    output = Path(mkdtemp(prefix="genesis6-", dir=folder))
    try:
        body = output / "cuerpo-base.glb"
        trimesh.Scene({"cuerpo-base": mesh}).export(body, file_type="glb")
        modular = None
        if set(patches.geometry) - {"sin-asignar"}:
            modular = output / "parches.glb"
            patches.export(modular, file_type="glb")
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


def partition_surfaces(mesh, matrices, offsets, masks, palette):
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

    for matrix, offset, mask in zip(matrices, offsets, masks):
        if mask is None:
            continue
        pixels = np.floor(centers @ matrix.T + offset).astype(int)
        x, y = pixels.T
        # Pixel Y points down. The negative cross product points toward the camera.
        direction = -np.cross(*matrix)
        direction = direction / np.linalg.norm(direction)
        front = mesh.face_normals @ direction > 1e-8
        visible = front & (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
        indices = np.flatnonzero(visible)
        sampled = mask[y[indices], x[indices]]
        for label, color in enumerate(colors, start=1):
            candidates = indices[np.all(sampled == color, axis=1)]
            matched = _unoccluded_faces(locator, centers, candidates, direction, span, tolerance)
            ambiguous[matched] |= (labels[matched] != 0) & (labels[matched] != label)
            labels[matched] = label
    labels[ambiguous] = 0
    scene = trimesh.Scene()
    for label, name in enumerate(["sin-asignar", *names]):
        faces = np.flatnonzero(labels == label)
        if len(faces):
            piece = mesh.submesh([faces], repair=False)[0]
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
