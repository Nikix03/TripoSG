"""Fit a supplied humanoid template to calibrated orthographic references."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
import trimesh


def reconstruct(manifest_path):
    """Return a fitted mesh, editable surface patches and reprojection errors."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    mesh = trimesh.load(root / data["template"], force="mesh", process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("La plantilla debe contener una malla triangular válida.")
    ids = np.asarray(data["vertex_indices"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 4
            or len(np.unique(ids)) != len(ids) or ids.min() < 0
            or ids.max() >= len(mesh.vertices)):
        raise ValueError("Se requieren al menos cuatro índices de vértice distintos.")
    anchors = mesh.vertices[ids]
    if len(np.unique(anchors, axis=0)) != len(anchors):
        raise ValueError("Los puntos de control deben ocupar posiciones distintas.")

    matrices, offsets, observations, masks, visibility = [], [], [], [], []
    if set(data["views"]) != {"front", "side", "back"}:
        raise ValueError("Define exactamente las vistas front, side y back.")
    for name in ("front", "side", "back"):
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
            ("front", "side", "back"), matrices, offsets, observations, visibility,
        )
    }
    return mesh, partition_surfaces(mesh, matrices, offsets, masks, data.get("palette", {})), errors


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
