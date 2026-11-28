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


def frontal_hair_outline(mask, color, projection):
    """Sample robust horizontal limits in the model frame from a frontal hair mask."""
    matrix = np.asarray(projection, dtype=float)
    if (mask is None or matrix.shape != (2, 4) or matrix[0, 0] <= 0 or matrix[1, 1] >= 0
            or not np.allclose(matrix[0, 1:3], 0) or not np.allclose(matrix[1, (0, 2)], 0)):
        return None
    selected = np.all(mask == color, axis=2)
    occupied = np.flatnonzero(selected.sum(axis=1) >= 8)
    if len(occupied) < 2:
        return None
    rows = occupied[np.unique(np.linspace(0, len(occupied) - 1, min(40, len(occupied))).astype(int))]
    outline = []
    for row in rows[::-1]:
        limits = (np.quantile(np.flatnonzero(selected[row]), [.03, .97]) - matrix[0, 3]) / matrix[0, 0]
        outline.append([(row - matrix[1, 3]) / matrix[1, 1], *limits.tolist()])
    return outline


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

    matrices, offsets, observations, masks, visibility, reference_colors, axes, confidence = [], [], [], [], [], [], [], []
    names = [name for name in VIEWS if name in data["views"]]
    partial = data.get("allow_inferred_depth") is True
    if not names or set(names) != set(data["views"]) or (len(names) != 3 and not partial):
        raise ValueError("Define de una a tres vistas calibradas. El modo parcial requiere autorizar profundidad inferida.")
    for name in names:
        view = data["views"][name]
        projection = np.asarray(view["projection"], dtype=float)
        points = np.asarray(view["landmarks"], dtype=float)
        with Image.open(root / view["image"]) as image:
            width, height = image.size
            reference_colors.append(np.asarray(image.convert("RGB")) if view.get("mask") else None)
        if projection.shape != (2, 4) or points.shape != (len(ids), 2):
            raise ValueError(f"Calibración o puntos inválidos en {name}.")
        visible = np.isfinite(points).all(axis=1)
        weights = np.asarray(view.get("landmark_weights", np.ones(len(ids))), dtype=float)
        if weights.shape != (len(ids),) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
            raise ValueError(f"Las ponderaciones de {name} deben estar entre 0 y 1.")
        confidence.append(weights)
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
        axes.append((True, not (name == "side" and view.get("horizontal_only") is True)))
        mask = None
        if view.get("mask"):
            with Image.open(root / view["mask"]) as image:
                if image.size != (width, height):
                    raise ValueError(f"La máscara de {name} debe coincidir con su imagen.")
                mask = np.asarray(image.convert("RGB"))
        masks.append(mask)

    system = np.vstack(matrices)
    active_rows = np.asarray(axes).ravel()
    if np.linalg.matrix_rank(system[active_rows]) != 3 and not partial:
        raise ValueError("Las cámaras no permiten recuperar profundidad.")
    targets = []
    weights = np.asarray(confidence).T.repeat(2, axis=1)
    measured = np.asarray(visibility).T.repeat(2, axis=1) & active_rows & (weights > 0)
    values = np.hstack([points - offset for points, offset in zip(observations, offsets)])
    excluded_vertical = {}
    if data.get("reconcile_pose") is True and "front" in names:
        # Compare calibrated vertical observations in body-height units, retaining depth and width.
        front_row = 2 * names.index("front") + 1
        if np.allclose(system[front_row, (0, 2)], 0) and system[front_row, 1] != 0:
            frontal_y = values[:, front_row] / system[front_row, 1]
            for number, name in enumerate(names):
                row = 2 * number + 1
                if row == front_row or not np.allclose(system[row, (0, 2)], 0) or system[row, 1] == 0:
                    continue
                y = values[:, row] / system[row, 1]
                rejected = measured[:, row] & measured[:, front_row] & (abs(y - frontal_y) > .06 * mesh.extents[1])
                measured[rejected, row] = False
                if rejected.any():
                    excluded_vertical[name] = ids[rejected].tolist()
    ranks = {}
    for index, rows in enumerate(measured):
        observed = system[rows]
        scale = np.sqrt(weights[index, rows])
        delta, _, rank, _ = np.linalg.lstsq(
            observed * scale[:, None], (values[index, rows] - observed @ anchors[index]) * scale, rcond=None,
        )
        if rank < 2 or (rank != 3 and not partial):
            raise ValueError(f"Las observaciones no resuelven la profundidad del vértice {ids[index]}.")
        ranks[str(int(ids[index]))] = int(rank)
        targets.append(anchors[index] + delta)
    smoothing = float(data.get("smoothing", 0))
    if not 0 <= smoothing <= .1:
        raise ValueError("El suavizado del ajuste debe estar entre 0 y 0,1.")
    deform = RBFInterpolator(anchors, targets - anchors, kernel="linear", degree=0,
                             smoothing=smoothing * mesh.extents[1])
    mesh.vertices = mesh.vertices + deform(mesh.vertices)
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 0):
        raise ValueError("El ajuste produjo geometría inválida o caras colapsadas.")
    errors = {}
    for index, (name, matrix, offset, points) in enumerate(zip(names, matrices, offsets, observations)):
        used = measured[:, 2 * index:2 * index + 2]
        residual = np.where(used, mesh.vertices[ids] @ matrix.T + offset - points, 0)
        if used.any():
            errors[name] = float(np.sqrt(np.sum(residual ** 2) / np.count_nonzero(used.any(axis=1))))
    patches = partition_surfaces(mesh, matrices, offsets, masks, data.get("palette", {}),
                                 reference_colors=reference_colors)
    hull_class = data.get("visual_hull_class")
    if hull_class:
        palette = data.get("palette", {})
        if hull_class not in palette or any(mask is None for mask in masks):
            raise ValueError("El volumen semántico necesita la misma clase en tres máscaras calibradas.")
        volume = semantic_visual_hull(mesh.bounds, matrices, offsets, masks, palette[hull_class])
        patches.add_geometry(volume, node_name=f"{hull_class}-volumen", geom_name=f"{hull_class}-volumen")
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
        observed_colors = {}
        for label in ("hair", "clothing"):
            if label not in data.get("palette", {}):
                continue
            color = data["palette"][label]
            samples = []
            for mask, rgb in zip(masks, reference_colors):
                if mask is not None:
                    pixels = rgb[np.all(mask == color, axis=2)]
                    if len(pixels) >= 32:
                        samples.append(np.median(pixels, axis=0))
            if samples:
                value = np.rint(np.median(samples, axis=0)).astype(int)
                observed_colors[label] = "#%02x%02x%02x" % tuple(value)
        deformation_report["reference_evidence"] = {
            "observed_views": names, "missing_views": [name for name in VIEWS if name not in names],
            "anchor_rank": ranks, "inferred_anchor_depths": sum(rank < 3 for rank in ranks.values()),
            "surface_model": "deformed_template",
            "visible_regional_colors": observed_colors,
            "reprojection_axes": {name: "xy" if use_vertical else "x"
                                  for name, (_, use_vertical) in zip(names, axes)},
            "excluded_vertical_observations": excluded_vertical,
        }
        if "front" in names and "hair" in data.get("palette", {}):
            index = names.index("front")
            outline = frontal_hair_outline(masks[index], data["palette"]["hair"], data["views"]["front"]["projection"])
            if outline is not None:
                deformation_report["reference_evidence"]["frontal_hair_outline"] = outline
    return mesh, patches, errors


def semantic_visual_hull(bounds, matrices, offsets, masks, color, resolution=100):
    """Intersect labeled orthographic silhouettes in world coordinates."""
    from skimage.measure import marching_cubes

    if not 16 <= resolution <= 192 or len(matrices) != 3 or len(offsets) != 3 or len(masks) != 3:
        raise ValueError("El volumen requiere tres vistas y una resolución entre 16 y 192.")
    low, high = np.asarray(bounds, dtype=float)
    step = float(max(high - low)) / resolution
    if not np.isfinite(step) or step <= 0:
        raise ValueError("El volumen necesita límites tridimensionales válidos.")
    margin = .12 * (high[1] - low[1])
    origin = low - margin
    shape = np.ceil((high + margin - origin) / step).astype(int) + 1
    points = np.indices(shape, dtype=np.float32).reshape(3, -1).T * step + origin
    occupied = np.ones(len(points), dtype=bool)
    cameras = []
    for matrix, offset, mask in zip(matrices, offsets, masks):
        matrix = np.asarray(matrix)
        cameras.append(np.cross(*matrix))
        selection = np.all(mask == color, axis=2)
        if not selection.any():
            raise ValueError("La clase seleccionada debe aparecer en las tres máscaras.")
        active = np.flatnonzero(occupied)
        x, y = np.floor(points[active] @ matrix.T + offset).astype(int).T
        valid = (x >= 0) & (y >= 0) & (x < selection.shape[1]) & (y < selection.shape[0])
        occupied[active[~valid]] = False
        occupied[active[valid]] = selection[y[valid], x[valid]]
    if np.linalg.matrix_rank(np.asarray(cameras)) < 2:
        raise ValueError("Las cámaras no ofrecen direcciones diferentes para el volumen.")
    grid = occupied.reshape(shape)
    if not grid.any():
        raise ValueError("Las máscaras de la clase no se cruzan en el espacio calibrado.")
    if any(np.take(grid, (0, -1), axis=axis).any() for axis in range(3)):
        raise ValueError("El volumen alcanza el límite del espacio de búsqueda. Revisa el encuadre y las máscaras.")
    vertices, faces, _, _ = marching_cubes(grid.astype(np.float32), .5, spacing=(step,) * 3)
    mesh = trimesh.Trimesh(vertices=vertices + origin, faces=faces, process=False)
    mesh.fix_normals(multibody=True)
    if not mesh.is_volume:
        raise ValueError("El volumen semántico no forma un cierre orientado.")
    mesh.metadata["aethryx_visual_hull"] = {"resolution": resolution, "views": 3}
    return mesh


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
        if data.get("export_hybrid") is True:
            from aethryx_anime import fitted_eye_details
            from aethryx_materials import update_pbr_maps, woven_normal_map
            from aethryx_modular import (
                EYES, apply_material, avatar_eye, body_contacts, clear_hair_penetrations,
                fit_garment_with_clearance, fitted_eyelids, regional_hair,
            )

            colors = report.get("reference_evidence", {}).get("visible_regional_colors", {})
            hair_options = {"root": colors["hair"]} if "hair" in colors else {}
            outline = report.get("reference_evidence", {}).get("frontal_hair_outline")
            hair_review = {}
            if outline:
                from aethryx_anime import layered_hair
                hair = layered_hair(mesh, follow_scalp=True, frontal_outline=outline, **hair_options)
                baseline = layered_hair(mesh, follow_scalp=True, **hair_options)
                for name, piece in hair.items():
                    contacts = body_contacts(mesh, piece)
                    retained = True
                    if not piece.is_volume or any(contacts.values()):
                        original_contacts = body_contacts(mesh, baseline[name])
                        if not piece.is_volume or any(
                            contacts[key] > original_contacts[key] for key in contacts
                        ):
                            hair[name], contacts, retained = baseline[name], original_contacts, False
                    hair_review[name] = {**contacts, "guided_shape_retained": retained}
                    if name.startswith(("cabello-largo-", "cabello-flequillo-")):
                        hair[name], repair = clear_hair_penetrations(mesh, hair[name])
                        hair_review[name]["clearance_repair"] = repair
            else:
                hair = regional_hair(mesh, **hair_options)
            pieces = {"cuerpo-base": mesh, **hair}
            for eye_name, left, color in zip(EYES, (True, False), ("#6589d4", "#ae6798")):
                eye = avatar_eye(mesh, left, color, refined=True)
                pieces[eye_name] = eye
                for lid, part in fitted_eyelids(eye).items():
                    pieces[f"{eye_name}-parpado-{lid}"] = part
                for detail, part in fitted_eye_details(eye, left=left).items():
                    pieces[f"{eye_name}-{detail}"] = part
            shirt, clearance = fit_garment_with_clearance(mesh, .016, .006, "top")
            if "clothing" in colors:
                shirt = apply_material(shirt, colors["clothing"], projection="cylindrical", roughness=.8)
            update_pbr_maps(shirt, {"normal": woven_normal_map()}, resolution=512)
            pieces["prenda-camiseta"] = shirt
            if outline and data.get("repair_hair_in_poses") is True:
                from aethryx_rig import PosePreview, hair_clearance_bodies
                preview = PosePreview(trimesh.Scene(pieces))
                posed = hair_clearance_bodies(preview)
                for name in hair:
                    cap = name == "cabello-fondo"
                    pieces[name], repair = clear_hair_penetrations(
                        mesh, pieces[name], clearance=.0035 if cap else .002, posed_bodies=posed,
                        repair_roots=True, paired_shell=cap,
                    )
                    hair_review[name]["pose_repair"] = repair
            provenance = {
                name: ("deformed_template" if name == "cuerpo-base" else
                       "procedural_reference_guided" if outline and name.startswith("cabello-") else "procedural")
                for name in pieces
            }
            hybrid = trimesh.Scene(pieces)
            hybrid.export(output / "avatar-modular.glb", file_type="glb")
            components.append({"name": "avatar-modular", "role": "modular-avatar", "file": "avatar-modular.glb"})
            report = {**report, "hybrid_evidence": {
                "body": "deformed_template", "procedural_parts": sorted(set(pieces) - {"cuerpo-base"}),
                "shirt_rest_clearance_height_fraction": clearance,
                "transferred_base_colors": colors,
                "hair_guidance": "frontal_semantic_outline" if outline else "procedural_body_frame",
                "hair_rest_review": hair_review,
                "component_provenance": provenance,
            }}
        else:
            hybrid = None
        if data.get("export_rig") is True or hybrid is not None:
            from aethryx_bridge import inspect_avatar_glb
            from aethryx_rig import export_rigged_avatar

            name = "avatar-articulado" if hybrid is not None else "cuerpo-articulado"
            rigged = output / f"{name}.glb"
            export_rigged_avatar(hybrid if hybrid is not None else trimesh.Scene({"cuerpo-base": mesh}), rigged)
            diagnostic = inspect_avatar_glb(rigged)
            if not diagnostic["valid_native_glb"]:
                raise ValueError("El rig del cuerpo no pasó la comprobación GLB.")
            components.append({"name": name, "role": "rigged-avatar" if hybrid is not None else "rigged-body",
                               "file": rigged.name})
        if set(patches.geometry) - {"sin-asignar"}:
            modular = output / "parches.glb"
            patches.export(modular, file_type="glb")
            for name, piece in patches.geometry.items():
                if name == "sin-asignar":
                    continue
                path = output / f"componente-{len(components):02d}.glb"
                trimesh.Scene({name: piece}).export(path, file_type="glb")
                role = "visual-hull" if piece.metadata.get("aethryx_visual_hull") else "surface-patch"
                components.append({"name": name, "role": role, "file": path.name})
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
