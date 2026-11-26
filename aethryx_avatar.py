"""Deterministic humanoid avatar scene graph for AETHRYX Studio."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AvatarNode:
    id: str
    name: str
    parent_id: str | None
    kind: str
    status: str
    source: str
    confidence: float | None = None


def build_avatar_scene_graph(references=None, mesh_stats=None):
    references = references or []
    nodes = [
        AvatarNode("avatar-root", "Avatar Root", None, "root", "available", "deterministic"),
        AvatarNode("body-base", "Body Base", "avatar-root", "body", "unavailable", "mesh"),
        AvatarNode("head", "Head", "avatar-root", "face", "unavailable", "mesh"),
        AvatarNode("eyes", "Eyes", "head", "face", "unavailable", "vision"),
        AvatarNode("hair", "Hair", "head", "hair", "unavailable", "vision"),
        AvatarNode("clothing", "Clothing", "body-base", "clothing", "unavailable", "vision"),
        AvatarNode("accessories", "Accessories", "avatar-root", "accessory", "unavailable", "vision"),
        AvatarNode("armature", "Humanoid Armature", "avatar-root", "rig", "unavailable", "rig"),
    ]

    views = {
        region.get("view")
        for region in references
        if isinstance(region, dict)
    }
    origins = {
        region.get("origin")
        for region in references
        if isinstance(region, dict)
    }
    pose_present = "mediapipe-pose" in origins
    face_present = "mediapipe-face" in origins or "opencv-haar" in origins

    updated = []
    for node in nodes:
        status = node.status
        source = node.source
        if node.id == "eyes" and face_present:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "head" and face_present:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "body-base" and pose_present:
            status = "detected-needs-confirmation"
            source = "mediapipe-pose"
        if node.id == "hair" and "hair_detail" in views:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "clothing" and "clothing_detail" in views:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "accessories" and "accessory_detail" in views:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "body-base" and mesh_stats:
            status = "mesh-imported-not-segmented"
            source = "mesh"
        updated.append(AvatarNode(
            node.id,
            node.name,
            node.parent_id,
            node.kind,
            status,
            source,
            node.confidence,
        ))
    return [asdict(node) for node in updated]


def _mesh_components(model_path):
    from pathlib import Path
    import trimesh

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError("Malla no encontrada")
    scene = trimesh.load(str(path), force="scene", process=False)

    pieces_by_geometry = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        if geometry_name not in pieces_by_geometry:
            pieces = geometry.split(only_watertight=False, repair=False)
            if sum(len(piece.faces) for piece in pieces) != len(geometry.faces):
                raise ValueError(f"La separación de {geometry_name} no conservó todas las caras")
            pieces_by_geometry[geometry_name] = pieces
        for index, piece in enumerate(pieces_by_geometry[geometry_name]):
            yield node, index, piece, transform


def split_disconnected_components(model_path):
    import trimesh

    components = []
    try:
        for node, index, piece, transform in _mesh_components(model_path):
            vertices = trimesh.transform_points(piece.vertices, transform)
            components.append({
                "id": f"{node}-component-{index}",
                "name": f"Componente geométrico {len(components) + 1}",
                "vertices": int(len(piece.vertices)),
                "faces": int(len(piece.faces)),
                "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                "classification": "unassigned",
                "editable": True,
            })
    except Exception as exc:
        return {"status": "unavailable", "reason": f"No se pudo separar la malla: {exc}"}
    if not components:
        return {"status": "unavailable", "reason": "La escena no contiene mallas triangulares"}
    return {
        "status": "available",
        "components": components,
        "note": "Separación topológica real; anatomía no inferida.",
    }


def export_disconnected_components(model_path, output_dir):
    from pathlib import Path
    import shutil
    import tempfile
    import trimesh

    destination = None
    try:
        components = list(_mesh_components(model_path))
        if not components:
            return {"status": "unavailable", "reason": "La escena no contiene mallas triangulares"}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="componentes-", dir=output_dir))
        scene = trimesh.Scene()
        exported = []
        for index, (_, _, piece, transform) in enumerate(components, start=1):
            name = f"component-{index}"
            output_path = destination / f"{name}.glb"
            single = trimesh.Scene()
            for target in (scene, single):
                target.add_geometry(piece, node_name=name, geom_name=name, transform=transform)
            single.export(str(output_path), file_type="glb")
            exported.append({
                "path": str(output_path),
                "vertices": int(len(piece.vertices)),
                "faces": int(len(piece.faces)),
            })
        scene_path = destination / "scene.glb"
        scene.export(str(scene_path), file_type="glb")
    except Exception as exc:
        if destination is not None:
            shutil.rmtree(destination, ignore_errors=True)
        return {"status": "unavailable", "reason": f"No se pudo exportar la separación: {exc}"}
    return {
        "status": "available",
        "components": exported,
        "scene_path": str(scene_path),
        "classification": "unassigned",
    }


def transform_scene_object(scene, node, values):
    import numpy as np
    from trimesh import transformations

    if len(values) != 7 or not np.isfinite(values).all() or values[6] <= 0:
        raise ValueError("Se requieren siete números finitos y escala positiva.")
    if node not in scene.graph.nodes_geometry:
        raise ValueError("El objeto seleccionado no pertenece a la escena.")
    parent = scene.graph.transforms.parents.get(node)
    if parent is None:
        raise ValueError("El nodo raíz de la escena no admite una transformación local.")
    result = scene.copy()
    local = scene.graph.transforms.edge_data[(parent, node)]["matrix"]
    translation = transformations.translation_matrix(values[:3])
    rotation = transformations.euler_matrix(*np.radians(values[3:6]))
    scale = transformations.scale_matrix(values[6])
    geometry = scene.graph[node][1]
    result.graph.update(
        frame_from=parent, frame_to=node, matrix=translation @ local @ rotation @ scale,
        geometry=geometry,
    )
    return result


def _repair_regression(original, repaired):
    import numpy as np

    if (not len(repaired.faces) or not np.isfinite(repaired.vertices).all()
            or np.any(repaired.area_faces <= 0)):
        return "La reparación produjo geometría vacía, degenerada o no finita."
    if original.is_watertight and not repaired.is_watertight:
        return "La reparación abrió una superficie que estaba cerrada."
    if original.is_winding_consistent and not repaired.is_winding_consistent:
        return "La reparación deterioró la orientación de las caras."

    def defect_lengths(mesh):
        counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
        edges = np.asarray(mesh.edges_unique)
        lengths = np.linalg.norm(mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]], axis=1)
        return np.array([lengths[counts == 1].sum(), lengths[counts > 2].sum()])

    if np.any(defect_lengths(repaired) > defect_lengths(original) * 1.001 + 1e-8):
        return "La reparación aumentó la longitud de bordes abiertos o no manifold."
    extent = np.maximum(original.extents, 1e-6)
    if np.any(np.abs(repaired.bounds - original.bounds) > extent * 0.05):
        return "La reparación desplazó los límites de la geometría más de 5%."
    return None


def repair_mesh_copy(model_path, output_path):
    from pathlib import Path
    import os
    import tempfile
    import trimesh
    import numpy as np

    path = Path(model_path)
    output_path = Path(output_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    if output_path.resolve() == path.resolve():
        return {"status": "rejected", "reason": "Selecciona una ruta distinta para la copia."}
    try:
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"No se pudo cargar la malla: {exc}"}

    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        return {"status": "rejected", "reason": "La malla original está vacía o contiene coordenadas inválidas."}
    before_vertices = int(len(mesh.vertices))
    before_faces = int(len(mesh.faces))
    before_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    original = mesh
    mesh = mesh.copy()
    mesh.merge_vertices()
    face_keys = np.sort(np.asarray(mesh.faces, dtype=np.int64), axis=1)
    _, unique_indices = np.unique(face_keys, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    mesh.update_faces(np.isin(np.arange(len(mesh.faces)), unique_indices))
    triangles = mesh.triangles
    if len(triangles):
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        degenerate = np.linalg.norm(cross, axis=1) <= 1e-12
        mesh.update_faces(~degenerate)
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    regression = _repair_regression(original, mesh)
    if regression:
        return {"status": "rejected", "reason": regression}
    after_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    dimensions_before = before_bounds[1] - before_bounds[0]
    dimensions_after = after_bounds[1] - after_bounds[0]
    dimension_delta = (
        np.abs(dimensions_after - dimensions_before)
        / np.maximum(dimensions_before, 1e-6)
    )
    if float(dimension_delta.max()) > 0.05:
        return {
            "status": "rejected",
            "reason": "La reparación alteró las dimensiones más de 5%; original preservada.",
        }
    temporary = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".glb", dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        trimesh.Scene([mesh]).export(str(temporary), file_type="glb")
        validate = trimesh.load(str(temporary), force="mesh", process=False)
        regression = _repair_regression(original, validate)
        if regression:
            return {"status": "rejected", "reason": regression}
        os.replace(temporary, output_path)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"No se pudo guardar la reparación: {exc}"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "status": "available",
        "path": str(output_path),
        "before": {"vertices": before_vertices, "faces": before_faces},
        "after": {
            "vertices": int(len(validate.vertices)),
            "faces": int(len(validate.faces)),
            "watertight": bool(validate.is_watertight),
        },
    }


def analyze_mesh_symmetry(model_path, axis="x"):
    from pathlib import Path
    import numpy as np
    import trimesh

    path = Path(model_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if axis_index is None:
        return {"status": "unavailable", "reason": "Eje de simetría no compatible"}
    if len(vertices) == 0:
        return {"status": "unavailable", "reason": "Malla sin vértices"}
    mirrored = vertices.copy()
    mirrored[:, axis_index] *= -1
    kd = None
    try:
        from scipy.spatial import cKDTree
        kd = cKDTree(mirrored)
    except ImportError:
        pass
    if kd is None:
        distances = [
            float(np.min(np.linalg.norm(vertices - point, axis=1)))
            for point in mirrored
        ]
    else:
        distances = kd.query(vertices, k=1)[0]
    return {
        "status": "available",
        "axis": axis.lower(),
        "mean_vertex_mirror_distance": float(np.mean(distances)),
        "max_vertex_mirror_distance": float(np.max(distances)),
    }


def retopologize_mesh_copy(model_path, output_path, target_faces):
    from pathlib import Path
    import numpy as np
    import trimesh

    path = Path(model_path)
    output_path = Path(output_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    if type(target_faces) is not int or not 100 <= target_faces <= 200000:
        return {"status": "unavailable", "reason": "Objetivo de caras no válido"}
    mesh = trimesh.load(str(path), force="mesh", process=False)
    before_faces = int(len(mesh.faces))
    before_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    try:
        import fast_simplification
        vertices, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            target_count=target_faces,
            lossless=False,
        )
        reduced = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"Retopología no disponible: {exc}"}
    after_bounds = np.asarray(reduced.bounds, dtype=np.float64)
    dimensions_before = before_bounds[1] - before_bounds[0]
    dimensions_after = after_bounds[1] - after_bounds[0]
    dimension_delta = (
        np.abs(dimensions_after - dimensions_before)
        / np.maximum(dimensions_before, 1e-6)
    )
    if float(dimension_delta.max()) > 0.05:
        return {
            "status": "rejected",
            "reason": "La retopología alteró las dimensiones más de 5%; malla original preservada.",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene([reduced]).export(str(output_path), file_type="glb")
    return {
        "status": "available",
        "path": str(output_path),
        "before": {"vertices": int(len(mesh.vertices)), "faces": before_faces},
        "after": {
            "vertices": int(len(reduced.vertices)),
            "faces": int(len(reduced.faces)),
            "watertight": bool(reduced.is_watertight),
        },
        "maximum_dimension_delta": float(dimension_delta.max()),
    }
