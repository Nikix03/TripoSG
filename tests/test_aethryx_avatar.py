import tempfile
import unittest
from pathlib import Path
from PIL import Image

import numpy as np
import trimesh

from aethryx_avatar import (
    analyze_mesh_symmetry,
    build_avatar_scene_graph,
    export_disconnected_components,
    repair_mesh_copy,
    retopologize_mesh_copy,
    split_disconnected_components,
)
from studio_project import save_project, load_project


class AvatarSceneGraphTests(unittest.TestCase):
    def test_component_exports_keep_instances_and_open_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.glb"
            triangle = trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 2]],
                process=False,
            )
            pair = trimesh.util.concatenate([
                triangle, triangle.copy().apply_translation([2, 0, 0]),
            ])
            source = trimesh.Scene()
            source.add_geometry(pair, geom_name="pair", node_name="first")
            source.graph.update(
                frame_to="second", geometry="pair",
                matrix=trimesh.transformations.translation_matrix([0, 5, 1]),
            )
            source.export(str(path), file_type="glb")
            original = path.read_bytes()
            analysis = split_disconnected_components(path)
            self.assertEqual(analysis["status"], "available", analysis)
            components = analysis["components"]
            self.assertEqual(len(components), 4)
            self.assertEqual(len({component["id"] for component in components}), 4)
            self.assertEqual(sorted(tuple(c["bounds"][0]) for c in components), [
                (0, 0, 0), (0, 5, 1), (2, 0, 0), (2, 5, 1),
            ])

            result = export_disconnected_components(path, folder)
            self.assertEqual(result["status"], "available", result)
            scene = trimesh.load(result["scene_path"], force="scene", process=False)
            self.assertEqual(len(scene.graph.nodes_geometry), 4)
            for report, exported in zip(components, result["components"]):
                piece = trimesh.load(exported["path"], force="mesh", process=False)
                self.assertEqual(len(piece.faces), 1)
                np.testing.assert_allclose(piece.bounds, report["bounds"])
            repeat = export_disconnected_components(path, folder)
            self.assertEqual(repeat["status"], "available", repeat)
            self.assertNotEqual(repeat["scene_path"], result["scene_path"])
            self.assertEqual(path.read_bytes(), original)

    def test_scene_graph_has_humanoid_required_nodes(self):
        nodes = build_avatar_scene_graph()
        ids = {node["id"] for node in nodes}
        for required in (
            "avatar-root",
            "body-base",
            "head",
            "eyes",
            "hair",
            "clothing",
            "accessories",
            "armature",
        ):
            self.assertIn(required, ids)

    def test_pose_evidence_marks_body_detected_not_rigged(self):
        nodes = build_avatar_scene_graph(
            [{"view": "other", "origin": "mediapipe-pose", "confirmed": False}],
        )
        body = next(node for node in nodes if node["id"] == "body-base")
        armature = next(node for node in nodes if node["id"] == "armature")
        self.assertEqual(body["status"], "detected-needs-confirmation")
        self.assertEqual(armature["status"], "unavailable")

    def test_disconnected_geometry_components_are_split_without_anatomy_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "components.glb"
            scene = trimesh.Scene([
                trimesh.creation.box(extents=(1, 1, 1)),
                trimesh.creation.box(extents=(0.5, 0.5, 0.5)),
            ])
            scene.export(str(path), file_type="glb")
            result = split_disconnected_components(path)
            self.assertEqual(result["status"], "available")
            self.assertGreaterEqual(len(result["components"]), 2)
            self.assertTrue(all(
                component["classification"] == "unassigned"
                for component in result["components"]
            ))

    def test_repair_copy_is_real_and_preserves_original_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.glb"
            output = Path(folder) / "repaired.glb"
            mesh = trimesh.creation.box(extents=(2, 2, 2))
            mesh.export(str(path), file_type="glb")
            result = repair_mesh_copy(path, output)
            self.assertEqual(result["status"], "available")
            self.assertTrue(output.is_file())
            original = trimesh.load(str(path), force="mesh", process=False)
            self.assertEqual(result["before"]["vertices"], len(original.vertices))
            self.assertGreaterEqual(result["after"]["vertices"], 1)

    def test_retopology_target_is_real_and_dimension_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.glb"
            output = Path(folder) / "retopo.glb"
            trimesh.creation.icosphere(subdivisions=3).export(str(path), file_type="glb")
            result = retopologize_mesh_copy(path, output, 100)
            self.assertIn(result["status"], ("available", "rejected"))
            if result["status"] == "available":
                self.assertLessEqual(result["after"]["faces"], 100)

    def test_symmetry_analysis_runs_on_a_real_mesh(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "symmetric.glb"
            trimesh.creation.box(extents=(2, 2, 2)).export(str(path), file_type="glb")
            result = analyze_mesh_symmetry(path, axis="x")
            self.assertEqual(result["status"], "available")
            self.assertGreaterEqual(result["mean_vertex_mirror_distance"], 0)

    def test_avatar_scene_graph_round_trip_in_project_version_2(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "character.png"
            Image.new("RGB", (16, 16), "white").save(image)
            project = root / "avatar.aeth"
            nodes = build_avatar_scene_graph([{
                "view": "face_detail",
                "origin": "mediapipe-face",
                "confirmed": False,
            }])
            save_project(
                project,
                image=image,
                model=None,
                output=root / "mesh.glb",
                steps=50,
                faces=5000,
                seed=42,
                guidance=7.0,
                references=[],
                avatar_nodes=nodes,
            )
            loaded = load_project(project)
            self.assertEqual(loaded["avatar_nodes"], nodes)


if __name__ == "__main__":
    unittest.main()
