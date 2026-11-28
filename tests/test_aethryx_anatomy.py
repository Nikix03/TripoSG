import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest

import numpy as np
from PIL import Image
import trimesh

from aethryx_anatomy import (
    PROPORTIONS, anime_proportions, blocking_geometry_issues, export_studio_reconstruction,
    load_body_template, orthographic_projection,
    partition_surfaces, reconstruct, reconstruct_data, validate_deformation,
)
from aethryx_profiles import BODY_PROFILES, BUNDLED_BODY, POSE_VERTICES


class ReconstructionTests(unittest.TestCase):
    def test_lateral_horizontal_observations_preserve_frontal_height(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            template = trimesh.creation.box()
            template.export(root / "template.ply")
            Image.new("RGB", (100, 100), "gray").save(root / "view.png")
            ids = [0, 1, 2, 4]
            anchors = template.vertices[ids]
            front = [[30, 0, 0, 50], [0, -30, 0, 50]]
            side = [[0, 0, -30, 50], [0, -30, 0, 50]]
            views = {
                "front": {"image": "view.png", "projection": front,
                          "landmarks": (anchors @ np.asarray(front)[:, :3].T + [50, 50]).tolist()},
                "side": {"image": "view.png", "projection": side, "horizontal_only": True,
                         "landmarks": (anchors @ np.asarray(side)[:, :3].T + [44, 60]).tolist()},
            }
            report = {}
            fitted, _, errors = reconstruct_data({
                "template": "template.ply", "vertex_indices": ids, "views": views,
                "allow_inferred_depth": True,
            }, root, deformation_report=report)
            np.testing.assert_allclose(fitted.vertices, template.vertices + [0, 0, .2])
            self.assertLess(max(errors.values()), 1e-9)
            self.assertEqual(report["reference_evidence"]["reprojection_axes"], {"front": "xy", "side": "x"})
            views["side"].pop("horizontal_only")
            views["side"]["landmarks"][0][1] += 20
            views["side"]["landmark_weights"] = [.5, .5, .5, .5]
            fitted, _, _ = reconstruct_data({
                "template": "template.ply", "vertex_indices": ids, "views": views,
                "allow_inferred_depth": True, "reconcile_pose": True,
            }, root, deformation_report=report)
            np.testing.assert_allclose(fitted.vertices, template.vertices + [0, 0, .2])
            self.assertEqual(report["reference_evidence"]["excluded_vertical_observations"], {"side": ids})

    def test_hybrid_export_keeps_editable_components_and_valid_rig(self):
        from aethryx_bridge import inspect_avatar_glb

        body = load_body_template(BUNDLED_BODY)
        report = validate_deformation(body, body)
        with tempfile.TemporaryDirectory() as folder:
            path, _ = export_studio_reconstruction(
                folder, {"export_hybrid": True}, body, trimesh.Scene(), report, {},
            )
            modular = trimesh.load(path.parent / "avatar-modular.glb", force="scene", process=False)
            self.assertIn("cuerpo-base", modular.graph.nodes_geometry)
            self.assertIn("prenda-camiseta", modular.graph.nodes_geometry)
            rig = inspect_avatar_glb(path.parent / "avatar-articulado.glb")
            self.assertTrue(rig["valid_native_glb"], rig["errors"])
            self.assertEqual(rig["joints"], 53)
            self.assertEqual(rig["visemes"], 15)
            self.assertGreater(rig["skinned_nodes"], 1)

    def test_studio_wizard_opens_as_a_real_tk_dialog(self):
        from anatomy_dialog import AnatomyDialog

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.addCleanup(root.destroy)
        root.withdraw()
        dialog = AnatomyDialog(root)
        self.assertEqual(dialog.view, "front")
        self.assertEqual(tuple(dialog.views), ("front", "side", "back"))
        self.assertEqual(len(dialog.pose_links), 13)
        self.assertEqual(dialog.template_path, str(BUNDLED_BODY))
        self.assertIsNone(dialog.result)
        dialog.destroy()

    def test_bundled_profiles_use_valid_bilateral_anchors_and_all_controls_move_vertices(self):
        mesh = load_body_template(BUNDLED_BODY)
        points = mesh.vertices
        self.assertEqual(len(mesh.faces), 26756)
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(len(set(POSE_VERTICES.values())), 13)
        for left in (11, 13, 15, 23, 25, 27):
            np.testing.assert_allclose(
                points[POSE_VERTICES[left]] * [-1, 1, 1], points[POSE_VERTICES[left + 1]], atol=1e-6,
            )
        np.testing.assert_array_equal(anime_proportions(points, mesh.bounds, {}), points)
        for key in PROPORTIONS:
            with self.subTest(proportion=key):
                result = anime_proportions(points, mesh.bounds, {key: 1.1})
                self.assertGreater(np.max(abs(result - points)), 1e-3)
        self.assertEqual(len(BODY_PROFILES), 3)

    def test_visual_calibration_aligns_height_and_all_view_axes(self):
        bounds = np.array([[-1, 2, -0.5], [1, 4, 0.5]])
        points = np.array([[0, 2, 0, 1], [0, 4, 0, 1], [1, 3, 0.5, 1]])
        for view, side, x in (("front", 1, 200), ("back", 1, 0),
                              ("side", 1, 50), ("side", -1, 150)):
            projection = orthographic_projection(view, [100, 250], 250, 50, bounds, side)
            np.testing.assert_allclose(points @ projection.T, [[100, 250], [100, 50], [x, 150]])

    def test_deformation_reports_crossing_and_known_symmetry(self):
        vertices = [[-2, 0, 0], [2, 0, 0], [0, 2, 0],
                    [-0.5, 1, -1], [0.5, 1, -1], [0, 1, 1]]
        crossed = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False)
        defective = validate_deformation(crossed, crossed)
        self.assertEqual(defective["intersection_pairs_up_to_100"], 1)
        self.assertEqual(len(blocking_geometry_issues(defective)), 2)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "autointersecciones"):
                export_studio_reconstruction(folder, {}, crossed, trimesh.Scene(), defective, {})
            self.assertFalse(list(Path(folder).iterdir()))
        sphere = trimesh.creation.icosphere(subdivisions=1)
        report = validate_deformation(sphere, sphere)
        self.assertEqual(report["intersection_pairs_up_to_100"], 0)
        self.assertAlmostEqual(report["symmetry_p95_height"], 0)

    def test_projection_keeps_occluded_faces_unassigned(self):
        planes = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
            faces=[[0, 1, 2], [3, 4, 5]], process=False,
        )
        scene = partition_surfaces(
            planes, [np.array([[1, 0, 0], [0, -1, 0]])], [np.array([1, 1])],
            [np.tile([255, 0, 0], (3, 3, 1))], {"visible": [255, 0, 0]},
            reference_colors=[np.tile([13, 122, 230], (3, 3, 1))],
        )
        np.testing.assert_array_equal(scene.geometry["visible"].vertices[:, 2], np.ones(3))
        np.testing.assert_array_equal(scene.geometry["sin-asignar"].vertices[:, 2], np.zeros(3))
        np.testing.assert_array_equal(scene.geometry["visible"].visual.face_colors[0, :3], [13, 122, 230])

    def test_three_calibrated_views_fit_depth_and_split_connected_surface(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            template = trimesh.creation.box()
            # PLY preserves the template's explicit vertex order.
            template.export(root / "template.ply")
            vertices = np.asarray(template.vertices)
            target = vertices + [0, 0, 0.2]
            indexes = [0, 1, 2, 4]
            matrices = {
                "front": [[40, 0, 0, 50], [0, -40, 0, 50]],
                "side": [[0, 0, -40, 50], [0, -40, 0, 50]],
                "back": [[-40, 0, 0, 50], [0, -40, 0, 50]],
            }
            views = {}
            for name, matrix in matrices.items():
                projection = np.asarray(matrix)
                file = f"{name}.png"
                Image.new("RGB", (100, 100), "gray").save(root / file)
                views[name] = {
                    "image": file, "projection": matrix,
                    "landmarks": (target[indexes] @ projection[:, :3].T + projection[:, 3]).tolist(),
                }
            views["back"]["landmarks"][0] = [None, None]
            Image.new("RGB", (100, 100), "red").save(root / "front-mask.png")
            views["front"]["mask"] = "front-mask.png"
            manifest = root / "fit.json"
            manifest.write_text(json.dumps({
                "template": "template.ply", "vertex_indices": indexes,
                "views": views, "palette": {"frontal": [255, 0, 0]},
            }), encoding="utf-8")
            fitted, pieces, errors = reconstruct(manifest)
            np.testing.assert_allclose(fitted.vertices, target)
            np.testing.assert_array_equal(fitted.faces, template.faces)
            self.assertEqual(set(pieces.graph.nodes_geometry), {"frontal", "sin-asignar"})
            self.assertEqual(len(pieces.geometry["frontal"].faces), 2)
            self.assertEqual(sum(len(piece.faces) for piece in pieces.geometry.values()), 12)
            self.assertLess(max(errors.values()), 1e-10)

            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["proportions"] = {"head": 1.2, "legs": 1.2}
            report = {}
            styled, patches, errors = reconstruct_data(data, root, deformation_report=report)
            np.testing.assert_allclose(
                styled.vertices, anime_proportions(fitted.vertices, fitted.bounds, data["proportions"]),
            )
            body, separate_patches = export_studio_reconstruction(root, data, styled, patches, report, errors)
            full_body = trimesh.load(body, force="mesh", process=False)
            self.assertEqual(len(full_body.faces), 12)
            self.assertTrue(full_body.is_watertight)
            self.assertEqual(len(trimesh.load(separate_patches, force="scene").geometry), 2)
            self.assertTrue((body.parent / "validacion.json").is_file())
            components = json.loads((body.parent / "componentes.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["role"] for entry in components], ["body", "surface-patch"])
            self.assertEqual(len(trimesh.load(body.parent / components[1]["file"], force="mesh").faces), 2)

            views["side"]["projection"] = matrices["front"]
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["views"] = views
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profundidad"):
                reconstruct(manifest)

    def test_partial_reconstruction_retains_only_unobserved_template_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            template = trimesh.creation.box()
            template.export(root / "template.ply")
            indices = [0, 1, 2, 4]
            translation = np.array([.08, .12, .21])
            target = template.vertices + translation
            Image.new("RGB", (100, 100), "gray").save(root / "ref.png")
            front = np.array([[40, 0, 0, 50], [0, -40, 0, 50]])
            side = np.array([[0, 0, 40, 50], [0, -40, 0, 50]])

            def view(projection):
                return {"image": "ref.png", "projection": projection.tolist(),
                        "landmarks": (target[indices] @ projection[:, :3].T + projection[:, 3]).tolist()}

            data = {"template": "template.ply", "vertex_indices": indices, "allow_inferred_depth": True,
                    "views": {"front": view(front)}}
            report = {}
            fitted, _, errors = reconstruct_data(data, root, deformation_report=report)
            np.testing.assert_allclose(fitted.vertices, template.vertices + [.08, .12, 0])
            self.assertEqual(report["reference_evidence"]["inferred_anchor_depths"], 4)
            self.assertEqual(tuple(errors), ("front",))
            data["views"]["side"] = view(side)
            fitted, _, _ = reconstruct_data(data, root, deformation_report=report)
            np.testing.assert_allclose(fitted.vertices, target)
            self.assertEqual(report["reference_evidence"]["missing_views"], ["back"])
            self.assertEqual(report["reference_evidence"]["inferred_anchor_depths"], 0)

    def test_batch_filename_views_refuse_ambiguous_assignments(self):
        from aethryx_reference import named_reference_views
        self.assertEqual(named_reference_views(["hero-lateral.png", "hero-frontal.png"]),
                         {"side": "hero-lateral.png", "front": "hero-frontal.png"})
        with self.assertRaises(ValueError):
            named_reference_views(["hero-frontal.png", "hero-front.png"])


if __name__ == "__main__":
    unittest.main()
