import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest

import numpy as np
from PIL import Image
import trimesh

from aethryx_anatomy import (
    anime_proportions, export_studio_reconstruction, orthographic_projection,
    partition_surfaces, reconstruct, reconstruct_data, validate_deformation,
)


class ReconstructionTests(unittest.TestCase):
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
        self.assertIsNone(dialog.result)
        dialog.destroy()

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
        self.assertEqual(validate_deformation(crossed, crossed)["intersection_pairs_up_to_100"], 1)
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
        )
        np.testing.assert_array_equal(scene.geometry["visible"].vertices[:, 2], np.ones(3))
        np.testing.assert_array_equal(scene.geometry["sin-asignar"].vertices[:, 2], np.zeros(3))

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

            views["side"]["projection"] = matrices["front"]
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["views"] = views
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profundidad"):
                reconstruct(manifest)


if __name__ == "__main__":
    unittest.main()
