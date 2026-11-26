import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from aethryx_anatomy import partition_surfaces, reconstruct


class ReconstructionTests(unittest.TestCase):
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

            views["side"]["projection"] = matrices["front"]
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["views"] = views
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profundidad"):
                reconstruct(manifest)


if __name__ == "__main__":
    unittest.main()
