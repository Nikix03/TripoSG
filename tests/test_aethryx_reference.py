from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from aethryx_reference import (
    SEMANTIC_COLORS, compare_silhouette, decode_semantic_scores, import_see_through_v2,
    reference_silhouette, silhouette_calibration,
)


class ReferenceTests(unittest.TestCase):
    def test_see_through_bridge_preserves_visible_labels_without_inventing_body(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v2.npy"
            masks = np.zeros((19, 4, 6), dtype=bool)
            for pixel, index in enumerate((9, 2, 11, 0, 18)):
                masks[index, :, pixel] = True
            masks[0, :, 4] = True
            np.save(path, masks)
            image = import_see_through_v2(path, (6, 4))
            order = ("body", "face", "clothing", "hair", "accessories", "background")
            np.testing.assert_array_equal(np.asarray(image)[0], [SEMANTIC_COLORS[key] for key in order])
            with self.assertRaises(ValueError):
                import_see_through_v2(path, (4, 6))
    def test_alpha_silhouette_alignment_and_bounded_comparison(self):
        image = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
        ImageDraw.Draw(image).rectangle((30, 30, 70, 70), fill="white")
        valid = reference_silhouette(image)
        self.assertEqual(silhouette_calibration(valid, {11: [40, 40], 12: [62, 40]}), ([51, 70], 30))
        mesh = trimesh.creation.box()
        projection = [[40, 0, 0, 50], [0, -40, 0, 50]]
        overlay, iou = compare_silhouette(mesh, image, projection)
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(iou, 1)
        _, lowered = compare_silhouette(mesh, image, projection, maximum_size=50)
        self.assertGreater(lowered, .9)
        self.assertIsNone(reference_silhouette(Image.new("RGB", (100, 100), "white")))

    def test_multiclass_scores_decode_exact_canonical_colors(self):
        labels = tuple(reversed(SEMANTIC_COLORS))
        scores = np.zeros((1, len(labels), 1, len(labels)), dtype=np.float32)
        for index in range(len(labels)):
            scores[0, index, 0, index] = 1
        result = decode_semantic_scores(scores, (12, 2), labels)
        np.testing.assert_array_equal(result[0, ::2], [SEMANTIC_COLORS[label] for label in labels])
        with self.assertRaises(ValueError):
            decode_semantic_scores(scores[:, :2], (12, 2), labels)
