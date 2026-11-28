from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from aethryx_reference import (
    SEMANTIC_COLORS, compare_silhouette, decode_semantic_scores, import_see_through_v2,
    foreground_crop_diagnostics, pose_points_for_view, reference_silhouette, retain_mask_polygon,
    silhouette_calibration, vertical_pose_disagreements,
)


class ReferenceTests(unittest.TestCase):
    def test_reviewed_polygon_preserves_labels_and_source(self):
        mask = Image.new("RGB", (12, 12), "red")
        mask.putpixel((5, 5), (0, 255, 0))
        cleaned = retain_mask_polygon(mask, [(1, 1), (10, 1), (10, 9), (1, 9)])
        self.assertEqual(cleaned.getpixel((5, 5)), (0, 255, 0))
        self.assertEqual(cleaned.getpixel((0, 5)), (0, 0, 0))
        self.assertEqual(cleaned.getpixel((5, 10)), (0, 0, 0))
        self.assertEqual(mask.getpixel((5, 10)), (255, 0, 0))
        with self.assertRaises(ValueError):
            retain_mask_polygon(mask, [(2, 2), (3, 3), (4, 4)])

    def test_rear_view_filters_face_and_pose_check_is_scale_normalized(self):
        front = {0: (10, 10), 11: (20, 40), 15: (20, 70)}
        rear = pose_points_for_view({0: (10, 10), 11: (20, 80), 15: (20, 112)}, "back")
        self.assertNotIn(0, rear)
        self.assertEqual(pose_points_for_view(front, "front"), front)
        self.assertEqual(vertical_pose_disagreements([
            (([50, 100], 0), front), (([80, 200], 0), rear), (None, {11: (0, 0)}),
        ]), [15])

    def test_crop_diagnostics_report_edge_and_detached_foreground(self):
        mask = Image.new("L", (20, 20))
        draw = ImageDraw.Draw(mask)
        draw.rectangle((4, 4, 8, 19), fill=255)
        draw.rectangle((18, 0, 19, 1), fill=255)
        report = foreground_crop_diagnostics(mask)
        self.assertEqual(report["edge_pixels"], {"top": 2, "bottom": 5, "left": 0, "right": 2})
        self.assertEqual(report["components"], 2)
        self.assertAlmostEqual(report["secondary_fraction"], 4 / 84)
        self.assertIsNone(reference_silhouette(mask, mask))
        self.assertEqual(foreground_crop_diagnostics(Image.new("L", (3, 3)))["components"], 0)

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
