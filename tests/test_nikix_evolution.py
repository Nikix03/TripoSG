import unittest

import numpy as np
import trimesh

from aethryx_anime import nikix_preset
from aethryx_materials import unwrap_uv, uv_quality
from aethryx_modular import BODY, body_contacts, fit_component_to_patch


class NikixEvolutionTests(unittest.TestCase):
    def test_v4_ribbons_footwear_and_emissive_trim(self):
        parts = nikix_preset(version=4)
        self.assertTrue(all(piece.is_volume for piece in parts.values()))
        ribbon = parts["cabello-largo-01"]
        points = np.asarray(ribbon.vertices)[14 * 18:14 * 19]
        widths = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        self.assertLess(widths[1] / widths[0], .25)
        left = parts["prenda-zapato-izquierdo"]
        self.assertGreater(len(np.unique(left.vertices[:, 1].round(5))), 10)
        trim = parts["ropa-ribete-emisivo"]
        self.assertFalse(any(body_contacts(parts[BODY], trim).values()))
        self.assertGreater(trim.visual.material.emissiveFactor[0], .1)

    def test_component_fit_preserves_sources_and_uses_world_target(self):
        body = trimesh.creation.icosphere()
        target = trimesh.creation.box(extents=(2., 4., .5))
        target.apply_translation([6, 3, 2])
        generated = trimesh.creation.box(extents=(1., 2., .5))
        source_vertices = generated.vertices.copy()
        result, overlap = fit_component_to_patch(body, target, generated)
        np.testing.assert_allclose(result.bounds.mean(axis=0), [6, 3, 2])
        np.testing.assert_allclose(result.extents, [2, 4, 1])
        np.testing.assert_allclose(generated.vertices, source_vertices)
        self.assertFalse(any(overlap.values()))

    def test_uv_atlas_separates_islands_without_changing_triangles(self):
        try:
            import xatlas
        except ImportError as exc:
            self.skipTest(str(exc))
        original = trimesh.creation.box()
        packed = unwrap_uv(original)
        np.testing.assert_allclose(packed.triangles, original.triangles)
        self.assertGreater(len(packed.vertices), len(original.vertices))
        metrics = uv_quality(packed)
        self.assertEqual(metrics["outside_vertices"], 0)
        self.assertEqual(metrics["overlap_fraction"], 0.)
