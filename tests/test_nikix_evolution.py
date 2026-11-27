import unittest

import numpy as np
import trimesh

from aethryx_anime import nikix_preset
from aethryx_materials import unwrap_uv, uv_quality
from aethryx_modular import BODY, body_contacts, fit_component_to_patch, refine_component_fit
from aethryx_rig import profile_parts


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

    def test_refinement_does_not_damage_a_clear_component(self):
        body = trimesh.creation.icosphere()
        mesh = trimesh.creation.box(extents=(.4, .6, .25))
        patch = mesh.copy()
        patch.apply_translation([3., 2., .5])
        refined, report = refine_component_fit(body, patch, mesh)
        self.assertFalse(any(report.values()))
        np.testing.assert_allclose(refined.bounds, patch.bounds)
        self.assertTrue(refined.is_volume)

    def test_export_profile_preserves_body_and_independent_attachments(self):
        studio = nikix_preset(version=4)
        original_total = sum(len(piece.faces) for piece in studio.values())
        optimized = profile_parts(studio, "VRChat PC Optimized")
        mobile = profile_parts(studio, "VRChat Mobile")
        self.assertLess(sum(len(piece.faces) for piece in mobile.values()),
                        sum(len(piece.faces) for piece in optimized.values()))
        self.assertLess(sum(len(piece.faces) for piece in optimized.values()), original_total)
        self.assertEqual(sum(len(piece.faces) for piece in studio.values()), original_total)
        for part in (BODY, "ojo-izquierdo", "ojo-derecho", "accesorio-hamsa-izquierdo", "ropa-ribete-emisivo"):
            self.assertIs(optimized[part], studio[part])
        self.assertIn("cabello-largo-01", optimized)
