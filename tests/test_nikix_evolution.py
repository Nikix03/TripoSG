import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import trimesh

from aethryx_anime import layered_hair, nikix_preset, sculpted_body
from aethryx_anatomy import load_body_template, semantic_visual_hull
from aethryx_materials import unwrap_uv, uv_quality
from aethryx_modular import (BODY, _repair_hair_seams, body_contacts, fit_component_to_patch,
                           fit_fingerless_gloves, fit_trousers,
                           generated_hair_passes_review, refine_component_fit,
                           remove_nonmanifold_flaps, review_generated_hair)
from aethryx_rig import PosePreview, component_pose_contacts, demo_rotations, profile_parts
from aethryx_profiles import BUNDLED_BODY
from studio_project import generation_recovery_path, load_project, save_project
from aethryx_tools import execute_tool


class NikixEvolutionTests(unittest.TestCase):
    def test_trousers_and_fingerless_gloves_fit_v6_in_three_poses(self):
        body = sculpted_body("Anime femenino", refined=True, facial_loops=True)
        pieces = {**fit_trousers(body), **fit_fingerless_gloves(body)}
        self.assertEqual(set(pieces), {"prenda-pantalon", "prenda-guante-izquierdo", "prenda-guante-derecho"})
        scene = trimesh.Scene({BODY: body, **pieces})
        for name, piece in pieces.items():
            with self.subTest(piece=name):
                self.assertTrue(piece.is_volume)
                self.assertEqual(list(component_pose_contacts(scene, name).values()),
                                 [{"contacts": 0, "inside": 0}] * 3)

    def test_layered_hair_uses_head_bounds_on_an_unstyled_template(self):
        body = load_body_template(BUNDLED_BODY)
        self.assertNotIn("aethryx_eye_guide", body.metadata)
        hair = layered_hair(body, ribbons=False, length=.5)
        self.assertEqual(len(hair), 21)
        self.assertTrue(all(piece.is_volume for piece in hair.values()))
        self.assertNotIn("aethryx_eye_guide", body.metadata)

    def test_nonmanifold_flap_cleanup_preserves_uv_and_source(self):
        sphere = trimesh.creation.icosphere(subdivisions=1)
        vertices = np.vstack((sphere.vertices, [5, 5, 5]))
        faces = np.vstack((sphere.faces, [*sphere.faces[0, :2], len(vertices) - 1]))
        uv = np.linspace(0, 1, len(vertices) * 2).reshape(-1, 2)
        mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uv), process=False)
        result, removed = remove_nonmanifold_flaps(mesh)
        self.assertEqual(removed, 1)
        self.assertTrue(result.is_volume)
        np.testing.assert_array_equal(result.visual.uv, uv[:-1])
        np.testing.assert_array_equal(mesh.faces, faces)

    def test_collision_free_hair_still_needs_scalp_and_silhouette_coverage(self):
        review = {"closed_volume": True, "contacts": 0, "inside": 0, "crown_coverage_after": .72,
                  "guide_after": {"front": .45, "side": .41, "top": .35}}
        self.assertTrue(generated_hair_passes_review(review))
        self.assertFalse(generated_hair_passes_review({**review, "crown_coverage_after": .22}))
        self.assertFalse(generated_hair_passes_review({**review, "guide_after": {"front": .02}}))

    def test_layered_hair_length_and_colors_are_reusable(self):
        body = sculpted_body("Anime masculino", refined=True, facial_loops=True)
        short = layered_hair(body, length=.25, root="#223344", tips=("#aabbcc", "#aa6600"))
        long = layered_hair(body, length=.85, root="#223344", tips=("#aabbcc", "#aa6600"))
        lock = "cabello-largo-01"
        self.assertGreater(long[lock].extents[1], short[lock].extents[1])
        self.assertTrue(short[lock].is_volume and long[lock].is_volume)
        self.assertEqual(tuple(short[lock].visual.material.baseColorTexture.getpixel((0, 0))), (34, 51, 68))
    def test_separate_sleeves_clear_body_in_two_poses(self):
        from kiria_dialog import apply_scene_plan, plan_scene_command
        body = nikix_preset(version=6)[BODY]
        scene = trimesh.Scene({BODY: body})
        result = apply_scene_plan(plan_scene_command("mangas", [BODY], [BODY]), scene, ())
        sleeves = {name: result.get(name) for name in result.nodes if name != BODY}
        self.assertEqual(set(scene.graph.nodes_geometry), {BODY})
        self.assertEqual(len(sleeves), 2)
        source = {BODY: body, **sleeves}
        preview = PosePreview(trimesh.Scene(source))
        poses = (demo_rotations(), {"LeftLowerArm": (0, 0, 45), "RightLowerArm": (0, 0, -45)})
        for pose in poses:
            vertices, _ = preview.evaluate(pose)
            posed = {name: mesh.copy() for name, mesh in source.items()}
            for name, mesh in posed.items():
                mesh.vertices = vertices[name]
            for name in sleeves:
                self.assertFalse(any(body_contacts(posed[BODY], posed[name]).values()))
    def test_kiria_reports_open_edges_without_writing_the_model(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "open.ply"
            mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                   faces=[[0, 1, 2]], process=False)
            mesh.export(path)
            original = path.read_bytes()
            report = execute_tool("inspect_component_geometry", {"model_path": str(path)})
            self.assertEqual(report["status"], "available")
            piece = next(iter(report["components"].values()))
            self.assertEqual(piece["boundary_edges"], 3)
            self.assertFalse(piece["closed_volume"])
            self.assertEqual(path.read_bytes(), original)
    def test_semantic_volume_preserves_three_calibrated_silhouettes(self):
        from PIL import Image, ImageDraw
        color = (220, 15, 130)
        masks = []
        for width in (28, 42, 28):
            mask = Image.new("RGB", (100, 100))
            ImageDraw.Draw(mask).ellipse((50-width/2, 25, 50+width/2, 75), fill=color)
            masks.append(np.asarray(mask))
        matrices = [np.array([[30, 0, 0], [0, -30, 0]]),
                    np.array([[0, 0, 30], [0, -30, 0]]),
                    np.array([[-30, 0, 0], [0, -30, 0]])]
        mesh = semantic_visual_hull([[-1, -1, -1], [1, 1, 1]], matrices,
                                    [np.array([50, 50])] * 3, masks, color, resolution=52)
        self.assertTrue(mesh.is_volume)
        np.testing.assert_allclose(mesh.extents, [.97, 1.70, 1.43], atol=.095)

    def test_hair_cleanup_preserves_source_and_removes_tiny_island(self):
        body = trimesh.creation.icosphere()
        guide = trimesh.creation.icosphere(subdivisions=2, radius=.42)
        guide.apply_translation([0, 2, 0])
        artifact = trimesh.creation.icosphere(subdivisions=1, radius=.006)
        artifact.apply_translation([0, 2, .55])
        source = trimesh.util.concatenate((guide, artifact))
        before = source.vertices.copy()
        cleaned, overlap = review_generated_hair(body, guide, source)
        report = cleaned.metadata["aethryx_hair_review"]
        np.testing.assert_array_equal(source.vertices, before)
        self.assertTrue(report["cleanup_kept"])
        self.assertEqual(report["discarded_islands"], 1)
        self.assertFalse(any(overlap.values()))

    def test_local_seam_repair_closes_a_small_hole_without_editing_source(self):
        closed = trimesh.creation.icosphere(subdivisions=1)
        open_piece = closed.copy()
        open_piece.update_faces(np.arange(1, len(closed.faces)))
        patched = _repair_hair_seams(open_piece)
        self.assertIsNotNone(patched)
        self.assertTrue(patched.is_volume)
        self.assertEqual(len(patched.faces), len(closed.faces))
        self.assertFalse(open_piece.is_watertight)

    def test_generation_recovery_uses_atomic_project_manifest(self):
        with TemporaryDirectory() as folder, patch.dict("os.environ", {"LOCALAPPDATA": folder}):
            root = Path(folder)
            source = root / "reference.png"
            source.touch()
            path = generation_recovery_path()
            self.assertNotEqual(path, generation_recovery_path())
            path.parent.mkdir(parents=True)
            save_project(path, image=source, model=None, output=root / "avatar.glb",
                         steps=30, faces=16000, seed=42, guidance=7.)
            attempt = path.with_name(path.name.replace("-started", "-failed"))
            path.rename(attempt)
            loaded = load_project(attempt)
            self.assertEqual(Path(loaded["image"]), source.resolve())
            self.assertEqual(loaded["settings"]["seed"], 42)
            self.assertEqual(loaded["settings"]["faces"], 16000)

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

    def test_mobile_texture_budget_does_not_change_the_editable_material(self):
        studio = nikix_preset(version=6)
        mobile = profile_parts(studio, "VRChat Mobile")
        original = studio[BODY]
        converted = mobile[BODY]
        self.assertEqual(original.visual.material.baseColorTexture.size, (1024, 1024))
        self.assertEqual(converted.visual.material.baseColorTexture.size, (256, 256))
        np.testing.assert_array_equal(converted.triangles, original.triangles)
        np.testing.assert_array_equal(converted.visual.uv, original.visual.uv)
