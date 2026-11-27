from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from aethryx_anatomy import blocking_geometry_issues, validate_deformation
from aethryx_anime import hair_locks, nikix_preset, sculpted_body, sculpted_face_parts
from aethryx_bridge import inspect_avatar_glb
from aethryx_modular import BODY, EYES, anime_body, avatar_eye, body_contacts
from aethryx_rig import PosePreview, export_rigged_avatar
from aethryx_reference import AnimeSilhouette, profile_opposite
from kiria_dialog import apply_scene_plan, plan_scene_command
from studio_project import load_project, save_project, unsaved_recovery_path, validate_glb


class GenesisQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = sculpted_body("Anime femenino")

    def test_vtk_viewer_preserves_visible_mesh_instances(self):
        try:
            import pyvista as pv
        except ImportError as exc:
            self.skipTest(str(exc))
        from app import AethryxStudio

        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(), geom_name="shared", node_name="base")
        for name, offset in (("moved", 3), ("hidden", -5)):
            scene.graph.update(frame_to=name, frame_from=scene.graph.base_frame, geometry="shared",
                               matrix=trimesh.transformations.translation_matrix([offset, 0, 0]))
        plotter = pv.Plotter(off_screen=True, window_size=(300, 260))
        try:
            AethryxStudio._populate_pyvista(plotter, pv, scene, {"hidden"})
            self.assertNotIn("hidden", plotter.renderer.actors)
            np.testing.assert_allclose(plotter.renderer.actors["base"].bounds[:2], [-.5, .5])
            np.testing.assert_allclose(plotter.renderer.actors["moved"].bounds[:2], [2.5, 3.5])
        finally:
            plotter.close()

    def test_nikix_preset_remains_modular_and_exportable(self):
        parts = nikix_preset()
        body = parts[BODY]
        self.assertTrue(all(mesh.is_volume for mesh in parts.values()))
        self.assertFalse(blocking_geometry_issues(validate_deformation(body, body)))
        self.assertFalse(any(body_contacts(body, parts["ropa-vestido"]).values()))
        self.assertIn("accesorio-collar-arbol", parts)
        self.assertEqual(sum(name.startswith("accesorio-hamsa") for name in parts), 2)
        for left, right in ((EYES[0], EYES[1]), ("cabello-largo-01", "cabello-largo-19")):
            a, b = (np.asarray(parts[name].visual.material.baseColorTexture) for name in (left, right))
            self.assertFalse(np.array_equal(a, b))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "nikix.glb"
            exported = export_rigged_avatar(trimesh.Scene(parts), target)
            self.assertEqual(exported["skinned_nodes"], len(parts))
            report = inspect_avatar_glb(target)
            self.assertTrue(report["valid_native_glb"])
            self.assertEqual(report["visemes"], 15)

    def test_isnet_removes_padding_and_keeps_original_image_size(self):
        from PIL import Image
        from types import SimpleNamespace

        class FixtureSession:
            def run(self, _, inputs):
                tensor = inputs["img"]
                self.input_shape = tensor.shape
                return [tensor[:, :1]]

        provider = AnimeSilhouette.__new__(AnimeSilhouette)
        provider.session = FixtureSession()
        provider.input = SimpleNamespace(name="img")
        image = Image.new("RGB", (80, 160), "black")
        image.paste("red", (22, 40, 58, 120))
        mask = provider.predict(image)
        self.assertEqual(provider.session.input_shape, (1, 3, 1024, 1024))
        self.assertEqual(mask.size, image.size)
        self.assertEqual(mask.getbbox(), (22, 40, 58, 120))

    def test_new_head_is_welded_and_face_geometry_is_modular(self):
        body = self.body
        legacy = anime_body("Anime femenino")
        self.assertNotEqual(len(legacy.faces), len(body.faces))
        self.assertTrue(body.is_volume)
        self.assertFalse(blocking_geometry_issues(validate_deformation(body, body)))
        facial = sculpted_face_parts(body)
        self.assertIn("boca-linea", facial)
        for eye in EYES:
            self.assertGreater(facial[eye].extents[1], avatar_eye(legacy, eye == EYES[0], "#6688dd").extents[1])
            for suffix in ("pestana", "ceja", "parpado-superior", "parpado-inferior"):
                self.assertTrue(facial[f"{eye}-{suffix}"].is_volume)
        hair = hair_locks(body, .54)
        self.assertIn("cabello-fondo", hair)
        self.assertGreaterEqual(len(hair), 18)
        self.assertTrue(all(lock.is_volume for lock in hair.values()))

    def test_procedural_face_exports_real_morphs_and_lip_line_follows_jaw(self):
        scene = trimesh.Scene({BODY: self.body, **sculpted_face_parts(self.body)})
        preview = PosePreview(scene)
        rest, _ = preview.evaluate({})
        motion, _ = preview.evaluate({}, "JawOpen", 1)
        self.assertGreater(np.max(abs(motion["boca-linea"] - rest["boca-linea"])), .0003)
        np.testing.assert_allclose(motion[EYES[0]], rest[EYES[0]], atol=1e-9)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "anime-v2.glb"
            export_rigged_avatar(scene, path)
            report = inspect_avatar_glb(path)
            self.assertTrue(report["valid_native_glb"])
            self.assertEqual(report["visemes"], 15)

    def test_joint_edits_are_saved_in_glb_and_expressions_blend(self):
        scene = trimesh.Scene({BODY: self.body})
        default = PosePreview(scene)
        scene.metadata["aethryx_joint_offsets"] = {"LeftLowerArm": [0, .025, 0]}
        edited = PosePreview(scene)
        index = default.skeleton[0].index("LeftLowerArm")
        self.assertAlmostEqual(edited.skeleton[2][index, 1] - default.skeleton[2][index, 1],
                               .025 * self.body.extents[1])
        rest, _ = edited.evaluate({})
        left, _ = edited.evaluate({}, "Smile", .25)
        right, _ = edited.evaluate({}, "JawOpen", .4)
        combined, _ = edited.evaluate({}, {"Smile": .25, "JawOpen": .4})
        np.testing.assert_allclose(combined[BODY], left[BODY] + right[BODY] - rest[BODY], atol=1e-8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "offset.glb"
            export_rigged_avatar(scene, path)
            nodes = {node["name"]: node for node in validate_glb(path)["nodes"]}
            parent = edited.skeleton[1][index]
            np.testing.assert_allclose(nodes["LeftLowerArm"]["translation"],
                                       edited.skeleton[2][index] - edited.skeleton[2][parent])

    def test_kiria_can_edit_body_and_iris_without_writing_original(self):
        parts = sculpted_face_parts(self.body)
        scene = trimesh.Scene({BODY: self.body, EYES[0]: parts[EYES[0]], EYES[1]: parts[EYES[1]]})
        plan = plan_scene_command("altura", [BODY], scene.graph.nodes_geometry, "1.12")
        edited = apply_scene_plan(plan, scene, ())
        self.assertAlmostEqual(edited.get(BODY).extents[1], self.body.extents[1] * 1.12)
        np.testing.assert_allclose(edited.get(BODY).bounds[0, 1], self.body.bounds[0, 1])
        right = np.asarray(parts[EYES[1]].visual.material.baseColorTexture)
        applied = apply_scene_plan(plan_scene_command("color_iris", [EYES[0]], edited.nodes, "#e1468c"),
                                   edited.scene, ())
        np.testing.assert_array_equal(np.asarray(applied.get(EYES[1]).visual.material.baseColorTexture), right)
        self.assertFalse(np.array_equal(np.asarray(applied.get(EYES[0]).visual.material.baseColorTexture),
                                        np.asarray(parts[EYES[0]].visual.material.baseColorTexture)))

    def test_profile_alignment_rejects_ambiguous_landmarks(self):
        self.assertIs(profile_opposite({0: (25, 40), 7: (39, 39)}, 170), False)
        self.assertIs(profile_opposite({0: (63, 40), 8: (40, 40)}, 170), True)
        self.assertIsNone(profile_opposite({0: (63, 40), 7: (61, 40), 8: (66, 40)}, 170))
        self.assertIsNone(profile_opposite({7: (22, 40)}, 170))

    def test_unsaved_recovery_names_are_distinct_and_readable(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"LOCALAPPDATA": folder}):
            first, second = unsaved_recovery_path(), unsaved_recovery_path()
            self.assertNotEqual(first, second)
            first.parent.mkdir(parents=True)
            save_project(first, image="", model=None, output="", steps=37, faces=-1, seed=42, guidance=7)
            self.assertEqual(load_project(first)["settings"]["steps"], 37)
            self.assertFalse(second.exists())
