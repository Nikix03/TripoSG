import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.texture import TextureVisuals

from aethryx_materials import pbr_map_images, transform_uv, update_pbr_maps, uv_quality
from aethryx_bridge import inspect_avatar_glb
from kiria_dialog import apply_scene_plan, plan_scene_command
from aethryx_modular import (
    BODY, EYES, ModularAvatar, anime_body, avatar_eye, fit_garment, fitted_eyelids,
    mouth_components, regional_hair, visible_body_faces,
)
from aethryx_rig import (
    _GlbBuffers, PosePreview, deform_pose, export_rigged_avatar, facial_morphs,
    humanoid_skeleton, pose_matrices, skin_weights,
)
from studio_project import validate_glb


class GenesisExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = anime_body("Anime femenino")

    def test_lids_hair_and_mouth_are_separate_geometry(self):
        eye = avatar_eye(self.body, True, "#164489")
        self.assertTrue(all(mesh.is_volume for mesh in fitted_eyelids(eye).values()))
        hair = regional_hair(self.body)
        self.assertEqual(len(hair), 4)
        for mesh in hair.values():
            self.assertTrue(mesh.is_watertight)
            colors = np.asarray(mesh.visual.material.baseColorTexture)
            self.assertFalse(np.array_equal(colors[0], colors[-1]))
        mouth = mouth_components(self.body)
        self.assertEqual(len(mouth), 4)
        self.assertFalse(mouth["boca-cavidad"].is_watertight)  # Open at the lips.
        self.assertTrue(mouth["lengua"].is_volume)

    def test_pbr_updates_multiple_maps_and_preserves_unsupplied_channels(self):
        eye = avatar_eye(self.body, True, "#164489")
        update_pbr_maps(eye, {
            "roughness": Image.new("L", (4, 4), 39), "metallic": Image.new("L", (4, 4), 171),
            "occlusion": Image.new("L", (4, 4), 202), "normal": Image.new("RGB", (4, 4), (128, 127, 255)),
            "emissive": Image.new("RGB", (4, 4), (24, 60, 33)),
        }, 512, "MASK", .37)
        material = eye.visual.material
        self.assertEqual(material.metallicRoughnessTexture.getpixel((0, 0))[1:], (39, 171))
        self.assertEqual(material.baseColorTexture.size, (192, 192))
        maps = pbr_map_images(eye)
        self.assertEqual(maps["occlusion"].getpixel((1, 1)), 202)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pbr.glb"
            trimesh.Scene({"test": eye}).export(path)
            saved = validate_glb(path)["materials"][0]
            self.assertEqual(saved["alphaMode"], "MASK")
            self.assertAlmostEqual(saved["alphaCutoff"], .37)
            self.assertTrue({"normalTexture", "emissiveTexture", "occlusionTexture"}.issubset(saved))

    def test_uv_overlap_is_detected_and_offset_does_not_touch_positions(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False)
        coordinates = np.asarray([[.1, .1], [.8, .1], [.1, .8]] * 2)
        mesh.visual = TextureVisuals(uv=coordinates)
        self.assertEqual(uv_quality(mesh, 64)["overlap_fraction"], 1)
        transform_uv(mesh, [.5, .5], [.2, .2])
        np.testing.assert_allclose(mesh.visual.uv, coordinates * .5 + .2)
        np.testing.assert_array_equal(mesh.vertices, vertices)

    def test_dress_skirt_and_outfit_reopen_without_changing_body(self):
        avatar = ModularAvatar(trimesh.Scene({BODY: self.body}))
        for style in ("falda", "vestido"):
            garment, contact = fit_garment(self.body, style=style)
            self.assertEqual(contact, {"contacts": 0, "inside": 0})
            self.assertTrue(garment.is_volume)
            avatar.add(f"prenda-{style}", garment)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outfit.glb"
            avatar.export_outfit(path)
            restored = ModularAvatar(trimesh.Scene({BODY: self.body}))
            restored.apply_outfit(path)
            self.assertEqual(set(restored.nodes), {BODY, "prenda-falda", "prenda-vestido"})
            np.testing.assert_array_equal(restored.get(BODY).faces, self.body.faces)

    def test_skin_rest_is_identity_and_movement_propagates(self):
        body = self.body
        skeleton = humanoid_skeleton(body)
        joints, weights = skin_weights(body.vertices, skeleton, 3)
        self.assertEqual(len(skeleton[0]), 53)
        self.assertTrue(np.all(weights[:, 3] == 0))
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
        rest = deform_pose(body.vertices, joints, weights, pose_matrices(skeleton, {}))
        np.testing.assert_allclose(rest, body.vertices, atol=1e-7)
        moved = deform_pose(body.vertices, joints, weights, pose_matrices(skeleton, {"LeftUpperArm": (0, 0, 30)}))
        elbow_side = body.vertices[:, 0] > .20
        self.assertGreater(np.max(np.linalg.norm((moved - rest)[elbow_side], axis=1)), .05)
        feet = body.vertices[:, 1] < .10
        self.assertLess(np.max(np.linalg.norm((moved - rest)[feet], axis=1)), .001)

    def test_pose_t_moves_arms_and_an_eye_rotates_independently(self):
        source = self.body
        scene = trimesh.Scene({
            BODY: source, EYES[0]: avatar_eye(source, True, "#3366cc"),
            EYES[1]: avatar_eye(source, False, "#aa4499"),
        })
        left_lid, right_lid = (f"{eye}-parpado-superior" for eye in EYES)
        for eye, name in zip(EYES, (left_lid, right_lid)):
            scene.add_geometry(fitted_eyelids(scene.geometry[eye])["superior"], node_name=name, geom_name=name)
        preview = PosePreview(scene)
        rest, bind = preview.evaluate({})
        changed, t_points = preview.evaluate(preview.t_pose())
        left_shoulder = preview.skeleton[0].index("LeftUpperArm")
        left_hand = preview.skeleton[0].index("LeftHand")
        self.assertLess(abs(t_points[left_shoulder, 1] - t_points[left_hand, 1]), 1e-6)
        self.assertGreater(abs(bind[left_shoulder, 1] - bind[left_hand, 1]), .10)
        self.assertGreater(np.max(abs(changed[BODY] - rest[BODY])), .06)
        eye, _ = preview.evaluate({"LeftEye": (0, 23, 0)})
        self.assertGreater(np.max(abs(eye[EYES[0]] - rest[EYES[0]])), .001)
        np.testing.assert_allclose(eye[EYES[1]], rest[EYES[1]], atol=1e-10)
        blinking, _ = preview.evaluate({}, "BlinkLeft", 1)
        self.assertGreater(np.max(abs(blinking[left_lid] - rest[left_lid])), .001)
        np.testing.assert_allclose(blinking[right_lid], rest[right_lid], atol=1e-10)

    def test_preview_hides_regions_without_removing_body_faces(self):
        body = self.body.copy()
        body.metadata["aethryx_hidden_body_bands"] = ["torso"]
        visible = visible_body_faces(body)
        self.assertGreater(visible.sum(), 0)
        self.assertLess(visible.sum(), len(body.faces))
        np.testing.assert_array_equal(body.faces, self.body.faces)
        body.metadata["aethryx_hidden_body_bands"].clear()
        self.assertTrue(visible_body_faces(body).all())

    def test_glb_contains_bind_weights_animation_and_local_morph_deltas(self):
        body = self.body
        shapes = facial_morphs(body)
        self.assertEqual(len([key for key in shapes if key.startswith("vrc.v_")]), 15)
        left = shapes["BlinkLeft"]
        self.assertGreater(np.linalg.norm(left[body.vertices[:, 0] > 0]), 50 * np.linalg.norm(left[body.vertices[:, 0] < 0]))
        self.assertLess(np.max(np.abs(shapes["Smile"][body.vertices[:, 1] < .60])), 1e-9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rig.glb"
            scene = trimesh.Scene({BODY: body, EYES[0]: avatar_eye(body, True, "#164489")})
            result = export_rigged_avatar(scene, path)
            self.assertEqual(result["skinned_nodes"], 2)
            glb = _GlbBuffers(path.read_bytes())
            doc = validate_glb(path)
            self.assertEqual(len(doc["skins"][0]["joints"]), 53)
            self.assertEqual(len(doc["animations"][0]["channels"]), 10)
            node = next(node for node in doc["nodes"] if node["name"] == BODY)
            mesh = doc["meshes"][node["mesh"]]
            primitive = mesh["primitives"][0]
            self.assertEqual(len(primitive["targets"]), len(shapes))
            indices = primitive["attributes"]
            weights = doc["accessors"][indices["WEIGHTS_0"]]
            view = doc["bufferViews"][weights["bufferView"]]
            stored = np.frombuffer(glb.binary, dtype="<f4", count=weights["count"] * 4, offset=view["byteOffset"])
            np.testing.assert_allclose(stored.reshape(-1, 4).sum(axis=1), 1, atol=1e-6)
            report = inspect_avatar_glb(path)
            self.assertTrue(report["valid_native_glb"])
            self.assertEqual(report["visemes"], 15)
            self.assertGreater(report["sample_body_displacement"], .05)
            glb.binary[view["byteOffset"]:view["byteOffset"] + 16] = b"\0" * 16
            corrupt = Path(tmp) / "bad.glb"
            corrupt.write_bytes(glb.encode())
            self.assertFalse(inspect_avatar_glb(corrupt)["valid_native_glb"])

    def test_kiria_previews_allowlisted_changes_without_modifying_input_scene(self):
        original = trimesh.Scene({BODY: self.body})
        show = plan_scene_command("mostrar", [BODY], {BODY})
        self.assertEqual(apply_scene_plan(show, original, [BODY]).hidden, set())
        apply_color = plan_scene_command("color_base", [BODY], {BODY}, "#44aaff")
        revised = apply_scene_plan(apply_color, original, ())
        material = revised.get(BODY).visual.material
        self.assertEqual(material.baseColorFactor[:3].tolist(), [68, 170, 255])
        self.assertNotEqual(original.geometry[BODY].visual.material.baseColorFactor[:3].tolist(), [68, 170, 255])
        with self.assertRaises(ValueError):
            plan_scene_command("python", [BODY], {BODY}, "")
        with self.assertRaises(ValueError):
            plan_scene_command("ocultar", ["missing"], {BODY})
