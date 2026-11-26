from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from aethryx_anatomy import load_body_template
from aethryx_modular import (
    BODY, EYES, FACE_CONTROLS, ModularAvatar, anime_body, avatar_eye, body_contacts,
    facial_vertices, fit_garment, fit_hair, recolor_iris, write_new_glb,
)
from aethryx_profiles import BUNDLED_BODY
from studio_project import validate_glb


class ModularAvatarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = anime_body()

    def test_styled_body_is_closed_and_face_controls_keep_vertex_order(self):
        source = load_body_template(BUNDLED_BODY)
        body = self.body
        np.testing.assert_array_equal(source.faces, body.faces)
        self.assertTrue(body.is_volume)
        self.assertGreater(np.max(abs(source.vertices - body.vertices)), .01)
        for control in FACE_CONTROLS:
            with self.subTest(control=control):
                points = facial_vertices(body, {control: 1.16})
                self.assertEqual(points.shape, body.vertices.shape)
                self.assertGreater(np.max(abs(points - body.vertices)), 1e-4)

    def test_independent_eye_materials_uv_and_body_reopen_from_glb(self):
        avatar = ModularAvatar(None)
        avatar.add(BODY, self.body.copy())
        avatar.add(EYES[0], avatar_eye(self.body, True, "#2255ee"))
        avatar.add(EYES[1], avatar_eye(self.body, False, "#e83355"))
        right_before = np.asarray(avatar.get(EYES[1]).visual.material.baseColorTexture).copy()
        eye = avatar.get(EYES[0])
        recolor_iris(eye, "#30aa40", .42)
        avatar.replace(EYES[0], eye)
        np.testing.assert_array_equal(right_before, np.asarray(avatar.get(EYES[1]).visual.material.baseColorTexture))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "modular.glb"
            write_new_glb(avatar.scene, path)
            gltf = validate_glb(path)
            self.assertTrue(all("TEXCOORD_0" in entry["primitives"][0]["attributes"] for entry in gltf["meshes"]))
            reopened = ModularAvatar(trimesh.load(path, force="scene", process=False))
            self.assertEqual(set(reopened.nodes), {BODY, *EYES})
            self.assertEqual(len(reopened.get(BODY).faces), 26756)
            self.assertEqual(reopened.get(BODY).metadata["aethryx_style"], "genesis-anime-v1")
            self.assertAlmostEqual(reopened.get(EYES[0]).visual.material.roughnessFactor, .42)
            self.assertGreater(reopened.get(EYES[0]).centroid[0], reopened.get(EYES[1]).centroid[0])
            self.assertFalse(np.array_equal(
                np.asarray(reopened.get(EYES[0]).visual.material.baseColorTexture),
                np.asarray(reopened.get(EYES[1]).visual.material.baseColorTexture),
            ))

    def test_hair_and_garment_are_volumes_with_independent_fit(self):
        short, long = fit_hair(self.body, length=.1), fit_hair(self.body, length=.7)
        self.assertTrue(short.is_volume and long.is_volume)
        self.assertLess(long.bounds[0, 1], short.bounds[0, 1])
        garment, report = fit_garment(self.body)
        self.assertTrue(garment.is_volume)
        self.assertEqual(report, {"contacts": 0, "inside": 0})
        inside = body_contacts(trimesh.creation.box(extents=(2, 2, 2)), trimesh.creation.box())
        self.assertEqual(inside, {"contacts": 0, "inside": 8})

    def test_replacement_preserves_transform_shared_instance_and_original_scene(self):
        scene = trimesh.Scene()
        transform = trimesh.transformations.rotation_matrix(.5, [0, 1, 0])
        transform[:3, 3] = [2, 3, 4]
        scene.add_geometry(trimesh.creation.box(), geom_name="original", node_name="first", transform=transform)
        scene.graph.update(frame_to="second", geometry="original")
        avatar = ModularAvatar(scene, {"first"})
        avatar.replace("first", trimesh.creation.icosphere(subdivisions=1))
        np.testing.assert_allclose(avatar.scene.graph["first"][0], transform)
        self.assertEqual(len(avatar.get("second").faces), 12)
        self.assertEqual(len(scene.geometry), 1)
        avatar.remove("first")
        self.assertEqual(avatar.nodes, ["second"])
        self.assertFalse(avatar.hidden)
        avatar.add(BODY, self.body.copy())
        with self.assertRaises(ValueError):
            avatar.remove(BODY)
