from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from aethryx_anatomy import blocking_geometry_issues, validate_deformation
from aethryx_anime import hair_locks, sculpted_body, sculpted_face_parts
from aethryx_bridge import inspect_avatar_glb
from aethryx_modular import BODY, EYES, anime_body, avatar_eye
from aethryx_rig import PosePreview, export_rigged_avatar


class GenesisQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = sculpted_body("Anime femenino")

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
