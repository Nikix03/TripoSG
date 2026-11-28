from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch
import unittest

import numpy as np

from aethryx_blender import blender_version, export_with_blender, find_blender
from aethryx_rig import hair_chain_weights


class EvolutionXTests(unittest.TestCase):
    def test_backend_checks_version_and_honors_cancellation_before_launch(self):
        with TemporaryDirectory() as folder:
            executable = Path(folder) / "blender.exe"
            executable.touch()
            self.assertEqual(find_blender(executable), executable.resolve())
            self.assertIsNone(find_blender(executable.with_name("missing.exe")))
            with patch("aethryx_blender.subprocess.run") as run:
                run.return_value.stdout = "Blender 4.5.1"
                self.assertEqual(blender_version(executable), "Blender 4.5.1")
                run.return_value.stdout = "Blender 3.6.0"
                with self.assertRaises(ValueError):
                    blender_version(executable)
                run.reset_mock()
                cancelled = Event()
                cancelled.set()
                self.assertIsNone(export_with_blender(executable, "missing.glb", "new.fbx", cancelled))
                run.assert_not_called()

    def test_chain_uses_all_four_joints_and_respects_the_influence_limit(self):
        uv = np.array([1., .85, .7, .35, 0.])
        indices, weights = hair_chain_weights(uv, 53)
        self.assertEqual(set(indices[weights > 0]), {53, 54, 55, 56})
        np.testing.assert_allclose(weights.sum(axis=1), 1.)
        np.testing.assert_allclose(weights[1, :2], [.5, .5])
        _, rigid = hair_chain_weights(uv, 53, influences=1)
        np.testing.assert_array_equal(np.count_nonzero(rigid, axis=1), np.ones(len(uv)))
