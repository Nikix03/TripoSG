import unittest

import numpy as np

from aethryx_rendering import orthographic_camera


class OrthographicCameraTests(unittest.TestCase):
    def test_full_bounds_fit_front_side_and_rear_non_origin_mesh(self):
        bounds = np.array([[3., -2., -1.], [5., 4., 2.]])
        size = (900, 600)
        corners = np.array([
            [x, y, z]
            for x in bounds[:, 0]
            for y in bounds[:, 1]
            for z in bounds[:, 2]
        ])
        for direction in ((0, 0, 1), (1, 0, 0), (0, 0, -1)):
            position, target, up, scale = orthographic_camera(
                bounds, direction, (0, 1, 0), size,
            )
            view = (target - position) / np.linalg.norm(target - position)
            right = np.cross(view, up)
            right /= np.linalg.norm(right)
            projected = corners - target
            self.assertLessEqual(np.max(np.abs(projected @ up)), scale)
            self.assertLessEqual(np.max(np.abs(projected @ right)), scale * size[0] / size[1])
            np.testing.assert_allclose(target, bounds.mean(axis=0))

    def test_rejects_degenerate_bounds_and_parallel_camera_up(self):
        with self.assertRaises(ValueError):
            orthographic_camera([[0, 0, 0], [0, 1, 1]], (0, 0, 1), (0, 1, 0), (900, 600))
        with self.assertRaises(ValueError):
            orthographic_camera([[0, 0, 0], [1, 1, 1]], (0, 1, 0), (0, 1, 0), (900, 600))

    def test_normalizes_camera_vectors_without_mutating_inputs(self):
        direction = np.array([0., 0., 3.])
        up = np.array([0., 2., 0.])
        orthographic_camera([[0, 0, 0], [1, 1, 1]], direction, up, (900, 600))
        np.testing.assert_array_equal(direction, [0., 0., 3.])
        np.testing.assert_array_equal(up, [0., 2., 0.])
