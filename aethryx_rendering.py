"""Camera fitting helpers for consistent full-mesh review renders."""

import numpy as np


def orthographic_camera(bounds, direction, up, window_size, margin=1.12):
    """Return a fitted camera position, focal point, up vector, and parallel scale."""
    bounds = np.asarray(bounds, dtype=float)
    direction = np.array(direction, dtype=float, copy=True)
    up = np.array(up, dtype=float, copy=True)
    width, height = window_size
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Los límites de la malla deben ser finitos y tener volumen.")
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("La dirección de cámara debe ser un vector finito no nulo.")
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
        raise ValueError("El vector vertical de cámara debe ser finito y no nulo.")
    if width <= 0 or height <= 0 or not np.isfinite(margin) or margin <= 1:
        raise ValueError("El tamaño del visor debe ser positivo y el margen mayor que 1.")

    direction /= np.linalg.norm(direction)
    up /= np.linalg.norm(up)
    view = -direction
    right = np.cross(view, up)
    right_length = np.linalg.norm(right)
    if right_length <= 1e-8:
        raise ValueError("La dirección de cámara y el vector vertical no pueden ser paralelos.")
    right /= right_length
    up = np.cross(right, view)
    up /= np.linalg.norm(up)

    corners = np.array([
        [x, y, z]
        for x in bounds[:, 0]
        for y in bounds[:, 1]
        for z in bounds[:, 2]
    ])
    target = bounds.mean(axis=0)
    relative = corners - target
    projected_width = np.ptp(relative @ right)
    projected_height = np.ptp(relative @ up)
    aspect = width / height
    parallel_scale = max(projected_height / 2, projected_width / (2 * aspect)) * margin
    distance = max(np.linalg.norm(bounds[1] - bounds[0]) * 2, parallel_scale * 4, 1.)
    position = target + direction * distance
    return position, target, up, parallel_scale
