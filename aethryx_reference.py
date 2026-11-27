"""Silhouettes and local multiclass masks for the Genesis reference views."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SEMANTIC_COLORS = {
    "background": (0, 0, 0), "body": (78, 210, 145), "face": (255, 182, 110),
    "hair": (163, 101, 245), "clothing": (50, 159, 255), "accessories": (255, 63, 147),
}


def import_see_through_v2(path, size):
    """Collapse externally inferred 19-layer V2 masks to visible semantic classes."""
    width, height = size
    masks = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(masks, np.ndarray) or masks.dtype != np.bool_ or masks.shape != (19, height, width):
        raise ValueError("Se requiere el NPY booleano V2 con forma [19, alto, ancho] de la referencia.")
    categories = (
        ("body", (9,)), ("face", (2, 3, 5, 7, 8)), ("clothing", (11, 12, 13, 14, 15)),
        ("hair", (0,)), ("accessories", (1, 4, 6, 10, 16, 17, 18)),
    )
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label, indices in categories:
        foreground = np.zeros((height, width), dtype=bool)
        for index in indices:
            foreground |= masks[index]
        rgb[foreground] = SEMANTIC_COLORS[label]
    return Image.fromarray(rgb)


def reference_silhouette(image, semantic_mask=None):
    explicit = semantic_mask is not None
    if semantic_mask is not None:
        if semantic_mask.size != image.size:
            raise ValueError("La máscara y la referencia deben tener las mismas dimensiones.")
        foreground = np.asarray(semantic_mask.convert("RGB")).any(axis=2)
    else:
        rgba = np.asarray(image.convert("RGBA"))
        if np.any(rgba[:, :, 3] < 32):
            foreground = rgba[:, :, 3] >= 128
            explicit = True
        else:
            rgb = rgba[:, :, :3].astype(np.int16)
            border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
            background = np.median(border, axis=0)
            if np.quantile(np.max(np.abs(border - background), axis=1), .9) > 25:
                return None
            foreground = np.max(np.abs(rgb - background), axis=2) > 32
    if not explicit:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
        if count <= 1:
            return None
        foreground = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = np.count_nonzero(foreground)
    if not .02 * foreground.size <= area <= .93 * foreground.size:
        return None
    if foreground[[0, -1]].any() or foreground[:, [0, -1]].any():
        return None
    return foreground


def silhouette_calibration(foreground, points):
    """Use the silhouette's vertical extent and stable torso anchors for centering."""
    if foreground is None:
        return None
    rows, _ = np.where(foreground)
    crown, floor = float(rows.min()), float(rows.max())
    height = floor - crown
    if height < 8:
        return None
    torso = [
        (points[a][0] + points[b][0]) / 2
        for a, b in ((11, 12), (23, 24)) if a in points and b in points
    ]
    if not torso:
        for y in np.linspace(crown + .32 * height, crown + .53 * height, 7).astype(int):
            xs = np.flatnonzero(foreground[y])
            if len(xs):
                torso.append(float(np.median(xs)))
    if not torso:
        return None
    return [float(np.median(torso)), floor], crown


def profile_opposite(points, width):
    """Return a profile direction only when a visible facial landmark supports it."""
    if 0 not in points:
        return None
    ears = [points[index][0] for index in (7, 8) if index in points]
    if not ears:
        return None
    nose = points[0][0]
    differences = [nose - ear for ear in ears]
    threshold = max(3., .025 * width)
    if min(differences) > threshold:
        return True
    if max(differences) < -threshold:
        return False
    return None


def segment_anime_reference(image, model_path, labels):
    """Run a local RGB [0,1] NCHW ONNX parser and return an exact RGB class mask."""
    if len(labels) != len(SEMANTIC_COLORS) or set(labels) != set(SEMANTIC_COLORS):
        raise ValueError("Define los seis canales de fondo, cuerpo, rostro, cabello, ropa y accesorios.")
    if not Path(model_path).is_file():
        raise ValueError("No se encontró el modelo ONNX multiclase local.")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4:
        raise ValueError("El ONNX debe aceptar una sola entrada float32 NCHW.")
    batch, channels, height, width = inputs[0].shape
    if batch != 1 or channels != 3 or not all(isinstance(dim, int) and dim > 0 for dim in (height, width)):
        raise ValueError("El ONNX requiere dimensiones estáticas [1, 3, alto, ancho].")
    rgb = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    tensor = np.ascontiguousarray((rgb / 255.0).transpose(2, 0, 1)[None])
    output = session.run(None, {inputs[0].name: tensor})[0]
    mask = decode_semantic_scores(output, image.size, labels)
    return Image.fromarray(mask)


def decode_semantic_scores(output, size, labels):
    if (output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != len(labels)
            or not np.isfinite(output).all()):
        raise ValueError("La salida debe contener puntuaciones finitas [1, clases, alto, ancho].")
    indices = np.argmax(output[0], axis=0).astype(np.uint8)
    indices = cv2.resize(indices, size, interpolation=cv2.INTER_NEAREST)
    return np.asarray([SEMANTIC_COLORS[label] for label in labels], dtype=np.uint8)[indices]


def compare_silhouette(mesh, image, projection, semantic_mask=None, maximum_size=640):
    original_size = image.size
    reference = reference_silhouette(image, semantic_mask)
    image = image.convert("RGB")
    image.thumbnail((maximum_size, maximum_size), Image.Resampling.LANCZOS)
    width, height = image.size
    scale = np.array([width, height]) / original_size
    projection = np.asarray(projection)
    points = (mesh.vertices @ projection[:, :3].T + projection[:, 3]) * scale
    pixels = np.rint(np.clip(points, -1e6, 1e6)).astype(np.int32)
    raster = np.zeros((height, width), dtype=np.uint8)
    for triangle in pixels[mesh.faces]:
        cv2.fillConvexPoly(raster, triangle, 1)
    model = raster.astype(bool)
    overlay = np.asarray(image).copy()
    def tint(mask, color):
        overlay[mask] = np.round(overlay[mask] * .38 + np.asarray(color) * .62).astype(np.uint8)
    if reference is None:
        tint(model, (0, 220, 255))
        return Image.fromarray(overlay), None
    reference = cv2.resize(reference.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    tint(reference & model, (77, 231, 151))
    tint(reference & ~model, (255, 68, 156))
    tint(model & ~reference, (0, 220, 255))
    union = int(np.count_nonzero(reference | model))
    return Image.fromarray(overlay), float(np.count_nonzero(reference & model) / union) if union else None
