"""Silhouettes and local multiclass masks for the Genesis reference views."""

from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw


SEMANTIC_COLORS = {
    "background": (0, 0, 0), "body": (78, 210, 145), "face": (255, 182, 110),
    "hair": (163, 101, 245), "clothing": (50, 159, 255), "accessories": (255, 63, 147),
}

ISNET_MODEL = Path(__file__).resolve().parent / "models" / "anime" / "isnetis.onnx"
SEE_THROUGH_COLORS = {**SEMANTIC_COLORS, "eyes": (244, 219, 85)}


def named_reference_views(paths):
    """Read explicit angle names from filenames without guessing the image content."""
    aliases = {"front": {"front", "frontal"}, "side": {"side", "perfil", "lateral"},
               "back": {"back", "rear", "posterior", "espalda"}}
    if not 1 <= len(paths) <= 3:
        raise ValueError("Selecciona de una a tres imágenes.")
    result = {}
    for path in paths:
        words = set(re.findall(r"[a-z]+", Path(path).stem.lower()))
        matches = [view for view, names in aliases.items() if words & names]
        if len(matches) != 1 or matches[0] in result:
            raise ValueError(f"No se pudo asignar una vista única a {Path(path).name}. Cárgala en su vista manualmente.")
        result[matches[0]] = path
    return result


class AnimeSilhouette:
    """ISNet character foreground model using its native letterboxed RGB input."""

    def __init__(self, provider="CPUExecutionProvider", path=ISNET_MODEL):
        import onnxruntime as ort

        if not Path(path).is_file():
            raise ValueError("No se encontró el modelo local ISNet.")
        if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
        if self.session.get_providers()[0] != provider:
            raise RuntimeError(f"ONNX Runtime no pudo iniciar {provider}.")
        (self.input,) = self.session.get_inputs()
        if self.input.type != "tensor(float)" or self.input.shape[1:] != [3, 1024, 1024]:
            raise ValueError("ISNet requiere una entrada float32 con forma [N,3,1024,1024].")

    def predict(self, image):
        width, height = image.size
        scale = 1024 / max(width, height)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        x, y = (1024 - w) // 2, (1024 - h) // 2
        padded = np.zeros((1024, 1024, 3), dtype=np.float32)
        rgb = np.asarray(image.convert("RGB"))
        padded[y:y+h, x:x+w] = cv2.resize(rgb, (w, h)).astype(np.float32) / 255.
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None])
        (scores,) = self.session.run(None, {self.input.name: tensor})
        if scores.shape != (1, 1, 1024, 1024) or not np.isfinite(scores).all():
            raise ValueError("La salida de ISNet no contiene un mapa de primer plano válido.")
        foreground = cv2.resize(scores[0, 0, y:y+h, x:x+w], image.size, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(np.uint8(foreground >= .5) * 255)


def import_see_through_v2(path, size, *, include_eyes=False):
    """Collapse externally inferred 19-layer V2 masks to visible semantic classes."""
    width, height = size
    masks = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(masks, np.ndarray) or masks.dtype != np.bool_ or masks.shape != (19, height, width):
        raise ValueError("Se requiere el NPY booleano V2 con forma [19, alto, ancho] de la referencia.")
    categories = (
        ("body", (9,)), ("face", (2, 3, 5, 7, 8)), ("clothing", (11, 12, 13, 14, 15)),
        ("hair", (0,)), ("accessories", (1, 4, 6, 10, 16, 17, 18)),
    )
    palette = SEE_THROUGH_COLORS if include_eyes else SEMANTIC_COLORS
    if include_eyes:
        categories = (*categories[:2], ("eyes", (3,)), *categories[2:])
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label, indices in categories:
        foreground = np.zeros((height, width), dtype=bool)
        for index in indices:
            foreground |= masks[index]
        rgb[foreground] = palette[label]
    return Image.fromarray(rgb)


def segment_see_through_v2(image_path, size, python, source, checkpoint, *, cancelled):
    worker = Path(__file__).resolve().parent / "scripts" / "segment_see_through.py"
    with TemporaryDirectory(prefix="aethryx-segment-") as folder:
        output = Path(folder) / "parts.npy"
        args = [python, str(worker), source, checkpoint, image_path, str(output)]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              encoding="utf-8", errors="replace", creationflags=flags) as child:
            while True:
                if cancelled.is_set():
                    child.kill()
                    child.communicate()
                    return None
                try:
                    _, error = child.communicate(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if child.returncode:
                raise RuntimeError(error.strip()[-1200:] or f"See-through terminó con código {child.returncode}.")
        if cancelled.is_set():
            return None
        return import_see_through_v2(output, size, include_eyes=True)


def semantic_reference_regions(image, mask, colors=SEE_THROUGH_COLORS):
    """Summarize observed pixels and produce transparent per-class crops."""
    if image.size != mask.size:
        raise ValueError("La máscara semántica debe coincidir con la referencia.")
    source = np.asarray(image.convert("RGBA"))
    labels = np.asarray(mask.convert("RGB"))
    regions = {}
    for name, color in colors.items():
        if name == "background":
            continue
        selection = np.all(labels == color, axis=2) & (source[:, :, 3] >= 128)
        y, x = np.where(selection)
        if not len(x):
            continue
        bounds = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
        x0, y0, x1, y1 = bounds
        crop = source[y0:y1, x0:x1].copy()
        crop[~selection[y0:y1, x0:x1]] = 0
        median = np.rint(np.median(source[selection, :3], axis=0)).astype(np.uint8)
        regions[name] = {
            "bounds": bounds, "pixels": len(x), "color": "#%02x%02x%02x" % tuple(median),
            "crop": Image.fromarray(crop),
        }
    return regions


def refine_visible_mask(image, mask, silhouette):
    """Use foreground and face chroma to propose exposed skin on visible pixels."""
    if image.size != mask.size or image.size != silhouette.size:
        raise ValueError("Las tres imágenes deben tener la misma resolución.")
    labels = np.asarray(mask.convert("RGB")).copy()
    foreground = np.asarray(silhouette.convert("L")) >= 128
    labels[~foreground] = 0
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    face = np.all(labels == SEE_THROUGH_COLORS["face"], axis=2)
    if face.sum() >= 36:
        chroma = lab[:, :, 1:]
        center = np.median(chroma[face], axis=0)
        distance = np.linalg.norm(chroma - center, axis=2)
        radius = np.clip(np.quantile(distance[face], .75) * 1.5, 8, 22)
        unassigned = ~labels.any(axis=2)
        clothes = np.all(labels == SEE_THROUGH_COLORS["clothing"], axis=2)
        eligible = foreground & (unassigned | clothes)
        low_light = np.quantile(lab[:, :, 0][face], .15) - 35
        exposed = eligible & (distance <= radius) & (lab[:, :, 0] >= low_light)
        count, regions, stats, _ = cv2.connectedComponentsWithStats(exposed.astype(np.uint8))
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(12, foreground.size // 30000)
        labels[keep[regions]] = SEE_THROUGH_COLORS["body"]
    return Image.fromarray(labels)


def remove_mask_islands(mask, palette, minimum_area):
    labels = np.asarray(mask.convert("RGB")).copy()
    for name, color in palette.items():
        if name != "background":
            selected = np.all(labels == color, axis=2)
            count, components, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8))
            discard = np.zeros(count, dtype=bool)
            discard[1:] = stats[1:, cv2.CC_STAT_AREA] < minimum_area
            labels[discard[components]] = 0
    return Image.fromarray(labels)


def retain_mask_polygon(mask, vertices):
    """Keep existing labels inside an operator-marked panel or silhouette."""
    points = np.asarray(vertices, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or np.any(points < 0)
            or np.any(points >= mask.size) or cv2.contourArea(points.astype(np.float32)) < 1):
        raise ValueError("Marca al menos tres puntos de un contorno válido dentro de la imagen.")
    keep = Image.new("L", mask.size)
    ImageDraw.Draw(keep).polygon([tuple(point) for point in points], fill=255)
    result = Image.new(mask.mode, mask.size)
    result.paste(mask, mask=keep)
    return result


def pose_points_for_view(points, view):
    """Exclude automatic facial landmarks from rear-view correspondences."""
    return {joint: point for joint, point in points.items() if view != "back" or joint >= 11}


def vertical_pose_disagreements(observations, tolerance=.06):
    """Flag projected height differences between calibrated views for review."""
    heights = {}
    for calibration, points in observations:
        if calibration is None:
            continue
        (_, floor), crown = calibration
        for joint, (_, y) in points.items():
            heights.setdefault(joint, []).append((floor - y) / (floor - crown))
    return sorted(joint for joint, values in heights.items() if max(values) - min(values) > tolerance)


def foreground_crop_diagnostics(mask):
    """Report edge contact and secondary foreground without changing the mask.

    The largest connected component is the subject candidate. Detached pixels
    may be valid details, so their area is reported without removing them.
    """
    foreground = np.asarray(mask.convert("RGB")).any(axis=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8,
    )
    pixels = int(foreground.sum())
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    edges = {
        "top": int(foreground[0].sum()),
        "bottom": int(foreground[-1].sum()),
        "left": int(foreground[:, 0].sum()),
        "right": int(foreground[:, -1].sum()),
    }
    return {"edge_pixels": edges, "components": count - 1,
            "secondary_fraction": (pixels - largest) / pixels if pixels else 0.0}


def reference_silhouette(image, semantic_mask=None, *, allow_border=False):
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
    if not allow_border and (foreground[[0, -1]].any() or foreground[:, [0, -1]].any()):
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


def compare_silhouette(mesh, image, projection, semantic_mask=None, maximum_size=640, *, allow_border=False):
    original_size = image.size
    reference = reference_silhouette(image, semantic_mask, allow_border=allow_border)
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
