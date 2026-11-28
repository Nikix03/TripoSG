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


def default_see_through_paths(root=None):
    """Return (python, source, checkpoint) if the local Apache-2.0 See-through V2 assets exist."""
    import sys

    base = Path(root) if root is not None else Path(__file__).resolve().parent
    source = base / "external" / "see-through"
    checkpoint = base / "pretrained_weights" / "See-through" / "checkpoint-18000.pt"
    if (source / "common" / "modules" / "semanticsam.py").is_file() and checkpoint.is_file():
        return sys.executable, str(source), str(checkpoint)
    return None


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


def analyze_view_alignment(source, image, view, body_mask=None, segmenter=None):
    from aethryx_engines import suggest_anatomical_points

    try:
        points = pose_points_for_view(suggest_anatomical_points(source), view)
    except (OSError, ImportError, RuntimeError, ValueError):
        points = {}
    mask = None
    if body_mask:
        with Image.open(body_mask) as saved:
            mask = saved.convert("RGB")
    elif segmenter is not None and image.convert("RGBA").getchannel("A").getextrema()[0] >= 32:
        mask = segmenter.predict(image)
    foreground = reference_silhouette(image, mask)
    diagnostics = foreground_crop_diagnostics(mask) if mask is not None else None
    return silhouette_calibration(foreground, points), points, diagnostics


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


def _hex_color(pixels):
    rgb = np.rint(np.median(pixels, axis=0)).astype(np.uint8)
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def extract_reference_palette_and_style(image, mask=None, silhouette=None):
    """Extract conservative color & garment-style hints from visible reference pixels."""
    rgba = np.asarray(image.convert("RGBA"))
    rgb = rgba[:, :, :3]
    fg = np.asarray(silhouette.convert("L")) >= 128 if silhouette is not None else (rgba[:, :, 3] >= 128)
    result = {}
    if mask is not None:
        labels = np.asarray(mask.convert("RGB"))
        for key, target in (("hair", "hair"), ("face", "skin"), ("clothing", "clothing"), ("body", "skin")):
            sel = np.all(labels == SEE_THROUGH_COLORS[key], axis=2) & fg
            if sel.sum() >= 24 and target not in result:
                result[target] = _hex_color(rgb[sel])
        hair_sel = np.all(labels == SEE_THROUGH_COLORS["hair"], axis=2) & fg
        if hair_sel.sum() >= 64:
            hy, _ = np.where(hair_sel)
            y_split_top = np.quantile(hy, .35)
            y_split_bot = np.quantile(hy, .75)
            root_px = rgb[hair_sel & (np.arange(rgb.shape[0])[:, None] <= y_split_top)]
            tip_px = rgb[hair_sel & (np.arange(rgb.shape[0])[:, None] >= y_split_bot)]
            if len(root_px) >= 16:
                result["hair_root"] = _hex_color(root_px)
            if len(tip_px) >= 16:
                result["hair_tip"] = _hex_color(tip_px)
        eye_sel = np.all(labels == SEE_THROUGH_COLORS["eyes"], axis=2) & fg
        if eye_sel.sum() >= 20:
            ey, ex = np.where(eye_sel)
            mid_x = float(np.median(ex))
            eyes = []
            for side_mask in (ex >= mid_x, ex < mid_x):
                pts = rgb[ey[side_mask], ex[side_mask]]
                if len(pts) >= 6:
                    # Pick chromatic non-sclera pixels for iris tint.
                    chroma = pts.max(axis=1).astype(int) - pts.min(axis=1).astype(int)
                    dark = pts.mean(axis=1) < 225
                    iris = pts[(chroma >= 18) & dark]
                    eyes.append(_hex_color(iris if len(iris) >= 4 else pts))
            if len(eyes) == 2:
                result["eyes"] = tuple(eyes)
        cloth_sel = np.all(labels == SEE_THROUGH_COLORS["clothing"], axis=2) & fg
        if cloth_sel.sum() >= 48:
            cy, cx = np.where(cloth_sel)
            mid_y = float(np.quantile(cy, .50))
            top_px = rgb[cloth_sel & (np.arange(rgb.shape[0])[:, None] <= mid_y)]
            bot_px = rgb[cloth_sel & (np.arange(rgb.shape[0])[:, None] > mid_y)]
            if len(top_px) >= 24:
                result["topwear"] = _hex_color(top_px)
            if len(bot_px) >= 24:
                result["bottomwear"] = _hex_color(bot_px)
            # Heuristic: dress vs top+trousers based on lower-garment horizontal continuity.
            lower_rows = np.unique(cy[cy > mid_y])
            if len(lower_rows) >= 6:
                span_ratios = []
                for r in lower_rows[::max(1, len(lower_rows) // 6)]:
                    row_xs = np.flatnonzero(cloth_sel[r])
                    if len(row_xs) >= 4:
                        span = row_xs.max() - row_xs.min() + 1
                        span_ratios.append(len(row_xs) / max(1, span))
                if span_ratios and float(np.median(span_ratios)) >= 0.85:
                    result["garment_style"] = "vestido"
                else:
                    result["garment_style"] = "top"
    elif fg.any():
        ys, xs = np.where(fg)
        y0, y1 = int(ys.min()), int(ys.max())
        h = max(1, y1 - y0)
        hair_band = fg & (np.arange(rgb.shape[0])[:, None] <= y0 + .16 * h)
        torso_band = fg & (np.arange(rgb.shape[0])[:, None] >= y0 + .28 * h) & (np.arange(rgb.shape[0])[:, None] <= y0 + .55 * h)
        if hair_band.sum() >= 24:
            result["hair_root"] = _hex_color(rgb[hair_band])
        if torso_band.sum() >= 24:
            result["topwear"] = _hex_color(rgb[torso_band])
    return result


def estimate_silhouette_limbs(foreground, pose_points, calibration):
    """Fill missing wrist or ankle points safely using silhouette geometry around detected parent joints."""
    if foreground is None or calibration is None:
        return dict(pose_points)
    points = dict(pose_points)
    (cx, floor), crown = calibration
    height = max(8.0, floor - crown)
    h, w = foreground.shape
    # Estimate missing wrists from shoulder/elbow outer silhouette reach in mid-body band.
    for side, shoulder_id, elbow_id, wrist_id in (("left", 11, 13, 15), ("right", 12, 14, 16)):
        if wrist_id in points or shoulder_id not in points:
            continue
        sx, sy = points[shoulder_id]
        y_lo = int(max(0, min(h - 1, sy)))
        y_hi = int(max(y_lo + 2, min(h, crown + .68 * height)))
        band = foreground[y_lo:y_hi]
        if not band.any():
            continue
        if side == "left":
            # Image-left (character right) vs image-right (character left): follow shoulder sign relative to cx.
            is_right_in_img = sx >= cx
            xs_mask = (np.arange(w) >= (cx + .08 * height)) if is_right_in_img else (np.arange(w) <= (cx - .08 * height))
        else:
            is_right_in_img = sx >= cx
            xs_mask = (np.arange(w) >= (cx + .08 * height)) if is_right_in_img else (np.arange(w) <= (cx - .08 * height))
        arm_fg = band[:, xs_mask]
        if not arm_fg.any():
            continue
        ys_local, xs_local = np.where(arm_fg)
        x_offset = int(cx + .08 * height) if is_right_in_img else 0
        # Distal tip is furthest along shoulder->elbow vector (or lowest-outer point).
        if elbow_id in points:
            ex, ey = points[elbow_id]
            dx, dy = ex - sx, ey - sy
        else:
            dx, dy = (1.0 if is_right_in_img else -1.0), 1.2
        proj = (xs_local + x_offset - sx) * dx + (ys_local + y_lo - sy) * dy
        tip_band = proj >= np.quantile(proj, .90)
        wx = float(np.median(xs_local[tip_band]) + x_offset)
        wy = float(np.median(ys_local[tip_band]) + y_lo)
        points[wrist_id] = [wx, wy]
    # Estimate missing ankles near bottom of silhouette on each side of cx.
    for side, hip_id, ankle_id in (("left", 23, 27), ("right", 24, 28)):
        if ankle_id in points:
            continue
        y_lo = int(max(0, crown + .82 * height))
        y_hi = int(min(h, floor + 1))
        if y_hi <= y_lo:
            continue
        is_right = (points[hip_id][0] >= cx) if hip_id in points else (side == "left")
        xs_mask = (np.arange(w) >= cx) if is_right else (np.arange(w) < cx)
        leg_fg = foreground[y_lo:y_hi, xs_mask]
        if leg_fg.any():
            ys_l, xs_l = np.where(leg_fg)
            x_off = int(cx) if is_right else 0
            points[ankle_id] = [float(np.median(xs_l) + x_off), float(np.quantile(ys_l, .60) + y_lo)]
    return points


def _estimate_pre_pose_and_proportions(foreground, pose_points, semantic_mask, calibration, template_mesh):
    """Estimate anime proportions and XY limb rotations from calibrated front evidence."""
    from aethryx_rig import humanoid_skeleton

    (cx, floor), crown = calibration
    img_h = max(8.0, floor - crown)
    h, w = foreground.shape
    proportions = {}

    # Estimate head scale from semantic face/hair or upper silhouette width/neck pinch.
    head_ratio = None
    if semantic_mask is not None:
        labels = np.asarray(semantic_mask.convert("RGB"))
        face_px = np.all(labels == SEE_THROUGH_COLORS["face"], axis=2) & foreground
        if face_px.sum() >= 48:
            fy, _ = np.where(face_px)
            chin_y = float(np.quantile(fy, .98))
            head_ratio = (chin_y - crown) / img_h
    if head_ratio is None and 0 in pose_points and 11 in pose_points and 12 in pose_points:
        shoulder_y = 0.5 * (pose_points[11][1] + pose_points[12][1])
        head_ratio = ((shoulder_y - crown) / img_h) * 0.85
    if head_ratio is not None and 0.10 <= head_ratio <= 0.35:
        proportions["head"] = float(np.clip(head_ratio / 0.155, 0.85, 1.32))

    # Estimate shoulder/hip/waist/leg proportions when shoulders & hips are detected.
    tmpl_h = float(template_mesh.extents[1])
    tmpl_w = float(template_mesh.extents[0])
    if 11 in pose_points and 12 in pose_points and 23 in pose_points and 24 in pose_points:
        sh_w = abs(pose_points[11][0] - pose_points[12][0]) / img_h
        hip_w = abs(pose_points[23][0] - pose_points[24][0]) / img_h
        sh_y = 0.5 * (pose_points[11][1] + pose_points[12][1])
        hip_y = 0.5 * (pose_points[23][1] + pose_points[24][1])
        # Bundled shoulder span ~0.239/height, hip span ~0.165/height.
        proportions["shoulders"] = float(np.clip(sh_w / 0.239, 0.82, 1.25))
        proportions["hips"] = float(np.clip(hip_w / 0.165, 0.82, 1.28))
        leg_frac = (floor - hip_y) / img_h
        if 0.35 <= leg_frac <= 0.65:
            proportions["legs"] = float(np.clip(leg_frac / 0.50, 0.85, 1.25))
        torso_frac = (hip_y - sh_y) / img_h
        if 0.18 <= torso_frac <= 0.42:
            proportions["torso"] = float(np.clip(torso_frac / 0.29, 0.82, 1.22))

    # Compute frontal XY arm & leg angles relative to template skeleton for clean pre-pose alignment.
    pre_pose = {}
    names, _, pos = humanoid_skeleton(template_mesh)
    for side, sh_id, el_id, wr_id in (("Left", 11, 13, 15), ("Right", 12, 14, 16)):
        if sh_id in pose_points and (el_id in pose_points or wr_id in pose_points):
            target_id = el_id if el_id in pose_points else wr_id
            sx, sy = pose_points[sh_id]
            tx, ty = pose_points[target_id]
            # Image Y points down; 3D Y points up.
            ref_vec = np.array([tx - sx, -(ty - sy)], dtype=float)
            p_sh = pos[names.index(side + "UpperArm")][:2]
            p_tg = pos[names.index(side + ("LowerArm" if el_id in pose_points else "Hand"))][:2]
            tmpl_vec = p_tg - p_sh
            if np.linalg.norm(ref_vec) > 4.0 and np.linalg.norm(tmpl_vec) > 1e-5:
                ang_ref = np.degrees(np.arctan2(ref_vec[1], ref_vec[0]))
                ang_tmpl = np.degrees(np.arctan2(tmpl_vec[1], tmpl_vec[0]))
                delta = float(((ang_ref - ang_tmpl + 180) % 360) - 180)
                delta = float(np.clip(delta, -55.0, 55.0))
                if abs(delta) > 3.0:
                    pre_pose[side + "UpperArm"] = [0.0, 0.0, delta]
                    if el_id in pose_points and wr_id in pose_points:
                        ex, ey = pose_points[el_id]
                        wx, wy = pose_points[wr_id]
                        fore_ref = np.array([wx - ex, -(wy - ey)], dtype=float)
                        p_wr = pos[names.index(side + "Hand")][:2]
                        fore_tmpl = p_wr - pos[names.index(side + "LowerArm")][:2]
                        if np.linalg.norm(fore_ref) > 4.0 and np.linalg.norm(fore_tmpl) > 1e-5:
                            f_delta = float(((np.degrees(np.arctan2(fore_ref[1], fore_ref[0]))
                                              - np.degrees(np.arctan2(fore_tmpl[1], fore_tmpl[0])) - delta + 180) % 360) - 180)
                            f_delta = float(np.clip(f_delta, -50.0, 50.0))
                            if abs(f_delta) > 3.0:
                                pre_pose[side + "LowerArm"] = [0.0, 0.0, f_delta]
    return proportions, pre_pose


def build_auto_front_setup(image_path, *, work_dir=None, use_see_through=False, use_sculpted_body=True,
                           profile="Anime neutro", root=None):
    """Build an automatic, validated single-image Genesis setup with pose, proportions, and palette."""
    from tempfile import mkdtemp
    from threading import Event
    from aethryx_anatomy import orthographic_projection
    from aethryx_anime import sculpted_body
    from aethryx_engines import suggest_anatomical_points, suggest_facial_points
    from aethryx_profiles import BUNDLED_BODY, resolve_pose_vertices

    src = Path(image_path).expanduser().resolve(strict=True)
    work = Path(work_dir) if work_dir is not None else Path(mkdtemp(prefix="aethryx-auto-"))
    work.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as loaded:
        rgba = loaded.convert("RGBA")

    # Obtain foreground silhouette via alpha or ISNet.
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] < 32:
        fg_mask = Image.fromarray((np.asarray(alpha) >= 128).astype(np.uint8) * 255)
    else:
        fg_mask = AnimeSilhouette().predict(rgba.convert("RGB"))
        rgba_arr = np.asarray(rgba).copy()
        rgba_arr[:, :, 3] = np.asarray(fg_mask)
        rgba = Image.fromarray(rgba_arr, "RGBA")

    # If the subject touches the image border, pad the working copy slightly so calibration stays valid.
    fg_arr = np.asarray(fg_mask) >= 128
    if fg_arr[[0, -1]].any() or fg_arr[:, [0, -1]].any():
        pad = max(12, int(round(0.06 * max(rgba.size))))
        padded_rgba = Image.new("RGBA", (rgba.width + 2 * pad, rgba.height + 2 * pad), (0, 0, 0, 0))
        padded_rgba.paste(rgba, (pad, pad))
        rgba = padded_rgba
        fg_mask = rgba.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
        fg_arr = np.asarray(fg_mask) >= 128

    working_img = work / "front-auto.png"
    rgba.save(working_img, format="PNG")
    body_mask_path = work / "front-body-mask.png"
    fg_mask.convert("RGB").save(body_mask_path, format="PNG")

    semantic_mask = None
    semantic_mask_path = None
    if use_see_through:
        st_paths = default_see_through_paths(root)
        if st_paths is not None:
            try:
                raw_mask = segment_see_through_v2(str(working_img), rgba.size, *st_paths, cancelled=Event())
                if raw_mask is not None:
                    semantic_mask = refine_visible_mask(rgba, raw_mask, fg_mask)
                    semantic_mask_path = work / "front-semantic-mask.png"
                    semantic_mask.save(semantic_mask_path, format="PNG")
            except Exception:
                semantic_mask = None

    try:
        raw_pose = suggest_anatomical_points(working_img, return_confidence=True)
    except (ImportError, OSError, RuntimeError, ValueError):
        raw_pose = {}
    pose_points = {}
    pose_weights = {}
    # Filter landmarks against a mildly dilated foreground silhouette to discard background hallucinations.
    dilated = cv2.dilate(fg_arr.astype(np.uint8), np.ones((max(5, rgba.height // 35), max(5, rgba.height // 35)), np.uint8)) > 0
    for joint, entry in raw_pose.items():
        px, py = entry["xy"]
        ix, iy = int(np.clip(round(px), 0, rgba.width - 1)), int(np.clip(round(py), 0, rgba.height - 1))
        if dilated[iy, ix]:
            pose_points[joint] = [px, py]
            pose_weights[joint] = float(np.clip(entry["visibility"], 0.45, 1.0))

    calibration = silhouette_calibration(fg_arr, pose_points)
    if calibration is None:
        raise ValueError("No se pudo calibrar la silueta frontal de la imagen.")
    pose_points = estimate_silhouette_limbs(fg_arr, pose_points, calibration)
    for j in pose_points:
        pose_weights.setdefault(j, 0.45)

    # Detect facial anchor refinements if face_landmarker is available.
    face_bbox = None
    if semantic_mask is not None:
        face_sel = np.all(np.asarray(semantic_mask) == SEE_THROUGH_COLORS["face"], axis=2) & fg_arr
        if face_sel.sum() >= 24:
            fy, fx = np.where(face_sel)
            face_bbox = (int(fx.min()), int(fy.min()), int(fx.max()) + 1, int(fy.max()) + 1)
    face_points = suggest_facial_points(working_img, face_bbox=face_bbox)
    if "nose_tip" in face_points and 0 not in pose_points:
        pose_points[0] = face_points["nose_tip"]
        pose_weights[0] = 0.75

    template_mesh = sculpted_body(profile, refined=True, facial_loops=True) if use_sculpted_body else None
    if template_mesh is None:
        from aethryx_anatomy import load_body_template
        template_mesh = load_body_template(BUNDLED_BODY)

    proportions, pre_pose = _estimate_pre_pose_and_proportions(fg_arr, pose_points, semantic_mask, calibration, template_mesh)
    pose_map = resolve_pose_vertices(template_mesh)
    center, crown = calibration
    projection = orthographic_projection("front", center, center[1], crown, template_mesh.bounds).tolist()

    # Build vertex_indices + landmarks + contour anchors (crown, temples, ankles).
    used_joints = []
    used_vertices = set()
    for joint, vertex in pose_map.items():
        if joint in pose_points and vertex not in used_vertices:
            used_joints.append(joint)
            used_vertices.add(vertex)
    vertex_indices = [pose_map[j] for j in used_joints]
    landmarks = [pose_points[j] for j in used_joints]
    weights = [pose_weights[j] for j in used_joints]

    # Add crown & temple contour anchors so head silhouette matches the reference width/top.
    crown_vid = int(np.argmax(template_mesh.vertices[:, 1]))
    if crown_vid not in vertex_indices:
        vertex_indices.append(crown_vid)
        landmarks.append([float(center[0]), float(crown)])
        weights.append(0.9)
    eye_y_world = template_mesh.bounds[0, 1] + 0.90 * template_mesh.extents[1]
    head_band = np.abs(template_mesh.vertices[:, 1] - eye_y_world) < 0.03 * template_mesh.extents[1]
    if head_band.any():
        band_ids = np.flatnonzero(head_band)
        left_temple = int(band_ids[np.argmax(template_mesh.vertices[band_ids, 0])])
        right_temple = int(band_ids[np.argmin(template_mesh.vertices[band_ids, 0])])
        if face_bbox is not None:
            fx0, fy0, fx1, fy1 = face_bbox
            temple_y = float(0.5 * (fy0 + fy1))
            left_x, right_x = float(fx1), float(fx0)
        else:
            temple_y = float(crown + 0.08 * (center[1] - crown))
            row_xs = np.flatnonzero(fg_arr[int(np.clip(round(temple_y), 0, rgba.height - 1))])
            if len(row_xs) >= 4:
                right_x, left_x = float(np.quantile(row_xs, .12)), float(np.quantile(row_xs, .88))
            else:
                right_x = left_x = None
        if left_x is not None and left_temple not in vertex_indices and right_temple not in vertex_indices:
            vertex_indices.extend((left_temple, right_temple))
            landmarks.extend(([left_x, temple_y], [right_x, temple_y]))
            weights.extend((0.65, 0.65))

    if len(vertex_indices) < 4:
        raise ValueError("No se encontraron suficientes correspondencias anatómicas automáticas en la imagen.")

    appearance = extract_reference_palette_and_style(rgba, semantic_mask, fg_mask)
    palette = {name: list(color) for name, color in SEE_THROUGH_COLORS.items() if name != "background"} if semantic_mask_path else {}

    front_view = {
        "image": str(working_img),
        "body_mask": str(body_mask_path),
        "calibration": [center, crown],
        "projection": projection,
        "landmarks": landmarks,
        "landmark_weights": weights,
        "suggested": list(vertex_indices),
        "hidden": [],
    }
    if semantic_mask_path is not None:
        front_view["mask"] = str(semantic_mask_path)

    return {
        "template": str(BUNDLED_BODY),
        "bundled_template": not use_sculpted_body,
        "use_sculpted_body": bool(use_sculpted_body),
        "profile": profile,
        "vertex_indices": vertex_indices,
        "views": {"front": front_view},
        "palette": palette,
        "pose_links": {str(pose_map[j]): j for j in used_joints},
        "allow_inferred_depth": True,
        "export_rig": True,
        "export_hybrid": True,
        "repair_hair_in_poses": True,
        "smoothing": 0.005,
        "proportions": {},
        "pre_proportions": proportions,
        "pre_pose": pre_pose,
        "appearance": appearance,
        "auto_generated": True,
    }
