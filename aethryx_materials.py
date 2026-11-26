"""Per-component glTF material maps and UV validation."""

from pathlib import Path

import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


CHANNELS = {
    "base": "Color base", "normal": "Normal", "roughness": "Rugosidad",
    "metallic": "Metalizado", "occlusion": "Oclusión", "emissive": "Emisión",
}
RESOLUTIONS = (512, 1024, 2048, 4096)


def color_gradient(top, bottom, resolution=512):
    first = np.asarray(Image.new("RGB", (1, 1), top))[0, 0]
    last = np.asarray(Image.new("RGB", (1, 1), bottom))[0, 0]
    blend = np.linspace(1, 0, resolution)[:, None, None]
    return Image.fromarray(np.repeat(np.rint(blend * first + (1 - blend) * last), 4, axis=1).astype(np.uint8))


def update_pbr_maps(mesh, layers, resolution=1024, alpha_mode="OPAQUE", cutoff=.5):
    """Replace supplied channels while preserving unrelated channels and mesh UVs."""
    if resolution not in RESOLUTIONS or alpha_mode not in ("OPAQUE", "MASK", "BLEND") or not 0 <= cutoff <= 1:
        raise ValueError("Resolución, modo alfa o corte inválidos.")
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or np.shape(uv) != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
        raise ValueError("Genera coordenadas UV válidas antes de incorporar mapas.")
    current = getattr(mesh.visual, "material", None)
    material = current.copy() if isinstance(current, PBRMaterial) else PBRMaterial()
    side = (resolution, resolution)
    fields = {"base": ("baseColorTexture", "RGBA"), "normal": ("normalTexture", "RGB"),
              "occlusion": ("occlusionTexture", "L"), "emissive": ("emissiveTexture", "RGB")}
    for channel, (field, mode) in fields.items():
        if channel in layers:
            setattr(material, field, layers[channel].convert(mode).resize(side, Image.Resampling.BILINEAR))
    if "base" in layers:
        material.baseColorFactor = [255, 255, 255, 255]
    if "emissive" in layers:
        material.emissiveFactor = [1., 1., 1.]
    if "roughness" in layers or "metallic" in layers:
        existing = material.metallicRoughnessTexture
        if existing is not None:
            channels = list(existing.convert("RGB").resize(side, Image.Resampling.BILINEAR).split())
        else:
            channels = [Image.new("L", side, 255) for _ in range(3)]
        for index, key, field in ((1, "roughness", "roughnessFactor"), (2, "metallic", "metallicFactor")):
            if key in layers:
                channels[index] = layers[key].convert("L").resize(side, Image.Resampling.BILINEAR)
                setattr(material, field, 1.)
        material.metallicRoughnessTexture = Image.merge("RGB", channels)
    material.alphaMode = alpha_mode
    material.alphaCutoff = cutoff if alpha_mode == "MASK" else None
    mesh.visual = TextureVisuals(uv=np.asarray(uv).copy(), material=material)
    return mesh


def pbr_map_images(mesh):
    material = getattr(mesh.visual, "material", None)
    fields = {"base": "baseColorTexture", "normal": "normalTexture",
              "occlusion": "occlusionTexture", "emissive": "emissiveTexture"}
    result = {channel: texture.copy() for channel, field in fields.items()
              if (texture := getattr(material, field, None)) is not None}
    combined = getattr(material, "metallicRoughnessTexture", None)
    if combined is not None:
        _, result["roughness"], result["metallic"] = combined.convert("RGB").split()
    return result


def transform_uv(mesh, scale, offset):
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise ValueError("La pieza no tiene UV.")
    if not np.isfinite([*scale, *offset]).all() or np.any(np.asarray(scale) <= 0):
        raise ValueError("La transformación UV requiere escalas positivas.")
    mesh.visual.uv = uv * scale + offset
    return mesh


def uv_quality(mesh, samples=256):
    """Sample overlaps at pixel centers within the main UV tile."""
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or not np.isfinite(uv).all():
        raise ValueError("La pieza no tiene UV finitas.")
    triangles = np.asarray(uv)[mesh.faces]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    determinants = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    valid = (np.abs(determinants) > 1e-12) & (mesh.area_faces > 1e-12)
    coverage = np.zeros((samples, samples), dtype=np.uint16)
    for tri, first, second, determinant in zip(triangles[valid], e1[valid], e2[valid], determinants[valid]):
        lo = np.clip(np.floor(tri.min(axis=0) * samples).astype(int), 0, samples)
        hi = np.clip(np.ceil(tri.max(axis=0) * samples).astype(int), 0, samples)
        if np.any(hi <= lo):
            continue
        x = (np.arange(lo[0], hi[0])[None, :] + .5) / samples - tri[0, 0]
        y = (np.arange(lo[1], hi[1])[:, None] + .5) / samples - tri[0, 1]
        a = (x * second[1] - y * second[0]) / determinant
        b = (first[0] * y - first[1] * x) / determinant
        coverage[lo[1]:hi[1], lo[0]:hi[0]] += (a > 1e-9) & (b > 1e-9) & (a + b < 1 - 1e-9)
    ratios = np.abs(determinants[valid]) / mesh.area_faces[valid]
    distortion = np.abs(np.log2(ratios / np.median(ratios))) if len(ratios) else []
    covered = int(np.count_nonzero(coverage))
    return {"triangles": len(triangles), "degenerate": int((~valid).sum()),
            "outside_vertices": int(np.any((uv < 0) | (uv > 1), axis=1).sum()),
            "overlap_fraction": np.count_nonzero(coverage > 1) / covered if covered else 0.,
            "area_distortion_p95_stops": float(np.quantile(distortion, .95)) if len(ratios) else None}


def save_pbr_maps(mesh, folder):
    images = pbr_map_images(mesh)
    if not images:
        raise ValueError("El material no tiene mapas PNG para exportar.")
    folder = Path(folder)
    if any((folder / f"{key}.png").exists() for key in images):
        raise FileExistsError("Elige una carpeta sin archivos con esos nombres.")
    for key, image in images.items():
        with (folder / f"{key}.png").open("xb") as output:
            image.save(output, format="PNG")
    return len(images)
