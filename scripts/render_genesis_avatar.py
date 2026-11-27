"""Render a four-view review of an AETHRYX modular avatar GLB."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyvista as pv
import trimesh


def render_review(source, output, *, toon=False):
    scene = trimesh.load(source, force="scene", process=False)
    center = scene.bounds.mean(axis=0)
    height = scene.extents[1]
    closeup = center + [0, .377 * height, 0]
    cameras = (
        ("Frontal", center, [0, 0, 1], .57),
        ("Lateral", center, [1, 0, 0], .57),
        ("Posterior", center, [0, 0, -1], .57),
        ("Rostro", closeup, [0, 0, 1], .112),
    )
    size = (430, 540)
    board = Image.new("RGB", (2 * size[0], 2 * size[1]), "#121621")
    for index, (label, target, direction, scale) in enumerate(cameras):
        plotter = pv.Plotter(off_screen=True, window_size=size)
        try:
            plotter.set_background("#121621")
            for node in scene.graph.nodes_geometry:
                transform, geometry = scene.graph[node]
                mesh = scene.geometry[geometry].copy()
                mesh.apply_transform(transform)
                poly = pv.PolyData(mesh.vertices, np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)))
                material = getattr(mesh.visual, "material", None)
                texture = getattr(material, "baseColorTexture", None)
                if texture is not None and getattr(mesh.visual, "uv", None) is not None:
                    poly.active_texture_coordinates = mesh.visual.uv
                    plotter.add_mesh(poly, texture=pv.numpy_to_texture(np.asarray(texture.convert("RGB"))),
                                     smooth_shading=True)
                else:
                    factor = getattr(material, "baseColorFactor", None)
                    color = tuple(factor[:3] / 255) if factor is not None else "#c6d4df"
                    if toon and factor is not None:
                        light = np.array([-.32, .64, .70])
                        power = mesh.vertex_normals @ (light / np.linalg.norm(light))
                        bands = np.select((power > .55, power > .13), (1., .83), default=.64)
                        poly.point_data["toon_rgb"] = np.rint(bands[:, None] * factor[:3]).astype(np.uint8)
                        plotter.add_mesh(poly, scalars="toon_rgb", rgb=True, lighting=False)
                    else:
                        plotter.add_mesh(poly, color=color, smooth_shading=True)
            plotter.camera_position = (target + np.asarray(direction) * height * 3, target, (0, 1, 0))
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = scale * height
            pixels = plotter.screenshot(return_img=True)
        finally:
            plotter.close()
        offset = (index % 2 * size[0], index // 2 * size[1])
        board.paste(Image.fromarray(pixels[:, :, :3]), offset)
        ImageDraw.Draw(board).text((offset[0] + 14, offset[1] + 12), label, fill="#00D9FF")
    path = Path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("La revisión se guarda en PNG.")
    with path.open("xb") as stream:
        board.save(stream, format="PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("output")
    parser.add_argument("--toon", action="store_true")
    args = parser.parse_args()
    render_review(args.scene, args.output, toon=args.toon)
