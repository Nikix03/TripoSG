"""Reproduce the Origin review using the CC BY 4.0 GDQuest mannequin GLB.

Run as a module from the repository with the source GLB and a new output folder.
The output folder contains the reference renders, fit projections and optional rig.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyvista as pv

from aethryx_anatomy import (blocking_geometry_issues, export_studio_reconstruction,
                            load_body_template, orthographic_projection, reconstruct_data)
from aethryx_profiles import BUNDLED_BODY, POSE_VERTICES
from aethryx_reference import compare_silhouette, reference_silhouette, silhouette_calibration


def render_references(source, folder):
    shape = load_body_template(source)
    shape.apply_translation(-shape.bounds.mean(axis=0))
    shape.apply_scale(1 / shape.extents[1])
    views = {"front": [0, 0, 2], "side": [2, 0, 0], "back": [0, 0, -2]}
    for name, direction in views.items():
        plot = pv.Plotter(off_screen=True, window_size=(480, 480))
        try:
            plot.set_background("white")
            plot.add_mesh(pv.wrap(shape), color="#3b73a8", smooth_shading=True)
            plot.camera_position = (direction, (0, 0, 0), (0, 1, 0))
            plot.enable_parallel_projection()
            plot.camera.parallel_scale = .59
            pixels = plot.screenshot(transparent_background=True, return_img=True)
            Image.fromarray(pixels).save(folder / f"{name}.png")
        finally:
            plot.close()


def evaluate(folder):
    template = load_body_template(BUNDLED_BODY)
    # Manual landmark annotations in the 400px review frame. Occluded-side points are omitted.
    points = {
        "front": [(223, 99), (177, 99), (213, 187), (187, 187), (224, 268), (176, 268), (233, 345), (167, 345)],
        "side": [(207, 99), None, (206, 187), None, (211, 268), None, (215, 345), None],
        "back": [(177, 99), (223, 99), (187, 187), (213, 187), (176, 268), (224, 268), (167, 345), (233, 345)],
    }
    data = {
        "bundled_template": True,
        "vertex_indices": [POSE_VERTICES[key] for key in (11, 12, 23, 24, 25, 26, 27, 28)],
        "allow_inferred_depth": True,
        "export_rig": True,
        "smoothing": .024,
        "views": {},
    }
    images = {}
    for name, landmarks in points.items():
        path = folder / f"{name}.png"
        with Image.open(path) as source:
            image = source.convert("RGBA")
        images[name] = image
        center, crown = silhouette_calibration(reference_silhouette(image), {})
        data["views"][name] = {
            "image": str(path),
            "projection": orthographic_projection(name, center, center[1], crown, template.bounds).tolist(),
            "landmarks": [[point[0] * 1.2, point[1] * 1.2] if point else [None, None] for point in landmarks],
        }
    report = {}
    fitted, patches, errors = reconstruct_data(data, deformation_report=report)
    board = Image.new("RGB", (1200, 800), "#dddfe4")
    draw = ImageDraw.Draw(board)
    comparisons, ious = {}, {}
    for index, (name, image) in enumerate(images.items()):
        comparisons[name], ious[name] = compare_silhouette(fitted, image, data["views"][name]["projection"])
        preview = image.resize((400, 400))
        board.paste(preview, (index * 400, 0), preview)
        board.paste(comparisons[name].resize((400, 400)), (index * 400, 400))
        draw.text((index * 400 + 12, 12), name, fill="#25294a")
    board.save(folder / "origin-mannequiny.png")
    issues = blocking_geometry_issues(report)
    result = {"iou": ious, "rmse_px": errors, "geometry_issues": issues, "evidence": report["reference_evidence"]}
    if not issues:
        body, _ = export_studio_reconstruction(folder, data, fitted, patches, report, errors, comparisons)
        result["output"] = str(body.parent)
    (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    render_references(args.model, output)
    evaluate(output)

