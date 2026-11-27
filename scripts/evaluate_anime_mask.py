"""Score an exact RGB semantic mask on manually annotated rectangles."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from aethryx_reference import SEE_THROUGH_COLORS


def evaluate(prediction, regions):
    labels = list(SEE_THROUGH_COLORS)
    prediction = np.asarray(prediction.convert("RGB"))
    expected = np.full(prediction.shape[:2], -1, dtype=np.int8)
    for name, boxes in regions.items():
        for x0, y0, x1, y1 in boxes:
            if not 0 <= x0 < x1 <= expected.shape[1] or not 0 <= y0 < y1 <= expected.shape[0]:
                raise ValueError(f"Anotación fuera de la imagen {name}.")
            sample = expected[y0:y1, x0:x1]
            if np.any(sample != -1):
                raise ValueError("Las anotaciones se superponen.")
            sample[:] = labels.index(name)
    known = expected >= 0
    if not np.any(known):
        raise ValueError("Se necesita al menos una región anotada.")
    target = expected[known]
    samples = prediction[known]
    predicted = np.full(len(samples), -1, dtype=np.int8)
    for index, color in enumerate(SEE_THROUGH_COLORS.values()):
        predicted[np.all(samples == color, axis=1)] = index
    classes = {}
    for index, name in enumerate(labels):
        actual = target == index
        total = (actual | (predicted == index)).sum()
        if total:
            classes[name] = {
                "iou": float((actual & (predicted == index)).sum() / total),
                "annotated_pixels": int(actual.sum()),
            }
    return {"accuracy": float(np.mean(target == predicted)), "annotated_pixels": len(target), "classes": classes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask", type=Path)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    with Image.open(args.mask) as mask, Image.open(args.annotations.parent / annotation["image"]) as source:
        if mask.size != source.size:
            parser.error("La máscara y la referencia anotada deben tener el mismo tamaño.")
        print(json.dumps(evaluate(mask, annotation["regions"]), indent=2))
