"""Run the upstream See-through V2 parser in a separate local Python process."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="See-through checkout")
    parser.add_argument("checkpoint", type=Path, help="Official V2 checkpoint-18000.pt")
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    common = args.source.resolve() / "common"
    if not (common / "modules" / "semanticsam.py").is_file() or not args.checkpoint.is_file():
        parser.error("Se necesitan el código fuente de See-through y el checkpoint V2 local.")
    if args.warmups < 0 or args.iterations < 1:
        parser.error("Parámetros de medición fuera de rango.")
    sys.path.insert(0, str(common))
    import torch
    from modules.semanticsam import SemanticSam
    from utils.torch_utils import init_model_from_pretrained

    if not torch.cuda.is_available():
        parser.error("See-through V2 requiere CUDA en esta integración.")
    with Image.open(args.image) as source:
        rgb = np.asarray(source.convert("RGB"))
    start = perf_counter()
    model = init_model_from_pretrained(
        str(args.checkpoint.resolve()), SemanticSam, model_args={"class_num": 19}, download_from_hf=False,
    ).eval().to("cuda")
    torch.cuda.synchronize()
    load_seconds = perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    timings = []
    with torch.inference_mode():
        for index in range(args.warmups + args.iterations):
            start = perf_counter()
            scores = model.inference(rgb)[0]
            torch.cuda.synchronize()
            if index >= args.warmups:
                timings.append(perf_counter() - start)
            if index == args.warmups + args.iterations - 1:
                masks = (scores > 0).cpu().numpy()
            del scores
    if masks.shape != (19, *rgb.shape[:2]):
        raise ValueError(f"Salida V2 inesperada {masks.shape}.")
    with args.output.open("xb") as stream:
        np.save(stream, masks, allow_pickle=False)
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "model_load_seconds": load_seconds,
        "warmups": args.warmups,
        "inference_seconds": timings,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "class_pixels": masks.sum(axis=(1, 2)).tolist(),
    }))


if __name__ == "__main__":
    main()
