"""Run the local TripoSR checkpoint using CPU marching cubes and CUDA inference."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
from PIL import Image


def extract_cpu_surface(level, iso):
    import torch
    from skimage.measure import marching_cubes

    vertices, faces, _, _ = marching_cubes(
        level.cpu().numpy(), level=iso, gradient_direction="ascent", allow_degenerate=False,
    )
    # TripoSR converts the torchmcubes convention back to the field's XYZ order.
    return torch.from_numpy(vertices[:, ::-1].copy()), torch.from_numpy(faces.copy()).long()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "config", "dino-config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=128, choices=(96, 128, 192, 256))
    args = parser.parse_args()
    if not args.probe and (not args.image or not args.output):
        parser.error("--image and --output are required")
    if not args.probe and (args.output.suffix.lower() != ".glb" or args.output.exists()):
        parser.error("--output must be a new GLB path")
    if not (args.source / "tsr" / "system.py").is_file():
        parser.error("--source must point to the TripoSR repository")

    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        parser.error("CUDA is required for this runner")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aethryx_adaptive import build_adaptive_plan

    plan = build_adaptive_plan(engine_id="triposr")
    if not plan.profile.generation_enabled:
        parser.error(plan.reason)
    sys.path.insert(0, str(args.source.resolve()))
    # Adapt the extractor only inside this worker process.
    extractor = ModuleType("torchmcubes")
    extractor.marching_cubes = extract_cpu_surface
    sys.modules["torchmcubes"] = extractor
    from tsr.models.tokenizers import image as tokenizer
    from tsr.system import TSR
    from tsr.utils import resize_foreground

    def local_dino_config(*, repo_id, filename):
        if (repo_id, filename) != ("facebook/dino-vitb16", "config.json"):
            raise ValueError(f"Unsupported tokenizer dependency {repo_id}/{filename}")
        return str(args.dino_config.resolve(strict=True))

    tokenizer.hf_hub_download = local_dino_config
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    with args.dino_config.open(encoding="utf-8") as stream:
        json.load(stream)
    if not args.checkpoint.is_file() or not args.checkpoint.stat().st_size:
        parser.error("The local checkpoint is missing or empty")
    if args.probe:
        from skimage.measure import marching_cubes

        torch.ones(1, device="cuda").sum().item()
        print(torch.cuda.get_device_name())
        return
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = TSR(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.renderer.set_chunk_size(8192)
    model.eval().to("cuda")
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    with Image.open(args.image) as source:
        rgba = source.convert("RGBA")
    if rgba.getchannel("A").getextrema() != (0, 255):
        parser.error("--image must contain an opaque subject against transparent background")
    rgba = resize_foreground(rgba, .85)
    image = Image.alpha_composite(Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba).convert("RGB")
    with torch.inference_mode():
        code = model([image], device="cuda")
        torch.cuda.synchronize()
        predicted = time.perf_counter()
        mesh = model.extract_mesh(code, True, resolution=args.resolution)[0]
    # TripoSR uses Z up and a front camera on +X. Studio uses Y up and front +Z.
    mesh.apply_transform(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
    with args.output.open("xb") as destination:
        destination.write(mesh.export(file_type="glb"))
    print(json.dumps({
        "device": torch.cuda.get_device_name(), "resolution": args.resolution, "seed": args.seed,
        "profile_vram_gib": plan.profile.target_vram_gib, "precision": plan.profile.dtype,
        "foreground_ratio": .85, "chunk_size": 8192, "load_seconds": loaded - start,
        "predict_seconds": predicted - loaded, "extract_and_write_seconds": time.perf_counter() - predicted,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "vertices": len(mesh.vertices), "faces": len(mesh.faces), "is_volume": bool(mesh.is_volume),
        "components": len(mesh.split(only_watertight=False, repair=False)),
    }))


if __name__ == "__main__":
    main()
