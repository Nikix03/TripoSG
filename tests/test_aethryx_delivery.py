import json
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from zipfile import ZipFile

from PIL import Image
import trimesh

from aethryx_anatomy import load_body_template
from aethryx_bridge import inspect_avatar_glb
from aethryx_delivery import DELIVERY_STAGES, DeliveryCancelled, build_reviewed_delivery
from aethryx_profiles import BUNDLED_BODY
from studio_project import load_project, save_project


class PortableDeliveryTests(unittest.TestCase):
    def test_delivery_can_move_and_retains_hidden_components(self):
        with TemporaryDirectory() as destination:
            target = Path(destination)
            with TemporaryDirectory() as source:
                root = Path(source)
                image, model = root / "reference.png", root / "editable.glb"
                Image.new("RGB", (16, 16)).save(image)
                body = load_body_template(BUNDLED_BODY)
                trimesh.Scene({"cuerpo-base": body, "borrador": trimesh.creation.box()}).export(model)
                project = root / "source.aeth"
                save_project(project, image=image, model=model, output=root / "output.glb",
                             steps=30, faces=-1, seed=42, guidance=7.,
                             hidden_components=("borrador",),
                             references=[{"id": "front", "path": str(image), "view": "front", "bbox": [0, 0, 1, 1]}],
                             anatomy_setup={"bundled_template": True, "views": {"front": {"image": str(image)}}})
                progress = []
                output = build_reviewed_delivery(project, root, on_stage=lambda i, label: progress.append((i, label)))
                self.assertEqual(progress, list(enumerate(DELIVERY_STAGES)))
                copytree(output, target / "avatar")
            relocated = target / "avatar"
            restored = load_project(relocated / "Avatar.aeth")
            self.assertEqual(restored["hidden_components"], ["borrador"])
            self.assertTrue(Path(restored["image"]).resolve().is_relative_to(relocated.resolve()))
            self.assertTrue(Path(restored["anatomy_setup"]["views"]["front"]["image"]).is_file())
            self.assertEqual(restored["references"][0]["path"], restored["image"])
            editable = trimesh.load(restored["model"], force="scene", process=False)
            self.assertIn("borrador", editable.graph.nodes_geometry)
            for name in ("avatar-articulado.glb", "avatar-pc.glb", "avatar-cadenas-capilares.glb"):
                audit = inspect_avatar_glb(relocated / name)
                self.assertTrue(audit["valid_native_glb"])
                self.assertEqual(audit["skinned_nodes"], 1)
            with ZipFile(relocated / "Unity-AETHRYX.zip") as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read("manifest.json"))
                self.assertIn(manifest["avatar"], archive.namelist())
                self.assertIn(manifest["hair_chain_variant"], archive.namelist())
                self.assertIn("unity_reference", manifest["dependencies"])
                self.assertFalse(any(name.endswith(".png") for name in archive.namelist()))

    def test_delivery_cancelled_before_start_leaves_output_empty(self):
        with TemporaryDirectory() as directory:
            cancelled = Event()
            cancelled.set()
            with self.assertRaises(DeliveryCancelled):
                build_reviewed_delivery("unused.aeth", directory, cancelled=cancelled)
            self.assertEqual(list(Path(directory).iterdir()), [])
