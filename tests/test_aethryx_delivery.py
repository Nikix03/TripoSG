import json
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image, ImageDraw
import trimesh

from aethryx_anatomy import load_body_template
from aethryx_bridge import inspect_avatar_glb
from aethryx_delivery import DELIVERY_STAGES, DeliveryCancelled, build_reviewed_delivery, verify_delivery
from aethryx_jobs import ComponentJobs
from aethryx_profiles import BUNDLED_BODY
from aethryx_reference import analyze_view_alignment, build_auto_front_setup
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
                cancelled = Event()

                def cancel_midway(index, _label):
                    if index == 2:
                        cancelled.set()

                with self.assertRaises(DeliveryCancelled):
                    build_reviewed_delivery(project, root, on_stage=cancel_midway, cancelled=cancelled)
                cancelled.clear()

                def cancel_after_verification(folder):
                    verify_delivery(folder)
                    cancelled.set()

                with patch("aethryx_delivery.verify_delivery", side_effect=cancel_after_verification):
                    with self.assertRaises(DeliveryCancelled):
                        build_reviewed_delivery(project, root, cancelled=cancelled)

                output = build_reviewed_delivery(project, root / "new-delivery",
                                                 on_stage=lambda i, label: progress.append((i, label)))
                self.assertEqual(output.parent.name, "new-delivery")
                inputs = verify_delivery(output)["inputs"]
                image.write_bytes(b"edited reference")
                with self.assertRaisesRegex(ValueError, "Las entradas cambiaron"):
                    build_reviewed_delivery(project, root, expected_inputs=inputs)
                self.assertEqual(progress, list(enumerate(DELIVERY_STAGES)))
                copytree(output, target / "avatar")
            relocated = target / "avatar"
            verify_delivery(relocated)
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

            (relocated / "procedencia.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "archivos faltantes o modificados"):
                verify_delivery(relocated)

    def test_delivery_cancelled_before_start_leaves_output_empty(self):
        with TemporaryDirectory() as directory:
            cancelled = Event()
            cancelled.set()
            with self.assertRaises(DeliveryCancelled):
                build_reviewed_delivery("unused.aeth", directory, cancelled=cancelled)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_retry_retains_first_input_hashes(self):
        with TemporaryDirectory() as directory:
            jobs = ComponentJobs(Path(directory) / "queue.sqlite3")
            identifier = jobs.add("Entrega", {"engine": "delivery"})
            first = jobs.start(identifier)
            hashes = {"reference.png": "sha256"}
            jobs.record_delivery_inputs(identifier, first, hashes)
            jobs.transition(identifier, "cancelled", attempt=first)
            second = jobs.start(identifier)
            self.assertEqual(jobs.entries()[0]["parameters"]["input_hashes"], hashes)
            self.assertEqual(jobs.history(identifier)[second - 1]["parameters"]["input_hashes"], hashes)

    def test_alignment_uses_alpha_without_inventing_pose_points(self):
        image = Image.new("RGBA", (80, 100))
        ImageDraw.Draw(image).rectangle((20, 10, 60, 90), fill="white")
        with patch("aethryx_engines.suggest_anatomical_points", return_value={}):
            calibration, points, _ = analyze_view_alignment("unused.png", image, "front")
        self.assertEqual(calibration, ([40., 90.], 10.))
        self.assertEqual(points, {})

    def test_auto_front_setup_keeps_inferred_anchors_subject_to_review(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            image = Image.new("RGBA", (160, 240))
            ImageDraw.Draw(image).rectangle((42, 20, 118, 216), fill="white")
            source = folder / "source.png"
            image.save(source)
            detections = {joint: {"xy": xy, "visibility": .8} for joint, xy in
                          {0: [80, 45], 11: [104, 80], 12: [56, 80], 23: [98, 135], 24: [62, 135]}.items()}
            with patch("aethryx_engines.suggest_anatomical_points", return_value=detections), patch(
                    "aethryx_engines.suggest_facial_points", return_value={}):
                setup = build_auto_front_setup(source, work_dir=folder / "analysis", use_sculpted_body=False)
            front = setup["views"]["front"]
            self.assertGreaterEqual(len(setup["vertex_indices"]), 4)
            self.assertEqual(front["suggested"], setup["vertex_indices"])
            self.assertEqual(len(set(front["suggested"])), len(front["landmark_weights"]))
            self.assertTrue(setup["allow_inferred_depth"])
            self.assertTrue(Path(front["body_mask"]).is_file())
