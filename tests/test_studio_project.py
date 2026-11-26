import json
from pathlib import Path
import struct
import tempfile
import unittest
from PIL import Image
from studio_project import save_project, load_project, validate_settings, validate_glb

class ProjectTests(unittest.TestCase):
    def test_modular_project_reopens_references_masks_controls_and_visibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root / "avatar.aeth"
            scene = root / "avatar.glb"
            scene.touch()
            anatomy = {"bundled_template": True, "template": "C:/unused.ply",
                       "proportions": {"head": 1.18},
                       "views": {"front": {kind: str(root / f"{kind}.png")
                                           for kind in ("image", "mask", "body_mask")}}}
            anatomy["views"]["front"] = {kind: str(Path(path).resolve())
                                             for kind, path in anatomy["views"]["front"].items()}
            save_project(project, image="", model=scene, output="", steps=50, faces=-1, seed=42,
                         guidance=7, hidden_components=["cabello"], anatomy_setup=anatomy)
            stored = json.loads(project.read_text(encoding="utf-8"))
            self.assertEqual(stored["version"], 3)
            self.assertEqual(stored["anatomy_setup"]["template"], "")
            self.assertEqual(stored["anatomy_setup"]["views"]["front"]["body_mask"], "body_mask.png")
            reopened = load_project(project)
            self.assertEqual(reopened["hidden_components"], ["cabello"])
            self.assertEqual(reopened["anatomy_setup"]["proportions"], {"head": 1.18})
            self.assertEqual(reopened["anatomy_setup"]["views"], anatomy["views"])
            stored.update(version=2)
            del stored["hidden_components"], stored["anatomy_setup"]
            project.write_text(json.dumps(stored), encoding="utf-8")
            self.assertEqual(load_project(project)["hidden_components"], [])
            self.assertIsNone(load_project(project)["anatomy_setup"])

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            image.touch()
            project = root / "test.aeth"
            save_project(project, image=image, model=None, output=root/"mesh.glb",
                         steps=50, faces=-1, seed=42, guidance=7.0)
            self.assertEqual(json.loads(project.read_text())["image"], "image.png")
            self.assertEqual(load_project(project)["image"], str(image.resolve()))
            image.unlink()
            with self.assertRaises(ValueError):
                load_project(project)
    def test_reference_regions_round_trip_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "sheet.png"
            image.touch()
            project = root / "character.aeth"
            region = {
                "id": "ref-001",
                "path": str(image),
                "view": "front",
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "primary": True,
            }
            save_project(project, image=image, model=None, output=root / "mesh.glb",
                         steps=50, faces=-1, seed=42, guidance=7.0,
                         references=[region])
            data = json.loads(project.read_text(encoding="utf-8"))
            self.assertEqual(data["references"][0]["path"], "sheet.png")
            self.assertEqual(load_project(project)["references"][0]["bbox"], region["bbox"])
            data.pop("references")
            project.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_project(project)["references"], [])
    def test_reference_provenance_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "sheet.png"
            Image.new("RGB", (32, 32), "white").save(image)
            project = root / "provenance.aeth"
            region = {
                "id": "ref-face",
                "path": str(image),
                "view": "face_detail",
                "bbox": [0.1, 0.1, 0.5, 0.5],
                "primary": False,
                "origin": "opencv-haar",
                "confidence": None,
                "confirmed": False,
            }
            save_project(project, image=image, model=None, output=root / "mesh.glb",
                         steps=50, faces=-1, seed=42, guidance=7.0,
                         references=[region])
            loaded = load_project(project)["references"][0]
            self.assertEqual(loaded["origin"], "opencv-haar")
            self.assertFalse(loaded["confirmed"])
    def test_invalid_settings(self):
        for values in [(0,-1,42,7), (50,0,42,7), (50,-1,-1,7), (50,-1,42,float("nan"))]:
            with self.assertRaises(ValueError):
                validate_settings(*values)
    def test_glb_header_and_chunks(self):
        data = b'{"meshes":[{}]}'
        data += b" " * (-len(data) % 4)
        content = struct.pack("<4sII", b"glTF", 2, 20+len(data)) + struct.pack("<I4s",len(data),b"JSON") + data
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"mesh.glb"
            path.write_bytes(content)
            self.assertIn("meshes", validate_glb(path))
            path.write_bytes(content[:-1])
            with self.assertRaises(ValueError):
                validate_glb(path)
if __name__ == "__main__":
    unittest.main()
