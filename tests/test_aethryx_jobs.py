from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from aethryx_engines import prepare_character_alpha
from aethryx_jobs import ComponentJobs
from aethryx_tools import execute_tool


class ComponentJobTests(unittest.TestCase):
    def test_queue_persists_attempts_errors_and_review(self):
        with TemporaryDirectory() as folder:
            database = Path(folder) / "components.sqlite3"
            first = ComponentJobs(database)
            params = {"image": "source.png", "settings": {"steps": 30}}
            identifier = first.add("Cabello", params)
            other = first.add("Camiseta", params)
            first.start(identifier)
            reopened = ComponentJobs(database)
            with self.assertRaises(ValueError):
                reopened.start(other)
            reopened.transition(identifier, "failed", error="CUDA", expected="running")
            row, pending = first.entries()
            self.assertEqual((row["attempt"], row["error"], row["parameters"]), (1, "CUDA", params))
            self.assertEqual(pending["status"], "pending")
            first.start(identifier)
            reopened.transition(identifier, "complete", result="hair.glb")
            reopened.transition(identifier, "rejected")
            row, _ = first.entries()
            self.assertEqual((row["attempt"], row["status"], row["result"]), (2, "rejected", "hair.glb"))
            first.transition(other, "cancelled", expected="pending")
            first.start(other)
            with self.assertRaises(ValueError):
                first.transition(other, "cancelled", expected="pending")
            first.transition(other, "complete", result="shirt.glb")
            first.transition(other, "accepted")
            self.assertEqual(first.entries()[1]["status"], "accepted")

    def test_existing_transparency_is_preserved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rgba = Image.new("RGBA", (3, 1))
            rgba.putdata([(5, 5, 5, 0), (20, 30, 40, 79), (9, 8, 7, 255)])
            rgba.save(root / "input.png")
            result = prepare_character_alpha(root / "input.png", root / "prepared.png")
            self.assertEqual(result["status"], "available")
            with Image.open(result["path"]) as prepared:
                self.assertEqual(prepared.tobytes(), rgba.tobytes())
            with self.assertRaises(ValueError):
                prepare_character_alpha(root / "input.png", root / "input.png")

    def test_kiria_cannot_override_local_job_counts_via_tool_input(self):
        result = execute_tool("get_component_job_counts", {"component_job_counts": {"fake": 5}},
                              {"component_job_counts": {"complete": 2}})
        self.assertEqual(result, {"status": "available", "counts": {"complete": 2}})
