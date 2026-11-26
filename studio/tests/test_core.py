import tempfile
import unittest
from pathlib import Path
from studio.core import generation_args, new_output

class CoreTests(unittest.TestCase):
    def test_args_and_unique_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "imagen con espacios.png"
            image.touch()
            first, second = new_output(tmp), new_output(tmp)
            self.assertNotEqual(first, second)
            args = generation_args(image, first, faces=50000)
            self.assertEqual(args[:3], ["-u", "-m", "scripts.inference_triposg"])
            self.assertEqual(args[args.index("--image-input")+1], str(image.resolve()))
            self.assertEqual(args[args.index("--faces")+1], "50000")
            self.assertTrue(first.parent.is_dir())
    def test_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.png"
            output = Path(tmp) / "model.glb"
            with self.assertRaises(ValueError): generation_args(image, output)
            image.touch()
            with self.assertRaises(ValueError): generation_args(image, output, steps=0)
            with self.assertRaises(ValueError): generation_args(image, output.with_suffix(".fbx"))
            output.touch()
            with self.assertRaises(ValueError): generation_args(image, output)

if __name__ == "__main__":
    unittest.main()
