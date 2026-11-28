import unittest
from types import SimpleNamespace

from app import AethryxStudio
from aethryx_tools import (
    analyze_project_diagnostics,
    execute_tool,
    get_project_context,
)


class AethryxToolTests(unittest.TestCase):
    def test_project_context_requires_identity(self):
        result = get_project_context({})
        self.assertEqual(result["status"], "unavailable")

    def test_diagnostics_use_real_project_state(self):
        context = {
            "project_id": "project-1",
            "revision": "4",
            "references": [],
            "mesh_stats": {"vertices": 2506, "faces": 5000},
            "model": "mesh.glb",
            "scene_components": [
                {"node": "body", "vertices": 2500, "faces": 4996, "visible": True},
                {"node": "hair-lock-01", "vertices": 6, "faces": 4, "visible": False},
            ],
        }
        result = analyze_project_diagnostics(context)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["revision"], "4")
        self.assertTrue(any("2506 vertices" in finding for finding in result["findings"]))
        self.assertIn("Objetos reales en la escena: 2; visibles 1, ocultos 1.", result["findings"])
        self.assertTrue(any("hair-lock-01: 6 vertices, 4 caras, oculto." == finding
                            for finding in result["findings"]))
        authorized = get_project_context(context)["context"]
        self.assertEqual(authorized["scene_components"], context["scene_components"])

    def test_diagnostics_reject_malformed_component_entries(self):
        result = analyze_project_diagnostics({
            "project_id": "project-1",
            "revision": "5",
            "scene_components": [{"node": "body", "vertices": "unknown"}],
        })
        self.assertIn("Los datos de componentes de escena no tienen un formato válido.", result["findings"])

    def test_studio_passes_loaded_scene_nodes_to_kiria(self):
        studio = SimpleNamespace(
            project_path="project.aeth",
            reference_regions=[],
            model_stats={"vertices": 12, "faces": 8},
            current_model_path="avatar.glb",
            image_path=SimpleNamespace(get=lambda: "reference.png"),
            model_parts=[
                {"node": "body", "source_vertices": 8, "source_faces": 6},
                {"node": "hair-lock-01", "source_vertices": 4, "source_faces": 2},
            ],
            _hidden_component_nodes=lambda: ["hair-lock-01"],
        )
        context = AethryxStudio._kiria_context(studio)
        self.assertEqual(context["scene_components"], [
            {"node": "body", "vertices": 8, "faces": 6, "visible": True},
            {"node": "hair-lock-01", "vertices": 4, "faces": 2, "visible": False},
        ])

    def test_unregistered_tool_is_not_executed(self):
        result = execute_tool("shell", {}, {})
        self.assertEqual(result["status"], "unavailable")

    def test_write_permission_is_not_granted_by_default(self):
        result = execute_tool("validate_avatar", {}, {}, write_allowed=False)
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
