import unittest
from unittest.mock import Mock

from app import AethryxStudio


class SceneHistoryTests(unittest.TestCase):
    def test_undo_and_redo_restore_prepared_scene_without_reloading(self):
        studio = Mock()
        studio.load_request_id = 5
        studio.scene_history = []
        studio.scene_future = []
        studio.part_to_tree = {0: "object-0"}
        studio.mesh_artists = [Mock()]
        before = ("before.glb", object(), [{"node": "body"}], {}, ("object-0",), [False])
        after = ("after.glb", object(), [{"node": "head"}], {}, (), [True])
        studio._scene_snapshot.side_effect = [before, after, before]

        AethryxStudio.checkpoint_scene_change(studio)
        AethryxStudio._restore_scene_change(studio, studio.scene_history, studio.scene_future, "deshacer")
        studio._show_model.assert_called_once_with(5, *before[:4])
        studio.scene_tree.selection_set.assert_called_once_with(("object-0",))
        studio.mesh_artists[0].set_visible.assert_called_once_with(False)
        self.assertEqual(studio.scene_future, [after])

        AethryxStudio._restore_scene_change(studio, studio.scene_future, studio.scene_history, "rehacer")
        studio._show_model.assert_called_with(5, *after[:4])
        studio.mesh_artists[0].set_visible.assert_called_with(True)
        self.assertEqual(studio.scene_history, [before])
        studio.load_model.assert_not_called()
