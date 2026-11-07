import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


@unittest.skipUnless(importlib.util.find_spec("torchvision"), "Optional temporal dependencies not installed")
class TemporalChecks(unittest.TestCase):
    def test_joint_decoder_zero_profile_reproduces_independent_decode(self):
        import numpy as np
        from joint_decoder import joint_segments
        from temporal_model import decode
        classes = dict(verb=['take', 'put'], noun=['cup', 'plate'])
        verbs = np.array([[.8, .2], [.4, .6], [.1, .9]])
        nouns = np.array([[.7, .3], [.1, .9], [.3, .7]])
        self.assertEqual(joint_segments(verbs, nouns, classes, 2.4, np.zeros((2, 2)), 0., 0.),
                         decode(verbs, nouns, classes, 2.4))

    def test_joint_decoder_continuity_and_pair_association(self):
        import numpy as np
        from joint_decoder import joint_segments
        classes = dict(verb=['take', 'put'], noun=['cup'])
        verbs = np.array([[.9, .1], [.49, .51], [.9, .1]])
        nouns = np.ones((3, 1))
        result = joint_segments(verbs, nouns, classes, 2.4, np.zeros((2, 1)), 0., 1.)
        self.assertEqual(result, [dict(verb='take', noun='cup', start=0., end=2.4)])
        associated = joint_segments(verbs, nouns, classes, 2.4, np.array([[-10.], [0.]]), 1., 0.)
        self.assertEqual(associated, [dict(verb='put', noun='cup', start=0., end=2.4)])

    def test_shift_does_not_mix_separate_clips(self):
        import torch
        from temporal_model import Shift

        clips = torch.cat([torch.ones(8, 8, 1, 1), torch.full((8, 8, 1, 1), 9.)])
        shifted = Shift(torch.nn.Identity())(clips).reshape(2, 8, 8)
        self.assertTrue(torch.all(shifted[0] <= 1))
        self.assertTrue(torch.all((shifted[1] == 0) | (shifted[1] == 9)))
        self.assertEqual(float(shifted[0, -1, 0]), 0.)
        self.assertEqual(float(shifted[1, 0, 1]), 0.)

    def test_merge_and_fractional_duration(self):
        from temporal_model import segments

        labels = {"verb": ["take", "put"], "noun": ["cup"]}
        self.assertEqual(segments([(0, 0), (0, 0), (1, 0), (1, 0)], labels, 2.4), [
            dict(verb="take", noun="cup", start=0., end=2.),
            dict(verb="put", noun="cup", start=2., end=2.4),
        ])

    def test_cache_changes_with_context_and_checkpoint(self):
        import numpy as np
        import torch
        import temporal_model

        model = temporal_model.TemporalRecognizer.__new__(temporal_model.TemporalRecognizer)
        torch.nn.Module.__init__(model)
        model.checkpoint_hash = "checkpoint-a"
        model.predict = Mock(return_value=(np.full((2, 97), 1 / 97), np.full((2, 300), 1 / 300)))
        raw = np.zeros((16, 224, 224, 3), dtype=np.uint8).tobytes()
        with tempfile.TemporaryDirectory() as folder, patch.object(temporal_model, "CACHE_ROOT", Path(folder)), \
                patch.object(temporal_model, "frames", return_value=raw):
            first = model.score_video(Path("video.mp4"), 2., 4)
            repeated = model.score_video(Path("video.mp4"), 2., 4)
            np.testing.assert_array_equal(first[0], repeated[0])
            self.assertEqual(model.predict.call_count, 1)
            model.score_video(Path("video.mp4"), 2., 2)
            self.assertEqual(model.predict.call_count, 2)
            model.checkpoint_hash = "checkpoint-b"
            model.score_video(Path("video.mp4"), 2., 2)
            self.assertEqual(model.predict.call_count, 3)

    def test_close_concurrent_hypothesis_and_gap(self):
        import numpy as np
        from temporal_model import decode

        verbs = np.array([[.55, .45], [.95, .05], [.55, .45]])
        nouns = np.array([[.95, .05]] * 3)
        classes = {"verb": ["take", "put"], "noun": ["cup", "bowl"]}
        actions = decode(verbs, nouns, classes, 2.5, top_k=2)
        self.assertIn(dict(verb="take", noun="cup", start=0., end=2.5), actions)
        secondary = [a for a in actions if a["verb"] == "put"]
        self.assertEqual(secondary, [dict(verb="put", noun="cup", start=0., end=1.),
                                     dict(verb="put", noun="cup", start=2., end=2.5)])
        self.assertEqual(len(decode(verbs, nouns, classes, 2.5, top_k=1)), 1)

    def test_temporal_selector_budget_and_bounds(self):
        import numpy as np
        from temporal_selection import select

        classes = {"verb": ["take", "put"], "noun": ["cup", "bowl"]}
        actions = [dict(verb="take" if i % 2 else "put", noun="cup", start=float(i), end=float(i + 1))
                   for i in range(10)]
        scores = np.tile([.6, .4], (11, 1))
        highlights = select(actions, scores, scores, classes, np.eye(11), 10.01)
        self.assertTrue(highlights)
        self.assertLessEqual(sum(h["end"] - h["start"] for h in highlights), 10.01 * .145)
        last = 0.
        for h in highlights:
            self.assertLessEqual(last, h["start"])
            self.assertLess(h["start"], h["end"])
            self.assertLessEqual(h["end"], 10.01)
            self.assertTrue(0 <= h["importance"] <= 1)
            self.assertTrue(all(isinstance(v, float) for v in h.values()))
            last = h["end"]


if __name__ == "__main__":
    unittest.main()
