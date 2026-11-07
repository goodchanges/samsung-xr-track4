import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from check_output import check
from selection import analyze

VOCAB = [dict(verb="cut", noun="carrot"), dict(verb="wash", noun="cup")]


def make_vectors(count):
    vectors = np.zeros((count, 2), dtype=float)
    vectors[:, 0] = 1.0
    vectors[::2, 1] = 0.2
    vectors[1::2, 1] = -0.2
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


class GoldenOutputChecks(unittest.TestCase):
    def assert_golden(self, name, duration, scores, idle):
        actions, highlights = analyze(scores, idle, make_vectors(len(scores)), VOCAB, duration)
        document = dict(video_id=name, label="test", actions=actions, highlights=highlights)
        json.dumps(document, allow_nan=False)
        check(document, name, duration)
        return document

    def test_short_clip_golden(self):
        document = self.assert_golden("short", 2.0, np.array([[0.4, 0.2], [0.4, 0.1]]), np.array([0.1, 0.1]))
        self.assertEqual(document["video_id"], "short")

    def test_long_clip_golden(self):
        n = 600
        scores = np.zeros((n, 2), dtype=float)
        scores[:20, 0] = 0.4
        scores[300:340, 1] = 0.5
        document = self.assert_golden("long", 600.0, scores, np.full(n, 0.1))
        self.assertLessEqual(sum(x["end"] - x["start"] for x in document["highlights"]), 87.0 + 1e-8)

    def test_idle_clip_golden(self):
        document = self.assert_golden("idle", 120.0, np.zeros((120, 2), dtype=float), np.ones(120))
        self.assertEqual(document["actions"], [])

    def test_non_kitchen_like_clip_golden(self):
        scores = np.tile([0.12, 0.1], (210, 1))
        document = self.assert_golden("non-kitchen", 210.0, scores, np.full(210, 0.11))
        self.assertIn("highlights", document)

    def test_kitchen_like_clip_golden(self):
        scores = np.tile([0.35, 0.12], (60, 1))
        document = self.assert_golden("kitchen-like", 60.0, scores, np.full(60, 0.1))
        self.assertGreaterEqual(len(document["actions"]), 1)

    def test_resolution_variant_like_clip_golden(self):
        scores = np.vstack([np.array([0.3, 0.2]) if i % 3 else np.array([0.2, 0.35]) for i in range(4)])
        document = self.assert_golden("resolution-fps", 4.0, scores, np.full(4, 0.1))
        self.assertTrue(all(0 <= x["importance"] <= 1 for x in document["highlights"]))


if __name__ == "__main__":
    unittest.main()
