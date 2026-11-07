import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from batch import manifest_entries
from config import CONFIG_VERSION
from output import clean_intervals
from pipeline import InferenceResult, PipelineOptions, PipelineState, infer_video
from perf_guardrail import validate_result


class FakeScorer:
    actions = [dict(verb="take", noun="cup"), dict(verb="wash", noun="cup")]

    def __init__(self):
        self.calls = 0

    def predict(self, raw):
        self.calls += 1
        scores = np.array([[0.5, 0.4], [0.6, 0.2], [0.2, 0.1]], dtype=float)
        idle = np.array([0.1, 0.1, 0.1], dtype=float)
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        return scores, idle, embeddings, "kitchen"

    def kitchen_confidence(self, embeddings):
        return False, -0.1, 0.0


class FakeOriginal(FakeScorer):
    actions = [dict(verb="open", noun="drawer")]

    def predict(self, raw):
        scores = np.array([[0.8], [0.8], [0.8]], dtype=float)
        idle = np.array([0.1, 0.1, 0.1], dtype=float)
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        return scores, idle, embeddings, "kitchen"


class HardeningChecks(unittest.TestCase):
    def test_perf_guardrail_rejects_failure_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out.json"
            output.write_text(json.dumps(dict(video_id="v", label="", actions=[], highlights=[])))
            with self.assertRaisesRegex(ValueError, "empty failure output"):
                validate_result(output, "v", 30.)

    def test_clip_route_does_not_query_kitchen_gate(self):
        scorer = FakeScorer()
        scorer.kitchen_confidence = Mock(side_effect=AssertionError("Unexpected kitchen gate"))
        options = PipelineOptions("preserved", False, 3, 1, None, 4)
        result = infer_video(Path("v.mp4"), 3., b"raw", options, PipelineState(scorer, FakeOriginal()))
        self.assertEqual(result.route, "clip")
        scorer.kitchen_confidence.assert_not_called()

    def test_rejected_kitchen_route_does_not_load_temporal_model(self):
        options = PipelineOptions("temporal", False, 1, 1, Path("absent.ckpt"), 2)
        state = PipelineState(FakeScorer())
        result = infer_video(Path("v.mp4"), 3., b"raw", options, state)
        self.assertEqual(result.route, "generic_clip_temporal_selector")
        self.assertIsNone(state.temporal)

    def test_accepted_kitchen_route_reuses_model_and_context(self):
        scorer = FakeScorer()
        scorer.kitchen_confidence = lambda _: (True, .1, 1.)
        temporal = Mock()
        temporal.recognize.return_value = [dict(verb="take", noun="cup", start=0., end=1.)]
        options = PipelineOptions("duration", False, 1, 1, Path("local.ckpt"), 4)
        state = PipelineState(scorer, temporal=temporal)
        for _ in range(2):
            result = infer_video(Path("v.mp4"), 3., b"raw", options, state)
            self.assertEqual(result.route, "kitchen_tsm")
            self.assertEqual(result.actions, temporal.recognize.return_value)
        self.assertEqual(temporal.recognize.call_count, 2)
        temporal.recognize.assert_called_with(Path("v.mp4"), 3., Path("classes.json"), 4)

    def test_config_version_is_pinned(self):
        self.assertRegex(CONFIG_VERSION, r"^\d{4}-\d{2}-\d{2}\.v\d+$")

    def test_manifest_property_like_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "v.mp4").touch()
            manifest = root / "manifest.csv"
            valid_ids = ["A1", "TEAM_1", "video-1", "xYz_123-abc"]
            for video_id in valid_ids:
                manifest.write_text(f"video_id,path\n{video_id},v.mp4\n", encoding="utf-8")
                self.assertEqual(manifest_entries(manifest, root), [(video_id, root / "v.mp4")])
            for invalid in ("", ".", "..", "bad/name", "bad\\name", "bad:name", "bad*name"):
                manifest.write_text(f"video_id,path\n{invalid},v.mp4\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    manifest_entries(manifest, root)

    def test_interval_normalization_property_like(self):
        duration = 10.0
        actions = [
            dict(verb="a", noun="b", start=-1, end=2),
            dict(verb="a", noun="b", start=2, end=12),
            dict(verb="a", noun="b", start=float("nan"), end=1),
        ]
        highlights = [
            dict(start=1.5, end=4.5, importance=2.0),
            dict(start=0.5, end=2.0, importance=-1.0),
            dict(start=3.0, end=3.0, importance=0.4),
        ]
        clean_actions, clean_highlights = clean_intervals(actions, highlights, duration)
        self.assertEqual(clean_actions, [
            dict(verb="a", noun="b", start=0.0, end=2.0),
            dict(verb="a", noun="b", start=2.0, end=10.0),
        ])
        self.assertEqual(clean_highlights[0]["start"], 0.5)
        self.assertEqual(clean_highlights[0]["importance"], 0.0)
        self.assertTrue(all(0 <= x["importance"] <= 1 for x in clean_highlights))
        self.assertTrue(all(clean_highlights[i]["end"] <= clean_highlights[i + 1]["start"] for i in range(len(clean_highlights) - 1)))

    def test_pipeline_typed_inference_result(self):
        options = PipelineOptions(
            selector="baseline", unsmoothed=False, top_k=1, threads=1, temporal_model=None, temporal_context=4
        )
        inferred = infer_video(Path("video.mp4"), 3.0, b"raw", options, PipelineState(FakeScorer(), FakeOriginal()))
        self.assertIsInstance(inferred, InferenceResult)
        payload = dict(video_id="v", label=inferred.label, actions=inferred.actions, highlights=inferred.highlights)
        json.dumps(payload, allow_nan=False)
        self.assertEqual(inferred.route, "clip")


if __name__ == "__main__":
    unittest.main()
