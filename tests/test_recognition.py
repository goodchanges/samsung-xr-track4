import json
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch, Mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from batch import manifest_entries
import batch
from evaluate import metrics
from localization import localize


class RecognitionChecks(unittest.TestCase):
    def test_concurrent_actions_and_idle(self):
        vocab = [dict(verb="open", noun="cupboard"), dict(verb="take", noun="cup")]
        scores = np.array([[.2, .19], [.2, .19], [0., 0.]])
        embeddings = np.eye(3)
        actions = localize(scores, np.full(3, .1), embeddings, vocab, 2.5, smoothing=False)
        self.assertEqual(len(actions), 2)
        self.assertTrue(all(x["start"] == 0 and x["end"] == 2 for x in actions))

    def test_missing_ground_truth_metrics_are_not_invented(self):
        result = metrics([], [], "cooking", {"actions": [], "label": None, "highlights": None}, 120)
        self.assertIsNone(result["highlight_f1_at_10hz"])
        self.assertIsNone(result["label_correct"])

    def test_manifest_ids_paths_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a video.mp4").touch()
            manifest = root / "manifest.csv"
            manifest.write_text('video_id,path\nCUSTOM_ID,a video.mp4\n', encoding="utf-8")
            entries = manifest_entries(manifest, root)
            self.assertEqual(entries, [("CUSTOM_ID", root / "a video.mp4")])
            manifest.write_text('video_id,path\nCUSTOM_ID,a video.mp4\ncustom_id,a video.mp4\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                manifest_entries(manifest, root)

    def test_supplied_validator_runs_and_failure_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "video.mp4").touch()
            manifest = root / "manifest.csv"
            manifest.write_text("video_id,path\nCUSTOM_ID,video.mp4\n", encoding="utf-8")
            validator = root / "validate.py"
            validator.write_text("import json,sys,pathlib\nassert json.loads((pathlib.Path(sys.argv[2])/'CUSTOM_ID.json').read_text())['video_id']=='CUSTOM_ID'\n", encoding="utf-8")
            scorer = Mock(actions=[dict(verb="take", noun="cup")])
            scorer.predict.return_value = (np.zeros((3, 1)), np.ones(3), np.ones((3, 2)), "test")
            argv = ["batch.py", str(root), str(root / "outputs"), "--manifest", str(manifest),
                    "--validator", str(validator), "--no-adapter", "--selector", "baseline", "--report", str(root / "report.json")]
            with patch.object(sys, "argv", argv), patch.object(batch, "Scorer", return_value=scorer), \
                    patch.object(batch, "frames", return_value=b"frames"), patch.object(batch, "probe", return_value=(3., False)):
                batch.main()
                validator.write_text("raise SystemExit(7)\n", encoding="utf-8")
                with self.assertRaises(subprocess.CalledProcessError):
                    batch.main()
            manifest.write_text('video_id,path\nCUSTOM_ID,missing.mp4\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing MP4"):
                manifest_entries(manifest, root)

    def test_batch_continues_after_corrupt_video_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a-corrupt.mp4").touch()
            (root / "b-valid.mp4").touch()
            scorer = Mock(actions=[dict(verb="take", noun="cup")])
            scorer.predict.return_value = (np.zeros((3, 1)), np.ones(3), np.ones((3, 2)), "test")
            def probe(path):
                if path.stem == "a-corrupt":
                    raise ValueError("Unreadable fixture")
                return 3., False
            argv = ["batch.py", str(root), str(root / "outputs"), "--selector", "baseline",
                    "--report", str(root / "report.json")]
            with patch.object(sys, "argv", argv), patch.object(batch, "Scorer", return_value=scorer), \
                    patch.object(batch, "frames", return_value=b"frames"), patch.object(batch, "probe", side_effect=probe):
                with self.assertRaises(SystemExit) as failure:
                    batch.main()
            self.assertEqual(failure.exception.code, 1)
            self.assertEqual(json.loads((root / "outputs/a-corrupt.json").read_text())["label"], "")
            self.assertEqual(json.loads((root / "outputs/b-valid.json").read_text())["label"], "test")
            self.assertEqual([r["status"] for r in json.loads((root / "report.json").read_text())], ["failed", "ok"])


if __name__ == "__main__":
    unittest.main()
