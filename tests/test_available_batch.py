import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import batch


class AvailableBatchChecks(unittest.TestCase):
    def test_missing_and_duration_failure_do_not_block_valid_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('wrong-duration', 'valid'):
                (root / (name + '.mp4')).touch()
            manifest = root / 'manifest.csv'
            manifest.write_text('video_id,duration_sec,dataset\nmissing,30,test\nwrong-duration,60,test\nvalid,30,test\n')
            report = root / 'report.json'
            result = SimpleNamespace(label='cooking', label_score=.5, actions=[],
                                     highlights=[dict(start=0., end=4., importance=.5)],
                                     route='clip')
            argv = ['batch.py', str(root), str(root / 'outputs'), '--manifest', str(manifest),
                    '--available-only', '--profile', 'clip', '--report', str(report)]
            with patch.object(sys, 'argv', argv), patch.object(batch, 'probe', return_value=(30., False)), \
                 patch.object(batch, 'frames', return_value=b'frames') as frames, \
                 patch.object(batch, 'Scorer'), patch.object(batch, 'infer_video', return_value=result):
                with self.assertRaises(SystemExit) as exit_status:
                    batch.main()
                self.assertEqual(exit_status.exception.code, 1)
                frames.assert_called_once()
            rows = json.loads(report.read_text())
            self.assertEqual([r['status'] for r in rows], ['missing', 'failed', 'ok'])
            self.assertIn('differs from manifest', rows[1]['reason'])
            self.assertTrue(all(r['elapsed_seconds'] >= 0 for r in rows))
            self.assertFalse((root / 'outputs/missing.json').exists())
            self.assertEqual(json.loads((root / 'outputs/valid.json').read_text())['video_id'], 'valid')
