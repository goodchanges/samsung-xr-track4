import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from scene_gate import decide, confidence
from pipeline import PipelineOptions, PipelineState, infer_video
from test_hardening import FakeScorer


class SceneGateChecks(unittest.TestCase):
    def test_scene_requires_margin_and_frame_support(self):
        self.assertTrue(decide(np.array([[.3,.1],[.25,.15]]))[0])
        self.assertFalse(decide(np.array([[.3,.31],[.25,.3]]))[0])
        # A strong single frame cannot override mostly non-kitchen frames.
        self.assertFalse(decide(np.array([[.9,.0],[.3,.31],[.3,.31]]))[0])

    def test_scene_profile_reaches_temporal_path_after_category_rejection(self):
        scorer = FakeScorer()  # Existing category gate rejects.
        temporal = Mock()
        temporal.recognize.return_value = []
        options = PipelineOptions('duration',False,1,1,Path('local.ckpt'),4,None,'scene')
        with patch('scene_gate.confidence',return_value=(True,.1,1.)):
            result = infer_video(Path('arbitrary.mp4'),3.,b'raw',options,PipelineState(scorer,temporal=temporal))
        self.assertEqual(result.route,'kitchen_tsm')
        temporal.recognize.assert_called_once()


if __name__ == '__main__':unittest.main()
