import sys
import tempfile
import subprocess
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from media import ffmpeg, frames, probe, SIZE
from selection import analyze
from selection_baseline import analyze as baseline
from check_output import check

VOCAB = [{"verb":"cut","noun":"carrot"}, {"verb":"wash","noun":"cup"}]


class PipelineChecks(unittest.TestCase):
    def test_budget_order_bounds_and_repeatability(self):
        rng = np.random.default_rng(31)
        for duration in (1.0, 120.01, 600.0):
            n = int(np.ceil(duration))
            scores = rng.uniform(.05,.35,(n,2))
            vectors = rng.normal(size=(n,8))
            vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
            inputs = (scores,np.full(n,.15),vectors,VOCAB,duration)
            actions, highlights = analyze(*inputs)
            check(dict(video_id="fixture",label="test",actions=actions,highlights=highlights),
                  "fixture",duration)
            self.assertEqual((actions, highlights), analyze(*inputs))
            for selected in (highlights, baseline(*inputs)[1]):
                self.assertLessEqual(sum(x["end"]-x["start"] for x in selected),duration*.145+1e-8)

    def test_partial_candidate_uses_retained_score(self):
        scores=np.array([[.12],[.4],[.4],[.4]])
        actions, highlights=analyze(scores,np.full(4,.1),np.tile([1.,0.],(4,1)),
                                   [VOCAB[0]],4)
        self.assertEqual(highlights[0]["start"],0)
        self.assertEqual(highlights[0]["end"],.58)
        self.assertAlmostEqual(highlights[0]["importance"],.532,places=3)

    def test_all_idle(self):
        actions, highlights = analyze(np.zeros((120,2)),np.ones(120),
                                      np.tile([1.,0.],(120,1)),VOCAB,120)
        self.assertEqual(actions,[])
        self.assertEqual(highlights,[{"start":51.3,"end":68.7,"importance":0.0}])

    def test_peripheral_content_and_bad_video(self):
        with tempfile.TemporaryDirectory(prefix="track4 spaces ") as folder:
            source=Path(folder)/"edge video.mp4"
            filters="color=c=black:s=640x240:r=12,drawbox=x=0:y=0:w=80:h=240:color=red:t=fill,drawbox=x=560:y=0:w=80:h=240:color=blue:t=fill"
            subprocess.run(ffmpeg("-v","error","-f","lavfi","-i",filters,"-t","2",
                                  "-c:v","libx264",source),check=True)
            raw=frames(source,1)
            image=np.frombuffer(raw,dtype=np.uint8).reshape(-1,SIZE,SIZE,3)[0]
            self.assertGreater(int(image[112,10,0]),150)
            self.assertGreater(int(image[112,210,2]),150)
            self.assertLess(int(image[10].max()),15)
            bad=Path(folder)/"broken.mp4"
            bad.write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                probe(bad)


if __name__ == "__main__":
    unittest.main()
