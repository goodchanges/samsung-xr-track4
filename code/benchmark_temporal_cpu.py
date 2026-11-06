"""Compare CPU layouts and fused inference; checks probability equivalence first."""
import json
import time

import numpy as np
import torch

from media import frames
from model import ROOT
from temporal_model import TemporalRecognizer


def main():
    checkpoint = ROOT / "weights/temporal/tsm_rgb.ckpt"
    original = TemporalRecognizer(checkpoint, optimize=False)
    optimized = TemporalRecognizer(checkpoint)
    pixels = np.frombuffer(frames(ROOT / "work/epic-preview.mp4", .5), dtype=np.uint8).reshape(-1, 224, 224, 3)
    clips = pixels[:32].reshape(4, 8, 224, 224, 3)
    expected, actual = original.predict(clips), optimized.predict(clips)
    errors = []
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, rtol=1e-4, atol=2e-6)
        np.testing.assert_array_equal(left.argmax(1), right.argmax(1))
        errors.append(float(np.abs(left - right).max()))
    records = []
    for threads in (2, 4):
        torch.set_num_threads(threads)
        for iteration in range(4):
            for name, model in (("original", original), ("fused_channels_last", optimized)):
                started = time.perf_counter()
                model.predict(clips)
                elapsed = time.perf_counter() - started
                if iteration:
                    records.append(dict(model=name, threads=threads, seconds=elapsed, clips=4))
    report = dict(max_probability_errors=errors, records=records)
    (ROOT / "work/temporal/cpu-benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
