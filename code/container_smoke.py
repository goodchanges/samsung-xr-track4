"""Offline runtime checks; explicitly reject silent inference-failure outputs."""
import argparse
import json
from pathlib import Path
import subprocess

from check_output import check
from media import ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "work/container-smoke"
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "idle sample.mp4"
    subprocess.run(ffmpeg("-loglevel", "error", "-y", "-f", "lavfi", "-i",
                          "color=c=gray:s=160x120:r=12", "-t", "120", "-c:v", "libx264", video), check=True)
    extra = []
    if args.temporal:
        import numpy as np
        from temporal_model import TemporalRecognizer
        model = TemporalRecognizer(root / "weights/temporal/tsm_rgb.ckpt")
        verbs, nouns = model.predict(np.zeros((1, 8, 224, 224, 3), dtype=np.uint8))
        assert verbs.shape == (1, 97) and nouns.shape == (1, 300)
        assert np.isfinite(verbs).all() and np.isfinite(nouns).all()
        np.testing.assert_allclose(verbs.sum(axis=1), 1., atol=1e-5)
        np.testing.assert_allclose(nouns.sum(axis=1), 1., atol=1e-5)
        from joint_decoder import joint_segments, load_profile
        classes = json.loads((root / "weights/temporal/classes.json").read_text())
        association, strength, switch_cost = load_profile(root / "weights/temporal_decoder.npz")
        decoded = joint_segments(verbs, nouns, classes, .5, association, strength, switch_cost)
        assert len(decoded) == 1 and decoded[0]["start"] == 0. and decoded[0]["end"] == .5
        # Exercise the exact two-argument contract used by the improved archive.
        (root / 'code/inference_profile.txt').write_text('improved\n')
        print("Local temporal checkpoint inference passed", flush=True)
    output = folder / "idle.json"
    run = subprocess.run([str(root / "run.sh"), str(video), str(output), *extra], capture_output=True, text=True)
    print(run.stderr, end="", flush=True)
    assert run.returncode == 0
    document = json.loads(output.read_text())
    assert document["label"], "Empty failure fallback is not a successful inference"
    assert "Track 4 failure:" not in run.stderr
    assert document["highlights"], "Expected a nonempty summary of valid media"
    check(document, video.stem, 120.)
    assert sum(h["end"] - h["start"] for h in document["highlights"]) <= 120 * .145 + 1e-8
    broken = folder / "corrupt.mp4"
    broken.write_bytes(b"invalid video")
    failed = subprocess.run([str(root / "run.sh"), str(broken), str(output)], capture_output=True, text=True)
    assert failed.returncode == 0 and "Track 4 failure:" in failed.stderr
    assert json.loads(output.read_text()) == dict(video_id="corrupt", label="", actions=[], highlights=[])
    print("Offline runner and corrupt-input containment passed", flush=True)


if __name__ == "__main__":
    main()
