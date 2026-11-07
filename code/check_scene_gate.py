"""Development domain routing diagnostic. No thresholds are tuned by this script."""
import json
from model import ROOT, Scorer
from media import frames
from scene_gate import confidence, PROMPTS, MARGIN, FRAME_FRACTION
from refine_temporal import DEVELOPMENT


def main():
    scorer = Scorer(ROOT / 'code/vocabulary.json', 4)
    rows = []
    for video_id, video in [(i, ROOT / 'work/temporal' / (i + '.mp4')) for i in DEVELOPMENT] + [
            ('ride', ROOT / 'work/non-kitchen/ride.mp4')]:
        _, _, embeddings, _ = scorer.predict(frames(video, 1))
        rows.append(dict(video_id=video_id, old_gate=scorer.kitchen_confidence(embeddings), scene_gate=confidence(scorer, embeddings)))
        print(rows[-1], flush=True)
    report = dict(prompts=PROMPTS, margin=MARGIN, frame_fraction=FRAME_FRACTION, rows=rows,
                  scope='Three kitchen development videos, one reused non-kitchen smoke video; not representative domain validation')
    (ROOT/'work/scene-gate-development.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
