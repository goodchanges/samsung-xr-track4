# Temporal refinement and action-aware selection

Measured 7 November 2026. These are local research metrics, not Samsung scores.

Development recordings: P28_18, P09_08, P07_17. Reserved evaluation recordings: P18_10, P08_14, P04_28 (114 annotated actions). All are public EPIC validation recordings, outside the checkpoint's training-video split. They are not a blind benchmark relative to the original checkpoint authors' model selection. No local training was performed.

Twelve development configurations compared two/four-second contexts, smoothing widths 1/3/5 and one/two hypotheses. The selected two-second, unsmoothed, single-hypothesis configuration was frozen before evaluation. It improved development F1 but regressed on evaluation, so it did not replace four-second recognition.

| Evaluation metric | CLIP with preserved selector | TSM 4-second recognition, preserved selector | TSM 2-second recognition, temporal selector |
| --- | ---: | ---: | ---: |
| Mean local action F1 | 0.37% | **9.69%** | 7.58% |
| Matched actions at tIoU 0.1 | 1/114 | **25/114** | 22/114 |
| Distinct-action coverage | 13.09% | 13.09% | **25.34%** |

F1 averages per-video values across tIoU 0.1, 0.3 and 0.5; it is not action mAP. Coverage counts distinct ground-truth pairs for which at least half of an annotated instance is retained. Uniform sampling at the same budget scored 15.22% coverage. The temporal selector was developed on the development recordings and frozen before evaluation coverage was inspected. Human highlight F1 and FPVSum category accuracy remain unmeasured. The table describes a tradeoff, not a calculated official competition score.

The selector ranks source clips using predicted action diversity, confidence and visual redundancy. It scores the actual retained interval, including shortened final clips, and uses integer milliseconds to enforce the 14.5% budget. It does not read annotations or video IDs. Source IDs are fixed only in evaluation scripts to make experiments reproducible. Renaming content must not change its predictions.

The kitchen-specific TSM classifier is not universal. It has a fixed published class set, predicts activity even in idle scenes, and can miss unfamiliar objects or actions. Of 57 evaluation actions shorter than two seconds, the four-second model matched 15 and the two-second model matched 14 at tIoU 0.1. Absolute recognition accuracy remains low. CLIP also has a finite provisional vocabulary; neither backend understands every video.

## Run the tested profiles

Install `requirements-temporal.txt` and run `python code/setup_temporal.py` once before using either optional profile.

Best measured recognition on this evaluation:

```sh
./run.sh recording.mp4 output.json --temporal-model weights/temporal/tsm_rgb.ckpt
```

Best measured action coverage on this evaluation:

```sh
./run.sh recording.mp4 output.json --temporal-model weights/temporal/tsm_rgb.ckpt --selector temporal
```

The temporal selector defaults to two-second contexts; otherwise the temporal model defaults to four seconds. `--temporal-context` is an explicit experimental override. Batch accepts the same options. Omitting the temporal model retains the existing CLIP behavior. JSON keys remain unchanged pending the student kit.

## Verification and reproduction

All 14 unit tests passed. A real `run.sh` invocation on P18_10 produced 120 action segments and 20.994 seconds of highlights from 144.79 seconds of source footage. The exported reel probed at 20.99 seconds, within 0.1 seconds of the selected intervals. Its JSON highlights exactly matched the evaluated temporal selector. The 9.2-second runner time used cached scores and is not cold inference performance. Cold temporal passes on these recordings took approximately 96–111 seconds each, excluding model loading and CLIP. No ten-minute performance claim is made.

The same media was renamed to `ordinary recording.mp4` and run through batch inference with Python socket connections blocked. Actions, highlights and category matched the original single-video output exactly; only `video_id` changed. This verifies filename independence for that sample and Python-level offline operation, not universal accuracy or OS-level network isolation. Dependency checks passed. No official validator was available.

Model probabilities are cached by source frames, checkpoint hash, duration and context. Cached development passes took 0.42–1.19 seconds each including decoding. CPU convolution/batch-normalization fusion preserved benchmark class predictions, with maximum probability difference below 0.000001. Timing gains were modest and noisy; raw timings are recorded.

Evidence files in this directory:

- `temporal-selected-settings.json`: frozen development choice and all configuration scores.
- `temporal-development-results.json`, `temporal-refinement-evaluation.json`, `temporal-refinement-summary.json`: predictions and metrics.
- `temporal-selector-selection.json`, `temporal-selector-development.json`, `temporal-selector-evaluation.json`: selector freeze and results.
- `temporal-duration-recall.json`: matched/ground-truth counts by action duration.
- `temporal-refinement-provenance.json`: pinned annotation and media hashes.
- `temporal-cpu-benchmark.json`: numerical comparison and CPU timings.

`code/refine_temporal.py` and `code/compare_temporal_selection.py` reproduce the two comparisons using the recorded local inputs. Do not repeatedly retune on the reserved evaluation recordings and continue calling them independent. See [temporal.md](temporal.md) for checkpoint attribution and data/model restrictions. Samsung's kit, exact class mapping and official validation are still pending.
