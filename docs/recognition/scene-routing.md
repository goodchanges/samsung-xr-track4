# Scene routing and a fresh comparison

The optional scene router checks whether frames depict a kitchen, rather than whether the person is preparing a meal. It reuses local CLIP image features and compares fixed text prompts for a kitchen against bathroom, office, bedroom, living room, workshop, vehicle and outdoor scenes. These are internal domain prompts, not new output categories or student-kit classes.

The prespecified rule requires a mean cosine margin of at least 0.01 over the strongest competing scene, with kitchen winning at least half of frames. This is an uncalibrated decision rule, not a probability. No thresholds were selected from the new comparison's scores. The existing category router remains the default; scene routing requires an explicit flag.

On the three development recordings P28_18, P09_08 and P07_17, the old category router rejected TSM in every case; scene routing accepted all three. Both routers rejected the reused 210-second amusement-ride recording. One non-kitchen example does not establish safe generalization to other indoor environments or mixed-domain footage. Scene routing uses a whole-video decision and can route individual out-of-domain intervals incorrectly.

## Run

```sh
./run.sh input.mp4 output.json --temporal-model weights/temporal/tsm_rgb.ckpt --temporal-context 4 --temporal-decoder weights/temporal_decoder.npz --kitchen-routing scene
python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --temporal-model weights/temporal/tsm_rgb.ckpt --temporal-context 4 --temporal-decoder weights/temporal_decoder.npz --kitchen-routing scene
```

These commands keep the existing highlight selector. Add `--selector temporal` only as an explicit experiment; highlight quality for that combination is unmeasured. JSON keys, provisional vocabulary, generic fallback and the 14.5% limit remain unchanged. Inference reads local weights only.

## Evaluation protocol

The previous six recordings have already been inspected. A new comparison therefore uses P01_12, P06_12 and P11_22, with 165 annotated actions. They were selected by sorted IDs, distinct participants outside the previous six-video comparison and a last annotated time of 120–240 seconds, before recognition scores were computed. No new neural training or decoder tuning was performed. The joint profile remains the one frozen on the earlier development split. Training statistics and the pretrained checkpoint can include these participants' other videos; none of these three video IDs appears in EPIC training annotations. The model authors' public validation use also prevents calling this a blind external benchmark.

The comparison records both original TSM and joint-decoder backend scores and the old versus scene-plus-joint full routed pipelines. All use four-second TSM contexts, identical source frames, canonical classes and the same local metric: mean per-video action F1 averaged over tIoU 0.1/0.3/0.5. Samsung scoring, human highlight F1 and FPVSum remain unmeasured.

## Measured results

| Fresh recording | Original TSM backend | Joint backend / scene-plus-joint pipeline | Old gated pipeline |
| --- | ---: | ---: | ---: |
| P01_12 | 12.16% | 16.67% | 1.53% |
| P06_12 | 4.58% | 15.19% | 0.00% |
| P11_22 | 7.07% | 8.15% | 0.00% |
| Mean local action F1 | **7.94%** | **13.34%** | **0.51%** |

On this new set, joint decoding improves backend F1 by 5.40 percentage points. Scene routing accepted TSM on all three whereas category routing rejected all three. Thus the scene-plus-joint end-to-end pipeline reaches the backend score. This compares two complete routing/recognition configurations, not the isolated effect of routing alone. Output category and the preserved highlight selector are unchanged.

At tIoU 0.1, mean per-video backend precision improved from 9.82% to 19.93% and recall from 18.72% to 22.17%. Both improved here; the older replay comparison had reduced recall. These three-video results have substantial uncertainty and must not be generalized to universal recognition. The 13.34% figure is from different recordings than the earlier 9.69%/11.86% comparison; those figures are not a before/after pair with this score.

All 34 local tests passed. [Linux CI run 42349474813](https://github.com/goodchanges/samsung-xr-track4/actions/runs/42349474813) passed both CPU images on revision 5743586, including offline inference. The temporal image directly executed the actual checkpoint plus joint decoder and exercised scene-routing flags through the runner. Docker remains unavailable on the laptop itself.

A real `run.sh` call with the scene/joint options on P01_12 renamed to `arbitrary recording.mp4` selected the kitchen TSM route. Its 54 actions exactly matched the fresh comparison predictions; only the output video ID changed. Its 25.202 selected seconds from 173.81 seconds passed the provisional interval checker. The exported reel probed at 25.19 seconds, within 0.1 seconds of the selected duration. Reported 4.20 seconds runtime reused cached model scores and is not cold inference speed. Fresh evaluation processing took 99.33, 100.30 and 154.82 seconds respectively, including multiple profile comparisons and feature processing; concurrent laptop activity makes these unsuitable for precise performance claims.

The actual runner also processed the amusement-ride clip using scene routing, rejected TSM and kept the generic path. Its output retained the same three unsupported actions and six-second highlight; neither non-kitchen action accuracy nor the underfilled summary was corrected by this change. It passed the interval checker. The scene option remains explicit until broader negative-domain validation is available.

`fresh-comparison-frozen.json` records media URLs/hashes, fixed prompts and thresholds, decoder/checkpoint hashes and implementation hashes before inference. The pinned media mirror and annotation revisions are included. `fresh-comparison-results.json` records predictions and per-video metrics; `fresh-comparison-summary.json` records the aggregate. `scene-gate-development.json` records the development routing check.

Reproduce the bounded comparison with `python code/fresh_comparison.py prepare` followed by `python code/fresh_comparison.py evaluate`. Preparation requires network access to retrieve the pinned public media; inference is offline. Existing freeze/completion files prevent overwriting this run. The required public metadata and local model setup are described in temporal.md.
