# Joint action decoder

Measured 7 November 2026. Same frozen TSM checkpoint, four-second contexts, existing local metric and media as the historical comparison. These are backend recognition measurements, not Samsung scores or measurements of the gated end-to-end runner.

| Recording | Original TSM F1 | Joint decoder F1 | Original / revised segments |
| --- | ---: | ---: | ---: |
| P18_10 | 9.46% | 10.46% | 102 / 56 |
| P08_14 | 9.82% | 12.90% | 86 / 50 |
| P04_28 | 9.80% | 12.22% | 77 / 35 |
| Mean | **9.69%** | **11.86%** | |

The metric averages per-video action F1 across tIoU 0.1, 0.3 and 0.5. Improvement is 2.17 percentage points, approximately 22.4% relative. Absolute accuracy remains low.

At tIoU 0.1, mean per-video precision increased from 9.46% to 15.81%, while recall decreased from 22.83% to 20.00%. This is a precision/recall tradeoff; action coverage and human highlight agreement have not been evaluated for this profile. It remains explicitly optional.

The decoder combines verb and noun probabilities with smoothed training-set log association scores. Exact Viterbi decoding applies a uniform cost to label changes, reducing fragmentation. It keeps unseen pairs possible. No video filename, ID, evaluation annotation or category shortcut is used during inference. Zero association strength and zero switch cost reproduce the original independent decoder exactly.

Pair counts use only the 67,217 public EPIC training annotations across 495 recordings. No images or neural weights were fitted. Add-one smoothing and the log-association formula are fixed; nine configurations compared association strengths 0/0.5/1 and switching costs 0/0.5/1 on development videos P28_18, P09_08 and P07_17. Development F1 improved from 4.82% to 9.15%. Strength 1 and switching cost 1 were frozen before one comparison on P18_10, P08_14 and P04_28. None of those six recordings is in the fitting video set.

The comparison recordings were already inspected during earlier work, so this replay is not a new blind holdout. The original model authors also evaluated on EPIC validation. Generalization to new participants, idle scenes and non-kitchen footage remains unmeasured. Lower segment counts may suppress valid short events; future evaluation must examine precision and recall separately. The profile only applies after the existing kitchen gate accepts TSM; gains cannot be assumed for the final gated pipeline.

Verification: all 32 unit tests passed locally, including joint decoding equivalence, continuity, pair associations and forwarding the profile through the shared pipeline. Calling the actual `TemporalRecognizer.recognize` with the saved profile on P18_10 exactly reproduced the evaluated 56 segments. A real `run.sh` invocation with the new flag wrote a valid local result with 20.994 seconds selected from 144.79 seconds. Its gate rejected TSM (margin -0.002642), so that run used generic CLIP and does not demonstrate the backend F1 gain end to end. The final modified routing test also passed separately. No new Linux/Docker claim is made for this revision.

## Run

With optional temporal dependencies and local weights installed:

```sh
./run.sh input.mp4 output.json --temporal-model weights/temporal/tsm_rgb.ckpt --temporal-context 4 --temporal-decoder weights/temporal_decoder.npz
python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --temporal-model weights/temporal/tsm_rgb.ckpt --temporal-context 4 --temporal-decoder weights/temporal_decoder.npz
```

Use `--selector temporal` with explicit `--temporal-context 4` to combine the decoder with temporal highlights. That combination's highlight quality has not been measured. Omitting `--temporal-decoder` preserves the previous decoder. Default CLIP, the kitchen gate, JSON schema and 14.5% highlight limit are unchanged.

Reproduce selection and the single comparison in a fresh work directory using `python code/improve_decoder.py development` followed by `python code/improve_decoder.py evaluation`. Development refuses to overwrite an existing freeze. Public metadata and the six videos must be prepared as described in refinement.md. Model scores are reused from content-keyed caches; labels and decoder choices are not inputs to those caches.

Evidence: `joint-decoder-frozen.json` records every fitting video ID, annotation revision/hash, checkpoint hash, selected settings, profile hash and freeze time. `joint-decoder-development.json` and `joint-decoder-evaluation.json` retain all predictions and per-video metrics. The tracked profile is `weights/temporal_decoder.npz` (about 136 KB), loaded locally with no downloads. Official Samsung vocabulary, allowed training data and model permissions remain pending. The profile depends on the existing published EPIC class order.
