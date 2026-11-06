# Recognition experiment — 6 November 2026

This is a small local experiment, not Samsung scoring or a competitive ranking.

## Data and separation

Nine continuous recordings were selected as the smallest 2–10 minute training-split recording available for each of participants P01–P09 in a 256p mirror. Selection used duration and file size, not model results. Four videos (P01–P04, 127 actions) fit 63 visual prototypes. Two videos (P05–P06, 67 actions) choose settings. Three different participants (P07–P09, 93 actions, 493.4 seconds) provide held-out evaluation. Short-recording selection and only three evaluation participants limit representativeness; the source includes very dark footage.

Annotations come from the public EPIC training split, mapped to canonical verb/noun class keys. No Samsung validation/test labels were used. The 512 most common action pairs are counted with development/evaluation participants excluded, then 16 missing baseline pairs are retained. Total: 528 pairs. A normalized CLIP image prototype is averaged per action from fitting intervals. Its centered cosine similarity adjusts text scores with strength 0.1. This is a visual prototype adapter, not a trained motion or hand-contact model.

The development search tested strengths 0, 0.1, 0.25, 0.5; one or three concurrent hypotheses; smoothing on/off. Mean macro action F1 over tIoU 0.1, 0.3, 0.5 selected strength 0.1, one hypothesis, smoothing. Selection was saved before held-out scoring. Development objective: 0.44% baseline versus 2.22% revised. Do not reuse the three held-out videos for another claimed independent model selection.

## Measured held-out results

| Local metric | Original | Revised recognizer |
| --- | ---: | ---: |
| Matched actions at tIoU 0.1 | 1 / 93 | 2 / 93 |
| Macro precision at tIoU 0.1 | 0.72% | 1.01% |
| Macro recall at tIoU 0.1 | 0.98% | 1.96% |
| Macro F1 at tIoU 0.1 | 0.83% | 1.33% |
| Macro F1 at tIoU 0.3 | 0.00% | 0.67% |
| Macro F1 at tIoU 0.5 | 0.00% | 0.67% |
| Mean macro F1 across the three thresholds | 0.28% | 0.89% |

Action precision/recall/F1 use greedy one-to-one chronological exact-pair matching, then average per-video metrics. They are NOT official action mAP. Absolute accuracy remains poor; neither a top-1% claim nor broad generalization is supported.

Feeding revised scores to the highlight selector reduced local distinct-action coverage from 18.33% to 13.06%. Therefore the shipped default preserves original highlight decisions using the original 29-pair scorer, sharing the same CLIP model and image features. The default's actual generated JSONs were checked against the measured revised action metrics and original 18.33% coverage. This coverage counts distinct pairs when at least half of one annotated instance is retained; it is not Samsung's unknown final scoring definition. The highlight-preservation decision was made after observing that regression; no new independent end-to-end evaluation is claimed.

No human consensus highlight labels or FPVSum category labels were available. Highlight F1 and category accuracy are null, not invented zeros or assumed cooking labels. All nine JSONs passed the provisional bounds/order/15% checker. They are stored under work/recognition/predictions, not official outputs/.

## Reproduce

From the repository root, using the project Python environment:

```text
python code/download_development.py work/recognition
python code/epic_development.py work/recognition work/recognition
python code/fit_adapter.py work/recognition/videos work/recognition/annotations work/recognition/vocabulary-epic.json work/recognition/adapter.npz
python code/recognition_experiment.py work/recognition code/vocabulary-baseline.json work/recognition/vocabulary-epic.json work/recognition/adapter.npz work/recognition/reproduced-report.json
python code/batch.py work/recognition/videos work/recognition/predictions
```

The download command verifies pinned source hashes. The fitting command reads only split=fit. Reports, annotations, split assignments and provenance hashes are checked in here. Original baseline inference is available using `--vocabulary code/vocabulary-baseline.json --no-adapter --selector baseline`. Alternative concurrent hypotheses remain an experiment (`--top-k 3`); they were not promoted.

## Sources and use

- Official annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations at ea8b40457a400c3fffa1c7f406ef3dc169cc2522.
- 256p video mirror: https://huggingface.co/datasets/mohammad2012191/epic_kitchens_100_256p at 3297646177d552b615fdb09d3a6986cb3a7cb527. The mirror documents spatial re-encoding and audio removal. Observed durations match official metadata to the probe's hundredth-second precision. This does not establish byte equivalence with original full-resolution videos.
- EPIC data is attributed to Damen et al., Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100, IJCV 2022, https://doi.org/10.1007/s11263-021-01531-2. Source data is CC BY-NC 4.0. Use this data-derived adapter for noncommercial research and verify challenge permitted-data/model rules when the kit arrives.

## Official submission still pending

Neither the connected mailbox search nor Downloads contained the student kit. The exact manifest, supported vocabulary, schema, supplied validator, all challenge source videos, FPVSum labels and highlight ground truth remain unavailable. No official outputs or official validation pass are claimed.

After inspecting and adopting the kit's schema/vocabulary, the batch runner supports an explicit manifest and supplied validator:

```text
python code/batch.py VIDEO_ROOT outputs --manifest KIT/manifest.csv --id-column video_id --path-column path --vocabulary KIT_ADAPTED_VOCABULARY.json --no-adapter --validator KIT/validate.py
```

Column names are configurable; no manifest columns or extra schema keys are guessed. The current writer still emits the four-key slide example. If the kit really requires six keys, it must be adapted first. The runner preflights every manifest file and ID, rejects duplicate IDs and unrelated output JSONs, reuses one loaded model, then runs the provided validator with `--dir`. Any validator failure exits unsuccessfully. A validator success alone does not supply accuracy ground truth.
