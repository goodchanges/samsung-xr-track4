# Optional temporal recognition experiment

Measured on 7 November 2026. This is a local diagnostic, not Samsung scoring.

| Recording | Annotated actions | CLIP mean F1 | TSM mean F1 | CLIP seconds | TSM seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| P28_18 | 26 | 0.00% | 4.67% | 9.6 | 111.9 |
| P09_08 | 27 | 5.39% | 7.05% | 18.7 | 72.3 |
| P07_17 | 26 | 0.00% | 2.75% | 18.8 | 111.2 |
| Mean | 79 total | **1.80%** | **4.82%** | | |

F1 is averaged equally across videos and temporal IoU thresholds 0.1, 0.3 and 0.5 using the existing local evaluator. At tIoU 0.1, CLIP matched 4/79 actions; TSM matched 11/79. These recordings differ from the earlier three-video adapter evaluation, so compare the two columns here, not these figures against the old 1.38% result. Timings include decoding and inference after model loading, exclude downloads, and are not full runner or ten-minute benchmarks. CPU load and warm-up can affect them.

The TSM recognizer uses the published EPIC-KITCHENS RGB ResNet50 checkpoint, eight frames from each four-second context, two sampled frames per second, and one prediction per second. Adjacent identical pairs are merged. Images retain their aspect ratio and full view through padding. This differs from the source model's original evaluation crops. Its documented normalization uses the ImageNet mean for both mean and standard deviation. All checkpoint parameters load strictly. No local training was performed.

The comparison is between complete recognizers: TSM predicts 97 verbs and 300 nouns, while the existing CLIP branch scores 528 provisional pairs. Improvements cannot be attributed solely to temporal modeling. TSM currently predicts an action in every time bin and can hallucinate activity during idle footage. It also fragments actions and remains weak on short events. It is kitchen-specific; generalization to FPVSum is unmeasured. Highlight selection and category classification still use CLIP. No improvement in highlight quality or official action coverage has been established.

## Data and reproducibility

The three source IDs are `P28_18`, `P09_08`, and `P07_17`. Their annotations are from the public `EPIC_100_validation.csv`, with canonical names from the corresponding verb/noun CSVs. An assertion confirms none of these video IDs appears in `EPIC_100_train.csv`. The published checkpoint was trained on EPIC training data; the authors used public validation data for evaluation/model selection, so this is not a blind held-out benchmark. Our earlier nine recordings are in EPIC training and must not be used as independent evaluation of this checkpoint. Samsung manifest overlap remains unknown.

Source revisions, media hashes and annotation hashes are in [temporal-provenance.json](temporal-provenance.json). Predictions, per-video metrics and the checkpoint hash are in [temporal-results.json](temporal-results.json). Downloaded media and weights are excluded from ordinary Git tracking.

With the project venv activated, reproduce the diagnostic:

```sh
python -m pip install -r requirements-temporal.txt
python code/setup_temporal.py
python code/download_development.py work/temporal-inputs --provenance docs/recognition/temporal-provenance.json
python code/temporal_experiment.py --data work/temporal-inputs --videos work/temporal-inputs/videos
```

The setup commands require network access. Inference only loads local weights. The checkpoint is 97,616,687 bytes, SHA-256 `b56b6e7210c89e7dfc93daadafa90fda9d4ea73dfec7e9d3399240bca664139c`.

## Attribution and limits on use

[EPIC-KITCHENS action-recognition baselines](https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM), University of Bristol, provides the checkpoint under CC BY-NC 4.0. The temporal-shift architecture is from Ji Lin, Chuang Gan and Song Han, *TSM: Temporal Shift Module for Efficient Video Understanding*, ICCV 2019. This project implements a small inference-only version; it does not include the source training framework. [Public annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) and the [256p video mirror](https://huggingface.co/datasets/mohammad2012191/epic_kitchens_100_256p) supply the research data. Confirm permitted external models/data and their licenses against Samsung's rules before selecting this optional model for submission.

## Runner verification

`python -m unittest discover -s tests -v` passed all 11 tests, including isolation of separate clips during temporal shifts and merging segments at fractional video boundaries. `pip check` passed. `git diff --check` passed.

The actual `run.sh` command with the optional checkpoint produced 7 actions and 2.610 seconds of highlights from `work/epic-preview.mp4` (18 seconds), in 24.6 seconds reported by the runner. The existing provisional interval checker passed; it is not Samsung's validator. The default CLIP run produced 6 actions with identical highlights and category, in 1.3 seconds using existing caches. These runtimes are not a controlled cold-start comparison.

Batch inference on the same preview, using input and output paths containing spaces, produced exactly the same action list as the single runner. Python socket connection methods were patched to fail during that batch run, which completed successfully. This tests Python-level network access, not an operating-system firewall or Docker network isolation. The local FFmpeg process read local media only. The student kit and official validation remain pending.
