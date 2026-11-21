# Strict mock Track 4 review

> Historical pre-kit mock review. Its rating, four-field/pending-kit notes and improved-profile archive do not describe the current default submission. Use [README](../README.md) and [technical defense](submission/TECHNICAL_DEFENSE.md) for current status; Samsung has issued no official score.

This is an independent technical assessment, not Samsung's opinion or an official score. Subjective prototype rating: **5/10**. Engineering is credible; the evidence does not yet support a finalist-quality action-understanding claim. Official challenge points cannot be estimated from the local metrics.

## Findings and improvement scope

| Priority | Finding | Evidence and implication | Action |
| --- | --- | --- | --- |
| High | Semantic recognition remains weak | The improved profile reaches 13.34% local F1 on three comparison recordings. At tIoU 0.1, only 33/165 annotated actions match. | Added a reproducible error audit. Future model/data work must target verb/noun errors rather than only timing adjustments. |
| High | Required bare command did not expose the measured profile | Previously the improved result required several additional flags. A judge running only the required two arguments would see CLIP. | Added an explicit deployment preset and an improved archive whose two-argument runner activates it. Plain CLIP remains the repository default and can be requested explicitly. |
| High | The best recognizer did not imply the best summary | Current scene/joint recognition with temporal selection regresses development coverage. The challenge assigns 40% to coverage. | Compared preserved, temporal and uniform selection at the same budget. Retained the development-selected preserved selector. |
| Medium | Optional demo failure discarded valid inference | Rendering ran before JSON was saved; encoder errors triggered empty fallback output. | Save inference JSON first; log demo failures without discarding valid predictions. Added regression coverage. |
| High, pending kit | Official compliance and accuracy are unproven | No organizer manifest, schema examples, validator or official scoring data is available locally. | Keep four output keys unchanged, generate official outputs and validate once the kit arrives. |
| High, research gap | Broader action/importance/category evaluation is missing | Kitchen recordings dominate; only one real non-kitchen negative has been tested. No FPVSum or independently human-rated highlight score exists. | Keep scene routing optional outside the explicitly chosen improved archive. Obtain additional non-kitchen videos and independent importance/category annotations before claiming generalization. |

## Recognition diagnosis

The descriptive audit follows the existing greedy pair matcher at tIoU 0.1 on the three reused comparison videos. Every GT action belongs to one bucket; this is a diagnostic, not an official metric.

| Bucket | Annotated actions |
| --- | ---: |
| Matched | 33 |
| Both verb and noun absent among temporally overlapping predictions | 63 |
| Correct noun present, correct verb missing | 32 |
| Correct verb present, correct noun missing | 26 |
| No prediction reaches tIoU 0.1 | 10 |
| Duplicate/competing annotated instance | 1 |
| Total | 165 |

Thus 121 of 132 unmatched annotations have overlapping predictions but the wrong semantic pair. Timing is still coarse and important, but this diagnostic does not support treating timing-only tuning as the main remedy. Examples and exact buckets are in `recognition/jury-recognition-errors.json`; reproduce with `python code/error_review.py`.

## Summary selection comparison

Same scene routing, joint decoder, four-second context, classes and maximum 14.5% duration budget. Coverage is the existing local distinct-pair metric: at least half of a GT action instance must be retained. It is not Samsung's official 40-point score.

| Selector | Development: P28_18, P09_08, P07_17 | Reused comparison: P01_12, P06_12, P11_22 |
| --- | ---: | ---: |
| Preserved | **20.37%** | 21.85% |
| Temporal | 15.97% | **24.99%** |
| Uniform | 10.19% | 18.01% |

The preserved selector was selected and frozen using development coverage before inspecting comparison coverage. The temporal selector wins on the comparison set but was not adopted after seeing that result. Three-video selector rankings are unstable. This round adds selection evidence and deployment correctness, not a new measured recognition F1 improvement. Human highlight F1 remains unmeasured.

Reproduce using `python code/judge_selector_review.py development` then `python code/judge_selector_review.py comparison` in a clean work directory. Existing freeze/comparison files prevent overwriting the run. The recognition scores are cached; metrics and selection are recomputed from the actual retained intervals. Reports and the freeze are in `docs/recognition/jury-selector-*.json`.

## Deployment

Source checkout, after installing optional dependencies and preparing local weights:

```sh
./run.sh input.mp4 output.json --profile improved
python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --profile improved
python code/package.py VelloreInstituteOfTechnologyVellore_VIT work/releases/track4-improved-prototype.zip --profile improved
```

The improved archive stores a small internal preset file in `code/` and includes torchvision in its root `requirements.txt`. It contains all local model assets and uses the scene gate, joint decoder, four-second contexts and preserved selector. After dependency installation, its required invocation is simply `./run.sh input.mp4 output.json`. Add `--profile clip` to opt out. The preset changes no submission JSON keys or class vocabulary. This is a prototype package, not an officially validated submission.

The archive excludes local video fixtures, caches and virtual environments. Missing temporal assets are rejected before packaging. Inference is offline; setup and downloading source fixtures require network access.

## Verification of implemented fixes

- All 36 local tests passed, including packaged-preset resolution and preserving valid JSON after demo errors.
- [Linux CI 42350434158](https://github.com/goodchanges/samsung-xr-track4/actions/runs/42350434158) passed both CPU images on code revision `b3d8780`. The temporal container exercised the improved preset through the bare two-argument runner with networking disabled. It also directly executed the actual TSM checkpoint and joint decoder. Docker remains unavailable on the laptop itself.
- The 502,995,556-byte improved archive passed ZIP CRC, layout, full-checkpoint and dependency checks. It contains 127 entries, including the internal improved preset, full 605,247,071-byte CLIP checkpoint, full 97,616,687-byte TSM checkpoint, joint decoder and torchvision in root requirements. It excludes venvs, caches and sample videos.
- The archive was extracted to a separate folder, a new Python 3.13 venv was created, and only its `requirements.txt` was installed. `pip check` returned `No broken requirements found.` The exact resolved environment is in `recognition/jury-clean-environment.txt`.
- The extracted package's `./run.sh <video.mp4> <out.json>` processed the entire 173.81-second P01_12 recording with an empty application cache, selected kitchen TSM, and wrote 54 actions plus 25.202 seconds of highlights. Actions, highlights and category exactly matched the previously evaluated profile. The provisional interval checker passed. Wall time including startup was **157.84 seconds**, with sampled peak summed process-tree RSS of **1,348,739,072 bytes** (50 ms sampling, not unique physical memory). This is one cold-cache Windows run, not a ten-minute or general performance claim. See `recognition/jury-package-verification.json` for archive SHA-256 and raw runtime.
- An actual runner invocation with an unusable demo destination logged `demo_failed`, retained its nonempty inference JSON and passed the interval checker. The simulated encoder-error regression test also passed.

The archive's code is the verified revision above; this report adds the verification record afterward. The package is provisional because no organizer validator is available. No new recognition-F1 improvement is claimed for this engineering revision.

## Remaining research and submission work

1. Validate the permitted model/data rules and adopt the actual kit's schema, labels, IDs and timing constraints.
2. Evaluate recognition and action coverage on a larger fixed set before any additional tuning; require precision/recall and short-action breakdowns.
3. Add representative non-kitchen and mixed-domain evaluation, FPVSum category labels and independently annotated highlight importance.
4. Refresh the deck and narrated demo using the final profile, data disclosure, official results when available, and limitations. A working browser reel alone does not establish a finished submission video.

No competition percentile, internship outcome, universal-video accuracy or official score is established by this prototype.
