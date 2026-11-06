# Revision verification

## Measured status

The inference runner supports full-view letterboxing, cached features, selectable smoothing, and a duration-aware experimental selector. The original selector remains the default because the small development check favored it. Baseline code is preserved in code/selection_baseline.py and tag baseline-v1.

A 126-second montage produced 18.90 seconds of original-footage highlights. A measured cached default run completed in 17.9 seconds including Python startup, with approximately 591 MiB sampled peak combined RSS. This uses 50 ms sampling and may miss brief memory peaks. The main runner's own timer excludes import time, which explains its shorter reported time.

A generated 600-second static video produced zero predicted actions and 90 seconds of fallback footage. The fallback retains a central interval to meet the demonstration budget and has zero importance. Final idle-only behavior must follow Samsung's validator.

## Small development check

The coffee-preview fixture is an 18-second continuous crop from the top-left tile of the public EPIC-KITCHENS preview. Its published action subtitles and frame review supplied approximate timestamps. The subtitle band was removed before inference. The assistant selected the highlight reference. This is a diagnostic sample with approximate annotations, not an independent human-consensus benchmark.

Both smoothed and unsmoothed predictions had zero matched action recall at the local tIoU thresholds. The original selector captured 25% of distinct annotated pairs under the locally defined half-action-overlap criterion; the duration-aware selector captured 0%. Highlight F1 at a 10 Hz grid was 0.259 for the original, 0.074 for the duration-aware smoothed version, and 0.085 for uniform sampling. These definitions differ from the organizer's unreleased scoring details.

The comparison shares padded-frame features across all selectors. The preserved Git tag is available to rerun the complete original center-crop pipeline.

The vocabulary lacks some annotated actions. Motion-direction recognition and category prediction remain weak. A larger supported vocabulary and an action-specialized model need evaluation once the kit and permitted model rules are available.

## Reproduction

Run python code/evaluate.py VIDEO_DIR ANNOTATION_DIR REPORT.json --split development.
Annotations contain split, label, actions and highlights. Use separate continuous recordings with split=evaluation for final reporting. Do not tune on the final evaluation set.

Run python code/measure.py REPORT.json PYTHON_EXECUTABLE code/main.py VIDEO OUTPUT.json to measure whole-process wall time and sampled memory.

Run python -m unittest discover -s tests -v for bounds, deterministic selection, idle fallback, corrupt inputs, paths with spaces and peripheral-content preservation.

## Sources and remaining work

Source preview: https://epic-kitchens.github.io/static/videos/03x03_videoWall.mp4
Dataset project: https://epic-kitchens.github.io/
Model: https://huggingface.co/openai/clip-vit-base-patch32

An attempt to download the continuous P01_10 training video from the official Bristol server timed out. No independent full-length evaluation score is claimed.

The official kit, all 187 manifest outputs and official metric evaluation remain pending. The private GitHub repository needs organizer access before submission.
