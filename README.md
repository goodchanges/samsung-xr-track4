# Samsung XR Challenge 2026 Â· Track 4

Solo prototype for action-aware video summarization. The runner reads a first-person MP4 and writes timestamped verbâ€“noun predictions, a category, and chronological highlight intervals within 15% of the source duration. The optional browser demo plays the original footage and a highlight reel and lets you click timestamps to seek.

## Run on this Windows laptop

Open PowerShell in this folder.

    .\run.cmd "path\to\recording.mp4" "outputs\recording.json" --demo "work\recording-demo"
    .\.venv\Scripts\python.exe code/check_output.py "path\to\recording.mp4" "outputs\recording.json"

Open work/recording-demo/index.html in your browser. Keep the source and demo files in place so the page can find both videos. Omit the demo flag and its value for JSON only. The video ID comes from the filename without the extension.

## Set up a fresh checkout

Use Python 3.13, which matches the verified container. Initial setup downloads dependencies and approximately 600 MB of model weights. Inference then runs offline.

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements.lock.txt
    .\.venv\Scripts\python.exe code/setup_model.py

On Linux, substitute .venv/bin/python for the Windows Python path, then run:

    ./run.sh recording.mp4 outputs/recording.json

## Method and limits

CLIP ViT-B/32 scores one frame per second against 528 action-pair prompts and six provisional categories. A small visual adapter fitted on four annotated EPIC training recordings adjusts action scores. Neighboring samples smooth the scores. An idle prompt and visual change gate weak action predictions. The selector rewards coverage of distinct predicted verbâ€“noun pairs and discounts visually similar clips. Importance scores are ranking signals.

The provisional vocabulary lives in code/vocabulary.json. Pass --vocabulary PATH to use the same structure with a different label set. CLIP sees still images, so it can confuse similar motions, miss short actions, or predict the wrong verb for a visible object. Challenge accuracy is unmeasured. Use --threads N to tune CPU usage.

## Local verification

Open `work/final-demo/index.html` for the prepared laptop demo. The 126-second input joins seven crops from the public EPIC-KITCHENS video wall. A cached run with the default selector produced 18.90 seconds of highlights and took 17.9 seconds including process startup. The synthetic 10-minute idle check produced zero predicted actions and 90 seconds of fallback footage. See [verification notes](docs/verification.md) for how these results were measured and what they establish.

## Official submission handoff

The four top-level JSON keys follow the slides' example. The slides also mention six required keys. Use the student kit's validator and examples to resolve the final schema. Replace the provisional labels, map IDs through the manifest if needed, and run the official validator on all outputs. Package under the registered identifier. Include run.sh, code/, weights/, requirements.txt, outputs/ and this README. Keep the demo and deck separate as directed by the organizers. Git excludes large weights, local media and generated outputs.

[OpenAI CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32) Â· [EPIC-KITCHENS project](https://epic-kitchens.github.io/)

## Revision commands

The default `--selector preserved` retains the original 29-pair scorer and highlight decisions while using revised recognition for action outputs. Both scorers share one CLIP model. Use `--selector duration` for the experimental duration-aware selector. Use `--unsmoothed` to compare action localization. Cached features live under `work/cache` and are keyed by the checkpoint content, sampled video content and text embeddings.

Batch inference reuses the model:

    python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --selector duration

Local evaluation and process-memory measurement instructions are in docs/verification.md. The local metric definitions are provisional.

Build the CPU container after fetching local weights:

    docker build -t track4 .
    docker run --rm --network none -v /absolute/data:/data track4 /data/video.mp4 /data/video.json

For a fresh checkout without weights, build with `--build-arg DOWNLOAD_WEIGHTS=1` while online, then run offline. The [Linux container check](https://github.com/goodchanges/samsung-xr-track4/actions/runs/42186160883) passed the regression tests and processed a generated two-minute MP4 through `run.sh` with networking disabled.

Create a prototype archive with the registered identifier:

    python code/package.py VelloreInstituteOfTechnologyVellore_VIT PATH_TO_NEW_ZIP

The archive includes local weights. Keep outputs/ reserved for official per-video outputs once the manifest arrives. Local fixture results belong under work/verification/.

## Measured recognition revision

See [the experiment and limitations](docs/recognition/README.md). On three held-out training-split participants, matched actions at tIoU 0.1 increased from 1/93 to 2/93; mean macro action F1 over tIoU 0.1, 0.3, 0.5 increased from 0.28% to 0.89%. Absolute accuracy remains low. These are local diagnostic metrics, not official scores. Nine real-video outputs passed the provisional checker. Official manifest outputs and validation remain blocked by the missing student kit and challenge data.

The checked-in `weights/action_adapter.npz` is part of the revised default. Custom vocabularies disable its automatic use; explicitly supplying an incompatible adapter fails. Use `--no-adapter` to disable it. Image and text caches allow experiments without repeating feature extraction. The manifest/organizer-validator command is documented in the experiment notes.
