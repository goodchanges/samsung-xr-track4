# Samsung XR Challenge 2026 · Track 4

Solo prototype for action-aware video summarization. The runner reads a first-person MP4 and writes timestamped verb–noun predictions, a category, and chronological highlight intervals within 15% of the source duration. The optional browser demo plays the original footage and a highlight reel and lets you click timestamps to seek.

## Run on this Windows laptop

Open PowerShell in this folder.

    .\run.cmd "path\to\recording.mp4" "outputs\recording.json" --demo "work\recording-demo"
    .\.venv\Scripts\python.exe code/check_output.py "path\to\recording.mp4" "outputs\recording.json"

Open work/recording-demo/index.html in your browser. Keep the source and demo files in place so the page can find both videos. Omit the demo flag and its value for JSON only. The video ID comes from the filename without the extension.

## Set up a fresh checkout

Use Python 3.11–3.13. Initial setup downloads dependencies and approximately 600 MB of model weights. Inference then runs offline.

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe code/setup_model.py

On Linux, substitute .venv/bin/python for the Windows Python path, then run:

    ./run.sh recording.mp4 outputs/recording.json

## Method and limits

CLIP ViT-B/32 scores one frame per second against prompts for each action and category. Neighboring samples smooth the scores. An idle prompt and visual change gate weak action predictions. The selector rewards coverage of distinct predicted verb–noun pairs and discounts visually similar clips. Importance scores are ranking signals.

The provisional vocabulary lives in code/vocabulary.json. Pass --vocabulary PATH to use the same structure with a different label set. CLIP sees still images, so it can confuse similar motions, miss short actions, or predict the wrong verb for a visible object. Challenge accuracy is unmeasured. Use --threads N to tune CPU usage.

## Local verification

Open work/montage-demo/index.html for the prepared laptop demo. Its input joins seven single-view crops of the public EPIC-KITCHENS video wall into a 126-second montage. The run produced 48 predicted actions and selected 18.90 seconds in 26.2 seconds on the laptop. The JSON passed the local checker and FFmpeg decoded the rendered highlight without errors. A two-minute static clip with synthetic audio also passed and produced zero predicted actions. These runs check the pipeline; they do not measure challenge accuracy.

## Official submission handoff

The four top-level JSON keys follow the slides' example. The slides also mention six required keys. Use the student kit's validator and examples to resolve the final schema. Replace the provisional labels, map IDs through the manifest if needed, and run the official validator on all outputs. Package under the registered identifier. Include run.sh, code/, weights/, requirements.txt, outputs/ and this README. Keep the demo and deck separate as directed by the organizers. Git excludes large weights, local media and generated outputs.

[OpenAI CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32) · [EPIC-KITCHENS project](https://epic-kitchens.github.io/)

## Revision commands

The default remains the original highlight selector after a small development comparison. Use --selector duration for the experimental duration-aware selector. Use --unsmoothed to compare action localization. Model features cache under work/cache; cache keys include the checkpoint content, sampled video content and text embeddings.

Batch inference reuses the model:

    python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --selector duration

Local evaluation and process-memory measurement instructions are in docs/verification.md. The local metric definitions are provisional.

Build the CPU container after fetching local weights:

    docker build -t track4 .
    docker run --rm --network none -v /absolute/data:/data track4 /data/video.mp4 /data/video.json

For a fresh checkout without weights, build with --build-arg DOWNLOAD_WEIGHTS=1 while online, then run offline. GitHub Actions runs the clean-container smoke check through run.sh.

Create a prototype archive with the registered identifier:

    python code/package.py VelloreInstituteOfTechnologyVellore_VIT PATH_TO_NEW_ZIP

The archive includes local weights. Keep outputs/ reserved for official per-video outputs once the manifest arrives. Local fixture results belong under work/verification/.
