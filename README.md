# Samsung XR Challenge 2026 Â· Track 4

Solo prototype for action-aware video summarization. The runner reads a first-person MP4 and writes timestamped verbâ€“noun predictions, a category, and chronological highlight intervals within 15% of the source duration. The optional browser demo plays the original footage and a highlight reel and lets you click timestamps to seek.

## Run on this Windows laptop

Open PowerShell in this folder.

    .\run.cmd "path\to\recording.mp4" "outputs\recording.json" --demo "work\recording-demo"
    .\.venv\Scripts\python.exe code/check_output.py "path\to\recording.mp4" "outputs\recording.json"

Open work/recording-demo/index.html in your browser. Keep the source and demo files in place so the page can find both videos. Omit the demo flag and its value for JSON only. The video ID comes from the filename without the extension.

## Set up a fresh checkout

Use Python 3.13. Install the dependencies, then follow the Weights section before running inference.

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements.lock.txt

On Linux, substitute .venv/bin/python for the Windows Python path, then run:

    ./run.sh recording.mp4 outputs/recording.json

## Weights

From a fresh clone, run `python code/setup_model.py` with the project venv active to download the pinned CLIP checkpoint and its processor and tokenizer files into `weights/clip/`. The download is approximately 605 MB.

The checkpoint `weights/clip/pytorch_model.bin` is tracked with Git LFS. Once the LFS object is available on the clone's remote, `git lfs pull` can retrieve the checkpoint. Use the setup command to populate the full CLIP directory. Inference reads the local files and runs offline.

## Method and limits

CLIP ViT-B/32 scores one frame per second against 528 action-pair prompts and six provisional categories. Neighboring samples smooth the scores. An idle prompt and visual change gate weak action predictions. The selector rewards coverage of distinct predicted verbâ€“noun pairs and discounts visually similar clips. Importance scores are ranking signals.

The provisional vocabulary lives in code/vocabulary.json. Pass --vocabulary PATH to use the same structure with a different label set. CLIP sees still images, so it can confuse similar motions, miss short actions, or predict the wrong verb for a visible object. Challenge accuracy is unmeasured. Use --threads N to tune CPU usage.

## Data used

The fitted action adapter (`weights/action_adapter.npz`) records four source videos from the 256p mirror of the EPIC-KITCHENS-100 training split. The fitting input used the timestamped action rows for these IDs from `EPIC_100_train.csv`. Verb and noun class IDs were mapped to canonical keys using `EPIC_100_verb_classes.csv` and `EPIC_100_noun_classes.csv`.

| Fitting video | Annotations used |
| --- | --- |
| `P01_10.mp4` | [P01_10.json](docs/recognition/annotations/P01_10.json) |
| `P02_103.mp4` | [P02_103.json](docs/recognition/annotations/P02_103.json) |
| `P03_17.mp4` | [P03_17.json](docs/recognition/annotations/P03_17.json) |
| `P04_07.mp4` | [P04_07.json](docs/recognition/annotations/P04_07.json) |

The local held-out evaluation used `P07_05`, `P08_19`, and `P09_03`, with no video or participant overlap with adapter fitting. The separate development split used `P05_05` and `P06_104` to select settings. Overlap with the official Samsung evaluation videos is unknown until the student-kit manifest is available. The [split assignments](docs/recognition/splits.json) and [source revisions and hashes](docs/recognition/provenance.json) record the inputs.

## Local verification

Open `work/final-demo/index.html` for the prepared laptop demo. The 126-second input joins seven crops from the public EPIC-KITCHENS video wall. A cached run with the default selector produced 18.90 seconds of highlights and took 17.9 seconds including process startup. The synthetic 10-minute idle check produced zero predicted actions and 90 seconds of fallback footage. See [verification notes](docs/verification.md) for how these results were measured and what they establish.

## Official submission handoff

The four top-level JSON keys follow the slides' example. The slides also mention six required keys. Use the student kit's validator and examples to resolve the final schema. Replace the provisional labels, map IDs through the manifest if needed, and run the official validator on all outputs. Package under the registered identifier. Include run.sh, code/, weights/, requirements.txt, outputs/ and this README. Keep the demo and deck separate as directed by the organizers. Git excludes local media and generated outputs.

[OpenAI CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32) Â· [EPIC-KITCHENS project](https://epic-kitchens.github.io/)

## Revision commands

The default `--selector preserved` retains the original 29-pair scorer and highlight decisions while using revised recognition for action outputs. Both scorers share one CLIP model. Use `--selector duration` for the experimental duration-aware selector. Use `--unsmoothed` to compare action localization. Cached features live under `work/cache` and are keyed by the checkpoint content, sampled video content and text embeddings.

Batch inference reuses the model:

    python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --selector duration

Local evaluation and process-memory measurement instructions are in docs/verification.md. The local metric definitions are provisional.

Create a prototype archive with the registered identifier:

    python code/package.py VelloreInstituteOfTechnologyVellore_VIT PATH_TO_NEW_ZIP

The archive includes local weights. Keep outputs/ reserved for official per-video outputs once the manifest arrives. Local fixture results belong under work/verification/.

## Run with Docker

1. Set up `weights/clip/` first, or add `--build-arg DOWNLOAD_WEIGHTS=1` to the build to fetch the checkpoint.
2. Build with `docker build -t track4 .`.
3. Run with `docker run --rm --network none -v /absolute/data:/data track4 /data/video.mp4 /data/video.json`.

The current Dockerfile is untested. Docker is unavailable on the development machine.

## Measured recognition revision

On three local held-out videos, plain CLIP scored **1.38% macro action F1**, compared with **0.89% with the adapter**. Both runs used the same 528 action pairs and averaged F1 across temporal IoU thresholds of 0.1, 0.3, and 0.5. See the [local experiment and its limitations](docs/recognition/README.md).

The adapter is off by default (strength `0`). Pass `--adapter-strength 0.1` to opt in with the bundled adapter and default vocabulary. For a custom vocabulary, pass a compatible adapter with `--adapter PATH` and a positive strength.
