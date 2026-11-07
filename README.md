# Samsung XR Challenge 2026 Â· Track 4

Solo prototype for action-aware video summarization. The runner reads a first-person MP4 and writes timestamped verbâ€“noun predictions, a category, and chronological highlight intervals within 14.5% of the source duration. The optional browser demo plays the original footage and a highlight reel and lets you click timestamps to seek.

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

Open `work/final-demo/index.html` for the prepared laptop demo. The 126-second input joins seven crops from the public EPIC-KITCHENS video wall. Those historical runs predate the 14.5% safety margin and are not final-settings outputs. See [verification notes](docs/verification.md) for how these results were measured and what they establish.

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

Docker is unavailable on the development machine. See [readiness checks](docs/readiness.md) for the latest Linux CI evidence. For the optional temporal model, build with `--build-arg ENABLE_TEMPORAL=1`; combine it with `--build-arg DOWNLOAD_WEIGHTS=1` to fetch both checkpoints at build time. Runtime remains offline.

## Measured recognition revision

On three local held-out videos, plain CLIP scored **1.38% macro action F1**, compared with **0.89% with the adapter**. Both runs used the same 528 action pairs and averaged F1 across temporal IoU thresholds of 0.1, 0.3, and 0.5. See the [local experiment and its limitations](docs/recognition/README.md).

The adapter is off by default (strength `0`). Pass `--adapter-strength 0.1` to opt in with the bundled adapter and default vocabulary. For a custom vocabulary, pass a compatible adapter with `--adapter PATH` and a positive strength.

## Optional temporal recognizer

**Final safety revision:** the historical TSM results below were measured before the confidence gate. They are not accuracy measurements of the final gated pipeline. TSM now runs only when the existing CLIP `cooking` category exceeds every other category by a mean cosine-similarity margin of at least 0.03 and wins at least half of sampled frames. This fixed, uncalibrated safety rule was not tuned. Otherwise the runner uses generic CLIP action predictions and the temporal selector; no kitchen-only decoding occurs. Generic pair scores are normalized ranking signals, not calibrated probabilities. With no detected actions, the existing budget-safe fallback highlight is retained.

The subsequent [separate evaluation and coverage improvement](docs/recognition/refinement.md) found 9.69% local action F1 for four-second TSM versus 0.37% for CLIP on three new recordings. An optional `--selector temporal` profile raised local action coverage from 13.09% to 25.34%, with a recognition tradeoff described in that report. It uses two-second contexts and the same 14.5% budget. Neither recognizer is universal; the TSM model is kitchen-specific. Input filenames and annotations do not determine production predictions.

A pretrained eight-frame TSM recognizer improved local macro action F1 from **1.80% to 4.82%** on three different public EPIC validation recordings (79 annotated actions). It matched 11 actions at tIoU 0.1 versus CLIP's 4. This is a small diagnostic with low absolute accuracy, not an official score or evidence of a competition ranking. See [results, provenance, runtime and limitations](docs/recognition/temporal.md).

With the venv active, install `python -m pip install -r requirements-temporal.txt`, then run `python code/setup_temporal.py` once online. The latter downloads and checks the 97.6 MB checkpoint and EPIC class names in `weights/temporal/`. These are published model labels, not a guessed student-kit vocabulary.

```sh
./run.sh recording.mp4 outputs/recording.json --temporal-model weights/temporal/tsm_rgb.ckpt
python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --temporal-model weights/temporal/tsm_rgb.ckpt
```

Omit the flag to retain plain CLIP. By default the optional model replaces action predictions only; CLIP still chooses highlights and the video-level label. Add `--selector temporal` to use the tested temporal highlight selector as well. The four JSON keys and 14.5% budget are unchanged. No training is needed, and inference makes no model downloads. The Docker image supports optional temporal dependencies with `--build-arg ENABLE_TEMPORAL=1`. Samsung model/data permissions and exact vocabulary remain pending.

## Final verification round

Video-level labels are data-driven: normalized CLIP frame embeddings are compared with category-text embeddings, similarities are averaged over frames, and the highest-scoring category is selected. The existing `categories` list in `code/vocabulary.json` is the **provisional category list**, not Samsung's vocabulary. No categories were added in this round. Exact kit mapping remains pending.

Historical development results: four-second TSM action F1 4.82%, selected two-second candidate 6.52%; preserved highlight coverage 20.37%, temporal selector 24.07%. Reserved three-video results: CLIP F1 0.37%, four-second TSM 9.69%, two-second TSM 7.58%; preserved coverage 13.09%, two-second temporal selection 25.34%, uniform coverage 15.22%. The shorter context was rejected as the recognition default. These are small, kitchen-specific, three-video local splits, using local metrics—not Samsung scores. Human highlight agreement and real non-kitchen accuracy remain unmeasured. See `docs/recognition/refinement.md` for definitions and frozen-selection evidence. Do not infer final gated accuracy from these historical ungated results.

Actual final-settings edge tests all exited zero: 2-second clip, 600-second synthetic color video, no-audio clip, 160x120/12-fps clip, physically rotated 240x426 clip, and corrupt input. Readable cases stayed within 14.5%; corrupt input wrote empty actions/highlights and an empty label. Raw results and stderr are in [final-audit.json](docs/final-audit.json). The ten-minute synthetic run took 76.787 seconds wall time with a sampled peak summed process-tree RSS of 854.531 MiB (50-ms sampling, not unique physical memory). It used the generic route and is not a ten-minute TSM or real-world accuracy benchmark. At that stage no real non-kitchen recording was available and all 14 unit tests passed; see the readiness update for later checks.

`run.sh` and `run.cmd` contain inference/import failures, log the reason to stderr, write the current four-key fallback JSON and exit zero. Empty label means classification was unavailable. This passes the existing provisional format checker, but official vocabulary/schema validity cannot be guaranteed before the kit arrives. A killed process, missing Python interpreter, absent output argument, unwritable output path, or a request to overwrite the input cannot be guaranteed to produce JSON. Downstream users must inspect stderr and empty outputs rather than treating exit zero as inference success.

Exact PowerShell run commands, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-temporal.txt
.\.venv\Scripts\python.exe code/setup_temporal.py
& 'C:\Program Files\Git\bin\bash.exe' ./run.sh 'input.mp4' 'outputs/result.json' --temporal-model weights/temporal/tsm_rgb.ckpt --selector temporal
# Generic CLIP only:
& 'C:\Program Files\Git\bin\bash.exe' ./run.sh 'input.mp4' 'outputs/result.json'
# Tests:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Setup requires network access; inference uses local weights. No further tuning experiments were performed in this final round.

## Kit-independent readiness update

A real 210-second non-kitchen recording has now been tested. Routing selected the generic path and labeled it `outdoors`, but its three action predictions were unsupported by inspected source frames and its highlight was only six seconds. This does not establish non-kitchen accuracy. Full provenance, output and limitations are in [readiness.md](docs/readiness.md).

Batch continues after per-video inference failures and writes a status report outside the submission folder. Inspect `work/batch-report.json` or pass `--report PATH`; a failed batch exits 1 after writing each video's result. Package preparation checks all required local CLIP assets and rejects LFS pointer files. Student-kit schema, vocabulary, official outputs and official validation remain pending.

Both CPU images passed [Linux CI](https://github.com/goodchanges/samsung-xr-track4/actions/runs/42295192380) with inference networking disabled. The temporal image passed all 16 tests and directly executed the local TSM checkpoint; the default image passed 11 tests and skipped five optional tests. Both processed a 120-second synthetic video and a corrupt input. The end-to-end temporal-image fixture used the generic route.
