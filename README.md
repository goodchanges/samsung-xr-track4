# Samsung PRISM XR Hackathon 2026 | Track 04 | VIT

The submission uses pretrained CLIP frame embeddings to score 16 video labels,
detect supported action candidates, and select diverse highlights from original
footage. A CLIP scene confidence gate allows cooking predictions for kitchen
footage and suppresses kitchen action guesses on other footage. The optional
pretrained TSM profile runs only when the kitchen gate accepts the video.
The historically calibrated joint decoder is excluded from the submission profile.

See [release status](docs/STATUS.md) and [submission materials](docs/submission/deck_outline.md).
Timing and accuracy experiments elsewhere in this README are historical local
measurements, including earlier development recordings later found in the kit.
They do not measure this release's accuracy or establish a Samsung score.

## Student kit integration status

The supplied kit uses six output keys and 16 labels, 98 verbs, and 301 nouns.
The runtime loads code/supported_classes.json locally and predicts categories
from CLIP image features. Predictions never consult the manifest or annotations.
Scores are uncalibrated: CLIP category softmax on mean cosine similarities,
CLIP action cosine rescaled to [0,1], and TSM mean verb/noun probability products.

Highlights are rounded inward to centiseconds. Short proposals expand to four
seconds where they fit without overlap and within 14.5%; remaining proposals
that cannot fit are dropped. When no segment survives, the highest-ranked original
highlight supplies the start for one contiguous interval. For videos whose
budget is below four seconds, the temporary assumption prioritizes one four-second
segment and logs the budget conflict. Inputs shorter than four seconds cannot
meet the guide's minimum. Samsung clarification is required.

The supplied ZIP omits validate.py. Local checks are based on the guide and
cannot establish an official pass. Historical development and comparison videos
overlap the official manifest; those results are not independent submission
scores. Do not tune further on manifest videos. The joint decoder's parameters
were selected on three videos now known to be in the manifest. Organizer
permission for using that component remains unresolved; the default source
runner continues to use CLIP.

Check all available outputs against the complete manifest:

```sh
python code/check_output.py --dir outputs --manifest kit/manifest.csv --team-id VIT --report work/local-submission-check.json
```

This local check requires exactly six fields, supported labels/verbs/nouns,
finite numeric values, scores in [0, 1], valid two-decimal segment timestamps,
at most 2,000 actions, and nonempty sorted nonoverlapping highlights of at
least four seconds. It targets the project's 14.5% margin and permits a single four-second highlight within the official 15% when necessary.
It also checks filenames, manifest durations within 0.5 seconds, team-ID
consistency, missing and unexpected files, duplicate manifest IDs, and
duplicate JSON keys. Touching highlight endpoints are allowed.

Exit code 0 means all local checks passed; 1 means invalid or missing outputs;
2 means only unresolved short-video budget conflicts remain. A four-second
highlight that cannot fit even the guide's 15% budget is reported as unresolved
and never counted as a clean pass. The report is stored outside outputs.
The existing single-video command remains available. The command-line checker
requires the six-key format; legacy four-key support is retained only in the
internal checker used by historical tests. These checks do not measure accuracy
or replace Samsung's validator.

To process available manifest videos while recording unavailable entries:

```sh
python code/batch.py VIDEO_ROOT outputs --manifest kit/manifest.csv --available-only --resume --profile clip --report work/manifest-run/report.json
```

The batch expects VIDEO_ROOT/<video_id>.mp4, or accepts a manifest copy with
an explicit path column. It checks source duration against the manifest before
inference, shares the loaded model, and records elapsed seconds, source path,
profile, result status and failure reasons. The report is updated atomically
after each entry. Missing inputs produce report entries without placeholder
JSONs. Inference failures are recorded without generating a new JSON.
Output-write errors are recorded and later videos continue. Any missing,
failed or unresolved entry makes the batch exit 1 and skips official validation.
The strict default still rejects missing files before processing.

# Samsung XR Challenge 2026 Â· Track 4

Solo prototype for action-aware video summarization. The runner reads a first-person MP4 and writes timestamped verbâ€“noun predictions, a category, and chronological highlight intervals within 14.5% of the source duration. The optional browser demo plays the original footage and a highlight reel and lets you click timestamps to seek.

## Run on this Windows laptop

For the optional pretrained temporal profile, install `requirements-temporal.txt`, prepare the temporal weights with `python code/setup_temporal.py`, then run `./run.sh input.mp4 output.json --profile improved` (or `run.cmd` on Windows). Plain CLIP remains the source checkout default. The [strict mock jury review](docs/jury-review.md) records current weaknesses, error counts and the selector comparison.

Open PowerShell in this folder.

    .\run.cmd "path\to\recording.mp4" "outputs\recording.json" --demo "work\recording-demo"
    .\.venv\Scripts\python.exe code/check_output.py "path\to\recording.mp4" "outputs\recording.json"

Open work/recording-demo/index.html in your browser. Keep the source and demo files in place so the page can find both videos. Omit the demo flag and its value for JSON only. The JSON video ID comes from the output filename without the extension.

## Set up a fresh checkout

Use Python 3.13. Install the dependencies, then follow the Weights section before running inference.

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements.lock.txt

On Linux, substitute .venv/bin/python for the Windows Python path, then run:

    ./run.sh recording.mp4 outputs/recording.json

## Weights

From a fresh clone, run `git lfs pull` to retrieve the approximately 605 MB CLIP checkpoint. The pinned processor and tokenizer files are committed alongside the LFS path. If LFS is unavailable, run `python code/setup_model.py` with the project venv active.

The checkpoint `weights/clip/pytorch_model.bin` is tracked with Git LFS. Once the LFS object is available on the clone's remote, `git lfs pull` can retrieve the checkpoint. Inference reads the local files. The package includes the weight file and all CLIP metadata.

## Method and limits

CLIP ViT-B/32 scores one frame per second against 519 supported action-pair prompts and 16 student-kit categories. Neighboring samples smooth the scores. An idle prompt and visual change gate weak action predictions. The selector rewards coverage of distinct predicted verbâ€“noun pairs and discounts visually similar clips. Importance scores are ranking signals.

The provisional vocabulary lives in code/vocabulary.json. Pass --vocabulary PATH to use the same structure with a different label set. CLIP sees still images, so it can confuse similar motions, miss short actions, or predict the wrong verb for a visible object. Challenge accuracy is unmeasured. Use --threads N to tune CPU usage.

## Data used

The fitted action adapter (`weights/action_adapter.npz`) records four source videos from the 256p mirror of the EPIC-KITCHENS-100 training split. The fitting input used the timestamped action rows for these IDs from `EPIC_100_train.csv`. Verb and noun class IDs were mapped to canonical keys using `EPIC_100_verb_classes.csv` and `EPIC_100_noun_classes.csv`.

| Fitting video | Annotations used |
| --- | --- |
| `P01_10.mp4` | [P01_10.json](docs/recognition/annotations/P01_10.json) |
| `P02_103.mp4` | [P02_103.json](docs/recognition/annotations/P02_103.json) |
| `P03_17.mp4` | [P03_17.json](docs/recognition/annotations/P03_17.json) |
| `P04_07.mp4` | [P04_07.json](docs/recognition/annotations/P04_07.json) |

The local held-out evaluation used `P07_05`, `P08_19`, and `P09_03`, with no video or participant overlap with adapter fitting. The separate development split used `P05_05` and `P06_104` to select settings. None of the four adapter fitting videos appears in the supplied manifest. The temporal decoder settings were selected on three manifest videos, as described above. The [split assignments](docs/recognition/splits.json) and [source revisions and hashes](docs/recognition/provenance.json) record the inputs.

## Local verification

Open `work/final-demo/index.html` for the prepared laptop demo. The 126-second input joins seven crops from the public EPIC-KITCHENS video wall. Those historical runs predate the 14.5% safety margin and are not final-settings outputs. See [verification notes](docs/verification.md) for how these results were measured and what they establish.

## Official submission handoff

The student guide specifies six top-level JSON fields. The runner uses that format and the supplied class list. The organizer validator was absent from the ZIP, so the format has not been officially validated. Process every manifest video and run the official validator once supplied. Package under the registered identifier with run.sh, code/, weights/, requirements.txt, outputs/ and this README.

The November 12 organizer email sets the final submission date to 22 November 2026, the top 12 announcement to 29 November, final demo to 9 December, and results to 12 December. It asks for code, resources, documents, demo, presentation, package, and README in a public GitHub repository, with its link submitted through the organizer form: https://forms.gle/86xENrtqhx988hBt8. Review redistribution permissions before making materials public. The final presentation template asks for team details, problem, solution, architecture, track output, differentiation, demo and setup, and optional additional slides. The repository and materials still require a public-release and completeness review.

[OpenAI CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32) Â· [EPIC-KITCHENS project](https://epic-kitchens.github.io/)

## Revision commands

The default `--selector preserved` retains the original 29-pair scorer and highlight decisions while using revised recognition for action outputs. Both scorers share one CLIP model. Use `--selector duration` for the experimental duration-aware selector. Use `--unsmoothed` to compare action localization. Cached features live under `work/cache` and are keyed by the checkpoint content, sampled video content and text embeddings.

Batch inference reuses the model:

    python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --selector duration

Local evaluation and process-memory measurement instructions are in docs/verification.md. The local metric definitions are provisional.

Create a prototype archive with the registered identifier:

    python code/package.py VIT PATH_TO_NEW_ZIP

The archive includes CLIP weights and the available manifest outputs. Read docs/STATUS.md for coverage before submitting. Use a fresh archive filename after regenerating outputs.

To package the improved profile for the required two-argument command, add `--profile improved` to the package command. That archive includes an internal preset and all temporal dependencies in its root `requirements.txt`; after installing dependencies, `./run.sh video.mp4 out.json` activates the measured profile. The current profile uses the pretrained TSM checkpoint without the historically fitted joint decoder. Use `--profile clip` to opt out. All packages remain provisional until organizer validation is available.

The jury-review revision passed 36 local tests and both offline Linux Docker configurations. A new Windows venv installed only the improved archive's requirements, then its bare runner processed a 173.81-second recording in 157.84 seconds with an empty application cache. Its outputs matched the measured profile. Detailed evidence and remaining limitations are in [the review](docs/jury-review.md).

## Run with Docker

1. Set up `weights/clip/` first, or add `--build-arg DOWNLOAD_WEIGHTS=1` to the build to fetch the checkpoint.
2. Build with `docker build -t track4 .`.
3. Run with `docker run --rm --network none -v /absolute/data:/data track4 /data/video.mp4 /data/video.json`.

Docker is unavailable on the development machine. See [readiness checks](docs/readiness.md) for the latest Linux CI evidence. For the optional temporal model, build with `--build-arg ENABLE_TEMPORAL=1`; combine it with `--build-arg DOWNLOAD_WEIGHTS=1` to fetch both checkpoints at build time. Runtime remains offline.

## Measured recognition revision

On three local held-out videos, plain CLIP scored **1.38% macro action F1**, compared with **0.89% with the adapter**. Both runs used the same 528 action pairs and averaged F1 across temporal IoU thresholds of 0.1, 0.3, and 0.5. See the [local experiment and its limitations](docs/recognition/README.md).

The adapter is off by default (strength `0`). Pass `--adapter-strength 0.1` to opt in with the bundled adapter and default vocabulary. For a custom vocabulary, pass a compatible adapter with `--adapter PATH` and a positive strength.

## Optional temporal recognizer

The subsequent [scene-routing comparison](docs/recognition/scene-routing.md) used three fresh recordings (165 annotated actions). Original TSM scored **7.94%** local action F1; joint decoding scored **13.34%**. With explicit scene routing, the full pipeline scored **13.34% versus 0.51% with the old gate** on these same recordings. This is a different set from the historical results below. Both precision and recall improved on the fresh set. All 34 local tests and both Linux Docker jobs passed.

Enable the experimentally improved kitchen profile with:

```sh
./run.sh input.mp4 output.json --temporal-model weights/temporal/tsm_rgb.ckpt --temporal-context 4 --temporal-decoder weights/temporal_decoder.npz --kitchen-routing scene
```

Scene routing is optional because only one real non-kitchen negative has been checked. Its internal scene prompts do not change output categories or the pending student-kit schema.

A subsequent optional [joint decoder](docs/recognition/joint-decoder.md) improved the same three-recording backend comparison from **9.69% to 11.86% local action F1**. Enable it with `--temporal-decoder weights/temporal_decoder.npz --temporal-context 4` alongside `--temporal-model`. It uses training-only pair associations and temporal continuity; the kitchen gate remains active. This is a small reused local comparison, not final gated accuracy or official scoring.

**Final safety revision:** the historical TSM results below were measured before the confidence gate. They are not accuracy measurements of the final gated pipeline. TSM now runs only when the existing CLIP `cooking` category exceeds every other category by a mean cosine-similarity margin of at least 0.03 and wins at least half of sampled frames. This fixed, uncalibrated safety rule was not tuned. Otherwise the runner uses generic CLIP action predictions and the temporal selector; no kitchen-only decoding occurs. Generic pair scores are normalized ranking signals, not calibrated probabilities. With no detected actions, the existing budget-safe fallback highlight is retained.

The subsequent [separate evaluation and coverage improvement](docs/recognition/refinement.md) found 9.69% local action F1 for four-second TSM versus 0.37% for CLIP on three new recordings. An optional `--selector temporal` profile raised local action coverage from 13.09% to 25.34%, with a recognition tradeoff described in that report. It uses two-second contexts and the same 14.5% budget. Neither recognizer is universal; the TSM model is kitchen-specific. Input filenames and annotations do not determine production predictions.

A pretrained eight-frame TSM recognizer improved local macro action F1 from **1.80% to 4.82%** on three different public EPIC validation recordings (79 annotated actions). It matched 11 actions at tIoU 0.1 versus CLIP's 4. This is a small diagnostic with low absolute accuracy, not an official score or evidence of a competition ranking. See [results, provenance, runtime and limitations](docs/recognition/temporal.md).

With the venv active, install `python -m pip install -r requirements-temporal.txt`, then run `python code/setup_temporal.py` once online. The latter downloads and checks the 97.6 MB checkpoint and EPIC class names in `weights/temporal/`. These are published model labels, not a guessed student-kit vocabulary.

```sh
./run.sh recording.mp4 outputs/recording.json --temporal-model weights/temporal/tsm_rgb.ckpt
python code/batch.py VIDEO_DIRECTORY OUTPUT_DIRECTORY --temporal-model weights/temporal/tsm_rgb.ckpt
```

Omit the flag to retain plain CLIP. By default the optional model replaces action predictions only; CLIP still chooses highlights and the video-level label. Add `--selector temporal` to use the tested temporal highlight selector as well. The runner uses the current six-key JSON schema. No training is needed, and inference makes no model downloads. The Docker image supports optional temporal dependencies with `--build-arg ENABLE_TEMPORAL=1`. Samsung model/data permissions and exact vocabulary remain pending.

## Final verification round

Video-level labels are data-driven: normalized CLIP frame embeddings are compared with category-text embeddings, similarities are averaged over frames, and the highest-scoring category is selected. The existing `categories` list in `code/vocabulary.json` is the **provisional category list**, not Samsung's vocabulary. No categories were added in this round. Exact kit mapping remains pending.

Historical development results: four-second TSM action F1 4.82%, selected two-second candidate 6.52%; preserved highlight coverage 20.37%, temporal selector 24.07%. Reserved three-video results: CLIP F1 0.37%, four-second TSM 9.69%, two-second TSM 7.58%; preserved coverage 13.09%, two-second temporal selection 25.34%, uniform coverage 15.22%. The shorter context was rejected as the recognition default. These are small, kitchen-specific, three-video local splits, using local metrics—not Samsung scores. Human highlight agreement and real non-kitchen accuracy remain unmeasured. See `docs/recognition/refinement.md` for definitions and frozen-selection evidence. Do not infer final gated accuracy from these historical ungated results.

Actual final-settings edge tests all exited zero: 2-second clip, 600-second synthetic color video, no-audio clip, 160x120/12-fps clip, physically rotated 240x426 clip, and corrupt input. Readable cases stayed within 14.5%; corrupt input wrote empty actions/highlights and an empty label. Raw results and stderr are in [final-audit.json](docs/final-audit.json). The ten-minute synthetic run took 76.787 seconds wall time with a sampled peak summed process-tree RSS of 854.531 MiB (50-ms sampling, not unique physical memory). It used the generic route and is not a ten-minute TSM or real-world accuracy benchmark. At that stage no real non-kitchen recording was available and all 14 unit tests passed; see the readiness update for later checks.

The runners log inference failures to stderr and attempt a six-key fallback JSON. If duration cannot be read, the fallback has zero duration and no highlights, so it cannot satisfy the student guide. A killed process, missing interpreter, absent output argument, or unwritable destination can prevent fallback output. Check stderr and batch reports for failures.

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

The shared inference path is in `code/pipeline.py`, with unchanged default recognition settings and versioned constants in `code/config.py`. Route and completion logs are JSON lines on stderr. CI measures coverage specifically for the refactored pipeline, localization, normalization and selection modules; it does not claim whole-project coverage. Its performance guardrail validates actual output as well as runtime and sampled memory. See `RELEASE_CHECKLIST.md` for release commands.

A real 210-second non-kitchen recording has now been tested. Routing selected the generic path and labeled it `outdoors`, but its three action predictions were unsupported by inspected source frames and its highlight was only six seconds. This does not establish non-kitchen accuracy. Full provenance, output and limitations are in [readiness.md](docs/readiness.md).

Batch continues after per-video inference failures and writes a status report outside the submission folder. Inspect `work/batch-report.json` or pass `--report PATH`; a failed batch exits 1 after writing each video's result. Package preparation checks all required local CLIP assets and rejects LFS pointer files. Student-kit schema, vocabulary, official outputs and official validation remain pending.

Both CPU images passed [Linux CI](https://github.com/goodchanges/samsung-xr-track4/actions/runs/42295192380) with inference networking disabled. The temporal image passed all 16 tests and directly executed the local TSM checkpoint; the default image passed 11 tests and skipped five optional tests. Both processed a 120-second synthetic video and a corrupt input. The end-to-end temporal-image fixture used the generic route.
## Submission checkpoint recovery

A fresh clone can contain a 134-byte LFS pointer if the weight transfer stalls.
Try `git lfs pull` first. If it cannot fetch the checkpoint, set up the venv
from requirements.txt and run one of these setup commands:

```sh
python code/setup_model.py
python code/setup_model.py --archive /path/to/VIT.zip
```

The first command fetches the pinned Hugging Face revision. The second can
recover the weight from a trusted VIT archive without a network connection.
Both verify 605,247,071 bytes and SHA-256
a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f.
The runner needs neither an account nor an internet connection once the model
files are present.

If the organizers confirm a different JSON team ID, make and check a separate
copy. Change the runner's configured team ID before regenerating published
outputs so a judge rerun agrees with them.

```sh
python code/copy_team_id.py outputs work/relabelled REGISTERED_ID
python code/check_output.py --dir work/relabelled --manifest kit/manifest.csv --team-id REGISTERED_ID
```

The checker still exits nonzero when the manifest is incomplete.
