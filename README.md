# Samsung PRISM XR Hackathon 2026 · Track 04 · VIT

Action-Aware Video Summarization turns a first-person MP4 into timestamped
action proposals and short highlights cut from the original footage.

## Submission status

**82 of 185 eligible videos passed local validation** (35 EPIC-KITCHENS and
47 FPVSum). The other 103 eligible entries have no output; 102 have no
completed local source and P19_05 has an unfinished, resumable transfer.
P07_18 and P26_38 were recovered from earlier failed attempts and passed
source verification, full decode, inference and the local output check.
The two organizer exclusions are
Scuba_KD74Ywuy9ng and Scuba_Um4kqB8bQBg. Validation checks format and timing.
EPIC inputs include official originals verified by MD5 and pinned 256p copies
verified by the mirror's SHA-256 and manifest duration. Prediction accuracy
requires the organizer's evaluation.

The local final-review archive is `work/final-2026/VIT-82-public.zip`. Earlier
archives, including `VIT-82.zip`, `VIT-82-final.zip` and
`VIT-82-complete.zip`, and their output bytes were preserved.
The unpublished GitHub draft holds the older `VIT.zip`. The 82 locally valid
output JSONs are tracked under `outputs/`; raw videos and ZIPs remain outside
version control. See the
[form checklist](docs/submission/FORM_CHECKLIST.md) for the current handoff.

## Install and recover the checkpoint

Use Python 3.13. On Linux or macOS use `.venv/bin/python` in place of the
Windows Python path below.

~~~powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe code/setup_model.py
~~~

The tested setup command downloads the pinned OpenAI CLIP ViT-B/32 files from
Hugging Face and verifies the 605,247,071-byte checkpoint against SHA-256
`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
It reproduced the preserved P01_12 output in a fresh clone. Git LFS tracks
the weight, but fresh remote pulls timed out and left a 134-byte pointer.
If you already have a trusted VIT archive, use
`python code/setup_model.py --archive /path/to/VIT.zip` from an active venv;
this recovery path was also tested. Inference uses local files.

## Run and validate

Run from the repo root using Bash (Git Bash on Windows). Use the manifest
video ID as the output filename stem.

~~~sh
./run.sh input.mp4 outputs/VIDEO_ID.json
python code/check_output.py input.mp4 outputs/VIDEO_ID.json
~~~

Run the checker with the project venv active. On Windows you can also invoke
`.venv/Scripts/python.exe` directly. A corrupt or unreadable input exits nonzero
without a placeholder prediction.

To validate a supplied dataset manifest locally, use:

~~~sh
python code/check_output.py --dir outputs --manifest kit/manifest.csv --team-id VIT --report work/local-submission-check.json
~~~

The manifest comes from the organizer kit. On a partial set the command
reports the valid files and exits 1 because of missing outputs.
Samsung's kit omitted `validate.py`. These checks cover the guide's schema,
vocabulary, bounds, ordering and coverage of manifest IDs.

The batch runner reuses the loaded model and records successes, missing sources,
failures and exclusions. It accepts videos named `<video_id>.mp4` or a local
manifest copy with a `path` column.

~~~sh
python code/batch.py VIDEO_ROOT outputs --manifest kit/manifest.csv --available-only --resume --profile clip --report work/batch-report.json
~~~

## How the pipeline works

FFmpeg decodes the source and samples frames. Pretrained CLIP ViT-B/32 scores
them against supported action and category prompts. A scene confidence gate
limits cooking labels and suppresses unsupported kitchen action guesses.
The selector favors varied predicted interactions and discounts visually
repeated moments. The default package uses CLIP. The optional pretrained
kitchen-specific TSM path is outside the default package.

Each output holds exactly six fields: `team_id`, `video_id`, `duration_sec`,
`label`, `actions` and `highlights`. The label contains `name` and `score`;
actions contain `segment`, `verb`, `noun` and `score`; highlights contain
`segment`. Segment endpoints are in seconds, and output identifiers do not
choose the predictions.

The selector normally targets at most 14.5% of the source and uses highlights
of at least four seconds. The local checker permits a single four-second
highlight within the official 15% cap when the 14.5% target is too small.
For P03_26, P26_30 and P28_23, four seconds exceeds 15% and requires organizer
clarification. None of those three has a submitted output.

CLIP's sampled still frames can miss short actions and confuse motions or
verb–noun pairs. Scores are model signals. Historical recognition comparisons
are documented in the [jury review](docs/jury-review.md) and
[kit integration results](docs/kit-integration-results.md). Several historical
development and comparison recordings were later found in the manifest, so
those measurements are not a current held-out or Samsung score. Model and
data redistribution notes are in the [repository access guide](docs/submission/REPO_ACCESS.md).

## Demo and materials

The verified local demo uses the genuine P01_12 source under
`work/fresh-comparison/`. The source is not distributed with the repo.
The exact command appears in the [2:45 recording script](docs/submission/demo_script.md).
The corrected [nine-slide PPTX](docs/submission/VIT_Submission_82.pptx) and
[matching PDF](docs/submission/VIT_Submission_82.pdf) still need the registered
member details filled in before submission.
The JSON selects 24.00 seconds from a 173.81-second source, and FFmpeg exports
a 23.99-second reel. These are measurements from one clip.

See the [AI disclosure](docs/submission/AI_DISCLOSURE.md),
[deck outline](docs/submission/deck_outline.md), [technical defense](docs/submission/TECHNICAL_DEFENSE.md),
and [release status](docs/STATUS.md).
