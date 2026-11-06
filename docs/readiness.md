# Kit-independent readiness checks

These checks cover execution, portability and packaging. They do not provide official Samsung scores. No new model tuning or category changes were made.

## Real non-kitchen smoke test

Source: [Kaninlandsbanan POV 2013](https://commons.wikimedia.org/wiki/File:Kaninlandsbanan_POV_2013.webm), Albin Olsson, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The Wikimedia 240p derivative was converted to H.264/AAC MP4, retaining the full 210 seconds. It is a public first-person amusement-ride recording, unrelated to Samsung's evaluation manifest. The downloaded video and generated reel remain under `work/`, outside Git and the submission archive.

Source SHA-256: `a4ea97f377fc53de596c2d5f641f33182750feada19537f66be1e25eb11c4a9f`.
MP4 SHA-256: `1ba613592c6a15f37141d5a6835c6897700ef51b3ccc78ea26aedcc22f9e4042`.

Run:

```sh
python code/download_smoke_video.py
./run.sh work/non-kitchen/ride.mp4 work/non-kitchen/result.json --temporal-model weights/temporal/tsm_rgb.ckpt --selector temporal --demo work/non-kitchen/demo
```

Observed stderr:

```text
Kitchen gate: accepted=False margin=-0.0705 frame_fraction=0.000
Route: generic CLIP + temporal selector
3 actions · 6.00/210.00s selected (2.9%) · 15.8s elapsed
```

The output label was `outdoors`. The local interval checker passed. The predicted actions were `turn container` once and `set alarm` twice. Inspection of source frames at 167 and 180 seconds shows the ride and tracks, without support for those interactions. This is a routing/execution success and a recognition-quality failure, not evidence of universal recognition. Only six seconds were selected; the summary respects the maximum budget but undershoots the approximately 15% target. There are no independent action or human-importance annotations for this fixture, so no accuracy or highlight F1 is claimed.

## Batch failures and packaging

Batch retains one loaded CLIP model and loads TSM only after the kitchen gate accepts a video. A failed video gets the existing empty fallback JSON; later videos are still processed. A separate status report defaults to `work/batch-report.json`. Batch exits 1 when any video fails, and does not run the supplied organizer validator in that case. The single-video `run.sh` failure contract still exits 0 as requested.

The package command rejects missing processor/tokenizer files, small LFS pointers and incomplete optional temporal assets. It excludes temporary downloads and video media, includes full local weights, and writes executable LF-terminated `run.sh` into the ZIP. This produces a prototype archive only; kit-dependent submission compliance remains pending.

## Container verification

The CI workflow now tests default and optional-temporal CPU images separately, using `--network none` at inference. It checks nonempty labels/highlights for valid media so an empty failure fallback cannot count as a pass. The temporal image additionally loads the real local checkpoint and runs the model directly. Docker is unavailable on the laptop; actual Linux CI results will be recorded after the run completes.
