# Demo script (2 min 45 s)

**0:00–0:25.** Introduce VIT, Track 04, and the goal of shortening first-person recordings to useful moments.

**0:25–0:55.** From the repository root, show setup and the local checkpoint. The input path points to a separately sourced manifest video.

```sh
python -m pip install -r requirements.txt
git lfs pull
./run.sh work/fresh-comparison/P01_12.mp4 outputs/P01_12.json
```

**0:55–1:25.** Open the JSON and point out the team ID, predicted label, action segments, and highlight intervals. Show the local check.

```sh
python code/check_output.py work/fresh-comparison/P01_12.mp4 outputs/P01_12.json
```

**1:25–2:10.** Play the highlight reel from the original recording.

```sh
./run.sh work/fresh-comparison/P01_12.mp4 work/demo/P01_12.json --demo work/demo
```

Open work/demo/index.html and show two selected moments. Run inference before recording if needed for timing.

**2:10–2:45.** Show the batch and validation reports. State measured coverage of the manifest, explain the three short-video conflicts, and distinguish local checks from Samsung validation. Use source footage cleared for the intended audience before publishing the recording.
