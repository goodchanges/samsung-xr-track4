# Demo recording script · 2 min 45 s

**Before recording.** Work from a clean repository checkout or the prepared local copy. Activate the project venv before recording (Git Bash on Windows: `source .venv/Scripts/activate`). Set the terminal to 18 pt or larger. Arrange the terminal beside the source/reel page and keep the JSON open in an editor. Record a short microphone test. This source and its reel have no audio track, so narrate over playback. Use footage cleared for your intended audience.

**0:00–0:25 · Source.** Play a few seconds of P01_12. “VIT’s Track 04 system turns first-person recordings into short highlights. The input is 173.81 seconds. The goal is to find distinct moments of interaction while staying within the requested viewing budget.”

**0:25–0:55 · Inference.** Show the exact Git Bash command from the repo root.

```sh
./run.sh work/fresh-comparison/P01_12.mp4 work/final-2026/demo-recording/P01_12.json --demo work/final-2026/demo-recording
```

“The runner uses local CLIP weights. It samples frames, scores categories and action proposals, and selects intervals. If the run is slow, say ‘Model loading takes time on CPU. Here is the verified output from this same command’ and open the prepared files.”

**0:55–1:25 · JSON.** Open work/final-2026/demo-recording/P01_12.json. “The team ID is VIT. The predicted label is cooking. This file contains 67 proposed actions and six highlight segments. The action count is a model output, not a count of verified correct actions.”

**1:25–1:55 · Reel.** Open work/final-2026/demo-recording/index.html. Play the reel or jump to two selected moments. “The selected intervals total 24 seconds, or 13.81% of the original. The exported reel measures 23.99 seconds.”

**1:55–2:20 · Local validation.** Run the checker and show its exit status.

```sh
python code/check_output.py work/fresh-comparison/P01_12.mp4 work/final-2026/demo-recording/P01_12.json
echo $?
```

“This checks the format and timing against the guide. It is not Samsung’s official validator.”

**2:20–2:45 · Architecture and repo.** “FFmpeg decodes and samples the MP4. CLIP supplies visual scores, and the selector determines which intervals go into the JSON and reel. The repository contains the offline runner and instructions. The local checker found 82 valid outputs among 185 eligible videos; 103 are still absent. Samsung has yet to score this submission.”
