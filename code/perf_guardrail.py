"""Lightweight runtime and memory guardrail for CI."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from media import ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--max-rss-bytes", type=int, default=2_200_000_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "work/perf-guardrail"
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "perf.mp4"
    output = folder / "perf.json"
    report = folder / "perf-report.json"
    subprocess.run(
        ffmpeg(
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=160x120:r=12",
            "-t",
            "30",
            "-c:v",
            "libx264",
            video,
        ),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "code/measure.py"),
            str(report),
            str(root / "run.sh"),
            str(video),
            str(output),
        ],
        check=True,
    )
    measured = json.loads(report.read_text(encoding="utf-8"))
    if measured["exit_code"] != 0:
        raise SystemExit("Guardrail failed: inference exited non-zero.")
    if measured["elapsed_seconds"] > args.max_seconds:
        raise SystemExit(
            f"Guardrail failed: elapsed {measured['elapsed_seconds']:.2f}s > {args.max_seconds:.2f}s."
        )
    if measured["peak_sampled_rss_bytes"] > args.max_rss_bytes:
        raise SystemExit(
            f"Guardrail failed: RSS {measured['peak_sampled_rss_bytes']} > {args.max_rss_bytes}."
        )
    print(json.dumps(dict(status="ok", elapsed_seconds=measured["elapsed_seconds"], peak_sampled_rss_bytes=measured["peak_sampled_rss_bytes"])))


if __name__ == "__main__":
    main()
