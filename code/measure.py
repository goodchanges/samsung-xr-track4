"""Measure child process elapsed time and sampled combined resident memory."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("Supply an executable and arguments after the report path.")
    started = time.perf_counter()
    process = subprocess.Popen(args.command)
    peak = 0
    root = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            family = [root, *root.children(recursive=True)]
            total = 0
            for member in family:
                try:
                    total += member.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, total)
        except psutil.Error:
            pass
        time.sleep(.05)
    result = dict(exit_code=process.returncode, elapsed_seconds=time.perf_counter()-started,
                  peak_sampled_rss_bytes=peak, sample_interval_seconds=.05)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
