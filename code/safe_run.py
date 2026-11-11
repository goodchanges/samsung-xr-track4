"""Contain inference failures without inventing observations about unreadable media."""
import json
import os
from pathlib import Path
import sys

from structured_log import log_event
from submission import failure_submission


def main():
    try:
        from main import main as infer
        infer()
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return
        print(f"Track 4 failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        if len(sys.argv) < 3:
            print("No output path was supplied; cannot write fallback JSON.", file=sys.stderr)
            return
        source, target = Path(sys.argv[1]), Path(sys.argv[2])
        try:
            if source.resolve() == target.resolve():
                raise ValueError("Refusing to overwrite the input file")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            duration = 0.0
            try:
                from media import probe
                duration, _ = probe(source)
            except Exception:
                print("Duration unavailable: fallback JSON cannot satisfy the official positive-duration and highlight rules.", file=sys.stderr)
            temporary.write_text(json.dumps(failure_submission(source.stem, duration)) + "\n", encoding="utf-8")
            temporary.replace(target)
            log_event("fallback_json_written", output=str(target))
        except Exception as write_error:
            print(f"Cannot write fallback JSON: {write_error}", file=sys.stderr)


if __name__ == "__main__":
    main()
