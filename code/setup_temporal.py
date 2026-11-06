"""Explicit online setup for the optional research TSM recognizer."""
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_URL = "https://www.dropbox.com/s/5yxnzubch7b6niu/tsm_rgb.ckpt?dl=1"
CHECKPOINT_SHA256 = "b56b6e7210c89e7dfc93daadafa90fda9d4ea73dfec7e9d3399240bca664139c"
ANNOTATIONS = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/"


def main():
    folder = ROOT / "weights/temporal"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "tsm_rgb.ckpt"
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".partial")
        try:
            with urlopen(CHECKPOINT_URL, timeout=90) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
                raise ValueError("Checkpoint checksum mismatch")
            temporary.replace(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Local checkpoint checksum mismatch")
    classes = {}
    for kind, count in (("verb", 97), ("noun", 300)):
        with urlopen(ANNOTATIONS + f"EPIC_100_{kind}_classes.csv", timeout=60) as response:
            rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))
        indexed = {int(row["id"]): row["key"] for row in rows}
        if set(indexed) != set(range(count)):
            raise ValueError("Unexpected EPIC class IDs")
        classes[kind] = [indexed[index] for index in range(count)]
    (folder / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    print(f"Ready: {checkpoint} ({checkpoint.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
