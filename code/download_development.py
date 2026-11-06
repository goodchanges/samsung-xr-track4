"""Download the pinned nine-video research subset; this is NOT Samsung's manifest."""
import argparse
import hashlib
import json
import urllib.request
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--provenance", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/recognition/provenance.json")
    args = parser.parse_args()
    provenance = args.provenance
    record = json.loads(provenance.read_text(encoding="utf-8"))
    for name, expected in record["files"].items():
        video = name.lower().endswith(".mp4")
        target = args.destination / ("videos" if video else "") / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected["sha256"]:
            continue
        if target.exists():
            raise ValueError(f"Existing file has different contents; refusing to overwrite {target}")
        if video:
            url = f"{record['mirror']}/resolve/{record['mirror_revision']}/videos/{Path(name).stem}.MP4"
        else:
            url = f"https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/{record['annotation_revision']}/{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != expected["sha256"] or temporary.stat().st_size != expected["bytes"]:
                raise ValueError(f"Source hash or size mismatch for {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}", flush=True)


if __name__ == "__main__":
    main()
