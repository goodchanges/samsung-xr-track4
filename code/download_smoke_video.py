"""Download a public real-world POV fixture, not challenge or training data."""
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

from media import ffmpeg

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://commons.wikimedia.org/wiki/File:Kaninlandsbanan_POV_2013.webm"
URL = "https://upload.wikimedia.org/wikipedia/commons/transcoded/0/05/Kaninlandsbanan_POV_2013.webm/Kaninlandsbanan_POV_2013.webm.240p.vp9.webm"


def main():
    folder = ROOT / "work/non-kitchen"
    folder.mkdir(parents=True, exist_ok=True)
    original, video = folder / "ride.webm", folder / "ride.mp4"
    if not original.exists():
        partial = original.with_suffix(".partial")
        try:
            with urlopen(Request(URL, headers={"User-Agent": "Track4Research/1.0"}), timeout=90) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            partial.replace(original)
        finally:
            partial.unlink(missing_ok=True)
    if not video.exists():
        subprocess.run(ffmpeg("-loglevel", "error", "-i", original, "-c:v", "libx264",
                              "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", video), check=True)
    record = dict(source_page=PAGE, download_url=URL, author="Albin Olsson",
                  license="CC BY-SA 3.0", license_url="https://creativecommons.org/licenses/by-sa/3.0/",
                  modifications="Wikimedia 240p transcode converted to H.264/AAC MP4; full duration retained",
                  use="Non-kitchen smoke test only; no training, tuning or action annotations",
                  source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                  mp4_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (folder / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Ready: {video}")


if __name__ == "__main__":
    main()
