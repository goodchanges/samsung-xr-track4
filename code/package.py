import argparse
import re
import tempfile
from pathlib import Path
from stat import S_IFREG
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def check_assets(root, profile='clip'):
    clip = root / "weights/clip"
    required = ("pytorch_model.bin", "config.json", "preprocessor_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
    missing = [name for name in required if not (clip / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete local CLIP assets: {missing}. Run code/setup_model.py first.")
    if (clip / "pytorch_model.bin").stat().st_size < 100_000_000:
        raise ValueError("CLIP checkpoint is too small; retrieve the full weights, not an LFS pointer.")
    temporal = root / "weights/temporal"
    if profile == 'improved' and (not (temporal/'tsm_rgb.ckpt').is_file()
                                or not (root/'weights/temporal_decoder.npz').is_file()):
        raise ValueError('Improved profile requires the local TSM checkpoint and joint decoder.')
    if temporal.exists() and any(temporal.iterdir()):
        if not (temporal / "classes.json").is_file() or not (temporal / "tsm_rgb.ckpt").is_file():
            raise ValueError("Incomplete optional temporal assets; run code/setup_temporal.py first.")
        if (temporal / 'tsm_rgb.ckpt').stat().st_size < 90_000_000:
            raise ValueError('Temporal checkpoint is too small; retrieve the full weights.')


def main():
    parser = argparse.ArgumentParser(description="Package local code, weights and outputs.")
    parser.add_argument("identifier", help="Actual registered college/team identifier")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--profile", choices=('clip','improved'), default='clip', help='Preset used by the packaged two-argument runner.')
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.identifier):
        parser.error("Use letters, digits, underscores and hyphens in the registered identifier.")
    try:
        check_assets(ROOT, args.profile)
    except ValueError as exc:
        parser.error(str(exc))
    if args.destination.exists():
        parser.error("Destination already exists. Choose a new file.")
    paths = [ROOT / name for name in ("run.sh", "run.cmd", "requirements.txt", "requirements-temporal.txt", "requirements.lock.txt", "README.md", "RELEASE_CHECKLIST.md", "Dockerfile", ".dockerignore")]
    for directory in ("code", "weights", "outputs", "tests", "docs"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(ROOT).parts)
                     and p.suffix.lower() not in (".pyc", ".tmp", ".part", ".partial", ".mp4", ".webm", ".mov", ".avi"))
    paths = [p for p in paths if p != ROOT/'code/inference_profile.txt']
    if args.profile == 'improved':
        paths = [p for p in paths if p != ROOT/'requirements.txt']
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.destination.parent, prefix=args.destination.stem + ".", suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(args.identifier + "/outputs/", "")
            archive.writestr(args.identifier + '/code/inference_profile.txt', args.profile + '\n')
            if args.profile == 'improved':
                extra = '\n'.join(line for line in (ROOT/'requirements-temporal.txt').read_text().splitlines()
                                  if not line.strip().startswith('-r '))
                archive.writestr(args.identifier+'/requirements.txt', (ROOT/'requirements.txt').read_text()+'\n'+extra+'\n')
            for path in paths:
                name = args.identifier + "/" + path.relative_to(ROOT).as_posix()
                if path == ROOT / "run.sh":
                    entry = ZipInfo.from_file(path, name)
                    entry.create_system = 3  # Store Unix file permissions even on Windows.
                    entry.external_attr = (S_IFREG | 0o755) << 16
                    archive.writestr(entry, path.read_bytes().replace(b"\r\n", b"\n"), compress_type=ZIP_DEFLATED)
                else:
                    archive.write(path, name)
        temporary.replace(args.destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created {args.destination}. Run Samsung's validator before final submission.")


if __name__ == "__main__":
    main()
