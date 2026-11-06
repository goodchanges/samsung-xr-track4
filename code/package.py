import argparse
import re
import tempfile
from pathlib import Path
from stat import S_IFREG
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Package local code, weights and outputs.")
    parser.add_argument("identifier", help="Actual registered college/team identifier")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.identifier):
        parser.error("Use letters, digits, underscores and hyphens in the registered identifier.")
    if not (ROOT / "weights/clip/pytorch_model.bin").is_file():
        parser.error("Model weights are missing.")
    if args.destination.exists():
        parser.error("Destination already exists. Choose a new file.")
    paths = [ROOT / name for name in ("run.sh", "run.cmd", "requirements.txt", "requirements.lock.txt", "README.md", "Dockerfile", ".dockerignore")]
    for directory in ("code", "weights", "outputs", "tests", "docs"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(ROOT).parts)
                     and p.suffix not in (".pyc", ".tmp", ".part"))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.destination.parent, prefix=args.destination.stem + ".", suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(args.identifier + "/outputs/", "")
            for path in paths:
                name = args.identifier + "/" + path.relative_to(ROOT).as_posix()
                if path == ROOT / "run.sh":
                    entry = ZipInfo.from_file(path, name)
                    entry.create_system = 3  # Store Unix file permissions even on Windows.
                    entry.external_attr = (S_IFREG | 0o755) << 16
                    archive.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED)
                else:
                    archive.write(path, name)
        temporary.replace(args.destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created {args.destination}. Run Samsung's validator before final submission.")


if __name__ == "__main__":
    main()
