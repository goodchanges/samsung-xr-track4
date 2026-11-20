"""Prepare the pinned CLIP checkpoint before running offline inference."""
import argparse
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

from model import CHECKPOINT, setup, verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Recover the checkpoint from a verified VIT.zip')
    args = parser.parse_args()
    if args.archive is None:
        setup()
    else:
        CHECKPOINT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CHECKPOINT) as folder:
            staged = Path(folder) / 'pytorch_model.bin'
            with ZipFile(args.archive) as archive:
                with archive.open('VIT/weights/clip/pytorch_model.bin') as source, staged.open('wb') as target:
                    shutil.copyfileobj(source, target)
            verify_checkpoint(staged)
            staged.replace(CHECKPOINT / staged.name)
    print(f"Saved model to {CHECKPOINT}")


if __name__ == "__main__":
    main()
