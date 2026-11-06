from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from package import check_assets


class PackageChecks(unittest.TestCase):
    def test_missing_processor_and_lfs_pointer_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "weights/clip"
            folder.mkdir(parents=True)
            (folder / "pytorch_model.bin").write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "Incomplete local CLIP"):
                check_assets(root)
            for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "vocab.json", "merges.txt"):
                (folder / name).touch()
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                check_assets(root)
