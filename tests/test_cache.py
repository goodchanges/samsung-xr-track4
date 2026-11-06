import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import model


class CacheChecks(unittest.TestCase):
    def test_label_change_invalidates_cache(self):
        scorer = model.Scorer.__new__(model.Scorer)
        scorer.checkpoint_hash = b"checkpoint"
        scorer.text = torch.zeros((1, 2))
        with tempfile.TemporaryDirectory() as folder, patch.object(model, "ROOT", Path(folder)):
            for label in ("old", "new"):
                scorer.vocabulary_signature = json.dumps({"label": label}).encode()
                result = (np.zeros((1, 1)), np.zeros(1), np.zeros((1, 2)), label)
                with patch.object(scorer, "_predict", return_value=result) as predict:
                    self.assertEqual(scorer.predict(b"frame")[3], label)
                    self.assertEqual(scorer.predict(b"frame")[3], label)
                    self.assertEqual(predict.call_count, 1)


if __name__ == "__main__":
    unittest.main()
