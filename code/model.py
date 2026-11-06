import json
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from media import SIZE

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "weights" / "clip"
MODEL_ID = "openai/clip-vit-base-patch32"


def load_supported_classes(path):
    """Read the kit document as JSON without assuming its class layout."""
    try:
        return path, json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None


def setup():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID, revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        local_dir=CHECKPOINT,
        allow_patterns=["*.json", "merges.txt", "pytorch_model.bin"],
    )


class Scorer:
    def __init__(self, vocabulary, threads, adapter=None, adapter_strength=0.0, shared=None):
        torch.set_num_threads(threads)
        if not np.isfinite(adapter_strength) or not 0 <= adapter_strength <= 1:
            raise ValueError("Adapter strength must be a finite number between zero and one.")
        vocab = json.loads(vocabulary.read_text(encoding="utf-8"))
        self.actions = vocab["actions"]
        self.categories = vocab["categories"]
        self.vocabulary_signature = json.dumps(vocab, sort_keys=True).encode("utf-8")
        self.adapter = None
        self.adapter_strength = adapter_strength
        if adapter is not None and adapter_strength > 0:
            with np.load(adapter, allow_pickle=False) as data:
                if str(data["vocabulary_signature"].item()) != self.vocabulary_signature.decode():
                    raise ValueError("Visual adapter was fitted for a different vocabulary.")
                self.adapter = (data["indices"], data["prototypes"])
            self.vocabulary_signature += adapter.read_bytes() + str(adapter_strength).encode()
        if not self.actions or not self.categories:
            raise ValueError("The vocabulary requires at least one action and category.")
        if not (CHECKPOINT / "pytorch_model.bin").is_file():
            raise ValueError("Model weights are missing. Run code/setup_model.py first.")
        if shared is None:
            self.processor = CLIPProcessor.from_pretrained(CHECKPOINT, use_fast=False, local_files_only=True)
            with (CHECKPOINT / "pytorch_model.bin").open("rb") as handle:
                self.checkpoint_hash = hashlib.file_digest(handle, "sha256").digest()
            self.model = CLIPModel.from_pretrained(CHECKPOINT, use_safetensors=False, local_files_only=True).eval()
        else:
            self.processor, self.model, self.checkpoint_hash = shared.processor, shared.model, shared.checkpoint_hash
        prompts = [
            prompt
            for item in self.actions
            for prompt in (
                f"First-person view of hands {item['description']}.",
                f"A close-up photo of someone {item['description']}.",
            )
        ]
        prompts += ["A first-person view with idle hands or no object interaction."]
        prompts += [f"First-person footage of {x['description']}." for x in self.categories]
        cache = ROOT / "work" / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        text_key = hashlib.sha256(self.checkpoint_hash + json.dumps(prompts).encode()).hexdigest()
        text_path = cache / ("text-" + text_key + ".npy")
        if text_path.exists():
            self.text = torch.from_numpy(np.load(text_path, allow_pickle=False))
            return
        text_vectors = []
        with torch.inference_mode():
            for offset in range(0, len(prompts), 32):
                tokens = self.processor(
                    text=prompts[offset:offset + 32], padding=True, truncation=True,
                    return_tensors="pt",
                )
                text_vectors.append(self.model.get_text_features(**tokens))
            self.text = torch.nn.functional.normalize(torch.cat(text_vectors), dim=-1)
        temporary = text_path.with_name(f".{text_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.save(handle, self.text.numpy(), allow_pickle=False)
            temporary.replace(text_path)
        finally:
            temporary.unlink(missing_ok=True)

    @torch.inference_mode()
    def predict(self, raw):
        key = hashlib.sha256(
            self.checkpoint_hash + self.vocabulary_signature + raw + self.text.numpy().tobytes()
        ).hexdigest()
        cache = ROOT / "work" / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (key + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                return saved["scores"], saved["idle"], saved["embeddings"], str(saved["label"].item())
        result = self._predict(raw)
        temporary = path.with_name(f"{key}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, scores=result[0], idle=result[1], embeddings=result[2], label=result[3], sample_timestamps_seconds=np.arange(len(result[0]), dtype=float))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return result

    @torch.inference_mode()
    def _predict(self, raw):
        # Reuse image features across vocabulary and prompt experiments.
        key = hashlib.sha256(b"clip-image-v1" + self.checkpoint_hash + raw).hexdigest()
        path = ROOT / "work" / "cache" / ("image-" + key + ".npy")
        if path.exists():
            embeddings = torch.from_numpy(np.load(path, allow_pickle=False))
        else:
            embeddings = self.encode_images(raw)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as handle:
                    np.save(handle, embeddings.numpy(), allow_pickle=False)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        scores = (embeddings @ self.text.T).numpy()
        stop = 2 * len(self.actions)
        action = scores[:, :stop].reshape(len(scores), -1, 2).mean(axis=-1)
        if self.adapter is not None:
            indices, prototypes = self.adapter
            similarities = embeddings.numpy() @ prototypes.T
            action[:, indices] += self.adapter_strength * (similarities - similarities.mean(axis=1, keepdims=True))
        idle = scores[:, stop]
        category = scores[:, stop + 1:].mean(axis=0).argmax()
        return action, idle, embeddings.numpy(), self.categories[category]["label"]

    @torch.inference_mode()
    def encode_images(self, raw):
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, SIZE, SIZE, 3)
        vectors = []
        for offset in range(0, len(pixels), 16):
            images = [Image.fromarray(x) for x in pixels[offset:offset + 16]]
            batch = self.processor(images=images, return_tensors="pt")
            vectors.append(self.model.get_image_features(**batch))
        return torch.nn.functional.normalize(torch.cat(vectors), dim=-1)
