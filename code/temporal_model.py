"""Offline EPIC TSM RGB inference (8 frames, ResNet50).

Architecture: Lin, Gan and Han, ICCV 2019, Temporal Shift Module.
Checkpoint: epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM (CC BY-NC 4.0).
No pretrained torchvision downloads and no training framework required.
"""
import json
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models import resnet50

from media import frames

CACHE_ROOT = Path(__file__).resolve().parents[1] / "work/cache/temporal"


class Shift(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, frames):
        sequence = frames.reshape(-1, 8, *frames.shape[1:])
        width = sequence.shape[2] // 8
        shifted = torch.zeros_like(sequence)
        shifted[:, :-1, :width] = sequence[:, 1:, :width]
        shifted[:, 1:, width:2 * width] = sequence[:, :-1, width:2 * width]
        shifted[:, :, 2 * width:] = sequence[:, :, 2 * width:]
        return self.net(shifted.reshape_as(frames))


class TemporalRecognizer(nn.Module):
    def __init__(self, checkpoint, threads=4, optimize=True):
        super().__init__()
        torch.set_num_threads(threads)
        self.base_model = resnet50(weights=None)
        for stage in (self.base_model.layer1, self.base_model.layer2,
                      self.base_model.layer3, self.base_model.layer4):
            for block in stage:
                block.conv1 = Shift(block.conv1)
        self.base_model.fc = nn.Dropout(0.7)
        self.new_fc = nn.Linear(2048, 397)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key.removeprefix("model."): value
                 for key, value in saved["state_dict"].items()}
        self.load_state_dict(state, strict=True)
        self.eval()
        if optimize:
            optimize_cpu(self)
        self.checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()

    @torch.inference_mode()
    def predict(self, clips):
        """Return verb/noun probabilities for B x 8 x 224 x 224 x RGB uint8."""
        values = torch.from_numpy(np.asarray(clips).copy()).float().permute(0, 1, 4, 2, 3) / 255
        mean = values.new_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
        # The released checkpoint explicitly uses std == mean.
        values = (values - mean) / mean
        logits = self.new_fc(self.base_model(values.flatten(0, 1)))
        logits = logits.reshape(-1, 8, 397).mean(dim=1)
        return logits[:, :97].softmax(-1).numpy(), logits[:, 97:].softmax(-1).numpy()

    def score_video(self, video, duration, context_seconds=4):
        """Cache probabilities independently of labels and localization settings."""
        if context_seconds not in (2, 4):
            raise ValueError("Temporal context must be 2 or 4 seconds")
        raw = frames(video, context_seconds / 8)
        signature = f"tsm-v1:{self.checkpoint_hash}:{duration}:{context_seconds}".encode()
        key = hashlib.sha256(signature + raw).hexdigest()
        cache = CACHE_ROOT / (key + ".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                return saved["verbs"], saved["nouns"]
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 224, 224, 3)
        count = int(np.ceil(duration))
        indices = np.floor((np.arange(count)[:, None] + .5) * 8 / context_seconds
                           + np.arange(8) - 3.5).astype(int)
        indices = np.clip(indices, 0, len(pixels) - 1)
        verb_rows, noun_rows = [], []
        for offset in range(0, count, 2):
            verbs, nouns = self.predict(pixels[indices[offset:offset + 2]])
            verb_rows.append(verbs)
            noun_rows.append(nouns)
        verbs, nouns = np.concatenate(verb_rows), np.concatenate(noun_rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, verbs=verbs, nouns=nouns)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
        return verbs, nouns

    def recognize(self, video, duration, classes_path, context_seconds=4, smoothing=1, top_k=1, decoder_path=None):
        """Decode cached clip scores; class names come from EPIC, not a kit guess."""
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
        for kind, count in (("verb", 97), ("noun", 300)):
            if len(classes[kind]) != count or not all(isinstance(x, str) and x for x in classes[kind]):
                raise ValueError(f"Expected {count} EPIC {kind} names")
        verbs, nouns = self.score_video(video, duration, context_seconds)
        if decoder_path is not None:
            if context_seconds != 4 or smoothing != 1 or top_k != 1:
                raise ValueError("The measured joint decoder requires four-second contexts, no smoothing and top-k 1")
            from joint_decoder import joint_segments, load_profile
            association, strength, switch_cost = load_profile(decoder_path)
            return joint_segments(verbs, nouns, classes, duration, association, strength, switch_cost)
        return decode(verbs, nouns, classes, duration, smoothing, top_k)


def smooth_probabilities(values, width):
    if width not in (1, 3, 5):
        raise ValueError("Smoothing width must be 1, 3 or 5")
    padded = np.pad(values, ((width // 2, width // 2), (0, 0)), mode="edge")
    return sum(padded[i:i + len(values)] for i in range(width)) / width


def optimize_cpu(model):
    """Fold fixed batch normalization into convolutions for equivalent inference."""
    net = model.base_model
    net.conv1 = fuse_conv_bn_eval(net.conv1, net.bn1)
    net.bn1 = nn.Identity()
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        for block in layer:
            block.conv1.net = fuse_conv_bn_eval(block.conv1.net, block.bn1)
            block.bn1 = nn.Identity()
            for number in (2, 3):
                conv, bn = f"conv{number}", f"bn{number}"
                setattr(block, conv, fuse_conv_bn_eval(getattr(block, conv), getattr(block, bn)))
                setattr(block, bn, nn.Identity())
            if block.downsample is not None:
                block.downsample[0] = fuse_conv_bn_eval(block.downsample[0], block.downsample[1])
                block.downsample[1] = nn.Identity()
    net.to(memory_format=torch.channels_last)


def decode(verbs, nouns, classes, duration, smoothing=1, top_k=1):
    """Optionally retain a close second hypothesis for overlapping actions."""
    if top_k not in (1, 2):
        raise ValueError("Temporal top-k must be 1 or 2")
    verbs, nouns = smooth_probabilities(verbs, smoothing), smooth_probabilities(nouns, smoothing)
    if top_k == 1:
        return segments(zip(verbs.argmax(1), nouns.argmax(1)), classes, duration)
    v_rank = np.argsort(-verbs, axis=1, kind="stable")[:, :2]
    n_rank = np.argsort(-nouns, axis=1, kind="stable")[:, :2]
    actions, latest = [], {}
    for index, (v, n) in enumerate(zip(v_rank, n_rank)):
        start, end = float(index), min(float(index + 1), duration)
        if start >= end:
            break
        pairs = [(int(v[0]), int(n[0]))]
        second = max(((int(v[1]), int(n[0])), (int(v[0]), int(n[1]))),
                     key=lambda pair: verbs[index, pair[0]] * nouns[index, pair[1]])
        if verbs[index, second[0]] * nouns[index, second[1]] >= .5 * verbs[index, v[0]] * nouns[index, n[0]]:
            pairs.append(second)
        for pair in pairs:
            previous = latest.get(pair)
            if previous is not None and previous["end"] == start:
                previous["end"] = end
            else:
                row = dict(verb=classes["verb"][pair[0]], noun=classes["noun"][pair[1]], start=start, end=end)
                actions.append(row)
                latest[pair] = row
    return sorted(actions, key=lambda row: (row["start"], row["end"], row["verb"], row["noun"]))


def segments(predictions, classes, duration):
    actions = []
    for index, (verb, noun) in enumerate(predictions):
        start, end = float(index), min(float(index + 1), duration)
        if start >= end:
            break
        verb, noun = classes["verb"][verb], classes["noun"][noun]
        if actions and actions[-1]["verb"] == verb and actions[-1]["noun"] == noun:
            actions[-1]["end"] = end
        else:
            actions.append(dict(verb=verb, noun=noun, start=start, end=end))
    return actions
