"""Prepare a small reproducible EPIC training-split experiment, not challenge scoring.

Reads organizer-independent public annotations. Evaluation participants never
contribute vocabulary frequencies or model fitting data.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SPLITS = {
    "fit": ["P01_10", "P02_103", "P03_17", "P04_07"],
    "development": ["P05_05", "P06_104"],
    "evaluation": ["P07_05", "P08_19", "P09_03"],
}


def seconds(value):
    h, m, s = map(float, value.split(":"))
    return h * 3600 + m * 60 + s


def prepare(folder, destination, baseline):
    def read(name):
        with (folder / name).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = read("EPIC_100_train.csv")
    verbs = {r["id"]: r["key"] for r in read("EPIC_100_verb_classes.csv")}
    nouns = {r["id"]: r["key"] for r in read("EPIC_100_noun_classes.csv")}
    excluded = {v.split("_")[0] for split in ("development", "evaluation") for v in SPLITS[split]}
    counts = Counter((verbs[r["verb_class"]], nouns[r["noun_class"]])
                     for r in rows if r["participant_id"] not in excluded)
    gerunds = {"take": "taking", "put": "putting", "cut": "cutting", "leave": "leaving",
               "remove": "removing", "close": "closing", "shake": "shaking", "squeeze": "squeezing",
               "move": "moving", "stir": "stirring", "dry": "drying", "turn-on": "turning on",
               "turn-off": "turning off", "switch-on": "switching on", "switch-off": "switching off"}
    vocab = json.loads(baseline.read_text(encoding="utf-8"))
    actions = []
    for (verb, noun), _ in counts.most_common(512):
        gerund = gerunds.get(verb, (verb[:-1] if verb.endswith("e") else verb) + "ing")
        natural_noun = " ".join(reversed(noun.split(":"))).replace("-", " ")
        actions.append(dict(verb=verb, noun=noun, description=f"{gerund} a {natural_noun}"))
    # Preserve non-kitchen coverage of the baseline.
    pairs = {(x["verb"], x["noun"]) for x in actions}
    actions.extend(x for x in vocab["actions"] if (x["verb"], x["noun"]) not in pairs)
    vocab["actions"] = actions
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "vocabulary-epic.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    annotations = destination / "annotations"
    annotations.mkdir(exist_ok=True)
    for split, videos in SPLITS.items():
        for video_id in videos:
            actions = [dict(verb=verbs[r["verb_class"]], noun=nouns[r["noun_class"]],
                            start=seconds(r["start_timestamp"]), end=seconds(r["stop_timestamp"]))
                       for r in rows if r["video_id"] == video_id]
            if not actions:
                raise ValueError(f"Missing training annotations for {video_id}")
            document = dict(video_id=video_id, split=split, actions=actions,
                            source="EPIC_100_train.csv; official canonical verb/noun class keys",
                            highlights=None, label=None)
            (annotations / (video_id + ".json")).write_text(json.dumps(document, indent=2), encoding="utf-8")
    (destination / "splits.json").write_text(json.dumps(SPLITS, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).with_name("vocabulary-baseline.json"))
    args = parser.parse_args()
    prepare(args.metadata, args.destination, args.baseline)
