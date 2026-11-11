# Stage 0 student kit audit

Inspected `Student package_track04 (1).zip` as supplied. SHA-256:
`ddf7ee8d2b8b9fb1363c1e72ae95da9144f64573cbca5a1b7a2378f41bbd0c09`.
Python's ZIP integrity check returned `None` (no bad members). The archive
contains `manifest.csv`, `supported_classes.json`, `TEAM_README.md`, and
two JSON examples at the archive's top level alongside those files. The guide
lists `validate.py` and an `examples/` directory, but neither is included.
The archive contains no raw videos.

| Required key | Type and constraints in the guide |
| --- | --- |
| `team_id` | Nonempty string, identical in all outputs |
| `video_id` | Nonempty string matching the manifest and the JSON filename stem |
| `duration_sec` | Finite positive number, within ±0.5 s of manifest duration |
| `label` | Object with a supported string `name` and numeric `score` in [0, 1]. The guide expects `cooking` for EPIC and a supported FPVSum category or `other` for FPVSum. Use video evidence for the prediction. |
| `actions` | Array of 0–2,000 objects with numeric `segment: [start, end]`, supported strings `verb` and `noun`, and numeric `score` in [0, 1]. May be empty for FPVSum. |
| `highlights` | Nonempty array of objects with numeric `segment: [start, end]` |

The guide uses seconds, up to two decimal places, and describes segment bounds
as `0 <= start < end <= duration_sec + 0.5`. Highlights must be sorted
chronologically with zero overlap. It requires at least one highlight of
four seconds and says shorter segments are warned about and ignored for
scoring. The opening paragraph says every highlight is at least four seconds.
It describes a target total of at most 15% with a warning and soft scoring
penalty for going over. Whether touching endpoints are accepted cannot be
confirmed without the validator. For `P03_26` (11.13 s), `P26_30` (15.28 s),
and `P28_23` (25.16 s), even the 15% budget is below four seconds. The two
rules cannot both be satisfied on those videos.

`supported_classes.json` is an object with strings `version` and `notes`
and string arrays `labels` (16), `verbs` (98), and `nouns` (301).
Version is `prism2026-theme01-v1`. The verb and noun lists use canonical
EPIC keys plus `other`. The category list uses `ski`, while the FPVSum
example emits `skiing`. Use the explicit class list for vocabulary.
Neither example ID (`P02_109`, `skiing_03`) appears in the manifest.

The manifest has 187 unique IDs in columns `video_id,duration_sec,dataset`.
There are 138 `epic_kitchens_100` entries (11.13–1,968.60 s) and 49
`fpvsum` entries (37.34–377.07 s). The longest video exceeds the guide's
typical 2–10 minute range.

The current runner writes four keys. It needs `team_id` and `duration_sec`,
a scored label object, scored actions with `segment` arrays, and highlights
with `segment` arrays in place of separate endpoints and `importance`.
The existing budget and segment handling do not establish the four-second
minimum. The empty failure output cannot meet the guide's nonempty highlight
requirement for an unreadable input with no known duration.

Compared exact manifest IDs with the adapter split record, the joint decoder
freeze record, and the public EPIC annotation CSVs already on disk:

| Source | Overlap with manifest |
| --- | ---: |
| Adapter fitting (4 videos) | 0 |
| Joint decoder fitting (495 videos) | 0 |
| Public EPIC training split (495 videos), the documented TSM training source | 0 |
| Public EPIC validation split (138 videos) | 138 |

The pretrained TSM authors used public validation for evaluation/model
selection. Our joint decoder settings used three manifest videos
(`P07_17`, `P09_08`, `P28_18`) for development. The earlier comparison
(`P04_28`, `P08_14`, `P18_10`) and later comparison
(`P01_12`, `P06_12`, `P11_22`) also overlap the manifest in full. These
scores are historical local measurements on manifest videos, so they cannot
serve as an independent submission benchmark. No more tuning on those IDs.
FFmpeg reported that the nine local copies match the corresponding manifest
durations to two decimal places.

Stage 1's schema work can use the guide, class list, and examples. Official
validation and any final decision on contradictory constraints require the
organizer's actual `validate.py` and clarification where it is silent.
