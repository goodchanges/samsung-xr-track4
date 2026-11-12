# Available manifest run

Executed with the shipped CLIP profile. The nine source videos were found under work/fresh-comparison and work/temporal; their durations matched the official manifest within 0.5 seconds. Existing features were cached. These timings include per-video processing and output writing; the first includes shared-model initialization. They exclude Python import startup and do not establish cold runtime.

```sh
python code/batch.py work outputs --manifest work/manifest-run/inputs.csv --available-only --profile clip --report work/manifest-run/report.json
python code/check_output.py --dir outputs --manifest kit/manifest.csv --team-id VelloreInstituteOfTechnologyVellore_VIT --report work/manifest-run/validation.json
```

The local input index adds source paths to all 187 original manifest rows. It does not supply prediction labels or annotations.

| Video | Status | Seconds | Actions | Highlight seconds |
| --- | --- | ---: | ---: | ---: |
| P01_12 | ok | 3.055000 | 67 | 24.00 |
| P04_28 | ok | 1.398293 | 83 | 20.00 |
| P06_12 | ok | 1.592917 | 92 | 28.00 |
| P07_17 | ok | 1.103088 | 77 | 16.00 |
| P08_14 | ok | 1.085618 | 75 | 20.00 |
| P09_08 | ok | 0.726400 | 69 | 16.00 |
| P11_22 | ok | 1.925933 | 122 | 32.00 |
| P18_10 | ok | 1.236255 | 72 | 20.00 |
| P28_18 | ok | 1.289393 | 75 | 16.00 |

Batch result: 187 entries, 9 ok, 0 failed, 178 missing, 0 unresolved. Exit 1 indicates incomplete input availability.

Local checker result: 9 found, 9 checked, 9 valid, no errors or warnings, 178 missing. Exit 1 indicates incomplete submission coverage. The organizer validator remains unavailable.

All nine available videos are EPIC-KITCHENS entries. No FPVSum source video from the manifest was available in the workspace. No downloads or model fitting were performed.

The two prior optional-temporal outputs were preserved under work/manifest-run/previous-outputs before generating the default-profile outputs. Active submission JSONs are in outputs/. Full reports and logs remain under work/manifest-run; these generated files are locally ignored by Git.

Final regression run: 42 tests passed in 9.694 seconds. The new test verifies missing-input and duration-mismatch reporting while continuing to a valid video. Batch P01_12 JSON exactly matched the previous default two-argument runner output.
