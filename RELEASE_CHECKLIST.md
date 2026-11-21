- [ ] Install dependencies and weights
  - `python -m pip install -r requirements.txt -c requirements.lock.txt`
  - `python code/setup_model.py`
  - Optional temporal path: `python -m pip install -r requirements-temporal.txt` and `python code/setup_temporal.py`

- [ ] Run unit and regression checks
  - `python -m unittest discover -s tests -v`
  - `python code/container_smoke.py`
  - Optional temporal smoke: `python code/container_smoke.py --temporal`

- [ ] Run performance guardrail
  - `python code/perf_guardrail.py --max-seconds 120 --max-rss-bytes 2200000000`

- [ ] Validate schema and bounds locally
  - `python code/check_output.py INPUT_VIDEO.mp4 OUTPUT.json`
  - `python code/check_output.py --dir outputs --manifest kit/manifest.csv --team-id VIT --report work/local-submission-check.json`
  - The current full-manifest command exits 1 because 103 eligible outputs are absent; 82 present files passed with zero errors or warnings. Do not call this official validation.

- [ ] Address unresolved organizer questions
  - The distributed kit has no `validate.py`. Ask the organizer how the three videos shorter than 26.67 seconds should satisfy both the four-second minimum and the 15% cap.
  - Confirm how partial manifest coverage is handled before claiming submission completeness.

- [ ] Verify package readiness
  - `python code/package.py REGISTERED_IDENTIFIER PATH_TO_NEW_ZIP`

- [ ] Go/no-go gates
  - No failed tests
  - No perf guardrail breach
  - No empty fallback outputs for valid media
  - No output schema/budget violations
  - Report partial coverage and the lack of an official validator honestly

- [ ] Fill the registered member placeholders in docs/submission/VIT_Submission_82.pptx, export the updated PDF, and record the narrated demo
- [ ] Review redistribution permissions for public release
- [ ] Put all required code, resources, documents, demo, presentation, package, outputs, and README in a public GitHub repository
- [ ] Submit the public repository link using the organizer form by 22 November 2026
