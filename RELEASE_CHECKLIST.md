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
  - `python code/check_output.py --dir outputs --manifest kit/manifest.csv --team-id VelloreInstituteOfTechnologyVellore_VIT --report work/local-submission-check.json`
  - Require exit 0 and resolve all missing files, invalid outputs, and short-video warnings

- [ ] Run manifest batch and organizer validator (when kit is available)
  - `python code/batch.py VIDEO_ROOT outputs --manifest kit/manifest.csv --validator kit/validate.py`

- [ ] Verify package readiness
  - `python code/package.py REGISTERED_IDENTIFIER PATH_TO_NEW_ZIP`

- [ ] Go/no-go gates
  - No failed tests
  - No perf guardrail breach
  - No empty fallback outputs for valid media
  - No output schema/budget violations
  - Organizer validator passes on the full manifest

- [ ] Complete the organizer presentation template and demo
- [ ] Review redistribution permissions for public release
- [ ] Put all required code, resources, documents, demo, presentation, package, outputs, and README in a public GitHub repository
- [ ] Submit the public repository link using the organizer form by 22 November 2026
