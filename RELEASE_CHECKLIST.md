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

- [ ] Run manifest batch and organizer validator (when kit is available)
  - `python code/batch.py VIDEO_ROOT outputs --manifest KIT/manifest.csv --id-column video_id --path-column path --validator KIT/validate.py`

- [ ] Verify package readiness
  - `python code/package.py REGISTERED_IDENTIFIER PATH_TO_NEW_ZIP`

- [ ] Go/no-go gates
  - No failed tests
  - No perf guardrail breach
  - No empty fallback outputs for valid media
  - No output schema/budget violations
  - Organizer validator passes on the full manifest
