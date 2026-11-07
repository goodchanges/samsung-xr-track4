import json
import sys
from datetime import datetime, timezone

from config import CONFIG_VERSION


def log_event(event, **fields):
    payload = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        event=event,
        config_version=CONFIG_VERSION,
        **fields,
    )
    print(json.dumps(payload, allow_nan=False, sort_keys=True), file=sys.stderr, flush=True)
