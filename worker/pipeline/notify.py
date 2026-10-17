"""Optional webhook: push the finished job to the frontend's backend.

Payload is signed: header X-Turtle-Signature = hex HMAC-SHA256(WEBHOOK_SECRET, raw body).
The receiver must recompute it and reject mismatches.
"""
import hashlib
import hmac
import json
import time

import httpx

from . import config


def send(payload: dict) -> None:
    if not config.WEBHOOK_URL:
        return
    body = json.dumps(payload, separators=(",", ":"), default=str).encode()
    sig = hmac.new(config.WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    headers = {"Content-Type": "application/json", "X-Turtle-Signature": sig}
    for attempt in range(3):
        try:
            r = httpx.post(config.WEBHOOK_URL, content=body, headers=headers, timeout=10)
            if r.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2 ** attempt)
    # Delivery failed: the result is still in the DB, the frontend can poll it.
