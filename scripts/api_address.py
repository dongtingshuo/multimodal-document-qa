"""Resolve the supervisor's actual port; explicit CLI address takes precedence."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docqa.config import Settings


def api_address(explicit=None):
    if explicit:
        return explicit.rstrip("/")
    lock = Settings().data_dir / "server.lock"
    try:
        base = json.loads(lock.read_text())["api_url"]
    except (OSError, ValueError, KeyError):
        base = os.getenv("DOCQA_API_URL", "http://127.0.0.1:8000")
    return base.rstrip("/") + "/api/v1"
