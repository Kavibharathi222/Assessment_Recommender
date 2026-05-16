from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/"
    "shl_product_catalog.json"
)
DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "shl_product_catalog.json"


def load_catalog() -> list[dict[str, Any]]:
    """Load the SHL catalog from local cache, then remote URL if needed."""
    if CATALOG_PATH.exists():
        return _read_catalog(CATALOG_PATH)

    url = os.getenv("SHL_CATALOG_URL", DEFAULT_CATALOG_URL)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    products = json.loads(response.text, strict=False)

    DATA_DIR.mkdir(exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(products, indent=2), encoding="utf-8")
    return products


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))
