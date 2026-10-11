"""Strict metadata filters shared by keyword and vector retrieval."""
from __future__ import annotations

import re


def matches_scope(chunk: dict, vehicle_filter: dict | None = None,
                  source_types: list[str] | None = None) -> bool:
    if source_types is not None and chunk.get("source_type") not in source_types:
        return False
    vehicle = chunk.get("vehicle") or {}
    for key, wanted in (vehicle_filter or {}).items():
        if wanted is None:
            continue
        have = vehicle.get(key, chunk.get(f"vehicle_{key}"))
        normalize = lambda value: re.sub(r"[^a-z0-9]", "", str(value).casefold())
        if have is None or normalize(wanted) != normalize(have):
            return False
    return True
