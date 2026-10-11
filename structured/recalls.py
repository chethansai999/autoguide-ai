"""Recalls lookup tool for resolved vehicles.

Order of sources:
  1. recall_cache table (fresh within RECALLS_CACHE_TTL_HOURS)
  2. live NHTSA Recalls API  GET {NHTSA_API_BASE}/recalls/recallsByVehicle
  3. local documents written by ingestion/api_collector.py + api_normalizer.py

Recall answers are informational: the UI always adds the VIN-check
disclaimer, because model-level recalls don't prove a specific car is
affected (proposal ethics: outdated / over-general recall info).
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from sqlalchemy.orm import Session, sessionmaker

from core.config import Settings, get_settings
from core.llm import is_connection_error
from core.netcheck import ensure_reachable, mark_down
from core.schemas import Evidence, VehicleRecord
from db.models import RecallCache

log = logging.getLogger(__name__)


def recall_text(record: dict, vehicle_label: str) -> str:
    fields = [
        ("Vehicle", vehicle_label),
        ("NHTSA Campaign Number", record.get("NHTSACampaignNumber")),
        ("Report Received Date", record.get("ReportReceivedDate")),
        ("Component", record.get("Component")),
        ("Summary", record.get("Summary")),
        ("Consequence", record.get("Consequence")),
        ("Remedy", record.get("Remedy")),
    ]
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


class RecallsTool:
    def __init__(self, session_factory: sessionmaker[Session], settings: Settings | None = None, http: httpx.Client | None = None):
        self.session_factory = session_factory
        self.settings = settings or get_settings()
        self.http = http

    def _url(self, make: str, model: str, year: int) -> str:
        query = urlencode({"make": make, "model": model, "modelYear": year})
        return f"{self.settings.nhtsa_api_base}/recalls/recallsByVehicle?{query}"

    def _fetch(self, make: str, model: str, year: int, db: Session | None = None) -> tuple[str, list[dict]] | None:
        key = f"{make}|{model}|{year}".upper()
        ttl = timedelta(hours=self.settings.recalls_cache_ttl_hours)
        # A chat turn already holds SQLite's write lock. Reuse its session so
        # the cache entry and the turn commit together, rather than deadlocking
        # a second writer. Standalone tool calls still own their transaction.
        with (nullcontext(db) if db is not None else self.session_factory()) as session:
            cached = session.get(RecallCache, key)
            if cached:
                fetched = cached.fetched_at if cached.fetched_at.tzinfo else cached.fetched_at.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - fetched < ttl:
                    return cached.source_url, cached.payload.get("results", [])
            if not self.settings.enable_live_recalls_api:
                return None
            url = self._url(make, model, year)
            host = httpx.URL(url).host
            try:
                if self.http is None:  # injected test clients never touch the network
                    ensure_reachable(host)
                client = self.http or httpx.Client(timeout=10, headers={"User-Agent": "AutoGuideAI-Academic/1.0"})
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
            except Exception as error:
                if self.http is None and is_connection_error(error):
                    mark_down(host)
                log.warning("Recalls API failed for %s: %s", key, error)
                return None
            session.merge(RecallCache(key=key, source_url=url, payload=payload, fetched_at=datetime.now(timezone.utc)))
            if db is None:
                session.commit()
            return url, payload.get("results", [])

    def _local(self, make: str, model: str, year: int) -> list[tuple[str, dict]]:
        found = []
        api_dir = self.settings.data_dir / "processed" / "api"
        for path in sorted(api_dir.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            vehicle = doc.get("vehicle") or {}
            if (
                str(vehicle.get("make", "")).upper() == make.upper()
                and str(vehicle.get("model", "")).upper() == model.upper()
                and str(vehicle.get("year", "")) == str(year)
            ):
                found.append((doc.get("url", ""), doc))
        return found

    def lookup(self, vehicles: list[VehicleRecord], start_ref: int = 1, limit: int = 5,
               db: Session | None = None) -> tuple[list[Evidence], str]:
        """Return (evidence, status). status: found | none | unavailable."""
        if not vehicles:
            return [], "none"
        first = vehicles[0]
        names = list(dict.fromkeys([first.model_family] + [v.model for v in vehicles][:3]))
        label = f"{first.model_year} {first.make} {first.model_family}"
        any_source = False
        for name in names:
            fetched = self._fetch(first.make, name, first.model_year, db=db)
            if fetched is not None:
                any_source = True
                url, records = fetched
                if records:
                    return [
                        Evidence(
                            ref=start_ref + i,
                            evidence_id=f"recall:{r.get('NHTSACampaignNumber', i)}",
                            kind="recall_record",
                            title=f"NHTSA Recall {r.get('NHTSACampaignNumber', '')}: {label} - {r.get('Component', '')}",
                            url=url,
                            text=recall_text(r, label),
                        )
                        for i, r in enumerate(records[:limit])
                    ], "found"
            local = self._local(first.make, name, first.model_year)
            if local:
                return [
                    Evidence(
                        ref=start_ref + i,
                        evidence_id=f"recall:{doc.get('campaign_number', i)}",
                        kind="recall_record",
                        title=doc.get("title", f"NHTSA recall: {label}"),
                        url=url,
                        text=doc.get("text", ""),
                    )
                    for i, (url, doc) in enumerate(local[:limit])
                ], "found"
        return [], ("none" if any_source else "unavailable")
