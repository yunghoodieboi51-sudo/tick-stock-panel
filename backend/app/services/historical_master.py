"""Immutable historical master-data snapshots and normalized research generations.

This is intentionally offline-only.  It neither reads nor replaces the live
``instruments.parquet`` snapshot, and no strategy/backtest code imports it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import polars as pl

from app.data_providers.historical_master import HistoricalSourcePayload

_SYMBOL = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
_EXCHANGE = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ", "SH": "SH", "SZ": "SZ", "BJ": "BJ"}
_STATUS = frozenset({"L", "D", "P", "G", "UN"})


class TruthValue(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Coverage:
    capability: str
    start: str | None
    end: str | None
    exchanges: tuple[str, ...] = ()
    security_types: tuple[str, ...] = ("stock",)
    complete: bool = False

    def covers(
        self,
        when: date,
        exchange: str | None = None,
        security_type: str | None = None,
    ) -> bool:
        if not self.complete or self.start is None or self.end is None:
            return False
        if not (date.fromisoformat(self.start) <= when <= date.fromisoformat(self.end)):
            return False
        if exchange is not None and self.exchanges and exchange not in self.exchanges:
            return False
        return (
            security_type is None or not self.security_types or security_type in self.security_types
        )


@dataclass(frozen=True)
class LifecycleRecord:
    symbol: str
    instrument_id: str
    exchange: str
    security_type: str
    name: str | None
    listing_date: str | None
    delisting_date: str | None
    status: str | None
    effective_from: str | None
    effective_to: str | None
    source: str
    source_as_of: str
    raw_snapshot_id: str


@dataclass(frozen=True)
class RiskStatusRecord:
    symbol: str
    status_type: str
    effective_from: str
    effective_to: str | None
    source: str
    source_as_of: str
    raw_snapshot_id: str


@dataclass(frozen=True)
class SuspensionRecord:
    symbol: str
    trade_date: str
    suspend_type: str
    suspend_time: str | None
    resume_time: str | None
    source: str
    source_as_of: str
    raw_snapshot_id: str


@dataclass(frozen=True)
class RejectedRow:
    table: str
    reason_code: str
    source_row: Mapping[str, object]


@dataclass(frozen=True)
class NormalizedMaster:
    lifecycle: tuple[LifecycleRecord, ...]
    risk_status: tuple[RiskStatusRecord, ...]
    suspensions: tuple[SuspensionRecord, ...]
    rejected: tuple[RejectedRow, ...]


@dataclass(frozen=True)
class HistoricalMasterManifest:
    schema_version: int
    generation_id: str
    provider: str
    provider_version: str | None
    raw_snapshot_ids: tuple[str, ...]
    retrieved_at: str
    coverage: Mapping[str, Coverage]
    row_counts: Mapping[str, int]
    rejected_counts: Mapping[str, int]
    fingerprints: Mapping[str, str]
    capabilities: Mapping[str, str]
    limitations: tuple[str, ...]
    historical_name_complete: bool
    announcement_pit_available: bool
    suspension_available: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "coverage", MappingProxyType(dict(self.coverage)))
        object.__setattr__(self, "row_counts", MappingProxyType(dict(self.row_counts)))
        object.__setattr__(self, "rejected_counts", MappingProxyType(dict(self.rejected_counts)))
        object.__setattr__(self, "fingerprints", MappingProxyType(dict(self.fingerprints)))
        object.__setattr__(self, "capabilities", MappingProxyType(dict(self.capabilities)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "raw_snapshot_ids": list(self.raw_snapshot_ids),
            "retrieved_at": self.retrieved_at,
            "coverage": {key: asdict(value) for key, value in self.coverage.items()},
            "row_counts": dict(self.row_counts),
            "rejected_counts": dict(self.rejected_counts),
            "fingerprints": dict(self.fingerprints),
            "capabilities": dict(self.capabilities),
            "limitations": list(self.limitations),
            "historical_name_complete": self.historical_name_complete,
            "announcement_pit_available": self.announcement_pit_available,
            "suspension_available": self.suspension_available,
        }


class HistoricalMasterStore:
    """Owns immutable raw snapshots and atomically published generations."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.source_root = data_dir / "research_sources"
        self.generation_root = data_dir / "research_master" / "generations"

    def write_snapshot(self, snapshot_id: str, payload: HistoricalSourcePayload) -> Path:
        _validate_id(snapshot_id)
        _validate_non_secret_params(payload.request_params)
        destination = self.source_root / payload.provider / snapshot_id
        if destination.exists():
            raise FileExistsError(f"snapshot already exists: {snapshot_id}")
        row_bytes = _canonical_bytes(list(payload.rows))
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "provider": payload.provider,
            "provider_version": payload.provider_version,
            "endpoint": payload.endpoint,
            "retrieved_at": payload.retrieved_at.isoformat(),
            "request_params": dict(payload.request_params),
            "row_count": len(payload.rows),
            "source_schema": payload.source_schema,
            "content_fingerprint": _sha(row_bytes),
        }
        staging = destination.with_name(destination.name + ".staging")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            _atomic_json(staging / "rows.json", list(payload.rows))
            _atomic_json(staging / "manifest.json", manifest)
            staging.replace(destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    def publish_generation(
        self,
        generation_id: str,
        normalized: NormalizedMaster,
        manifest: HistoricalMasterManifest,
    ) -> Path:
        _validate_id(generation_id)
        if generation_id != manifest.generation_id:
            raise ValueError("generation id does not match manifest")
        _validate_generation_manifest(normalized, manifest)
        destination = self.generation_root / generation_id
        if destination.exists():
            raise FileExistsError(f"generation already exists: {generation_id}")
        staging = destination.with_name(destination.name + ".staging")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            _records_frame(normalized.lifecycle).write_parquet(
                staging / "instrument_lifecycle.parquet"
            )
            _records_frame(normalized.risk_status).write_parquet(
                staging / "historical_risk_status.parquet"
            )
            _records_frame(normalized.suspensions).write_parquet(
                staging / "historical_suspension.parquet"
            )
            _atomic_json(
                staging / "rejected.json", [_rejected_dict(row) for row in normalized.rejected]
            )
            _atomic_json(staging / "manifest.json", manifest.to_dict())
            staging.replace(destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination


class HistoricalMasterQuery:
    """Tri-state helpers; missing records are never silently interpreted as false."""

    def __init__(
        self,
        lifecycle: Iterable[LifecycleRecord],
        risk_status: Iterable[RiskStatusRecord],
        suspensions: Iterable[SuspensionRecord],
        coverage: Mapping[str, Coverage],
    ) -> None:
        self.lifecycle = tuple(lifecycle)
        self.risk_status = tuple(risk_status)
        self.suspensions = tuple(suspensions)
        self.coverage = dict(coverage)

    def is_listed_at(self, symbol: str, when: date) -> TruthValue:
        rows = [row for row in self.lifecycle if row.symbol == symbol]
        if not rows:
            return TruthValue.UNKNOWN
        row = rows[0]
        if not row.listing_date:
            return TruthValue.UNKNOWN
        if when < date.fromisoformat(row.listing_date):
            return TruthValue.FALSE
        # Providers commonly do not document whether delisting_date itself was
        # tradeable.  Preserve that ambiguity rather than exclude the final bar.
        if row.delisting_date:
            delisting = date.fromisoformat(row.delisting_date)
            if when > delisting:
                return TruthValue.FALSE
            if when == delisting:
                return TruthValue.UNKNOWN
        return TruthValue.TRUE

    def risk_status_at(self, symbol: str, when: date) -> TruthValue:
        exchange = _exchange_from_symbol(symbol)
        coverage = self.coverage.get("historical_st", Coverage("", None, None))
        if not coverage.covers(when, exchange, "stock"):
            return TruthValue.UNKNOWN
        rows = [row for row in self.risk_status if row.symbol == symbol and _in_interval(row, when)]
        if rows:
            return TruthValue.TRUE
        return TruthValue.FALSE

    def trading_status_at(self, symbol: str, when: date) -> TruthValue:
        exchange = _exchange_from_symbol(symbol)
        coverage = self.coverage.get("suspension_status", Coverage("", None, None))
        if not coverage.covers(when, exchange, "stock"):
            return TruthValue.UNKNOWN
        rows = [
            row
            for row in self.suspensions
            if row.symbol == symbol and row.trade_date == when.isoformat()
        ]
        if any(row.suspend_type == "S" for row in rows):
            return TruthValue.TRUE
        if rows:
            return TruthValue.FALSE
        return TruthValue.FALSE


def normalize_tushare(
    lifecycle_payloads: Iterable[tuple[str, HistoricalSourcePayload]],
    st_payload: tuple[str, HistoricalSourcePayload] | None,
    suspension_payload: tuple[str, HistoricalSourcePayload] | None,
    trading_dates: Iterable[date],
) -> NormalizedMaster:
    """Normalize source rows without inventing historical name/PIT semantics."""
    calendar = tuple(sorted(set(trading_dates)))
    lifecycle: list[LifecycleRecord] = []
    risk: list[RiskStatusRecord] = []
    suspensions: list[SuspensionRecord] = []
    rejected: list[RejectedRow] = []
    seen_lifecycle: set[str] = set()
    for snapshot_id, payload in lifecycle_payloads:
        for raw in payload.rows:
            symbol = _text(raw.get("ts_code"))
            exchange = _EXCHANGE.get(_text(raw.get("exchange")) or "")
            listing = _iso_date(raw.get("list_date"))
            delisting = _iso_date(raw.get("delist_date"))
            status = _text(raw.get("list_status"))
            if not _valid_symbol(symbol) or not exchange:
                rejected.append(
                    RejectedRow("instrument_lifecycle", "INVALID_SYMBOL_OR_EXCHANGE", raw)
                )
            elif status not in _STATUS or not listing or (delisting and listing > delisting):
                rejected.append(
                    RejectedRow("instrument_lifecycle", "INVALID_LIFECYCLE_DATE_OR_STATUS", raw)
                )
            elif symbol in seen_lifecycle:
                rejected.append(RejectedRow("instrument_lifecycle", "DUPLICATE_INSTRUMENT", raw))
            else:
                seen_lifecycle.add(symbol)
                lifecycle.append(
                    LifecycleRecord(
                        symbol,
                        f"tushare:{symbol}",
                        exchange,
                        "stock",
                        _text(raw.get("name")),
                        listing,
                        delisting,
                        status,
                        listing,
                        None,
                        payload.provider,
                        payload.retrieved_at.isoformat(),
                        snapshot_id,
                    )
                )
    if st_payload:
        snapshot_id, payload = st_payload
        _normalize_st(payload, snapshot_id, calendar, risk, rejected)
    if suspension_payload:
        snapshot_id, payload = suspension_payload
        _normalize_suspensions(payload, snapshot_id, calendar, suspensions, rejected)
    known = {row.symbol for row in lifecycle}
    for row in [*risk, *suspensions]:
        if row.symbol not in known:
            rejected.append(
                RejectedRow(type(row).__name__, "NO_LIFECYCLE_MATCH", {"symbol": row.symbol})
            )
    risk = [row for row in risk if row.symbol in known]
    suspensions = [row for row in suspensions if row.symbol in known]
    return NormalizedMaster(tuple(lifecycle), tuple(risk), tuple(suspensions), tuple(rejected))


def build_manifest(
    generation_id: str,
    provider: str,
    raw_snapshot_ids: Iterable[str],
    normalized: NormalizedMaster,
    coverage: Mapping[str, Coverage],
    capabilities: Mapping[str, str],
) -> HistoricalMasterManifest:
    frames = {
        "instrument_lifecycle": _records_frame(normalized.lifecycle),
        "historical_risk_status": _records_frame(normalized.risk_status),
        "historical_suspension": _records_frame(normalized.suspensions),
    }
    rejected_counts: dict[str, int] = {}
    for row in normalized.rejected:
        rejected_counts[row.reason_code] = rejected_counts.get(row.reason_code, 0) + 1
    return HistoricalMasterManifest(
        1,
        generation_id,
        provider,
        None,
        tuple(raw_snapshot_ids),
        datetime.now(UTC).isoformat(),
        coverage,
        {name: frame.height for name, frame in frames.items()},
        rejected_counts,
        {name: _sha(_canonical_bytes(frame.to_dicts())) for name, frame in frames.items()},
        capabilities,
        ("historical_name_complete=false", "announcement_pit_available=false"),
        False,
        False,
        "suspension_status" in coverage,
    )


def _normalize_st(
    payload: HistoricalSourcePayload,
    snapshot_id: str,
    calendar: tuple[date, ...],
    out: list[RiskStatusRecord],
    rejected: list[RejectedRow],
) -> None:
    grouped: dict[tuple[str, str], list[date]] = {}
    candidates: list[tuple[str, str, date, Mapping[str, object]]] = []
    for raw in payload.rows:
        symbol, status, when = (
            _text(raw.get("ts_code")),
            _text(raw.get("type")),
            _date(raw.get("trade_date")),
        )
        if not _valid_symbol(symbol) or not status or when is None or when not in calendar:
            rejected.append(
                RejectedRow("historical_risk_status", "INVALID_OR_NON_TRADING_ST_ROW", raw)
            )
            continue
        candidates.append((symbol, status, when, raw))
    statuses_by_symbol_date: dict[tuple[str, date], set[str]] = {}
    for symbol, status, when, _raw in candidates:
        statuses_by_symbol_date.setdefault((symbol, when), set()).add(status)
    for symbol, status, when, raw in candidates:
        if len(statuses_by_symbol_date[(symbol, when)]) > 1:
            rejected.append(
                RejectedRow("historical_risk_status", "CONTRADICTORY_RISK_STATUS_ROW", raw)
            )
            continue
        grouped.setdefault((symbol, status), []).append(when)
    for (symbol, status), dates in grouped.items():
        dates = sorted(set(dates))
        start = previous = dates[0]
        for current in dates[1:]:
            if _next_calendar(calendar, previous) != current:
                end = _next_calendar(calendar, previous)
                out.append(
                    RiskStatusRecord(
                        symbol,
                        status,
                        start.isoformat(),
                        end.isoformat() if end else None,
                        payload.provider,
                        payload.retrieved_at.isoformat(),
                        snapshot_id,
                    )
                )
                start = current
            previous = current
        end = _next_calendar(calendar, previous)
        out.append(
            RiskStatusRecord(
                symbol,
                status,
                start.isoformat(),
                end.isoformat() if end else None,
                payload.provider,
                payload.retrieved_at.isoformat(),
                snapshot_id,
            )
        )


def _normalize_suspensions(
    payload: HistoricalSourcePayload,
    snapshot_id: str,
    calendar: tuple[date, ...],
    out: list[SuspensionRecord],
    rejected: list[RejectedRow],
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for raw in payload.rows:
        symbol, when, kind = (
            _text(raw.get("ts_code")),
            _date(raw.get("trade_date")),
            _text(raw.get("suspend_type")),
        )
        timing = _text(raw.get("suspend_timing"))
        key = (symbol or "", when.isoformat() if when else "", kind or "")
        if (
            not _valid_symbol(symbol)
            or when is None
            or when not in calendar
            or kind not in {"S", "R"}
        ):
            rejected.append(
                RejectedRow("historical_suspension", "INVALID_OR_NON_TRADING_SUSPENSION_ROW", raw)
            )
            continue
        if key in seen:
            rejected.append(RejectedRow("historical_suspension", "DUPLICATE_SUSPENSION_ROW", raw))
            continue
        seen.add(key)
        suspend_time, resume_time = _split_timing(timing)
        out.append(
            SuspensionRecord(
                symbol,
                when.isoformat(),
                kind,
                suspend_time,
                resume_time,
                payload.provider,
                payload.retrieved_at.isoformat(),
                snapshot_id,
            )
        )


def _records_frame(records: Iterable[object]) -> pl.DataFrame:
    rows = [asdict(item) for item in records]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
    ).encode()


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(_canonical_bytes(payload))
    temp.replace(path)


def _validate_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("invalid immutable id")


def _validate_non_secret_params(params: Mapping[str, object]) -> None:
    if _contains_credential_key(params):
        raise ValueError("snapshot request_params must not contain credentials")


def _contains_credential_key(value: object) -> bool:
    markers = (
        "token",
        "secret",
        "password",
        "authorization",
        "credential",
        "api_key",
        "apikey",
    )
    if isinstance(value, Mapping):
        return any(
            any(marker in str(key).casefold() for marker in markers)
            or _contains_credential_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_credential_key(item) for item in value)
    return False


def _validate_generation_manifest(
    normalized: NormalizedMaster, manifest: HistoricalMasterManifest
) -> None:
    frames = {
        "instrument_lifecycle": _records_frame(normalized.lifecycle),
        "historical_risk_status": _records_frame(normalized.risk_status),
        "historical_suspension": _records_frame(normalized.suspensions),
    }
    for name, frame in frames.items():
        if manifest.row_counts.get(name) != frame.height:
            raise ValueError(f"generation manifest row count mismatch: {name}")
        if manifest.fingerprints.get(name) != _sha(_canonical_bytes(frame.to_dicts())):
            raise ValueError(f"generation manifest fingerprint mismatch: {name}")


def _text(value: object) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _date(value: object) -> date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text) if "-" in text else datetime.strptime(text, "%Y%m%d").date()
    except ValueError:
        return None


def _iso_date(value: object) -> str | None:
    parsed = _date(value)
    return parsed.isoformat() if parsed else None


def _valid_symbol(value: str | None) -> bool:
    return bool(value and _SYMBOL.fullmatch(value))


def _exchange_from_symbol(symbol: str) -> str | None:
    return symbol.rsplit(".", 1)[-1] if _valid_symbol(symbol) else None


def _next_calendar(calendar: tuple[date, ...], value: date) -> date | None:
    try:
        index = calendar.index(value)
    except ValueError:
        return None
    return calendar[index + 1] if index + 1 < len(calendar) else None


def _in_interval(row: RiskStatusRecord, when: date) -> bool:
    return date.fromisoformat(row.effective_from) <= when and (
        row.effective_to is None or when < date.fromisoformat(row.effective_to)
    )


def _split_timing(value: str | None) -> tuple[str | None, str | None]:
    if not value or "-" not in value:
        return None, None
    left, right = value.split("-", 1)
    return left or None, right or None


def _rejected_dict(row: RejectedRow) -> dict[str, object]:
    return {"table": row.table, "reason_code": row.reason_code, "source_row": dict(row.source_row)}
