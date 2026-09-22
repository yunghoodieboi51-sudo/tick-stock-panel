"""Small, storage-agnostic contracts for reproducible historical research.

This module deliberately does not download market data or write canonical
Parquet.  A later backfill worker can use these value objects to record what
was requested, what the provider actually returned, and whether a run can be
resumed without treating an HTTP success as coverage success.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import polars as pl


class BackfillBatchStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    EMPTY = "EMPTY"
    FAILED = "FAILED"


_RETRYABLE_STATUSES = frozenset(
    {
        BackfillBatchStatus.PENDING,
        BackfillBatchStatus.PARTIAL,
        BackfillBatchStatus.FAILED,
        BackfillBatchStatus.EMPTY,
    }
)
_KNOWN_EMPTY_REASONS = frozenset({"pre_listing", "post_delisting", "expected_empty"})
_CHECKSUM_KINDS = frozenset({"normalized_logical_rows", "persisted_partition_fingerprint"})
_LISTING_DATE_PLACEHOLDERS = frozenset({date(1970, 1, 1)})


def is_valid_listing_date(value: str | None) -> bool:
    """Reject missing, malformed, and known sentinel listing dates.

    Callers calculating ``listed_days_at_T`` must treat ``False`` as unknown,
    never as a date far in the past.
    """
    if not value:
        return False
    try:
        return date.fromisoformat(value) not in _LISTING_DATE_PLACEHOLDERS
    except ValueError:
        return False


@dataclass(frozen=True)
class InstrumentHistoryRecord:
    """Point-in-time master-data row; ``effective_to`` is exclusive.

    A record covers ``effective_from <= T < effective_to``.  The current
    instruments snapshot cannot fill this contract retrospectively.
    """

    symbol: str
    exchange: str | None
    security_type: str | None
    listing_date: str | None
    delisting_date: str | None
    effective_from: str
    effective_to: str | None
    status: str | None
    name: str | None
    source: str
    as_of: str

    def is_effective_at(self, trade_date: date) -> bool:
        start = date.fromisoformat(self.effective_from)
        end = date.fromisoformat(self.effective_to) if self.effective_to else None
        return start <= trade_date and (end is None or trade_date < end)

    def was_listed_at(self, trade_date: date) -> bool | None:
        """Return ``None`` when this row cannot establish the answer at ``trade_date``."""
        if not self.is_effective_at(trade_date) or not is_valid_listing_date(self.listing_date):
            return None
        listing = date.fromisoformat(self.listing_date)
        if trade_date < listing:
            return False
        return not self.delisting_date or trade_date < date.fromisoformat(self.delisting_date)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InstrumentHistoryRecord:
        return cls(
            symbol=str(payload["symbol"]),
            exchange=payload.get("exchange"),
            security_type=payload.get("security_type"),
            listing_date=payload.get("listing_date"),
            delisting_date=payload.get("delisting_date"),
            effective_from=str(payload["effective_from"]),
            effective_to=payload.get("effective_to"),
            status=payload.get("status"),
            name=payload.get("name"),
            source=str(payload["source"]),
            as_of=str(payload["as_of"]),
        )


@dataclass(frozen=True)
class HistoricalBackfillBatch:
    batch_id: str
    asset_type: str
    symbols: tuple[str, ...]
    requested_start: str
    requested_end: str
    status: BackfillBatchStatus = BackfillBatchStatus.PENDING
    attempts: int = 0
    rows: int = 0
    actual_first_date: str | None = None
    actual_last_date: str | None = None
    checksum: str | None = None
    checksum_kind: str | None = None
    error: str | None = None
    reason: str | None = None
    expected_empty_symbols: tuple[str, ...] = ()

    def can_resume(self, *, force: bool = False) -> bool:
        if self.status is BackfillBatchStatus.EMPTY and self.reason in _KNOWN_EMPTY_REASONS:
            return force
        return force or self.status in _RETRYABLE_STATUSES

    def begin_attempt(self, *, force: bool = False) -> HistoricalBackfillBatch:
        if not self.can_resume(force=force):
            raise ValueError(f"batch {self.batch_id} is completed; use force to retry")
        return replace(
            self, status=BackfillBatchStatus.RUNNING, attempts=self.attempts + 1, error=None
        )

    def finish(
        self,
        status: BackfillBatchStatus,
        *,
        rows: int = 0,
        actual_first_date: str | None = None,
        actual_last_date: str | None = None,
        checksum: str | None = None,
        checksum_kind: str | None = None,
        error: str | None = None,
        reason: str | None = None,
    ) -> HistoricalBackfillBatch:
        if self.status is not BackfillBatchStatus.RUNNING:
            raise ValueError(f"batch {self.batch_id} must be RUNNING before it can finish")
        if status not in {
            BackfillBatchStatus.COMPLETED,
            BackfillBatchStatus.PARTIAL,
            BackfillBatchStatus.EMPTY,
            BackfillBatchStatus.FAILED,
        }:
            raise ValueError(f"invalid terminal batch status: {status}")
        if status is BackfillBatchStatus.COMPLETED and (
            rows <= 0 or not checksum or checksum_kind not in _CHECKSUM_KINDS
        ):
            raise ValueError("COMPLETED requires validated rows and a declared storage checksum")
        return replace(
            self,
            status=status,
            rows=rows,
            actual_first_date=actual_first_date,
            actual_last_date=actual_last_date,
            checksum=checksum,
            checksum_kind=checksum_kind,
            error=error,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["expected_empty_symbols"] = list(self.expected_empty_symbols)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HistoricalBackfillBatch:
        return cls(
            batch_id=str(payload["batch_id"]),
            asset_type=str(payload["asset_type"]),
            symbols=tuple(str(value) for value in payload.get("symbols", [])),
            requested_start=str(payload["requested_start"]),
            requested_end=str(payload["requested_end"]),
            status=BackfillBatchStatus(str(payload.get("status", BackfillBatchStatus.PENDING))),
            attempts=int(payload.get("attempts", 0)),
            rows=int(payload.get("rows", 0)),
            actual_first_date=payload.get("actual_first_date"),
            actual_last_date=payload.get("actual_last_date"),
            checksum=payload.get("checksum"),
            checksum_kind=payload.get("checksum_kind"),
            error=payload.get("error"),
            reason=payload.get("reason"),
            expected_empty_symbols=tuple(
                str(value) for value in payload.get("expected_empty_symbols", [])
            ),
        )


@dataclass(frozen=True)
class HistoricalBackfillManifest:
    schema_version: int
    dataset_version: str
    provider: str
    provider_version: str | None
    provider_capability: str
    requested_start: str
    requested_end: str
    warmup_start: str
    research_start: str
    universe_version: str
    survivorship_level: str
    adjustment_policy: str
    created_at: str
    updated_at: str
    batches: tuple[HistoricalBackfillBatch, ...] = ()

    def replace_batch(self, batch: HistoricalBackfillBatch) -> HistoricalBackfillManifest:
        if batch.batch_id not in {item.batch_id for item in self.batches}:
            raise KeyError(f"unknown batch: {batch.batch_id}")
        return replace(
            self,
            batches=tuple(
                batch if item.batch_id == batch.batch_id else item for item in self.batches
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["batches"] = [batch.to_dict() for batch in self.batches]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HistoricalBackfillManifest:
        return cls(
            schema_version=int(payload["schema_version"]),
            dataset_version=str(payload["dataset_version"]),
            provider=str(payload["provider"]),
            provider_version=payload.get("provider_version"),
            provider_capability=str(payload["provider_capability"]),
            requested_start=str(payload["requested_start"]),
            requested_end=str(payload["requested_end"]),
            warmup_start=str(payload["warmup_start"]),
            research_start=str(payload["research_start"]),
            universe_version=str(payload["universe_version"]),
            survivorship_level=str(payload["survivorship_level"]),
            adjustment_policy=str(payload["adjustment_policy"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            batches=tuple(
                HistoricalBackfillBatch.from_dict(item) for item in payload.get("batches", [])
            ),
        )


@dataclass(frozen=True)
class ResearchDatasetManifest:
    """Frozen description of a research dataset, not a mutable data store."""

    schema_version: int
    dataset_version: str
    raw_date_range: tuple[str, str]
    research_date_range: tuple[str, str]
    warmup_range: tuple[str, str]
    provider: str
    universe_version: str
    survivorship_level: str
    adjustment_policy: str
    index_coverage: str
    factor_coverage: str
    historical_st_available: bool
    financial_available: bool
    known_limitations: tuple[str, ...]
    partition_fingerprints: Mapping[str, str]
    generation_timestamp: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "partition_fingerprints", MappingProxyType(dict(self.partition_fingerprints))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "raw_date_range": list(self.raw_date_range),
            "research_date_range": list(self.research_date_range),
            "warmup_range": list(self.warmup_range),
            "provider": self.provider,
            "universe_version": self.universe_version,
            "survivorship_level": self.survivorship_level,
            "adjustment_policy": self.adjustment_policy,
            "index_coverage": self.index_coverage,
            "factor_coverage": self.factor_coverage,
            "historical_st_available": self.historical_st_available,
            "financial_available": self.financial_available,
            "known_limitations": list(self.known_limitations),
            "partition_fingerprints": dict(self.partition_fingerprints),
            "generation_timestamp": self.generation_timestamp,
        }


@dataclass(frozen=True)
class CoverageResult:
    """Coverage of the supplied market calendar, never per-symbol bar coverage."""

    requested_start: str
    requested_end: str
    actual_start: str | None
    actual_end: str | None
    expected_trading_days: int
    actual_days: int
    row_count: int
    duplicate_count: int
    missing_count: int
    first_date: str | None
    last_date: str | None
    coverage_ratio: float | None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def validate_daily_coverage(
    frame: pl.DataFrame,
    *,
    requested_start: date,
    requested_end: date,
    expected_trading_dates: Iterable[date],
) -> CoverageResult:
    """Validate market-calendar coverage separately from per-symbol bar coverage.

    Suspensions are valid: this function never treats a missing symbol/day row as
    a missing market day.  A later downloader must supply a real market calendar.
    """
    required = {"symbol", "date", "open", "high", "low", "close", "volume", "amount"}
    missing_columns = sorted(required - set(frame.columns))
    expected = sorted(
        value for value in set(expected_trading_dates) if requested_start <= value <= requested_end
    )
    errors: list[str] = [f"missing_columns:{','.join(missing_columns)}"] if missing_columns else []
    if missing_columns:
        return CoverageResult(
            requested_start=requested_start.isoformat(),
            requested_end=requested_end.isoformat(),
            actual_start=None,
            actual_end=None,
            expected_trading_days=len(expected),
            actual_days=0,
            row_count=frame.height,
            duplicate_count=0,
            missing_count=len(expected),
            first_date=None,
            last_date=None,
            coverage_ratio=0.0 if expected else None,
            warnings=("symbol_bar_coverage_not_enforced",),
            errors=tuple(errors),
        )

    with_dates = frame.with_columns(pl.col("date").cast(pl.Date, strict=False))
    invalid_date_count = with_dates.filter(pl.col("date").is_null()).height
    normalized = with_dates.drop_nulls(["date"]).with_columns(
        [
            pl.col(column).cast(pl.Float64, strict=False)
            for column in ("open", "high", "low", "close", "volume", "amount")
        ]
    )
    dates = sorted(set(normalized["date"].to_list()))
    actual_market_dates = {value for value in dates if requested_start <= value <= requested_end}
    missing_market_dates = set(expected) - actual_market_dates
    duplicate_count = int(
        normalized.group_by(["symbol", "date"])
        .len()
        .select((pl.col("len") - 1).clip(lower_bound=0).sum())
        .item()
    )
    integrity_errors, integrity_warnings = _daily_integrity_results(normalized)
    out_of_range_count = len(set(dates) - actual_market_dates)
    if invalid_date_count:
        integrity_errors.append(f"missing_or_invalid_date_rows:{invalid_date_count}")
    warnings = ["symbol_bar_coverage_not_enforced"]
    if out_of_range_count:
        warnings.append(f"out_of_requested_range_dates:{out_of_range_count}")
    warnings.extend(integrity_warnings)
    ratio = len(actual_market_dates & set(expected)) / len(expected) if expected else None
    return CoverageResult(
        requested_start=requested_start.isoformat(),
        requested_end=requested_end.isoformat(),
        actual_start=dates[0].isoformat() if dates else None,
        actual_end=dates[-1].isoformat() if dates else None,
        expected_trading_days=len(expected),
        actual_days=len(actual_market_dates),
        row_count=normalized.height,
        duplicate_count=duplicate_count,
        missing_count=len(missing_market_dates),
        first_date=dates[0].isoformat() if dates else None,
        last_date=dates[-1].isoformat() if dates else None,
        coverage_ratio=ratio,
        warnings=tuple(warnings),
        errors=tuple(integrity_errors),
    )


def _daily_integrity_results(frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    invalid_checks = {
        "non_positive_price": (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0),
        "high_below_open_or_close": (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close")),
        "low_above_open_or_close": (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close")),
        "high_below_low": pl.col("high") < pl.col("low"),
        "negative_volume": pl.col("volume") < 0,
        "negative_amount": pl.col("amount") < 0,
        "missing_ohlcv": pl.any_horizontal(
            [
                pl.col(column).is_null()
                for column in ("open", "high", "low", "close", "volume", "amount")
            ]
        ),
        "non_finite_ohlcv": pl.any_horizontal(
            [
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                for column in ("open", "high", "low", "close", "volume", "amount")
            ]
        ),
    }
    errors = [name for name, predicate in invalid_checks.items() if frame.filter(predicate).height]
    warnings: list[str] = []
    # Extreme returns are deliberately warning-only. Corporate actions and bad
    # adjustment data need investigation, not silent deletion by a downloader.
    if (
        frame.sort(["symbol", "date"])
        .with_columns(pl.col("close").pct_change().over("symbol").abs().alias("_return"))
        .filter(pl.col("_return") > 0.30)
        .height
    ):
        warnings.append("extreme_return")
    return errors, warnings


def unexpected_missing_symbols(
    requested_symbols: Iterable[str],
    returned_symbols: Iterable[str],
    expected_empty_symbols: Iterable[str] = (),
) -> tuple[str, ...]:
    """Missing symbols that make a provider response partial, not legitimate empty windows."""
    return tuple(
        sorted(set(requested_symbols) - set(returned_symbols) - set(expected_empty_symbols))
    )
