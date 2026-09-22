from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from app.services.historical_research import (
    BackfillBatchStatus,
    HistoricalBackfillBatch,
    HistoricalBackfillManifest,
    InstrumentHistoryRecord,
    ResearchDatasetManifest,
    is_valid_listing_date,
    unexpected_missing_symbols,
    validate_daily_coverage,
)
from scripts import probe_historical_capability as probe


def _batch(status: BackfillBatchStatus = BackfillBatchStatus.PENDING) -> HistoricalBackfillBatch:
    return HistoricalBackfillBatch(
        "stock-0001", "stock", ("600000.SH",), "2020-01-01", "2020-12-31", status
    )


def test_manifest_round_trip_and_dataset_manifest_are_json_ready():
    batch = (
        _batch()
        .begin_attempt()
        .finish(
            BackfillBatchStatus.COMPLETED,
            rows=2,
            checksum="sha256:ok",
            checksum_kind="normalized_logical_rows",
            actual_first_date="2020-01-02",
            actual_last_date="2020-01-03",
        )
    )
    manifest = HistoricalBackfillManifest(
        1,
        "research_daily_v1",
        "tickflow",
        "0.1.23",
        "daily",
        "2020-01-01",
        "2020-12-31",
        "2019-01-01",
        "2020-06-01",
        "u1",
        "LEVEL 1",
        "raw_execution_plus_research_adjusted",
        "t0",
        "t0",
        (batch,),
    )
    assert HistoricalBackfillManifest.from_dict(manifest.to_dict()) == manifest
    dataset = ResearchDatasetManifest(
        1,
        "research_daily_v1",
        ("2020-01-01", "2020-12-31"),
        ("2020-06-01", "2020-12-31"),
        ("2020-01-01", "2020-05-31"),
        "tickflow",
        "u1",
        "LEVEL 1",
        "raw",
        "complete",
        "missing",
        False,
        False,
        ("current universe",),
        {"date=2020-01-02": "sha256:x"},
        "t0",
    )
    assert dataset.to_dict()["known_limitations"] == ["current universe"]
    with pytest.raises(TypeError):
        dataset.partition_fingerprints["later"] = "sha256:y"  # type: ignore[index]


def test_resume_skips_completed_but_retries_partial_and_failed():
    completed = (
        _batch()
        .begin_attempt()
        .finish(
            BackfillBatchStatus.COMPLETED,
            rows=1,
            checksum="x",
            checksum_kind="normalized_logical_rows",
        )
    )
    assert not completed.can_resume()
    assert completed.begin_attempt(force=True).attempts == 2
    assert _batch(BackfillBatchStatus.PARTIAL).begin_attempt().attempts == 1
    assert _batch(BackfillBatchStatus.FAILED).begin_attempt().attempts == 1


def test_invalid_batch_state_transitions_are_rejected():
    with pytest.raises(ValueError, match="RUNNING"):
        _batch().finish(BackfillBatchStatus.FAILED)
    with pytest.raises(ValueError, match="checksum"):
        _batch().begin_attempt().finish(BackfillBatchStatus.COMPLETED, rows=1)


def test_expected_empty_window_does_not_retry_without_force():
    empty = _batch().begin_attempt().finish(BackfillBatchStatus.EMPTY, reason="pre_listing")
    assert not empty.can_resume()
    assert empty.can_resume(force=True)


def _daily(rows: dict) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


def test_coverage_uses_market_calendar_not_symbol_bar_coverage():
    frame = _daily(
        {
            "symbol": ["a", "b", "a"],
            "date": ["2020-01-02", "2020-01-02", "2020-01-03"],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [1, 1, 1],
            "amount": [1, 1, 1],
        }
    )
    result = validate_daily_coverage(
        frame,
        requested_start=date(2020, 1, 2),
        requested_end=date(2020, 1, 6),
        expected_trading_dates=[date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
    )
    assert result.actual_days == 2
    assert result.missing_count == 1
    assert result.coverage_ratio == pytest.approx(2 / 3)
    assert result.warnings == ("symbol_bar_coverage_not_enforced",)


def test_coverage_reports_duplicates_and_ohlc_errors_without_dropping_rows():
    frame = _daily(
        {
            "symbol": ["a", "a"],
            "date": ["2020-01-02", "2020-01-02"],
            "open": [2, 1],
            "high": [1, 1],
            "low": [3, 1],
            "close": [2, 1],
            "volume": [-1, 1],
            "amount": [-1, 1],
        }
    )
    result = validate_daily_coverage(
        frame,
        requested_start=date(2020, 1, 2),
        requested_end=date(2020, 1, 2),
        expected_trading_dates=[date(2020, 1, 2)],
    )
    assert result.row_count == 2
    assert result.duplicate_count == 1
    assert {
        "high_below_open_or_close",
        "low_above_open_or_close",
        "high_below_low",
        "negative_volume",
        "negative_amount",
    } <= set(result.errors)


def test_extreme_return_is_flagged_but_not_an_integrity_error():
    frame = _daily(
        {
            "symbol": ["a", "a"],
            "date": ["2020-01-02", "2020-01-03"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [1, 1],
            "amount": [1, 1],
        }
    )
    result = validate_daily_coverage(
        frame,
        requested_start=date(2020, 1, 2),
        requested_end=date(2020, 1, 3),
        expected_trading_dates=[date(2020, 1, 2), date(2020, 1, 3)],
    )
    assert result.errors == ()
    assert "extreme_return" in result.warnings


def test_non_finite_ohlcv_is_an_integrity_error():
    frame = _daily(
        {
            "symbol": ["a"],
            "date": ["2020-01-02"],
            "open": [float("inf")],
            "high": [float("nan")],
            "low": [1],
            "close": [1],
            "volume": [1],
            "amount": [1],
        }
    )
    result = validate_daily_coverage(
        frame,
        requested_start=date(2020, 1, 2),
        requested_end=date(2020, 1, 2),
        expected_trading_dates=[date(2020, 1, 2)],
    )
    assert "non_finite_ohlcv" in result.errors


def test_instrument_history_is_half_open_and_epoch_listing_is_unknown():
    record = InstrumentHistoryRecord(
        "600000.SH",
        "SH",
        "stock",
        "2000-01-01",
        None,
        "2000-01-01",
        "2000-01-03",
        "listed",
        "x",
        "test",
        "2000-01-03",
    )
    assert InstrumentHistoryRecord.from_dict(record.to_dict()) == record
    assert record.was_listed_at(date(2000, 1, 2)) is True
    assert record.was_listed_at(date(2000, 1, 3)) is None
    assert not is_valid_listing_date("1970-01-01")
    assert (
        InstrumentHistoryRecord(
            "x", "SH", "stock", "1970-01-01", None, "1970-01-01", None, None, "x", "test", "t"
        ).was_listed_at(date(2020, 1, 1))
        is None
    )


def test_partial_batch_excludes_explicit_expected_empty_symbols():
    assert unexpected_missing_symbols(["a", "b", "c"], ["a", "c"]) == ("b",)
    assert unexpected_missing_symbols(["a", "b", "c"], ["a", "c"], ["b"]) == ()


def test_probe_summary_handles_datetime_and_listing_boundary_without_false_truncation():
    frame = pl.DataFrame(
        {
            "symbol": ["new", "new"],
            "datetime": [datetime(2026, 8, 19), datetime(2026, 9, 11)],
            "volume": [1, 1],
        }
    )
    result = probe._summary(
        frame,
        datetime(2026, 7, 20),
        datetime(2026, 9, 11),
        expected_first_date=datetime(2026, 8, 19),
    )
    assert result["actual_first_date"] == "2026-08-19"
    assert result["truncated"] is False


def test_probe_reports_partial_batch_without_network(monkeypatch):
    monkeypatch.setattr(
        probe,
        "_raw_daily",
        lambda symbols, start, end: pl.DataFrame(
            {"symbol": [symbols[0]], "datetime": [start], "volume": [1]}
        ),
    )
    result = probe._probe_daily(["a", "b"], "stock", datetime(2020, 1, 2), datetime(2020, 1, 2))
    assert result["returned_symbols"] == ["a"]
    assert result["missing_symbols"] == ["b"]


def test_coverage_missing_required_columns_fails_closed():
    result = validate_daily_coverage(
        pl.DataFrame({"symbol": ["a"]}),
        requested_start=date(2020, 1, 2),
        requested_end=date(2020, 1, 2),
        expected_trading_dates=[date(2020, 1, 2)],
    )
    assert result.errors[0].startswith("missing_columns:")
    assert result.coverage_ratio == 0.0
