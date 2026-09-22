"""Manual-only, memory-only TickFlow historical capability probe.

Run from ``backend`` with ``uv run python scripts/probe_historical_capability.py``.
It prints JSON summaries and never writes canonical data or retained K-line payloads.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import polars as pl

from app.data_providers.tickflow_provider import TickFlowProvider
from app.services.kline_sync import (
    _compact_klines_to_df,
    _datetime_to_ms,
    _timestamp_to_beijing_datetime,
)
from app.tickflow.client import get_client

START = datetime(2020, 1, 2)  # first A-share trading day of 2020
END = datetime(2026, 9, 11)
STOCK_SYMBOLS = [
    "600519.SH",
    "000001.SZ",
    "300750.SZ",
    "688981.SH",
    "920000.BJ",
    "600000.SH",
    "000858.SZ",
    "002594.SZ",
    "601318.SH",
    "600735.SH",
]
INDEX_SYMBOLS = ["000001.SH", "399001.SZ", "000688.SH"]
FACTOR_SYMBOLS = ["600519.SH", "000001.SZ"]
NEW_LISTING_SYMBOL = "688836.SH"  # local snapshot: listed 2026-08-19
SUSPENSION_CANDIDATE = "600735.SH"  # local 241-day coverage has a long missing tail
EXPECTED_FIRST_DATES = {
    "688981.SH": datetime(2020, 7, 16),
    "920000.BJ": datetime(2020, 12, 23),
    NEW_LISTING_SYMBOL: datetime(2026, 8, 19),
}


def _raw_daily(symbols: list[str], start: datetime, end: datetime) -> pl.DataFrame:
    raw = get_client().klines.batch(
        symbols,
        period="1d",
        adjust="none",
        start_time=_datetime_to_ms(start),
        end_time=_datetime_to_ms(end),
        count=10000,
        as_dataframe=False,
        show_progress=False,
    )
    frame = _compact_klines_to_df(raw)
    if not frame.is_empty() and "timestamp" in frame.columns:
        frame = frame.with_columns(
            _timestamp_to_beijing_datetime(pl.col("timestamp")).alias("datetime")
        )
    return frame


def _summary(
    frame: pl.DataFrame,
    requested_start: datetime,
    requested_end: datetime,
    expected_first_date: datetime | None = None,
) -> dict:
    if (frame.is_empty() or "date" not in frame.columns) and "datetime" not in frame.columns:
        return {
            "rows": frame.height,
            "actual_first_date": None,
            "actual_last_date": None,
            "unique_dates": 0,
            "truncated": True,
        }
    date_col = "datetime" if "datetime" in frame.columns else "date"
    date_expr = (
        pl.col(date_col).dt.date()
        if date_col == "datetime"
        else pl.col(date_col).cast(pl.Date, strict=False)
    )
    dates = frame.select(date_expr.alias("_date"))["_date"].drop_nulls()
    first, last = dates.min(), dates.max()
    return {
        "rows": frame.height,
        "actual_first_date": first.isoformat() if first else None,
        "actual_last_date": last.isoformat() if last else None,
        "unique_dates": dates.n_unique(),
        "truncated": first is None
        or last is None
        or first > (expected_first_date or requested_start).date()
        or last < requested_end.date(),
        "raw_zero_volume_bars": frame.filter(pl.col("volume") == 0).height
        if "volume" in frame.columns
        else None,
    }


def _probe_daily(
    symbols: list[str],
    asset_type: str,
    start: datetime = START,
    end: datetime = END,
    expected_first_date: datetime | None = None,
) -> dict:
    started = time.perf_counter()
    try:
        frame = _raw_daily(symbols, start, end)
        returned_symbols = (
            sorted(frame["symbol"].unique().to_list()) if "symbol" in frame.columns else []
        )
        per_symbol = {
            symbol: _summary(
                frame.filter(pl.col("symbol") == symbol),
                start,
                end,
                EXPECTED_FIRST_DATES.get(symbol, expected_first_date),
            )
            for symbol in returned_symbols
        }
        missing_symbols = sorted(set(symbols) - set(returned_symbols))
        result = {
            "rows": frame.height,
            "date_union": _summary(frame, start, end, expected_first_date),
            "per_symbol": per_symbol,
            "returned_symbols": returned_symbols,
            "missing_symbols": missing_symbols,
            "truncated": bool(missing_symbols)
            or any(item["truncated"] for item in per_symbol.values()),
            "error": None,
        }
    except Exception as exc:  # live diagnostics must report, not disguise failures
        result = {
            "rows": 0,
            "date_union": None,
            "per_symbol": {},
            "returned_symbols": [],
            "missing_symbols": list(symbols),
            "error": f"{type(exc).__name__}: {exc}",
            "truncated": None,
        }
    result.update(
        {
            "symbols": symbols,
            "asset_type": asset_type,
            "requested_start": start.date().isoformat(),
            "requested_end": end.date().isoformat(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "retry": "SDK default (up to 3)",
        }
    )
    return result


def _probe_factors(provider: TickFlowProvider) -> dict:
    started = time.perf_counter()
    try:
        frame = provider.get_adj_factors(FACTOR_SYMBOLS, START, END, "stock")
        dates = (
            frame["trade_date"].drop_nulls()
            if "trade_date" in frame.columns
            else pl.Series([], dtype=pl.Date)
        )
        result = {
            "rows": frame.height,
            "first_factor_date": dates.min().isoformat() if len(dates) else None,
            "last_factor_date": dates.max().isoformat() if len(dates) else None,
            "factor_changes": frame.filter(pl.col("ex_factor") != 1).height
            if "ex_factor" in frame.columns
            else None,
            "error": None,
        }
    except Exception as exc:
        result = {"rows": 0, "error": f"{type(exc).__name__}: {exc}"}
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def main() -> None:
    provider = TickFlowProvider()
    report = {
        "notice": "Manual live probe only; no returned K-line or factor rows are persisted.",
        "stock_one": _probe_daily(STOCK_SYMBOLS[:1], "stock"),
        "stock_five": _probe_daily(STOCK_SYMBOLS[:5], "stock"),
        "stock_ten": _probe_daily(STOCK_SYMBOLS, "stock"),
        "new_listing": _probe_daily(
            [NEW_LISTING_SYMBOL],
            "stock",
            datetime(2026, 7, 20),
            END,
            expected_first_date=EXPECTED_FIRST_DATES[NEW_LISTING_SYMBOL],
        ),
        "suspension_candidate": _probe_daily(
            [SUSPENSION_CANDIDATE], "stock", datetime(2025, 9, 15), END
        ),
        "delisted": {
            "capability": "UNKNOWN",
            "reason": "current-universe endpoint has no trusted delisting master record; no symbol is invented for a probe",
        },
        "index_three": _probe_daily(INDEX_SYMBOLS, "index"),
        "adjustment_factors": _probe_factors(provider),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
