"""PIT coverage for the V4.6a trend-pullback research helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.backtest.matrix import build_market_data_matrix
from app.backtest.pullback_diagnostics import (
    DEPTH_BUCKETS,
    DURATION_BUCKETS,
    MA20_INTERACTIONS,
    build_pullback_diagnostics,
    event_date_aggregate,
    positive_return_outlier_dependency,
    summarize_external_event_relationship,
    summarize_pullback_diagnostics,
)

EVENT_TIME = 70


def _market():
    symbols = ("ABOVE", "TOUCH", "RECLAIM", "BELOW", "BREAKOUT")
    count = 100
    closes = {symbol: np.full(count, 10.0, dtype=np.float32) for symbol in symbols}
    peaks = {"ABOVE": 10.1, "TOUCH": 10.5, "RECLAIM": 10.7, "BELOW": 11.0, "BREAKOUT": 10.5}
    for symbol, peak in peaks.items():
        closes[symbol][EVENT_TIME - 5] = peak
    closes["RECLAIM"][EVENT_TIME - 1] = 9.6
    closes["TOUCH"][EVENT_TIME] = 10.2
    closes["BELOW"][EVENT_TIME] = 9.5
    closes["BREAKOUT"][EVENT_TIME] = 11.0
    for symbol in symbols:
        closes[symbol][EVENT_TIME + 3] = 13.0
        closes[symbol][EVENT_TIME + 5] = 14.0
        closes[symbol][EVENT_TIME + 10] = 15.0
        closes[symbol][EVENT_TIME + 20] = 16.0

    start = date(2026, 1, 2)
    rows = []
    for symbol in symbols:
        for time_id, close in enumerate(closes[symbol]):
            low = close - 0.1
            if symbol in {"TOUCH", "RECLAIM"} and time_id == EVENT_TIME:
                low = 9.7
            high = close + 0.2
            if symbol == "TOUCH" and time_id == EVENT_TIME - 1:
                high = 10.1
            rows.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "date": start + timedelta(days=time_id),
                    "open": 100.0 + time_id,
                    "high": high,
                    "low": low,
                    "close": float(close),
                    "volume": 1_000.0,
                    "amount": 100_000.0,
                }
            )
    market = build_market_data_matrix(pl.DataFrame(rows))
    fields = {
        "ma10": np.full(market.shape, 9.9, dtype=np.float32),
        "ma20": np.full(market.shape, 9.8, dtype=np.float32),
        "ma60": np.full(market.shape, 9.0, dtype=np.float32),
        "atr_pct": np.full(market.shape, 0.02, dtype=np.float32),
    }
    return replace(market, fields=fields)


def _asset_ids(market):
    return {symbol: asset_id for asset_id, symbol in enumerate(market.symbols)}


def _diagnostics():
    market = _market()
    return market, build_pullback_diagnostics(market)


def test_trend_universe_is_pit_and_excludes_breakout_events():
    market, diagnostics = _diagnostics()
    ids = _asset_ids(market)
    assert diagnostics.trend_qualified[EVENT_TIME, ids["ABOVE"]]
    assert diagnostics.breakout_excluded[EVENT_TIME, ids["BREAKOUT"]]
    assert not diagnostics.pullback[EVENT_TIME, ids["BREAKOUT"]]

    close = np.array(market.close, copy=True)
    close[EVENT_TIME + 1 :] = 100.0
    changed = build_pullback_diagnostics(replace(market, close=close))
    np.testing.assert_array_equal(
        changed.trend_qualified[EVENT_TIME], diagnostics.trend_qualified[EVENT_TIME]
    )
    np.testing.assert_array_equal(changed.pullback[EVENT_TIME], diagnostics.pullback[EVENT_TIME])


def test_prior_peak_depth_and_duration_use_only_pre_event_valid_bars():
    market, diagnostics = _diagnostics()
    above = _asset_ids(market)["ABOVE"]
    assert diagnostics.prior_peak[EVENT_TIME, above] == pytest.approx(10.1)
    assert diagnostics.pullback_depth_pct[EVENT_TIME, above] == pytest.approx(
        (10.1 - 10.0) / 10.1,
        abs=1e-6,
    )
    assert diagnostics.duration_bars[EVENT_TIME, above] == 5

    close = np.array(market.close, copy=True)
    close[EVENT_TIME] = 99.0
    close[EVENT_TIME + 1 :] = 200.0
    changed = build_pullback_diagnostics(replace(market, close=close))
    assert changed.prior_peak[EVENT_TIME, above] == diagnostics.prior_peak[EVENT_TIME, above]
    assert changed.duration_bars[EVENT_TIME, above] == diagnostics.duration_bars[EVENT_TIME, above]


def test_depth_duration_and_ma20_groups_are_exhaustive_for_pullbacks():
    market, diagnostics = _diagnostics()
    ids = _asset_ids(market)
    assert diagnostics.depth_buckets["VERY_SHALLOW"][EVENT_TIME, ids["ABOVE"]]
    assert diagnostics.duration_buckets["3-5"][EVENT_TIME, ids["ABOVE"]]
    assert diagnostics.ma20_interactions["ABOVE_MA20"][EVENT_TIME, ids["ABOVE"]]
    assert diagnostics.ma20_interactions["TOUCHED_MA20"][EVENT_TIME, ids["TOUCH"]]
    assert diagnostics.ma20_interactions["RECLAIMED_MA20"][EVENT_TIME, ids["RECLAIM"]]
    assert diagnostics.ma20_interactions["BELOW_MA20"][EVENT_TIME, ids["BELOW"]]
    for groups in (
        diagnostics.depth_buckets,
        diagnostics.duration_buckets,
        diagnostics.ma20_interactions,
    ):
        membership = sum(groups[name].astype(np.uint8) for name in groups)
        np.testing.assert_array_equal(membership, diagnostics.pullback.astype(np.uint8))


def test_reclaims_and_price_recovery_are_event_day_only():
    market, diagnostics = _diagnostics()
    ids = _asset_ids(market)
    reclaim = ids["RECLAIM"]
    touch = ids["TOUCH"]
    assert diagnostics.ma10_reclaim[EVENT_TIME, reclaim]
    assert diagnostics.ma20_reclaim[EVENT_TIME, reclaim]
    assert diagnostics.price_recovery[EVENT_TIME, touch]
    assert diagnostics.pullback_without_reclaim[EVENT_TIME, ids["ABOVE"]]

    high = np.array(market.high, copy=True)
    close = np.array(market.close, copy=True)
    high[EVENT_TIME + 1 :] = 1.0
    close[EVENT_TIME + 1 :] = 1.0
    changed = build_pullback_diagnostics(replace(market, high=high, close=close))
    for name in ("ma10_reclaim", "ma20_reclaim", "price_recovery"):
        np.testing.assert_array_equal(
            getattr(changed, name)[EVENT_TIME], getattr(diagnostics, name)[EVENT_TIME]
        )


@pytest.mark.parametrize("horizon", (3, 5, 10, 20))
def test_forward_returns_enter_next_valid_open_and_exit_at_t_plus_horizon(horizon: int):
    market, diagnostics = _diagnostics()
    above = _asset_ids(market)["ABOVE"]
    expected_close = {3: 13.0, 5: 14.0, 10: 15.0, 20: 16.0}[horizon]
    assert diagnostics.entry_open[EVENT_TIME, above] == 100.0 + EVENT_TIME + 1
    assert diagnostics.forward_returns[horizon][EVENT_TIME, above] == pytest.approx(
        expected_close / (100.0 + EVENT_TIME + 1) - 1.0
    )


def test_valid_bar_alignment_skips_suspension_for_entry_and_horizon():
    market, _ = _diagnostics()
    above = _asset_ids(market)["ABOVE"]
    open_ = np.array(market.open, copy=True)
    high = np.array(market.high, copy=True)
    low = np.array(market.low, copy=True)
    close = np.array(market.close, copy=True)
    for values in (open_, high, low, close):
        values[EVENT_TIME + 1, above] = np.nan
    diagnostics = build_pullback_diagnostics(
        replace(market, open=open_, high=high, low=low, close=close)
    )
    assert diagnostics.entry_open[EVENT_TIME, above] == 100.0 + EVENT_TIME + 2
    # T+3 valid close is row T+4 when T+1 is suspended.
    assert diagnostics.forward_returns[3][EVENT_TIME, above] == pytest.approx(
        10.0 / (100.0 + EVENT_TIME + 2) - 1.0
    )


def test_summary_continuous_and_external_event_relationship_are_json_safe():
    market, diagnostics = _diagnostics()
    events = np.zeros(market.shape, dtype=bool)
    events[EVENT_TIME, _asset_ids(market)["RECLAIM"]] = True
    summary = summarize_pullback_diagnostics(diagnostics)
    relationship = summarize_external_event_relationship(events, diagnostics)
    assert summary["counts"]["breakout_excluded"] >= 1
    assert set(summary["depth_buckets"]) == set(DEPTH_BUCKETS)
    assert set(summary["duration_buckets"]) == set(DURATION_BUCKETS)
    assert set(summary["ma20_interactions"]) == set(MA20_INTERACTIONS)
    assert set(summary["continuous_spearman"]) == {
        "pullback_depth_pct",
        "pullback_depth_atr",
        "duration_bars",
        "distance_to_ma20_pct",
        "distance_to_ma20_atr",
    }
    assert relationship["event_count"] == 1
    assert relationship["pullback_count"] == 1


def test_event_date_aggregate_and_outlier_dependency_use_event_level_values():
    values = np.array([[0.10, 0.20], [-0.10, 0.30], [np.nan, np.nan]], dtype=np.float32)
    mask = np.array([[True, True], [True, True], [False, False]])
    aggregate = event_date_aggregate(values, mask)
    dependency = positive_return_outlier_dependency(values, mask)
    assert aggregate == {
        "date_count": 2,
        "mean": 0.125,
        "median": 0.125,
        "positive_day_rate": 1.0,
        "p25": 0.1125,
        "p75": 0.1375,
    }
    assert dependency["count"] == 4
    assert dependency["top_1_positive_contribution"] == pytest.approx(0.5)
    assert dependency["top_5_positive_contribution"] == pytest.approx(1.0)
