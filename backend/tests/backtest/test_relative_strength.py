from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.backtest.matrix import build_market_data_matrix, matrix_feature
from app.backtest.relative_strength import (
    bucket_return_diagnostics,
    cross_sectional_percentile,
    forward_returns_from_next_open,
    percentile_bucket,
    relative_strength_eligible_universe,
    replace_signal_scores,
    spearman_correlation,
    trade_bucket_diagnostics,
)
from app.strategy.engine import StrategyEngine


def _market(rows: int = 8):
    start = date(2026, 1, 2)
    panel = pl.DataFrame(
        [
            {
                "symbol": symbol,
                "name": symbol,
                "date": start + timedelta(days=day),
                "open": 10.0 + asset + day * 0.1,
                "high": 10.3 + asset + day * 0.1,
                "low": 9.8 + asset + day * 0.1,
                "close": 10.1 + asset + day * (0.1 + asset * 0.02),
                "volume": 1000.0,
                "amount": 100000.0,
            }
            for day in range(rows)
            for asset, symbol in enumerate(("000001.SZ", "000002.SZ", "510300.SH"))
        ]
    )
    return build_market_data_matrix(panel)


def test_cross_sectional_percentile_uses_asset_axis_average_ties_and_nan():
    values = np.array([[1.0, 1.0, 3.0, np.nan], [4.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    eligible = np.array([[True, True, True, True], [True, False, True, True]])

    result = cross_sectional_percentile(values, eligible)

    assert np.allclose(result[0, :3], [0.25, 0.25, 1.0])
    assert np.isnan(result[0, 3])
    assert np.allclose(result[1, [0, 2, 3]], [1.0, 0.5, 0.0])
    assert np.isnan(result[1, 1])


def test_relative_strength_universe_and_forward_returns_are_pit_local():
    market = _market()
    momentum = matrix_feature(market, "momentum_3d")
    eligible = relative_strength_eligible_universe(market, momentum)
    returns = forward_returns_from_next_open(market, (3,))

    assert eligible.shape == market.shape
    assert not eligible[:3].any()
    assert np.isfinite(returns[3][0]).all()
    assert np.isnan(returns[3][-3:]).all()


def test_bucket_and_trade_diagnostics_are_fixed_and_json_safe():
    percentiles = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    returns = np.array([[0.01, -0.01, 0.02, 0.03, np.nan]], dtype=np.float32)
    events = np.ones(percentiles.shape, dtype=bool)

    assert percentile_bucket(percentiles).tolist() == [[0, 1, 2, 3, 4]]
    result = bucket_return_diagnostics(percentiles, returns, events)
    assert result["0-20"]["count"] == 1
    assert result["80-100"]["count"] == 0
    assert spearman_correlation([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])["rho"] == 1.0
    trades = [{"pnl_pct": 0.1, "duration": 2}, {"pnl_pct": -0.05, "duration": 4}]
    grouped = trade_bucket_diagnostics(trades, [0, 0])
    assert grouped["0-20"]["trade_count"] == 2
    assert grouped["0-20"]["average_hold_days"] == 3.0


def test_future_mutation_cannot_change_prior_relative_strength_or_v4_candidate_rank():
    # This retains the V4 signal's precomputed technical fields while momentum
    # itself is derived from close, so mutating T+1 onward exercises both paths.
    count = 70
    start = date(2026, 1, 2)
    symbols = ("000001.SZ", "000002.SZ", "000003.SZ")
    rows = []
    for day in range(count):
        for symbol in symbols:
            close = 10.0
            if day == 55:
                close = 8.0
            elif day == 67:
                close = 9.0
            elif day == 68:
                close = 10.5
            low = 10.3 if 63 <= day < 68 else close - 0.2
            rows.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "date": start + timedelta(days=day),
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": low,
                    "close": close,
                    "volume": 1000000.0,
                    "amount": 100000000.0,
                }
            )
    base = build_market_data_matrix(pl.DataFrame(rows))
    shape = base.shape
    fields = {
        "ma20": np.full(shape, 10.0, dtype=np.float32),
        "ma60": np.full(shape, 9.0, dtype=np.float32),
        "momentum_20d": np.full(shape, 0.10, dtype=np.float32),
        "momentum_60d": np.full(shape, 0.10, dtype=np.float32),
        "rsi_14": np.full(shape, 50.0, dtype=np.float32),
        "atr_pct": np.full(shape, 0.02, dtype=np.float32),
        "vol_ratio_5d": np.full(shape, 1.5, dtype=np.float32),
        "change_pct": np.full(shape, 0.01, dtype=np.float32),
    }
    market = replace(base, fields=fields)
    strategy = StrategyEngine._load_file(
        Path(__file__).resolve().parents[3]
        / "data"
        / "strategies"
        / "custom"
        / "v4_trend_strategy.py"
    ).matrix_strategy
    time_id = 68
    baseline_signals = strategy.compute_signals(market, {})
    baseline_rs = cross_sectional_percentile(
        matrix_feature(market, "momentum_20d"),
        relative_strength_eligible_universe(market, matrix_feature(market, "momentum_20d")),
    )
    candidate_ids = np.flatnonzero(baseline_signals.entry[time_id])
    assert len(candidate_ids) >= 2
    baseline_order = candidate_ids[
        np.argsort(-np.nan_to_num(baseline_rs[time_id, candidate_ids], nan=-1.0), kind="mergesort")
    ]

    mutated_close = np.array(market.close, copy=True)
    mutated_close[time_id + 1 :, 0] *= 9.0
    mutated = replace(market, close=mutated_close)
    mutated_signals = strategy.compute_signals(mutated, {})
    mutated_momentum = matrix_feature(mutated, "momentum_20d")
    mutated_rs = cross_sectional_percentile(
        mutated_momentum,
        relative_strength_eligible_universe(mutated, mutated_momentum),
    )
    mutated_candidate_ids = np.flatnonzero(mutated_signals.entry[time_id])
    mutated_order = mutated_candidate_ids[
        np.argsort(
            -np.nan_to_num(mutated_rs[time_id, mutated_candidate_ids], nan=-1.0), kind="mergesort"
        )
    ]

    assert np.allclose(baseline_rs[time_id], mutated_rs[time_id], equal_nan=True)
    assert np.array_equal(baseline_signals.entry[time_id], mutated_signals.entry[time_id])
    assert np.array_equal(baseline_order, mutated_order)
    assert replace_signal_scores(baseline_signals, baseline_rs).entry.shape == market.shape
