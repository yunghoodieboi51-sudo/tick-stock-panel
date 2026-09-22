"""V4.4.1 entry-pattern attribution regression coverage."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, MatcherConfig, TradeRecord
from app.backtest.matrix import (
    build_market_data_matrix,
    build_market_matrix_from_signals,
    make_signal_matrix,
    slice_market_data_matrix,
)
from app.backtest.strategy import (
    build_entry_pattern_breakdown,
    build_matched_pattern_diagnostics,
    trade_pnl_diagnostics,
)
from app.strategy.engine import StrategyDataContext, StrategyEngine
from app.strategy.fundamental_veto import FundamentalVetoResult, apply_fundamental_veto

REPO_ROOT = Path(__file__).resolve().parents[2]
PATTERN_IDS = (
    "BREAKOUT",
    "PULLBACK_RESTART",
    "CONSOLIDATION_BREAKOUT",
)


def _load_v4_strategy():
    return StrategyEngine._load_file(
        REPO_ROOT / "data" / "strategies" / "custom" / "v4_trend_strategy.py"
    ).matrix_strategy


def _market(kind: str):
    """Build a controlled, precomputed-feature market with one last-bar signal."""
    count = 70
    start = date(2026, 1, 1)
    closes = np.full(count, 10.0, dtype=np.float32)
    lows = closes - 0.1
    ma20 = np.full(count, 10.0, dtype=np.float32)
    ma60 = np.full(count, 9.0, dtype=np.float32)
    daily_return = np.full(count, 0.01, dtype=np.float32)

    if kind == "BREAKOUT":
        closes[-15] = 8.0  # makes the recent consolidation range deliberately wide
        closes[-2] = 9.0
        lows[-6:-1] = 10.3  # no MA20 touch in the pullback window
        closes[-1] = 10.5
        lows[-1] = 10.4
    elif kind == "PULLBACK_RESTART":
        closes[-15] = 8.0
        closes[-10] = 11.0  # prior 20-bar high prevents a price breakout
        closes[-5] = 8.5  # prevents a tight-range breakout
        closes[-1] = 10.5
        lows[-2] = 9.9  # prior-bar MA20 touch
        lows[-1] = 10.4
        daily_return[-1] = 0.02
    elif kind == "CONSOLIDATION_BREAKOUT":
        closes[-20] = 11.0  # outside 15-bar range, inside 20-bar stage-high window
        closes[-1] = 10.5
        lows[-1] = 10.4
        ma20.fill(9.5)
    elif kind == "MULTI":
        closes[-1] = 10.5
        lows[-2] = 9.9
        lows[-1] = 10.4
        daily_return[-1] = 0.02
    elif kind == "MULTI_EARLY":
        closes[-2] = 10.5
        lows[-3] = 9.9
        lows[-2] = 10.4
        daily_return[-2] = 0.02
    else:
        raise ValueError(kind)

    rows = [
        {
            "symbol": "000001.SZ",
            "name": "测试股",
            "date": start + timedelta(days=index),
            "open": float(close - 0.1),
            "high": float(close + 0.2),
            "low": float(lows[index]),
            "close": float(close),
            "volume": 1_000_000.0,
            "amount": 100_000_000.0,
        }
        for index, close in enumerate(closes)
    ]
    panel = pl.DataFrame(rows)
    market = build_market_data_matrix(panel)
    shape = market.shape
    fields = {
        "ma20": np.broadcast_to(ma20[:, None], shape).copy(),
        "ma60": np.broadcast_to(ma60[:, None], shape).copy(),
        "momentum_20d": np.full(shape, 0.10, dtype=np.float32),
        "momentum_60d": np.full(shape, 0.10, dtype=np.float32),
        "rsi_14": np.full(shape, 50.0, dtype=np.float32),
        "atr_pct": np.full(shape, 0.02, dtype=np.float32),
        "vol_ratio_5d": np.full(shape, 1.5, dtype=np.float32),
        "change_pct": np.broadcast_to(daily_return[:, None], shape).copy(),
    }
    return panel, replace(market, fields=fields)


@pytest.mark.parametrize(
    ("kind", "expected_mask", "expected_primary_code", "expected_score"),
    [
        ("BREAKOUT", 0b001, 0, 70.537),
        ("PULLBACK_RESTART", 0b010, 1, 68.204),
        ("CONSOLIDATION_BREAKOUT", 0b100, 2, 66.746),
    ],
)
def test_v4_each_entry_pattern_has_stable_primary_attribution(
    kind: str,
    expected_mask: int,
    expected_primary_code: int,
    expected_score: float,
):
    _, market = _market(kind)
    signals = _load_v4_strategy().compute_signals(market, {})

    assert signals.entry[-1, 0] == 1
    assert signals.entry_signal_code[-1, 0] == expected_primary_code
    assert signals.entry_pattern_ids == PATTERN_IDS
    assert signals.entry_pattern_mask is not None
    assert signals.entry_pattern_mask[-1, 0] == expected_mask
    # Fixed baseline protects V4.1's existing score as diagnostic metadata is added.
    assert signals.score[-1, 0] == pytest.approx(expected_score, abs=0.001)
    assert set(signals.diagnostics) == {"trend", "momentum", "breakout", "volume", "volatility"}
    assert all(values.shape == signals.shape and not values.flags.writeable for values in signals.diagnostics.values())


def test_v4_multi_pattern_mask_keeps_all_matches_and_primary_order():
    panel, market = _market("MULTI")
    strategy = _load_v4_strategy()
    signals = strategy.compute_signals(market, {})

    assert signals.entry[-1, 0] == 1
    assert signals.entry_signal_code[-1, 0] == 0
    assert signals.entry_pattern_mask is not None
    assert signals.entry_pattern_mask[-1, 0] == 0b111

    engine = StrategyEngine(strategy_dirs=[])
    engine._strategies["v4_trend_strategy"] = StrategyEngine._load_file(
        REPO_ROOT / "data" / "strategies" / "custom" / "v4_trend_strategy.py"
    )
    as_of = date.fromisoformat(market.timestamp_labels[-1][:10])
    result = engine.run(
        "v4_trend_strategy",
        StrategyDataContext(
            asset_type="stock",
            timeframe="1d",
            as_of=as_of,
            current=panel.filter(pl.col("date") == as_of),
            history=panel,
            market=market,
        ),
        overrides={
            "basic_filter": {"enabled": False},
            "fundamental_veto": {"enabled": False},
        },
        # A normal resolved-params path must not activate the offline-only
        # pattern selector, even when an untrusted caller supplies its name.
        params={"__experiment_enabled_entry_patterns": ["PULLBACK_RESTART"]},
    )
    assert result.rows[0]["primary_entry_pattern"] == "BREAKOUT"
    assert result.rows[0]["matched_entry_patterns"] == list(PATTERN_IDS)


def test_v4_research_pattern_override_requires_runner_context_and_keeps_score_formula():
    _, market = _market("MULTI")
    strategy = _load_v4_strategy()
    baseline = strategy.compute_signals(market, {})
    explicit_default = strategy.compute_signals(
        market,
        {"__experiment_enabled_entry_patterns": list(PATTERN_IDS)},
    )
    untrusted_pullback_only = strategy.compute_signals(
        market,
        {"__experiment_enabled_entry_patterns": ["PULLBACK_RESTART"]},
    )
    pullback_only = strategy.compute_signals(
        market,
        {
            "__experiment_enabled_entry_patterns": ["PULLBACK_RESTART"],
            "__v4_4_2_experiment_context": np.array([0x44], dtype=np.uint8),
        },
    )

    assert np.array_equal(baseline.entry, explicit_default.entry)
    assert np.array_equal(baseline.entry_signal_code, explicit_default.entry_signal_code)
    assert np.array_equal(baseline.entry_pattern_mask, explicit_default.entry_pattern_mask)
    assert np.array_equal(baseline.score, explicit_default.score)
    assert np.array_equal(baseline.entry, untrusted_pullback_only.entry)
    assert np.array_equal(baseline.entry_signal_code, untrusted_pullback_only.entry_signal_code)
    assert np.array_equal(baseline.entry_pattern_mask, untrusted_pullback_only.entry_pattern_mask)
    assert np.array_equal(baseline.score, untrusted_pullback_only.score)
    assert pullback_only.entry[-1, 0] == 1
    assert pullback_only.entry_signal_code[-1, 0] == 1
    assert pullback_only.entry_pattern_mask[-1, 0] == 0b010
    assert pullback_only.score[-1, 0] == baseline.score[-1, 0]


def test_v4_pattern_is_delayed_and_frozen_in_backtest_trade():
    _, market = _market("MULTI")
    signals = _load_v4_strategy().compute_signals(market, {})
    matrix = build_market_matrix_from_signals(
        market,
        signals,
        entry_delay_bars=1,
    )
    assert matrix.entry_pattern_mask is not None
    assert matrix.entry_pattern_mask[-1, 0] == 0

    # Place a frozen signal before the final two bars so T+1 can be executed and
    # closed.  This isolates the execution-time metadata contract from the
    # strategy's final-bar diagnostic signal above.
    entry = np.zeros(market.shape, dtype=np.uint8)
    entry[-3, 0] = 1
    entry_code = np.full(market.shape, -1, dtype=np.int16)
    entry_code[-3, 0] = 0
    pattern_mask = np.zeros(market.shape, dtype=np.uint8)
    pattern_mask[-3, 0] = 0b111
    frozen_signals = make_signal_matrix(
        market.shape,
        entry=entry,
        score=signals.score,
        entry_signal_code=entry_code,
        entry_pattern_mask=pattern_mask,
        entry_pattern_ids=PATTERN_IDS,
    )
    matrix = build_market_matrix_from_signals(market, frozen_signals, entry_delay_bars=1)
    trade = (
        BacktestEngine(repo=None)
        .simulate_market_matrix(
            matrix,
            MatcherConfig(
                matching="open_t+1",
                fees_pct=0,
                slippage_bps=0,
                max_positions=1,
            ),
        )
        .trades[0]
    )
    assert trade.primary_entry_pattern == "BREAKOUT"
    assert trade.matched_entry_patterns == PATTERN_IDS


def test_v4_non_entry_has_no_pattern_metadata():
    _, market = _market("MULTI")
    signals = _load_v4_strategy().compute_signals(market, {"min_total_score": 101})

    assert not signals.entry.any()
    assert signals.entry_pattern_mask is not None
    assert not signals.entry_pattern_mask.any()


def test_v4_pattern_metadata_does_not_use_future_bars():
    _, market = _market("MULTI_EARLY")
    full = _load_v4_strategy().compute_signals(market, {})
    prefix = _load_v4_strategy().compute_signals(slice_market_data_matrix(market, 0, 69), {})

    assert full.entry[-2, 0] == prefix.entry[-1, 0] == 1
    assert full.entry_signal_code[-2, 0] == prefix.entry_signal_code[-1, 0] == 0
    assert full.entry_pattern_mask is not None
    assert prefix.entry_pattern_mask is not None
    assert full.entry_pattern_mask[-2, 0] == prefix.entry_pattern_mask[-1, 0] == 0b111
    assert full.score[-2, 0] == pytest.approx(prefix.score[-1, 0])


def test_v4_fundamental_veto_only_removes_vetoed_pattern_metadata():
    _, market = _market("MULTI")
    signals = _load_v4_strategy().compute_signals(market, {})
    veto = np.zeros(market.shape, dtype=bool)
    veto[-1, 0] = True
    reason_mask = np.zeros(market.shape, dtype=np.uint16)
    filtered = apply_fundamental_veto(
        signals,
        FundamentalVetoResult(veto=veto, reason_mask=reason_mask),
    )

    assert filtered.entry_pattern_ids == PATTERN_IDS
    assert filtered.entry_pattern_mask is not None
    assert filtered.entry_pattern_mask[-1, 0] == 0
    assert filtered.entry[-1, 0] == 0


def test_strategy_without_pattern_metadata_keeps_the_legacy_contract():
    _, market = _market("MULTI")
    signals = make_signal_matrix(market.shape, entry=np.ones(market.shape, dtype=np.uint8))
    assert signals.entry_pattern_mask is None
    matrix = build_market_matrix_from_signals(market, signals)
    assert matrix.entry_pattern_mask is None
    assert matrix.entry_pattern_ids == ()


def test_entry_pattern_breakdown_uses_primary_once_and_handles_boundaries():
    trades = [
        TradeRecord(
            "A",
            date(2026, 1, 1),
            date(2026, 1, 3),
            10,
            11,
            0.10,
            2,
            "end",
            primary_entry_pattern="BREAKOUT",
            matched_entry_patterns=("BREAKOUT", "PULLBACK_RESTART"),
        ),
        TradeRecord(
            "B",
            date(2026, 1, 1),
            date(2026, 1, 2),
            10,
            9,
            -0.10,
            1,
            "end",
            primary_entry_pattern="BREAKOUT",
            matched_entry_patterns=("BREAKOUT",),
        ),
        TradeRecord(
            "C",
            date(2026, 1, 1),
            date(2026, 1, 4),
            10,
            12,
            0.20,
            3,
            "end",
            primary_entry_pattern="PULLBACK_RESTART",
            matched_entry_patterns=("PULLBACK_RESTART",),
        ),
    ]
    breakdown = build_entry_pattern_breakdown(trades, PATTERN_IDS)

    assert breakdown["BREAKOUT"]["trade_count"] == 2
    assert breakdown["BREAKOUT"]["win_count"] == 1
    assert breakdown["BREAKOUT"]["loss_count"] == 1
    assert breakdown["BREAKOUT"]["win_rate"] == 0.5
    assert breakdown["BREAKOUT"]["average_return"] == 0.0
    assert breakdown["BREAKOUT"]["gross_profit"] == 0.1
    assert breakdown["BREAKOUT"]["gross_loss"] == -0.1
    assert breakdown["BREAKOUT"]["standard_profit_factor"] == 1.0
    assert breakdown["BREAKOUT"]["profit_factor"] == 1.0
    assert breakdown["BREAKOUT"]["payoff_ratio"] == 1.0
    assert breakdown["BREAKOUT"]["average_hold_days"] == 1.5
    assert breakdown["PULLBACK_RESTART"]["standard_profit_factor"] is None
    assert breakdown["PULLBACK_RESTART"]["profit_factor"] is None
    assert breakdown["CONSOLIDATION_BREAKOUT"]["trade_count"] == 0
    assert breakdown["CONSOLIDATION_BREAKOUT"]["standard_profit_factor"] is None


def test_standard_profit_factor_boundaries_are_json_safe():
    assert trade_pnl_diagnostics(np.array([], dtype=float)) == {
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "standard_profit_factor": None,
        "payoff_ratio": None,
    }
    assert trade_pnl_diagnostics(np.array([0.10, 0.20]))["standard_profit_factor"] is None
    assert trade_pnl_diagnostics(np.array([-0.10, -0.20]))["standard_profit_factor"] == 0.0
    metrics = trade_pnl_diagnostics(np.array([0.10, 0.20, -0.10]))
    assert metrics["gross_profit"] == 0.3
    assert metrics["gross_loss"] == -0.1
    assert metrics["standard_profit_factor"] == 3.0
    assert metrics["payoff_ratio"] == 1.5


def test_matched_pattern_diagnostics_count_overlaps_without_pnl_duplication():
    trades = [
        TradeRecord(
            "A",
            date(2026, 1, 1),
            date(2026, 1, 2),
            10,
            11,
            0.1,
            1,
            "end",
            primary_entry_pattern="BREAKOUT",
            matched_entry_patterns=("BREAKOUT", "PULLBACK_RESTART"),
        ),
        TradeRecord(
            "B",
            date(2026, 1, 1),
            date(2026, 1, 2),
            10,
            11,
            0.1,
            1,
            "end",
            primary_entry_pattern="BREAKOUT",
            matched_entry_patterns=PATTERN_IDS,
        ),
        TradeRecord(
            "C",
            date(2026, 1, 1),
            date(2026, 1, 2),
            10,
            9,
            -0.1,
            1,
            "end",
            primary_entry_pattern="CONSOLIDATION_BREAKOUT",
            matched_entry_patterns=("CONSOLIDATION_BREAKOUT",),
        ),
    ]
    counts, combinations = build_matched_pattern_diagnostics(trades, PATTERN_IDS)
    breakdown = build_entry_pattern_breakdown(trades, PATTERN_IDS)

    assert counts == {
        "BREAKOUT": 2,
        "PULLBACK_RESTART": 2,
        "CONSOLIDATION_BREAKOUT": 2,
    }
    assert combinations == {
        "BREAKOUT_ONLY": 0,
        "PULLBACK_RESTART_ONLY": 0,
        "CONSOLIDATION_BREAKOUT_ONLY": 1,
        "BREAKOUT+PULLBACK_RESTART": 1,
        "BREAKOUT+CONSOLIDATION_BREAKOUT": 0,
        "PULLBACK_RESTART+CONSOLIDATION_BREAKOUT": 0,
        "ALL_THREE": 1,
    }
    assert sum(row["trade_count"] for row in breakdown.values()) == len(trades)
