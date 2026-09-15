"""PIT and ablation regression coverage for the V4.5c research strategy."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.backtest.matrix import (
    build_market_data_matrix,
    build_market_matrix_from_signals,
)
from app.backtest.strategy import StrategyDependencyResolver
from app.strategy.engine import StrategyEngine
from app.strategy.fundamental_veto import (
    V4_5_BREAKOUT_STRATEGY_ID,
    FundamentalVetoResult,
    apply_fundamental_veto,
    fundamental_veto_required_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = REPO_ROOT / "data" / "strategies" / "custom" / "v4_5_breakout_strategy.py"
V4_STRATEGY_PATH = REPO_ROOT / "data" / "strategies" / "custom" / "v4_trend_strategy.py"
PATTERN_ID = "BREAKOUT_SETUP::PIVOT_BREAK_CONFIRM"


def _strategy():
    return StrategyEngine._load_file(STRATEGY_PATH).matrix_strategy


def _v4_strategy():
    return StrategyEngine._load_file(V4_STRATEGY_PATH).matrix_strategy


def _market(*, extension: float = 0.15, prior_atr_pct: float = 0.018):
    """Return a market with a qualified pivot setup at ``signal_bar``."""
    count = 75
    signal_bar = 72
    start = date(2026, 1, 1)
    closes = np.full(count, 10.0, dtype=np.float32)
    closes[signal_bar - 1] = 9.85
    closes[signal_bar] = 10.0 + extension
    rows = [
        {
            "symbol": "000001.SZ",
            "name": "测试股",
            "date": start + timedelta(days=index),
            "open": float(close - 0.1),
            "high": float(close + 0.2),
            "low": float(close - 0.2),
            "close": float(close),
            "volume": 1_000_000.0,
            "amount": 100_000_000.0,
        }
        for index, close in enumerate(closes)
    ]
    market = build_market_data_matrix(pl.DataFrame(rows))
    shape = market.shape
    atr_pct = np.full(shape, 0.025, dtype=np.float32)
    atr_pct[signal_bar - 1, 0] = prior_atr_pct
    fields = {
        "ma20": np.full(shape, 9.5, dtype=np.float32),
        "ma60": np.full(shape, 9.0, dtype=np.float32),
        "momentum_20d": np.full(shape, 0.10, dtype=np.float32),
        "momentum_60d": np.full(shape, 0.10, dtype=np.float32),
        "rsi_14": np.full(shape, 50.0, dtype=np.float32),
        "atr_pct": atr_pct,
        "vol_ratio_5d": np.full(shape, 1.5, dtype=np.float32),
        "change_pct": np.full(shape, 0.01, dtype=np.float32),
    }
    return replace(market, fields=fields), signal_bar


def _mutate_market(market, *, close=None, high=None, low=None, fields=None):
    return replace(
        market,
        close=np.array(market.close if close is None else close, copy=True),
        high=np.array(market.high if high is None else high, copy=True),
        low=np.array(market.low if low is None else low, copy=True),
        fields={
            name: np.array(values, copy=True)
            for name, values in (market.fields if fields is None else fields).items()
        },
    )


def test_v45_loads_and_declares_the_research_contract():
    definition = StrategyEngine._load_file(STRATEGY_PATH)
    strategy = definition.matrix_strategy

    assert strategy.required_fields() == frozenset({"open", "high", "low", "close", "volume"})
    assert strategy.required_warmup_bars({}) == 70
    assert strategy.entry_pattern_ids == (PATTERN_ID,)
    expected_veto_fields = frozenset(
        {
            "roe_latest",
            "revenue_yoy_latest",
            "net_income_yoy_latest",
            "net_margin_latest",
            "gross_margin_latest",
        }
    )
    assert fundamental_veto_required_fields(V4_5_BREAKOUT_STRATEGY_ID) == expected_veto_fields
    plan = StrategyDependencyResolver().resolve(
        definition,
        params=StrategyEngine.resolve_params(definition),
        basic_filter=definition.basic_filter,
        entry_signals=definition.entry_signals,
        exit_signals=definition.exit_signals,
        overrides={},
    )
    assert plan.fundamental_columns == expected_veto_fields


def test_v45_pivot_excludes_t_and_setup_uses_only_t_minus_one():
    market, signal_bar = _market()
    strategy = _strategy()
    params = {"use_contraction": False, "use_extension": False}
    baseline = strategy.compute_signals(market, params)

    # Raising T itself must not raise the prior pivot.  If T were included in
    # pivot[T], the trigger would disappear instead of remaining qualified.
    close = np.array(market.close, copy=True)
    # Keep the current close within V4's unchanged MA20-bias hard limit so
    # this assertion isolates pivot membership rather than a trend veto.
    close[signal_bar, 0] = 10.6
    changed_t = _mutate_market(market, close=close)
    changed = strategy.compute_signals(changed_t, params)

    assert baseline.entry[signal_bar, 0] == changed.entry[signal_bar, 0] == 1
    assert (
        baseline.entry_pattern_mask[signal_bar, 0] == changed.entry_pattern_mask[signal_bar, 0] == 1
    )


def test_v45_contraction_uses_only_prebreakout_atr_data():
    market, signal_bar = _market()
    strategy = _strategy()
    params = {"use_contraction": True, "use_extension": False}
    baseline = strategy.compute_signals(market, params)

    fields = {name: np.array(values, copy=True) for name, values in market.fields.items()}
    # This changes ATR% at T but remains inside V4's hard volatility bound.
    # The T-1 contraction ratio must remain unchanged.
    fields["atr_pct"][signal_bar, 0] = 0.03
    changed = strategy.compute_signals(_mutate_market(market, fields=fields), params)

    assert baseline.entry[signal_bar, 0] == changed.entry[signal_bar, 0] == 1


def test_v45_trigger_uses_t_close_but_future_mutation_cannot_change_t():
    market, signal_bar = _market()
    strategy = _strategy()
    params = {"use_contraction": False, "use_extension": False}
    baseline = strategy.compute_signals(market, params)

    close = np.array(market.close, copy=True)
    close[signal_bar, 0] = 9.99
    not_broken = strategy.compute_signals(_mutate_market(market, close=close), params)
    assert baseline.entry[signal_bar, 0] == 1
    assert not_broken.entry[signal_bar, 0] == 0

    future_close = np.array(market.close, copy=True)
    future_high = np.array(market.high, copy=True)
    future_low = np.array(market.low, copy=True)
    future_close[signal_bar + 1 :, 0] = (30.0, 1.0)
    future_high[signal_bar + 1 :, 0] = (31.0, 1.1)
    future_low[signal_bar + 1 :, 0] = (29.0, 0.9)
    future = strategy.compute_signals(
        _mutate_market(market, close=future_close, high=future_high, low=future_low), params
    )
    assert future.entry[signal_bar, 0] == baseline.entry[signal_bar, 0]
    assert future.entry_pattern_mask[signal_bar, 0] == baseline.entry_pattern_mask[signal_bar, 0]
    assert future.score[signal_bar, 0] == baseline.score[signal_bar, 0]


def test_v45_contraction_and_extension_switches_are_independent():
    # This row has a non-contracting T-1 ATR% and an over-extended T close.
    market, signal_bar = _market(extension=0.35, prior_atr_pct=0.04)
    strategy = _strategy()
    signals = {
        name: strategy.compute_signals(market, params)
        for name, params in {
            "A1": {"use_contraction": False, "use_extension": False},
            "A2": {"use_contraction": True, "use_extension": False},
            "A3": {"use_contraction": False, "use_extension": True},
            "A4": {"use_contraction": True, "use_extension": True},
        }.items()
    }

    assert signals["A1"].entry[signal_bar, 0] == 1
    assert signals["A2"].entry[signal_bar, 0] == 0
    assert signals["A3"].entry[signal_bar, 0] == 0
    assert signals["A4"].entry[signal_bar, 0] == 0
    # Entry eligibility changes, but the shared V4 score remains invariant.
    assert signals["A1"].score[signal_bar, 0] == signals["A4"].score[signal_bar, 0]


def test_v45_reuses_v4_default_score_and_exit_semantics():
    market, _ = _market()
    v4 = _v4_strategy().compute_signals(
        market,
        {
            "__experiment_enabled_entry_patterns": ["BREAKOUT"],
            "__v4_4_2_experiment_context": np.array([0x44], dtype=np.uint8),
        },
    )
    v45 = _strategy().compute_signals(market, {"use_contraction": False, "use_extension": False})

    np.testing.assert_array_equal(v45.score, v4.score)
    np.testing.assert_array_equal(v45.exit, v4.exit)
    np.testing.assert_array_equal(v45.exit_signal_code, v4.exit_signal_code)


def test_v45_pattern_is_delayed_and_fundamental_veto_clears_it():
    market, signal_bar = _market()
    signals = _strategy().compute_signals(market, {})
    matrix = build_market_matrix_from_signals(market, signals, entry_delay_bars=1)
    assert signals.entry[signal_bar, 0] == 1
    assert matrix.entry[signal_bar, 0] == 0
    assert matrix.entry[signal_bar + 1, 0] == 1

    veto = np.zeros(market.shape, dtype=bool)
    veto[signal_bar, 0] = True
    filtered = apply_fundamental_veto(
        signals,
        FundamentalVetoResult(veto=veto, reason_mask=np.zeros(market.shape, dtype=np.uint16)),
    )
    assert filtered.entry[signal_bar, 0] == 0
    assert filtered.entry_pattern_mask[signal_bar, 0] == 0
