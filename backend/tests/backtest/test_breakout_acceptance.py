"""Point-in-time acceptance diagnostics for frozen V4 BREAKOUT events."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.backtest.breakout_acceptance import (
    ACCEPTANCE_STATES,
    build_breakout_acceptance_window,
    conditional_state_transitions,
    prebreakout_atr_ratio,
    prior_breakout_pivot,
    summarize_breakout_acceptance,
)
from app.backtest.matrix import build_market_data_matrix

EVENT_TIME = 22


def _market():
    symbols = ("HELD", "RECLAIMED", "FAILED", "FAILED_RECLAIMED")
    closes = {
        "HELD": (10, 10, 10, 11, 12, 13),
        "RECLAIMED": (10, 10, 10, 9, 11, 12),
        "FAILED": (10, 10, 10, 9, 8, 7),
        "FAILED_RECLAIMED": (10, 10, 10, 9, 8, 12),
    }
    start = date(2026, 1, 2)
    rows = []
    for symbol in symbols:
        values = (10,) * (EVENT_TIME + 1) + closes[symbol][3:] + (13,) * 24
        for time_id, close in enumerate(values):
            low = close - 0.2
            if symbol == "HELD" and time_id == EVENT_TIME + 1:
                low = 9.8  # A wick retest is orthogonal to the close state.
            rows.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "date": start + timedelta(days=time_id),
                    "open": 100.0 + time_id,
                    "high": close + 0.5,
                    "low": low,
                    "close": float(close),
                    "volume": 1_000.0,
                    "amount": 100_000.0,
                }
            )
    market = build_market_data_matrix(pl.DataFrame(rows))
    events = np.zeros(market.shape, dtype=bool)
    events[EVENT_TIME] = True
    return market, events


def _asset_ids(market):
    return {symbol: asset_id for asset_id, symbol in enumerate(market.symbols)}


def test_frozen_pivot_excludes_event_day_and_future_mutation():
    market, events = _market()
    pivot = prior_breakout_pivot(market)
    assert pivot[EVENT_TIME].tolist() == [10.0, 10.0, 10.0, 10.0]

    close = np.array(market.close, copy=True)
    close[EVENT_TIME] = 100.0
    close[EVENT_TIME + 1 :] = 500.0
    mutated = replace(market, close=close)
    np.testing.assert_array_equal(prior_breakout_pivot(mutated)[EVENT_TIME], pivot[EVENT_TIME])

    base = build_breakout_acceptance_window(market, events, confirmation_bars=1)
    post_confirmation_close = np.array(market.close, copy=True)
    post_confirmation_close[EVENT_TIME + 2 :] = 500.0
    future = build_breakout_acceptance_window(
        replace(market, close=post_confirmation_close), events, confirmation_bars=1
    )
    # The event itself is caller-frozen; mutation after K=1 cannot alter K=1.
    np.testing.assert_array_equal(base.state_masks["HELD_ABOVE"], future.state_masks["HELD_ABOVE"])


@pytest.mark.parametrize(
    ("confirmation_bars", "entry_time"),
    [(1, EVENT_TIME + 2), (2, EVENT_TIME + 3), (3, EVENT_TIME + 4)],
)
def test_confirmation_entry_open_is_strictly_after_confirmation(
    confirmation_bars: int,
    entry_time: int,
):
    market, events = _market()
    window = build_breakout_acceptance_window(
        market, events, confirmation_bars=confirmation_bars, horizons=(2,)
    )

    assert window.entry_open[EVENT_TIME].tolist() == [100.0 + entry_time] * 4


def test_states_are_mutually_exclusive_exhaustive_and_wick_is_orthogonal():
    market, events = _market()
    k1 = build_breakout_acceptance_window(market, events, confirmation_bars=1, horizons=(2,))
    k2 = build_breakout_acceptance_window(market, events, confirmation_bars=2, horizons=(2,))
    k3 = build_breakout_acceptance_window(market, events, confirmation_bars=3, horizons=(2,))

    asset_ids = _asset_ids(market)
    assert k1.state_masks["HELD_ABOVE"][EVENT_TIME, asset_ids["HELD"]]
    assert k2.state_masks["RECLAIMED"][EVENT_TIME, asset_ids["RECLAIMED"]]
    assert k3.state_masks["FAILED"][EVENT_TIME, asset_ids["FAILED"]]
    assert k1.intraday_retest[EVENT_TIME].tolist() == [True, True, True, True]
    for window in (k1, k2, k3):
        membership = sum(window.state_masks[state].astype(np.uint8) for state in ACCEPTANCE_STATES)
        np.testing.assert_array_equal(membership, window.eligible_mask.astype(np.uint8))


def test_forward_return_uses_confirmation_next_open_and_future_close():
    market, events = _market()
    window = build_breakout_acceptance_window(market, events, confirmation_bars=1, horizons=(2,))

    # K=1 confirms at T+1, enters T+2 open, and h=2 exits T+3 close.
    assert window.forward_returns[2][EVENT_TIME, _asset_ids(market)["HELD"]] == pytest.approx(
        13.0 / (100.0 + EVENT_TIME + 2) - 1.0
    )


def test_confirmation_state_ignores_data_after_its_allowed_boundary():
    market, events = _market()
    base = build_breakout_acceptance_window(market, events, confirmation_bars=2)
    close = np.array(market.close, copy=True)
    close[EVENT_TIME + 3 :] = 1.0  # K=2 confirmation ends at T+2.
    changed = build_breakout_acceptance_window(
        replace(market, close=close), events, confirmation_bars=2
    )

    for state in ACCEPTANCE_STATES:
        np.testing.assert_array_equal(base.state_masks[state], changed.state_masks[state])


def test_valid_bar_index_skips_missing_session_for_delayed_entry():
    market, events = _market()
    open_ = np.array(market.open, copy=True)
    high = np.array(market.high, copy=True)
    low = np.array(market.low, copy=True)
    close = np.array(market.close, copy=True)
    # HELD's T+2 is absent.  Its K=1 entry must use T+3's next valid open.
    for values in (open_, high, low, close):
        values[EVENT_TIME + 2, 0] = np.nan
    window = build_breakout_acceptance_window(
        replace(market, open=open_, high=high, low=low, close=close),
        events,
        confirmation_bars=1,
        horizons=(2,),
    )

    assert window.entry_open[EVENT_TIME, 0] == 100.0 + EVENT_TIME + 3


def test_contraction_ratio_is_prebreakout_and_transition_summary_is_stable():
    market, events = _market()
    baseline = prebreakout_atr_ratio(market)
    high = np.array(market.high, copy=True)
    high[EVENT_TIME:] *= 5.0
    changed = prebreakout_atr_ratio(replace(market, high=high))
    np.testing.assert_array_equal(baseline[EVENT_TIME], changed[EVENT_TIME])

    k1 = build_breakout_acceptance_window(market, events, confirmation_bars=1, horizons=(2,))
    k3 = build_breakout_acceptance_window(market, events, confirmation_bars=3, horizons=(2,))
    transitions = conditional_state_transitions(k1, k3)
    assert transitions["FAILED"]["RECLAIMED"] == {"count": 2, "rate": 0.6667}
    summary = summarize_breakout_acceptance(k3)
    assert summary["states"]["HELD_ABOVE"]["count"] == 1
    assert sum(row["count"] for row in summary["acceptance_atr_buckets"].values()) == 4
    assert all("2" in row["forward_returns"] for row in summary["acceptance_atr_buckets"].values())
