"""Point-in-time diagnostics for post-breakout price acceptance.

This module deliberately evaluates an already-frozen event mask.  It does not
generate entries, alter scores, or invoke the matching engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    matrix_feature,
    safe_divide,
    valid_rolling_max,
    valid_rolling_mean,
    valid_shift,
)
from app.backtest.relative_strength import spearman_correlation, summarize_returns

AcceptanceState = Literal["HELD_ABOVE", "RECLAIMED", "FAILED"]
ACCEPTANCE_STATES: tuple[AcceptanceState, ...] = ("HELD_ABOVE", "RECLAIMED", "FAILED")


@dataclass(frozen=True)
class BreakoutAcceptanceWindow:
    """Frozen-pivot diagnostics for one post-breakout confirmation horizon."""

    confirmation_bars: int
    event_mask: np.ndarray
    eligible_mask: np.ndarray
    pivot: np.ndarray
    atr: np.ndarray
    state_masks: dict[AcceptanceState, np.ndarray]
    intraday_retest: np.ndarray
    acceptance_atr: np.ndarray
    mae: np.ndarray
    mfe: np.ndarray
    mae_atr: np.ndarray
    mfe_atr: np.ndarray
    entry_open: np.ndarray
    forward_returns: dict[int, np.ndarray]


def prior_breakout_pivot(market: MarketDataMatrix, lookback: int = 20) -> np.ndarray:
    """Return T's pivot: the maximum close over T's preceding valid bars."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    close_valid = np.isfinite(market.close) & (market.close > 0)
    current_high = valid_rolling_max(market.close, close_valid, lookback)
    return valid_shift(current_high, 1, np.isfinite(current_high))


def prebreakout_atr_ratio(market: MarketDataMatrix, window: int = 20) -> np.ndarray:
    """Return V4.5c's T-1 ATR% divided by its trailing T-1 baseline."""
    if window < 1:
        raise ValueError("window must be positive")
    atr_pct = matrix_feature(market, "atr_pct")
    prior_atr_pct = valid_shift(atr_pct, 1, np.isfinite(atr_pct))
    baseline = valid_rolling_mean(prior_atr_pct, np.isfinite(prior_atr_pct), window)
    return safe_divide(prior_atr_pct, baseline)


def build_breakout_acceptance_window(
    market: MarketDataMatrix,
    event_mask: np.ndarray,
    *,
    confirmation_bars: int,
    pivot: np.ndarray | None = None,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> BreakoutAcceptanceWindow:
    """Classify frozen events after K valid bars and align post-K forward returns.

    A horizon ``h`` means confirmation-day-plus-``h`` close divided by the
    next valid-bar open after confirmation.  This preserves the project's
    established ``T+1 open -> T+h close`` convention after replacing T with
    the confirmation day.
    """
    if confirmation_bars not in (1, 2, 3):
        raise ValueError("confirmation_bars must be one of 1, 2, or 3")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive trading-bar counts")
    events = np.asarray(event_mask, dtype=bool)
    if events.shape != market.shape:
        raise ValueError("event_mask shape does not match MarketDataMatrix")

    close, high, low, open_ = market.close, market.high, market.low, market.open
    bar_valid = (
        np.isfinite(open_)
        & (open_ > 0)
        & np.isfinite(high)
        & (high > 0)
        & np.isfinite(low)
        & (low > 0)
        & np.isfinite(close)
        & (close > 0)
    )
    frozen_pivot = prior_breakout_pivot(market) if pivot is None else np.asarray(pivot)
    if frozen_pivot.shape != market.shape:
        raise ValueError("pivot shape does not match MarketDataMatrix")
    atr = matrix_feature(market, "atr_pct") * close

    future_close = tuple(
        valid_shift(close, -offset, bar_valid) for offset in range(1, confirmation_bars + 1)
    )
    future_low = tuple(
        valid_shift(low, -offset, bar_valid) for offset in range(1, confirmation_bars + 1)
    )
    future_high = tuple(
        valid_shift(high, -offset, bar_valid) for offset in range(1, confirmation_bars + 1)
    )
    eligible = (
        events
        & bar_valid
        & np.isfinite(frozen_pivot)
        & np.isfinite(atr)
        & (atr > 0)
        & np.logical_and.reduce([np.isfinite(values) for values in future_close])
        & np.logical_and.reduce([np.isfinite(values) for values in future_low])
        & np.logical_and.reduce([np.isfinite(values) for values in future_high])
    )
    final_close = future_close[-1]
    any_close_at_or_below = np.logical_or.reduce(
        [values <= frozen_pivot for values in future_close]
    )
    held = eligible & (final_close > frozen_pivot) & ~any_close_at_or_below
    reclaimed = eligible & (final_close > frozen_pivot) & any_close_at_or_below
    failed = eligible & (final_close <= frozen_pivot)
    state_masks: dict[AcceptanceState, np.ndarray] = {
        "HELD_ABOVE": held,
        "RECLAIMED": reclaimed,
        "FAILED": failed,
    }
    intraday_retest = eligible & np.logical_or.reduce(
        [values <= frozen_pivot for values in future_low]
    )
    minimum_low = np.minimum.reduce(future_low)
    maximum_high = np.maximum.reduce(future_high)
    mae = safe_divide(minimum_low, close) - 1.0
    mfe = safe_divide(maximum_high, close) - 1.0
    entry_open = valid_shift(open_, -(confirmation_bars + 1), bar_valid)
    forwards: dict[int, np.ndarray] = {}
    for horizon in horizons:
        future_exit_close = valid_shift(close, -(confirmation_bars + horizon), bar_valid)
        forwards[int(horizon)] = safe_divide(future_exit_close, entry_open) - 1.0

    return BreakoutAcceptanceWindow(
        confirmation_bars=confirmation_bars,
        event_mask=_freeze(events),
        eligible_mask=_freeze(eligible),
        pivot=_freeze(np.array(frozen_pivot, dtype=np.float32, copy=True)),
        atr=_freeze(np.array(atr, dtype=np.float32, copy=True)),
        state_masks={name: _freeze(values) for name, values in state_masks.items()},
        intraday_retest=_freeze(intraday_retest),
        acceptance_atr=_freeze(safe_divide(final_close - frozen_pivot, atr)),
        mae=_freeze(mae),
        mfe=_freeze(mfe),
        mae_atr=_freeze(safe_divide(minimum_low - close, atr)),
        mfe_atr=_freeze(safe_divide(maximum_high - close, atr)),
        entry_open=_freeze(entry_open),
        forward_returns={horizon: _freeze(values) for horizon, values in forwards.items()},
    )


def summarize_breakout_acceptance(window: BreakoutAcceptanceWindow) -> dict:
    """Return JSON-safe state, ATR, MAE/MFE, and forward-return diagnostics."""
    eligible_count = int(np.count_nonzero(window.eligible_mask))
    states = {}
    for state, mask in window.state_masks.items():
        states[state] = {
            "count": int(np.count_nonzero(mask)),
            "rate": _rate(mask, window.eligible_mask),
            "median_mae": _median(window.mae, mask),
            "median_mfe": _median(window.mfe, mask),
            "median_mae_atr": _median(window.mae_atr, mask),
            "median_mfe_atr": _median(window.mfe_atr, mask),
            "forward_returns": {
                str(horizon): summarize_returns(values[mask])
                for horizon, values in window.forward_returns.items()
            },
        }
    atr_buckets = {
        "<=0": window.eligible_mask & (window.acceptance_atr <= 0.0),
        "0-0.5": window.eligible_mask
        & (window.acceptance_atr > 0.0)
        & (window.acceptance_atr <= 0.5),
        "0.5-1": window.eligible_mask
        & (window.acceptance_atr > 0.5)
        & (window.acceptance_atr <= 1.0),
        ">1": window.eligible_mask & (window.acceptance_atr > 1.0),
    }
    return {
        "confirmation_bars": window.confirmation_bars,
        "event_count": int(np.count_nonzero(window.event_mask)),
        "eligible_count": eligible_count,
        "states": states,
        "intraday_retest_count": int(np.count_nonzero(window.intraday_retest)),
        "intraday_retest_rate": _rate(window.intraday_retest, window.eligible_mask),
        "acceptance_atr_buckets": {
            label: {
                "count": int(np.count_nonzero(mask)),
                "rate": _rate(mask, window.eligible_mask),
                "forward_returns": {
                    str(horizon): summarize_returns(values[mask])
                    for horizon, values in window.forward_returns.items()
                },
            }
            for label, mask in atr_buckets.items()
        },
        "acceptance_atr_spearman": {
            str(horizon): spearman_correlation(
                window.acceptance_atr[window.eligible_mask],
                values[window.eligible_mask],
            )
            for horizon, values in window.forward_returns.items()
        },
    }


def conditional_state_transitions(
    earlier: BreakoutAcceptanceWindow,
    later: BreakoutAcceptanceWindow,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Summarize later confirmation states conditional on an earlier state."""
    if earlier.confirmation_bars >= later.confirmation_bars:
        raise ValueError("later confirmation must follow earlier confirmation")
    common = earlier.eligible_mask & later.eligible_mask
    output: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for earlier_state, earlier_mask in earlier.state_masks.items():
        base = common & earlier_mask
        output[earlier_state] = {
            later_state: {
                "count": int(np.count_nonzero(base & later_mask)),
                "rate": _rate(base & later_mask, base),
            }
            for later_state, later_mask in later.state_masks.items()
        }
    return output


def _freeze(values: np.ndarray) -> np.ndarray:
    output = np.array(values, copy=True)
    output.flags.writeable = False
    return output


def _median(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    return round(float(np.median(selected)), 4) if len(selected) else None


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    total = int(np.count_nonzero(denominator))
    return round(float(np.count_nonzero(numerator) / total), 4) if total else None
