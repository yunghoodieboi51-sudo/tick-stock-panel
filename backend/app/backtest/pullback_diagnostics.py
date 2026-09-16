"""Point-in-time diagnostics for trend pullback and reclaim hypotheses.

The module intentionally receives a prepared market matrix and optional
framework-owned basic mask.  It neither creates a strategy signal nor invokes
the matcher, so its forward labels cannot affect production selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    matrix_feature,
    safe_divide,
    valid_rolling_max,
    valid_shift,
)
from app.backtest.relative_strength import spearman_correlation, summarize_returns

DEPTH_BUCKETS = ("VERY_SHALLOW", "SHALLOW", "MEDIUM", "DEEP")
DURATION_BUCKETS = ("1-2", "3-5", "6-10", ">10")
MA20_INTERACTIONS = ("ABOVE_MA20", "TOUCHED_MA20", "RECLAIMED_MA20", "BELOW_MA20")


@dataclass(frozen=True)
class PullbackDiagnostics:
    """Read-only event masks, PIT variables, and forward labels for V4.6a."""

    trend_qualified: np.ndarray
    breakout_excluded: np.ndarray
    trend_non_breakout: np.ndarray
    pullback: np.ndarray
    pullback_without_reclaim: np.ndarray
    any_reclaim: np.ndarray
    ma10_reclaim: np.ndarray
    ma20_reclaim: np.ndarray
    price_recovery: np.ndarray
    prior_peak: np.ndarray
    pullback_depth_pct: np.ndarray
    pullback_depth_atr: np.ndarray
    duration_bars: np.ndarray
    distance_to_ma20_pct: np.ndarray
    distance_to_ma20_atr: np.ndarray
    depth_buckets: dict[str, np.ndarray]
    duration_buckets: dict[str, np.ndarray]
    ma20_interactions: dict[str, np.ndarray]
    entry_open: np.ndarray
    forward_returns: dict[int, np.ndarray]


def build_pullback_diagnostics(
    market: MarketDataMatrix,
    *,
    basic_mask: np.ndarray | None = None,
    peak_window: int = 20,
    ma20_slope_bars: int = 5,
    horizons: tuple[int, ...] = (3, 5, 10, 20),
) -> PullbackDiagnostics:
    """Classify trend pullbacks using data available no later than event day T.

    The broad, fixed primary depth buckets are 0--2%, 2--5%, 5--10%, and
    greater than 10% below the prior 20-valid-bar close peak.  They are
    diagnostics, not strategy parameters or eligibility thresholds.
    """
    if peak_window < 1 or ma20_slope_bars < 1:
        raise ValueError("peak_window and ma20_slope_bars must be positive")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive trading-bar counts")

    close, high, low, open_ = market.close, market.high, market.low, market.open
    bar_valid = _valid_ohlc(market)
    basic = _basic_mask(market, basic_mask)
    ma10 = matrix_feature(market, "ma10")
    ma20 = matrix_feature(market, "ma20")
    ma60 = matrix_feature(market, "ma60")
    atr = matrix_feature(market, "atr_pct") * close

    ma20_previous = valid_shift(ma20, ma20_slope_bars, np.isfinite(ma20))
    ma20_slope = safe_divide(ma20, ma20_previous) - 1.0
    trend = (
        bar_valid
        & basic
        & np.asarray(market.tradable, dtype=bool)
        & np.isfinite(ma20)
        & np.isfinite(ma60)
        & np.isfinite(ma20_slope)
        & (close > ma60)
        & (ma20 > ma60)
        & (ma20_slope >= 0.0)
    )

    close_valid = np.isfinite(close) & (close > 0)
    rolling_peak = valid_rolling_max(close, close_valid, peak_window)
    prior_peak = valid_shift(rolling_peak, 1, np.isfinite(rolling_peak))
    duration = _prior_peak_duration(close, close_valid, peak_window)
    breakout = trend & np.isfinite(prior_peak) & (close > prior_peak)
    trend_non_breakout = trend & ~breakout
    depth_pct = safe_divide(prior_peak - close, prior_peak)
    depth_atr = safe_divide(prior_peak - close, atr)
    pullback = (
        trend_non_breakout
        & np.isfinite(depth_pct)
        & (depth_pct > 0.0)
        & np.isfinite(depth_atr)
        & np.isfinite(duration)
    )

    previous_close = valid_shift(close, 1, close_valid)
    previous_ma10 = valid_shift(ma10, 1, np.isfinite(ma10))
    previous_ma20 = valid_shift(ma20, 1, np.isfinite(ma20))
    previous_high = valid_shift(high, 1, bar_valid)
    ma10_reclaim = pullback & (previous_close <= previous_ma10) & (close > ma10)
    ma20_reclaim = pullback & (previous_close <= previous_ma20) & (close > ma20)
    price_recovery = pullback & (close > previous_high)
    any_reclaim = ma10_reclaim | ma20_reclaim | price_recovery

    distance_ma20_pct = safe_divide(close - ma20, ma20)
    distance_ma20_atr = safe_divide(close - ma20, atr)
    ma_interactions = _ma20_interactions(
        pullback,
        close,
        low,
        ma20,
        ma20_reclaim,
    )
    depth_buckets = _depth_buckets(pullback, depth_pct)
    duration_buckets = _duration_buckets(pullback, duration)

    entry_open = valid_shift(open_, -1, bar_valid)
    forward_returns = {
        int(horizon): safe_divide(valid_shift(close, -horizon, bar_valid), entry_open) - 1.0
        for horizon in horizons
    }
    return PullbackDiagnostics(
        trend_qualified=_freeze(trend),
        breakout_excluded=_freeze(breakout),
        trend_non_breakout=_freeze(trend_non_breakout),
        pullback=_freeze(pullback),
        pullback_without_reclaim=_freeze(pullback & ~any_reclaim),
        any_reclaim=_freeze(any_reclaim),
        ma10_reclaim=_freeze(ma10_reclaim),
        ma20_reclaim=_freeze(ma20_reclaim),
        price_recovery=_freeze(price_recovery),
        prior_peak=_freeze(prior_peak),
        pullback_depth_pct=_freeze(depth_pct),
        pullback_depth_atr=_freeze(depth_atr),
        duration_bars=_freeze(duration),
        distance_to_ma20_pct=_freeze(distance_ma20_pct),
        distance_to_ma20_atr=_freeze(distance_ma20_atr),
        depth_buckets={name: _freeze(mask) for name, mask in depth_buckets.items()},
        duration_buckets={name: _freeze(mask) for name, mask in duration_buckets.items()},
        ma20_interactions={name: _freeze(mask) for name, mask in ma_interactions.items()},
        entry_open=_freeze(entry_open),
        forward_returns={horizon: _freeze(values) for horizon, values in forward_returns.items()},
    )


def summarize_pullback_diagnostics(diagnostics: PullbackDiagnostics) -> dict:
    """Return JSON-safe fixed-group diagnostics without selecting a winner."""
    groups = {
        "trend_qualified": diagnostics.trend_qualified,
        "trend_non_breakout": diagnostics.trend_non_breakout,
        "pullback": diagnostics.pullback,
        "pullback_without_reclaim": diagnostics.pullback_without_reclaim,
        "any_reclaim": diagnostics.any_reclaim,
        "ma10_reclaim": diagnostics.ma10_reclaim,
        "ma20_reclaim": diagnostics.ma20_reclaim,
        "price_recovery": diagnostics.price_recovery,
    }
    continuous = {
        "pullback_depth_pct": diagnostics.pullback_depth_pct,
        "pullback_depth_atr": diagnostics.pullback_depth_atr,
        "duration_bars": diagnostics.duration_bars,
        "distance_to_ma20_pct": diagnostics.distance_to_ma20_pct,
        "distance_to_ma20_atr": diagnostics.distance_to_ma20_atr,
    }
    return {
        "counts": {
            "trend_qualified": _count(diagnostics.trend_qualified),
            "breakout_excluded": _count(diagnostics.breakout_excluded),
            "trend_non_breakout": _count(diagnostics.trend_non_breakout),
            "pullback": _count(diagnostics.pullback),
        },
        "groups": _summaries_by_mask(groups, diagnostics.forward_returns),
        "depth_buckets": _summaries_by_mask(diagnostics.depth_buckets, diagnostics.forward_returns),
        "duration_buckets": _summaries_by_mask(
            diagnostics.duration_buckets, diagnostics.forward_returns
        ),
        "ma20_interactions": _summaries_by_mask(
            diagnostics.ma20_interactions, diagnostics.forward_returns
        ),
        "continuous_spearman": {
            name: {
                str(horizon): spearman_correlation(
                    values[diagnostics.pullback], returns[diagnostics.pullback]
                )
                for horizon, returns in diagnostics.forward_returns.items()
            }
            for name, values in continuous.items()
        },
        "event_date_aggregate": {
            name: {
                str(horizon): event_date_aggregate(returns, mask)
                for horizon, returns in diagnostics.forward_returns.items()
            }
            for name, mask in groups.items()
        },
    }


def event_date_aggregate(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    """Summarize daily cross-sectional medians to limit sample-count inflation."""
    data = np.asarray(values, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    if data.shape != selected.shape:
        raise ValueError("values and mask must share one matrix shape")
    daily = []
    for time_id in range(data.shape[0]):
        row = data[time_id, selected[time_id]]
        row = row[np.isfinite(row)]
        if len(row):
            daily.append(float(np.median(row)))
    summary = summarize_returns(daily)
    return {
        "date_count": len(daily),
        "mean": summary["mean"],
        "median": summary["median"],
        "positive_day_rate": summary["win_rate"],
        "p25": summary["p25"],
        "p75": summary["p75"],
    }


def positive_return_outlier_dependency(
    values: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | None]:
    """Return event-level winner concentration; this is not portfolio PnL."""
    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return {
            "count": 0,
            "top_1_positive_contribution": None,
            "top_5_positive_contribution": None,
            "top_10_positive_contribution": None,
            "mean_without_top_5": None,
            "median_without_top_5": None,
        }
    ordered = np.sort(selected)[::-1]
    positive = ordered[ordered > 0]
    denominator = float(positive.sum())
    trimmed = ordered[5:]
    return {
        "count": len(ordered),
        "top_1_positive_contribution": _contribution(positive, 1, denominator),
        "top_5_positive_contribution": _contribution(positive, 5, denominator),
        "top_10_positive_contribution": _contribution(positive, 10, denominator),
        "mean_without_top_5": _round(float(np.mean(trimmed))) if len(trimmed) else None,
        "median_without_top_5": _round(float(np.median(trimmed))) if len(trimmed) else None,
    }


def summarize_external_event_relationship(
    event_mask: np.ndarray,
    diagnostics: PullbackDiagnostics,
) -> dict[str, int | float | None | dict[str, int]]:
    """Describe an existing event family without turning it into a new signal."""
    events = np.asarray(event_mask, dtype=bool)
    if events.shape != diagnostics.pullback.shape:
        raise ValueError("event_mask shape does not match pullback diagnostics")
    event_count = _count(events)
    return {
        "event_count": event_count,
        "trend_qualified_count": _count(events & diagnostics.trend_qualified),
        "breakout_excluded_count": _count(events & diagnostics.breakout_excluded),
        "pullback_count": _count(events & diagnostics.pullback),
        "any_reclaim_count": _count(events & diagnostics.any_reclaim),
        "pullback_rate": _ratio(events & diagnostics.pullback, events),
        "depth_bucket_counts": {
            name: _count(events & mask) for name, mask in diagnostics.depth_buckets.items()
        },
    }


def _valid_ohlc(market: MarketDataMatrix) -> np.ndarray:
    return (
        np.isfinite(market.open)
        & (market.open > 0)
        & np.isfinite(market.high)
        & (market.high > 0)
        & np.isfinite(market.low)
        & (market.low > 0)
        & np.isfinite(market.close)
        & (market.close > 0)
    )


def _basic_mask(market: MarketDataMatrix, basic_mask: np.ndarray | None) -> np.ndarray:
    if basic_mask is None:
        return np.ones(market.shape, dtype=bool)
    values = np.asarray(basic_mask, dtype=bool)
    if values.shape != market.shape:
        raise ValueError("basic_mask shape does not match MarketDataMatrix")
    return values


def _prior_peak_duration(close: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Bars from the most recent maximum in T's preceding valid-bar window."""
    output = np.full(close.shape, np.nan, dtype=np.float32)
    for asset_id in range(close.shape[1]):
        rows = np.flatnonzero(valid[:, asset_id])
        for position in range(window, len(rows)):
            history_rows = rows[position - window : position]
            history = close[history_rows, asset_id]
            peak = np.max(history)
            last_peak = np.flatnonzero(history == peak)[-1]
            output[rows[position], asset_id] = float(window - last_peak)
    return output


def _depth_buckets(pullback: np.ndarray, depth: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "VERY_SHALLOW": pullback & (depth <= 0.02),
        "SHALLOW": pullback & (depth > 0.02) & (depth <= 0.05),
        "MEDIUM": pullback & (depth > 0.05) & (depth <= 0.10),
        "DEEP": pullback & (depth > 0.10),
    }


def _duration_buckets(pullback: np.ndarray, duration: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "1-2": pullback & (duration <= 2),
        "3-5": pullback & (duration >= 3) & (duration <= 5),
        "6-10": pullback & (duration >= 6) & (duration <= 10),
        ">10": pullback & (duration > 10),
    }


def _ma20_interactions(
    pullback: np.ndarray,
    close: np.ndarray,
    low: np.ndarray,
    ma20: np.ndarray,
    reclaim: np.ndarray,
) -> dict[str, np.ndarray]:
    below = pullback & (close < ma20)
    above = pullback & ~reclaim & (close > ma20) & (low > ma20)
    touched = pullback & ~reclaim & (close >= ma20) & (low <= ma20)
    return {
        "ABOVE_MA20": above,
        "TOUCHED_MA20": touched,
        "RECLAIMED_MA20": reclaim,
        "BELOW_MA20": below,
    }


def _summaries_by_mask(masks: dict[str, np.ndarray], forwards: dict[int, np.ndarray]) -> dict:
    return {
        name: {
            "count": _count(mask),
            "forward_returns": {
                str(horizon): summarize_returns(values[mask])
                for horizon, values in forwards.items()
            },
        }
        for name, mask in masks.items()
    }


def _freeze(values: np.ndarray) -> np.ndarray:
    output = np.array(values, copy=True)
    output.flags.writeable = False
    return output


def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    total = _count(denominator)
    return _round(_count(numerator) / total) if total else None


def _contribution(positive: np.ndarray, count: int, denominator: float) -> float | None:
    return _round(float(positive[:count].sum()) / denominator) if denominator else None


def _round(value: float) -> float:
    return round(float(value), 4)
