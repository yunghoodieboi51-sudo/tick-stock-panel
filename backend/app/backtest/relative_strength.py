"""Point-in-time cross-sectional relative-strength research helpers.

These helpers are deliberately offline diagnostics.  They do not alter a
strategy's signal eligibility or production score; an experiment may pass a
returned percentile matrix to the existing matcher only to compare candidate
ordering under identical execution rules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from app.backtest.matrix import MarketDataMatrix, SignalMatrix, make_signal_matrix
from app.backtest.strategy import trade_pnl_diagnostics


def cross_sectional_percentile(
    values: np.ndarray,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Rank each time row across assets with average-rank ties.

    The first axis is time and the second is the asset universe.  Only finite,
    eligible values participate.  A row's weakest/strongest distinct values
    receive 0/1; a one-member or all-tied row receives 0.5.  Invalid inputs are
    represented as ``NaN`` rather than being silently ranked at the bottom.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("cross-sectional values must be a time x asset matrix")
    if eligible is None:
        eligible_mask = np.ones(array.shape, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible, dtype=bool)
        if eligible_mask.shape != array.shape:
            raise ValueError("eligible mask shape does not match values")

    result = np.full(array.shape, np.nan, dtype=np.float32)
    for time_id in range(array.shape[0]):
        row = array[time_id]
        valid_ids = np.flatnonzero(eligible_mask[time_id] & np.isfinite(row))
        count = len(valid_ids)
        if count == 0:
            continue
        if count == 1:
            result[time_id, valid_ids[0]] = np.float32(0.5)
            continue
        ordered_ids = valid_ids[np.argsort(row[valid_ids], kind="mergesort")]
        ordered_values = row[ordered_ids]
        start = 0
        while start < count:
            stop = start + 1
            while stop < count and ordered_values[stop] == ordered_values[start]:
                stop += 1
            # Average zero-based ordinal rank gives deterministic equal values
            # the same percentile, without depending on source row order.
            percentile = np.float32(((start + stop - 1) / 2) / (count - 1))
            result[time_id, ordered_ids[start:stop]] = percentile
            start = stop
    return result


def relative_strength_eligible_universe(
    market: MarketDataMatrix,
    momentum: np.ndarray,
    *,
    basic_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the existing stock matrix's valid, technically usable universe."""
    values = np.asarray(momentum)
    if values.shape != market.shape:
        raise ValueError("momentum shape does not match MarketDataMatrix")
    valid_ohlcv = (
        np.isfinite(market.open)
        & (market.open > 0)
        & np.isfinite(market.high)
        & (market.high > 0)
        & np.isfinite(market.low)
        & (market.low > 0)
        & np.isfinite(market.close)
        & (market.close > 0)
        & np.isfinite(market.volume)
        & (market.volume > 0)
    )
    eligible = np.asarray(market.tradable, dtype=bool) & valid_ohlcv & np.isfinite(values)
    if basic_mask is not None:
        basic = np.asarray(basic_mask, dtype=bool)
        if basic.shape != market.shape:
            raise ValueError("basic mask shape does not match MarketDataMatrix")
        eligible &= basic
    return eligible


def forward_returns_from_next_open(
    market: MarketDataMatrix,
    horizons: Sequence[int] = (3, 5, 10),
) -> dict[int, np.ndarray]:
    """Return T+h close versus T+1 open, without using it in signal creation."""
    result: dict[int, np.ndarray] = {}
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("forward-return horizon must be at least one trading day")
        values = np.full(market.shape, np.nan, dtype=np.float32)
        if horizon >= market.shape[0]:
            result[int(horizon)] = values
            continue
        entry_open = market.open[1 : market.shape[0] - horizon + 1]
        exit_close = market.close[horizon:]
        valid = (
            np.isfinite(entry_open)
            & (entry_open > 0)
            & np.isfinite(exit_close)
            & (exit_close > 0)
        )
        window = values[: market.shape[0] - horizon]
        np.divide(
            exit_close,
            entry_open,
            out=window,
            where=valid,
        )
        window[valid] -= np.float32(1.0)
        result[int(horizon)] = values
    return result


def percentile_bucket(percentiles: np.ndarray) -> np.ndarray:
    """Assign fixed broad 20-point buckets; invalid percentiles remain -1."""
    values = np.asarray(percentiles, dtype=np.float32)
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    result[valid] = np.minimum((values[valid] * 5).astype(np.int8), 4)
    return result


def summarize_returns(values: Iterable[float]) -> dict[str, float | int | None]:
    """JSON-safe distribution summary used by fixed-bucket diagnostics."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "win_rate": None,
            "p25": None,
            "p75": None,
        }
    return {
            "count": len(array),
        "mean": round(float(np.mean(array)), 4),
        "median": round(float(np.median(array)), 4),
        "win_rate": round(float(np.mean(array > 0)), 4),
        "p25": round(float(np.percentile(array, 25)), 4),
        "p75": round(float(np.percentile(array, 75)), 4),
    }


def bucket_return_diagnostics(
    percentiles: np.ndarray,
    returns: np.ndarray,
    event_mask: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize event returns in the five predeclared percentile buckets."""
    if percentiles.shape != returns.shape or returns.shape != event_mask.shape:
        raise ValueError("percentiles, returns, and event_mask must share one shape")
    buckets = percentile_bucket(percentiles)
    labels = ("0-20", "20-40", "40-60", "60-80", "80-100")
    return {
        label: summarize_returns(returns[(buckets == index) & event_mask])
        for index, label in enumerate(labels)
    }


def replace_signal_scores(signals: SignalMatrix, scores: np.ndarray) -> SignalMatrix:
    """Keep frozen signals/metadata while substituting an offline sort key."""
    return make_signal_matrix(
        signals.shape,
        entry=signals.entry,
        exit=signals.exit,
        score=np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
        entry_signal_code=signals.entry_signal_code,
        exit_signal_code=signals.exit_signal_code,
        entry_signal_ids=signals.entry_signal_ids,
        exit_signal_ids=signals.exit_signal_ids,
        entry_pattern_mask=signals.entry_pattern_mask,
        entry_pattern_ids=signals.entry_pattern_ids,
    )


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> dict[str, float | int | None]:
    """Return a tie-aware, dependency-free Spearman coefficient."""
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must share one shape")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return {"count": len(x), "rho": None}

    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = (start + stop - 1) / 2
            start = stop
        return ranks

    x_rank, y_rank = average_ranks(x), average_ranks(y)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        rho = None
    else:
        rho = round(float(np.corrcoef(x_rank, y_rank)[0, 1]), 4)
    return {"count": len(x), "rho": rho}


def trade_bucket_diagnostics(
    trades: Sequence[Mapping[str, Any]],
    buckets: Sequence[int],
) -> dict[str, dict[str, float | int | None]]:
    """Return realized-trade diagnostics without inferring exits from PnL."""
    if len(trades) != len(buckets):
        raise ValueError("trades and buckets must have equal lengths")
    labels = ("0-20", "20-40", "40-60", "60-80", "80-100")
    output: dict[str, dict[str, float | int | None]] = {}
    for bucket, label in enumerate(labels):
        selected = [trade for trade, value in zip(trades, buckets, strict=True) if value == bucket]
        pnls = np.asarray([trade.get("pnl_pct", np.nan) for trade in selected], dtype=float)
        holds = np.asarray([trade.get("duration", np.nan) for trade in selected], dtype=float)
        finite_pnls = pnls[np.isfinite(pnls)]
        summary = summarize_returns(finite_pnls)
        output[label] = {
            **summary,
            "trade_count": summary["count"],
            **trade_pnl_diagnostics(finite_pnls),
            "average_hold_days": (
                round(float(np.mean(holds[np.isfinite(holds)])), 1)
                if np.isfinite(holds).any()
                else None
            ),
        }
    return output
