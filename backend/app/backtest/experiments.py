"""Small, reproducible diagnostics for local strategy-parameter experiments.

This module is intentionally not an API or an optimizer.  It executes an
explicit ordered list of local trials through :class:`StrategyBacktestService`,
retains every trial (rather than choosing a winner), and writes JSON/CSV when
asked.  It is suitable for offline research only.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from app.backtest.strategy import (
    StrategyBacktestConfig,
    StrategyBacktestService,
    trade_pnl_diagnostics,
)

_EXPERIMENT_PATTERN_PARAM = "__experiment_enabled_entry_patterns"
_EXPERIMENT_CONTEXT_PARAM = "__v4_4_2_experiment_context"
_EXPERIMENT_CONTEXT_VALUE = 0x44
_CSV_METRICS = (
    "trade_count",
    "win_rate",
    "total_return",
    "max_drawdown",
    "standard_profit_factor",
    "payoff_ratio",
    "average_return",
    "median_return",
    "average_hold_days",
)


@dataclass(frozen=True)
class ExperimentRequest:
    """One explicit backtest trial; this does not generate parameter grids."""

    experiment_name: str
    strategy_id: str
    start: date
    end: date
    params: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    symbols: list[str] | None = None
    matching: str = "open_t+1"
    # Preserve StrategyBacktestConfig's portfolio-position default.  ``full``
    # intentionally evaluates every candidate independently and is therefore
    # not comparable to the V4.4.1 portfolio baseline.
    mode: str = "position"
    max_positions: int = 10
    position_sizing: str = "equal"
    # Offline-only pass-through to the existing backtest T-1 regime gate.
    # It deliberately has no connection to persisted strategy overrides or APIs.
    regime_filter: dict[str, Any] | None = None

    def with_params(self, **params: Any) -> ExperimentRequest:
        return replace(self, params={**self.params, **params})

    def with_overrides(self, **overrides: Any) -> ExperimentRequest:
        return replace(self, overrides={**self.overrides, **overrides})

    def with_enabled_patterns(self, patterns: Sequence[str]) -> ExperimentRequest:
        return self.with_params(**{_EXPERIMENT_PATTERN_PARAM: list(patterns)})


def local_parameter_requests(
    base: ExperimentRequest,
    parameter: str,
    values: Sequence[Any],
    *,
    prefix: str | None = None,
) -> list[ExperimentRequest]:
    """Create a one-factor list; callers must provide values deliberately."""
    name = prefix or parameter
    return [
        replace(
            base.with_params(**{parameter: value}),
            experiment_name=f"{name}={value}",
        )
        for value in values
    ]


def local_override_requests(
    base: ExperimentRequest,
    override: str,
    values: Sequence[Any],
    *,
    prefix: str | None = None,
) -> list[ExperimentRequest]:
    """Create a local risk-control sweep without changing strategy defaults."""
    name = prefix or override
    return [
        replace(
            base.with_overrides(**{override: value}),
            experiment_name=f"{name}={value}",
        )
        for value in values
    ]


def v4_4_2_local_plan(base: ExperimentRequest) -> dict[str, list[ExperimentRequest]]:
    """The declared V4.4.2 local-neighborhood research plan.

    This returns independent one-factor groups only.  It intentionally does
    not rank, combine, or apply any result as a strategy default.
    """
    return {
        "min_total_score": local_parameter_requests(base, "min_total_score", (50, 54, 58, 62, 66)),
        "stop_loss": local_override_requests(
            base, "stop_loss", (-0.06, -0.07, -0.08, -0.09, -0.10)
        ),
        "trailing_stop": local_override_requests(
            base, "trailing_stop", (-0.07, -0.085, -0.10, -0.115, -0.13)
        ),
        "max_hold_days": local_override_requests(base, "max_hold_days", (20, 30, 40, 50, 60)),
        "max_daily_gain": local_parameter_requests(
            base, "max_daily_gain", (0.06, 0.07, 0.085, 0.10)
        ),
        "max_ma20_bias": local_parameter_requests(base, "max_ma20_bias", (0.08, 0.10, 0.12, 0.14)),
        "rsi_max": local_parameter_requests(base, "rsi_max", (75, 78, 82, 85)),
        "ma20_slope_min": local_parameter_requests(
            base, "ma20_slope_min", (-0.005, 0.0, 0.005, 0.01)
        ),
        "ma60_slope_min": local_parameter_requests(
            base, "ma60_slope_min", (-0.01, -0.005, 0.0, 0.005)
        ),
        "entry_patterns": [
            replace(
                base.with_enabled_patterns(("BREAKOUT",)),
                experiment_name="entry_patterns=BREAKOUT_ONLY",
            ),
            replace(
                base.with_enabled_patterns(("PULLBACK_RESTART",)),
                experiment_name="entry_patterns=PULLBACK_RESTART_ONLY",
            ),
            replace(
                base.with_enabled_patterns(("BREAKOUT", "PULLBACK_RESTART")),
                experiment_name="entry_patterns=BREAKOUT+PULLBACK_RESTART",
            ),
        ],
    }


def split_requests_by_trading_days(
    base: ExperimentRequest,
    trading_days: Sequence[date],
    *,
    parts: int,
    prefix: str,
) -> list[ExperimentRequest]:
    """Split a request by observed trading-day indexes, never calendar days."""
    if parts < 2:
        raise ValueError("parts must be at least two")
    days = sorted({day for day in trading_days if base.start <= day <= base.end})
    if len(days) < parts:
        raise ValueError("not enough observed trading days for requested splits")
    requests = []
    for index in range(parts):
        lo = index * len(days) // parts
        hi = (index + 1) * len(days) // parts
        if lo == hi:
            continue
        requests.append(
            replace(
                base,
                experiment_name=f"{prefix}_{index + 1}_of_{parts}",
                start=days[lo],
                end=days[hi - 1],
            )
        )
    return requests


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _trade_diagnostics(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pnls = _finite(trade.get("pnl_pct") for trade in trades)
    holds = _finite(trade.get("duration") for trade in trades)
    wins = pnls[pnls > 0]
    metrics = {
        "trade_count": len(pnls),
        "win_rate": _round(len(wins) / len(pnls)) if len(pnls) else None,
        "average_return": _round(float(np.mean(pnls))) if len(pnls) else None,
        "median_return": _round(float(np.median(pnls))) if len(pnls) else None,
        "average_hold_days": _round(float(np.mean(holds)), 1) if len(holds) else None,
        "median_hold_days": _round(float(np.median(holds)), 1) if len(holds) else None,
    }
    metrics.update(trade_pnl_diagnostics(pnls))
    return metrics


def _exit_reason_diagnostics(trades: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.get("exit_reason") or "unknown")].append(trade)
    output = {}
    for reason, group in sorted(grouped.items()):
        pnls = _finite(trade.get("pnl_pct") for trade in group)
        holds = _finite(trade.get("duration") for trade in group)
        output[reason] = {
            "count": len(group),
            "average_return": _round(float(np.mean(pnls))) if len(pnls) else None,
            "median_hold_days": _round(float(np.median(holds)), 1) if len(holds) else None,
        }
    return output


def _loss_diagnostics(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pnls = _finite(trade.get("pnl_pct") for trade in trades)
    percentiles = {
        f"p{percentile}": _round(float(np.percentile(pnls, percentile))) if len(pnls) else None
        for percentile in (5, 10, 25, 50, 75, 90, 95)
    }
    return {
        "trade_pnl_percentiles": percentiles,
        "counts": {
            "lte_-5pct": int(np.count_nonzero(pnls <= -0.05)),
            "lte_-10pct": int(np.count_nonzero(pnls <= -0.10)),
            "lte_-15pct": int(np.count_nonzero(pnls <= -0.15)),
        },
    }


def _winner_dependency(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure trade-return concentration, not portfolio cash-PnL concentration."""
    pnls = _finite(trade.get("pnl_pct") for trade in trades)
    winners = np.sort(pnls[pnls > 0])[::-1]
    gross_profit = float(winners.sum()) if len(winners) else 0.0

    def contribution(count: int) -> float | None:
        if gross_profit <= 0:
            return None
        return _round(float(winners[:count].sum() / gross_profit))

    def removing_top(count: int) -> dict[str, Any]:
        remaining = np.concatenate((winners[count:], pnls[pnls <= 0]))
        return {
            "trade_count": len(remaining),
            "average_return": _round(float(np.mean(remaining))) if len(remaining) else None,
            "standard_profit_factor": trade_pnl_diagnostics(remaining)["standard_profit_factor"],
        }

    return {
        "gross_positive_trade_pnl": _round(gross_profit),
        "top_positive_trade_pnl_contribution": {
            "top_1": contribution(1),
            "top_5": contribution(5),
            "top_10": contribution(10),
        },
        "after_removing_top_positive_trades": {
            "top_1": removing_top(1),
            "top_5": removing_top(5),
        },
    }


def assess_local_stability(
    records: Sequence[dict[str, Any]],
    parameter: str,
    *,
    metric: str = "standard_profit_factor",
    min_trade_count: int = 100,
) -> dict[str, Any]:
    """Flag evidence limits without selecting or applying a "best" parameter."""
    ordered = sorted(records, key=lambda row: float(row["parameters"][parameter]))
    values = [row.get("metrics", {}).get(metric) for row in ordered]
    low_sample = [
        row["experiment_name"]
        for row in ordered
        if int(row.get("metrics", {}).get("trade_count") or 0) < min_trade_count
    ]
    isolated = []
    for index, value in enumerate(values):
        if value is None or index == 0 or index == len(values) - 1:
            continue
        left, right = values[index - 1], values[index + 1]
        if left is not None and right is not None and value > left and value > right:
            isolated.append(ordered[index]["experiment_name"])
    return {
        "parameter": parameter,
        "metric": metric,
        "min_trade_count": min_trade_count,
        "low_sample_experiments": low_sample,
        "isolated_local_peaks": isolated,
        "interpretation": (
            "An isolated local peak is an overfit-risk flag, not a parameter recommendation."
        ),
    }


class StrategyExperimentRunner:
    """Run explicit offline experiments through the existing backtest service."""

    def __init__(self, service, strategy_engine) -> None:
        self.service = service
        self.strategy_engine = strategy_engine

    def _config(self, request: ExperimentRequest) -> StrategyBacktestConfig:
        params = dict(request.params)
        # Only the offline runner can attach this non-JSON capability.
        # A similarly named value from API, screener, or persisted params is
        # deliberately ignored by the V4 strategy.
        if _EXPERIMENT_PATTERN_PARAM in params:
            params[_EXPERIMENT_CONTEXT_PARAM] = np.array(
                [_EXPERIMENT_CONTEXT_VALUE], dtype=np.uint8
            )
        return StrategyBacktestConfig(
            strategy_id=request.strategy_id,
            symbols=request.symbols,
            start=request.start,
            end=request.end,
            params=params,
            overrides=dict(request.overrides),
            matching=request.matching,  # type: ignore[arg-type]
            mode=request.mode,  # type: ignore[arg-type]
            max_positions=request.max_positions,
            position_sizing=request.position_sizing,  # type: ignore[arg-type]
            regime_filter=(
                deepcopy(request.regime_filter)
                if request.regime_filter is not None
                else None
            ),
        )

    def _effective_params(self, request: ExperimentRequest) -> dict[str, Any]:
        strategy = self.strategy_engine.get(request.strategy_id)
        # Match the service's actual clamp/default behavior, while retaining
        # the private research-only pattern input that META intentionally omits.
        return StrategyBacktestService._normalize_params(dict(request.params), strategy)

    def _record(self, request: ExperimentRequest, result) -> dict[str, Any]:
        trades = list(result.trades or [])
        metrics = _trade_diagnostics(trades)
        metrics.update(
            {
                "total_return": _round(result.stats.get("total_return")),
                "max_drawdown": _round(result.stats.get("max_drawdown")),
            }
        )
        return {
            "experiment_name": request.experiment_name,
            "strategy_id": request.strategy_id,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "parameters": self._effective_params(request),
            "execution_overrides": dict(request.overrides),
            "error": result.error,
            "metrics": metrics,
            "exit_reason_diagnostics": _exit_reason_diagnostics(trades),
            "loss_diagnostics": _loss_diagnostics(trades),
            "winner_dependency": _winner_dependency(trades),
            "entry_pattern_breakdown": result.entry_pattern_breakdown,
            "matched_pattern_counts": result.matched_pattern_counts,
            "matched_pattern_combinations": result.matched_pattern_combinations,
        }

    def run(self, requests: Sequence[ExperimentRequest]) -> list[dict[str, Any]]:
        """Run in caller order, grouping only identical preparation signatures."""
        if not requests:
            raise ValueError("at least one explicit experiment request is required")
        configs = [self._config(request) for request in requests]
        strategy = self.strategy_engine.get(requests[0].strategy_id)
        if any(request.strategy_id != requests[0].strategy_id for request in requests):
            raise ValueError("one runner invocation must use one strategy")

        groups: dict[tuple, list[tuple[int, StrategyBacktestConfig]]] = defaultdict(list)
        for index, config in enumerate(configs):
            groups[self.service._matrix_prepare_signature(config)].append((index, config))

        results: list[dict[str, Any] | None] = [None] * len(requests)
        for group in groups.values():
            prepared = None
            try:
                if strategy.execution_backend == "matrix_native":
                    prepared = self.service.prepare_matrix_optimization([cfg for _, cfg in group])
                for index, config in group:
                    result = self.service.run(config, prepared=prepared)
                    results[index] = self._record(requests[index], result)
            finally:
                if prepared is not None:
                    prepared.compute_cache.close()
        return [record for record in results if record is not None]

    @staticmethod
    def write_output(
        experiment_name: str,
        records: Sequence[dict[str, Any]],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Write explicit, JSON-safe results only to the caller-selected directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{experiment_name}.json"
        csv_path = output_dir / f"{experiment_name}.csv"
        payload = {"experiment_name": experiment_name, "results": list(records)}
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "experiment_name",
                "strategy_id",
                "start",
                "end",
                "error",
                *_CSV_METRICS,
                "parameters",
                "execution_overrides",
                "exit_reason_diagnostics",
                "loss_diagnostics",
                "winner_dependency",
                "entry_pattern_breakdown",
                "matched_pattern_counts",
                "matched_pattern_combinations",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = {key: record.get(key) for key in fields}
                row.update({key: record["metrics"].get(key) for key in _CSV_METRICS})
                for key in (
                    "parameters",
                    "execution_overrides",
                    "exit_reason_diagnostics",
                    "loss_diagnostics",
                    "winner_dependency",
                    "entry_pattern_breakdown",
                    "matched_pattern_counts",
                    "matched_pattern_combinations",
                ):
                    row[key] = json.dumps(record.get(key), ensure_ascii=False, allow_nan=False)
                writer.writerow(row)
        return json_path, csv_path
