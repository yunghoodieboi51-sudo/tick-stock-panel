"""Regression coverage for the explicit local experiment runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.backtest.experiments import (
    ExperimentRequest,
    StrategyExperimentRunner,
    assess_local_stability,
    local_parameter_requests,
    split_requests_by_trading_days,
    v4_4_2_local_plan,
)


@dataclass
class _Strategy:
    execution_backend: str = "matrix_native"
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        self.meta = {"params": [{"id": "min_total_score", "default": 58.0}]}


class _StrategyEngine:
    def get(self, _strategy_id):
        return _Strategy()


class _Cache:
    closed = False

    def close(self):
        self.closed = True


class _Prepared:
    def __init__(self):
        self.compute_cache = _Cache()


class _Service:
    def __init__(self):
        self.prepared_calls = []
        self.run_calls = []
        self.prepared = []

    @staticmethod
    def _matrix_prepare_signature(config):
        return (config.strategy_id, config.start, config.end, repr(config.overrides))

    def prepare_matrix_optimization(self, configs):
        self.prepared_calls.append(configs)
        prepared = _Prepared()
        self.prepared.append(prepared)
        return prepared

    def run(self, config, prepared=None):
        self.run_calls.append((config, prepared))
        score = float(config.params.get("min_total_score", 58.0))
        return type(
            "Result",
            (),
            {
                "trades": [
                    {"pnl_pct": 0.12, "duration": 4, "exit_reason": "signal"},
                    {"pnl_pct": -0.06, "duration": 8, "exit_reason": "stop_loss"},
                    {"pnl_pct": 0.03, "duration": 2, "exit_reason": "signal"},
                ],
                "stats": {"total_return": score / 1000, "max_drawdown": -0.1},
                "error": None,
                "entry_pattern_breakdown": {"BREAKOUT": {"trade_count": 3}},
                "matched_pattern_counts": {"BREAKOUT": 3},
                "matched_pattern_combinations": {"BREAKOUT_ONLY": 3},
            },
        )()


def _base_request() -> ExperimentRequest:
    return ExperimentRequest(
        experiment_name="baseline",
        strategy_id="v4_trend_strategy",
        start=date(2025, 1, 2),
        end=date(2025, 1, 10),
        overrides={"fundamental_veto": {"enabled": False}},
    )


def test_runner_reuses_matrix_preparation_and_reports_trade_diagnostics(tmp_path):
    service = _Service()
    runner = StrategyExperimentRunner(service, _StrategyEngine())
    requests = local_parameter_requests(_base_request(), "min_total_score", [54, 58, 62])

    records = runner.run(requests)

    assert len(service.prepared_calls) == 1
    assert len(service.prepared_calls[0]) == 3
    assert all(prepared is service.prepared[0] for _, prepared in service.run_calls)
    assert service.prepared[0].compute_cache.closed is True
    assert records[0]["parameters"]["min_total_score"] == 54
    assert records[0]["metrics"]["standard_profit_factor"] == 2.5
    assert records[0]["exit_reason_diagnostics"]["signal"]["count"] == 2
    assert records[0]["loss_diagnostics"]["counts"]["lte_-5pct"] == 1
    assert records[0]["winner_dependency"]["top_positive_trade_pnl_contribution"]["top_1"] == 0.8

    json_path, csv_path = runner.write_output("local_scores", records, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "local_scores"
    assert len(payload["results"]) == 3
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("experiment_name,")


def test_runner_marks_pattern_research_only_in_its_own_config():
    runner = StrategyExperimentRunner(_Service(), _StrategyEngine())
    ordinary = runner._config(_base_request())
    pattern = runner._config(_base_request().with_enabled_patterns(["BREAKOUT"]))

    assert "__v4_4_2_experiment_context" not in ordinary.params
    context = pattern.params["__v4_4_2_experiment_context"]
    assert isinstance(context, np.ndarray)
    assert context.tolist() == [0x44]
    assert pattern.params["__experiment_enabled_entry_patterns"] == ["BREAKOUT"]


def test_trading_day_split_uses_indexed_observed_days():
    base = _base_request()
    days = [date(2025, 1, day) for day in (2, 3, 6, 7, 8, 9, 10)]

    splits = split_requests_by_trading_days(base, days, parts=3, prefix="third")

    assert [(item.start, item.end) for item in splits] == [
        (date(2025, 1, 2), date(2025, 1, 3)),
        (date(2025, 1, 6), date(2025, 1, 7)),
        (date(2025, 1, 8), date(2025, 1, 10)),
    ]


def test_stability_flags_low_samples_and_does_not_select_a_winner():
    records = [
        {
            "experiment_name": "score=54",
            "parameters": {"score": 54},
            "metrics": {"trade_count": 99, "standard_profit_factor": 0.9},
        },
        {
            "experiment_name": "score=58",
            "parameters": {"score": 58},
            "metrics": {"trade_count": 120, "standard_profit_factor": 1.2},
        },
        {
            "experiment_name": "score=62",
            "parameters": {"score": 62},
            "metrics": {"trade_count": 120, "standard_profit_factor": 0.8},
        },
    ]

    assessment = assess_local_stability(records, "score")

    assert assessment["low_sample_experiments"] == ["score=54"]
    assert assessment["isolated_local_peaks"] == ["score=58"]
    assert "best" not in assessment


def test_v4_local_plan_contains_only_explicit_one_factor_groups():
    plan = v4_4_2_local_plan(_base_request())

    assert [item.params["min_total_score"] for item in plan["min_total_score"]] == [
        50,
        54,
        58,
        62,
        66,
    ]
    assert [item.overrides["stop_loss"] for item in plan["stop_loss"]] == [
        -0.06,
        -0.07,
        -0.08,
        -0.09,
        -0.10,
    ]
    assert [
        item.params["__experiment_enabled_entry_patterns"] for item in plan["entry_patterns"]
    ] == [["BREAKOUT"], ["PULLBACK_RESTART"], ["BREAKOUT", "PULLBACK_RESTART"]]
