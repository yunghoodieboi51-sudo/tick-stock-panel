"""V4.2 fundamental veto: PIT semantics, rule coverage, and shared execution paths."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import SimResult
from app.backtest.fundamentals import (
    attach_fundamental_factors,
    build_fundamental_matrices,
    load_fundamental_snapshot,
)
from app.backtest.matrix import build_market_data_matrix, make_signal_matrix
from app.backtest.strategy import (
    StrategyBacktestConfig,
    StrategyBacktestService,
    StrategyDependencyResolver,
)
from app.services.screener import ScreenerService
from app.strategy.engine import StrategyDataContext, StrategyDef, StrategyEngine
from app.strategy.fundamental_veto import (
    ABNORMAL_GROSS_MARGIN,
    HIGH_DEBT,
    MISSING_FINANCIAL_DATA,
    NEGATIVE_NET_MARGIN,
    NEGATIVE_ROE,
    SEVERE_PROFIT_DECLINE,
    SEVERE_REVENUE_DECLINE,
    V4_TREND_STRATEGY_ID,
    FundamentalVetoConfig,
    apply_fundamental_veto,
    evaluate_fundamental_veto,
    fundamental_veto_required_fields,
    resolve_fundamental_veto_config,
)


def _panel(
    *,
    symbols: tuple[str, ...] = ("000001.SZ",),
    dates: tuple[date, ...] = (date(2026, 8, 26),),
    financials: dict[str, list[float | None]] | None = None,
) -> pl.DataFrame:
    rows: list[dict] = []
    financials = financials or {}
    for day in dates:
        for asset_id, symbol in enumerate(symbols):
            row = {
                "symbol": symbol,
                "name": symbol,
                "date": day,
                "open": 10.0 + asset_id,
                "high": 10.5 + asset_id,
                "low": 9.5 + asset_id,
                "close": 10.0 + asset_id,
                "volume": 1_000.0,
                "amount": 10_000_000.0,
            }
            for name, values in financials.items():
                row[name] = values[asset_id]
            rows.append(row)
    return pl.DataFrame(rows).sort(["symbol", "date"])


def _safe_financials(asset_count: int = 1) -> dict[str, list[float]]:
    return {
        "roe_latest": [8.0] * asset_count,
        "revenue_yoy_latest": [5.0] * asset_count,
        "net_income_yoy_latest": [5.0] * asset_count,
        "net_margin_latest": [5.0] * asset_count,
        "gross_margin_latest": [20.0] * asset_count,
        "debt_ratio_latest": [40.0] * asset_count,
    }


def _market(financials: dict[str, list[float | None]]):
    panel = _panel(
        symbols=tuple(f"{index:06d}.SZ" for index in range(len(next(iter(financials.values()))))),
        financials=financials,
    )
    return build_market_data_matrix(panel, field_columns=set(financials))


@pytest.mark.parametrize(
    ("field", "value", "config", "reason"),
    [
        ("roe_latest", -0.1, FundamentalVetoConfig(), NEGATIVE_ROE),
        (
            "net_income_yoy_latest",
            -50.1,
            FundamentalVetoConfig(),
            SEVERE_PROFIT_DECLINE,
        ),
        (
            "revenue_yoy_latest",
            -30.1,
            FundamentalVetoConfig(),
            SEVERE_REVENUE_DECLINE,
        ),
        ("net_margin_latest", -5.1, FundamentalVetoConfig(), NEGATIVE_NET_MARGIN),
        (
            "debt_ratio_latest",
            85.1,
            FundamentalVetoConfig(debt_ratio_enabled=True),
            HIGH_DEBT,
        ),
        (
            "gross_margin_latest",
            -20.1,
            FundamentalVetoConfig(),
            ABNORMAL_GROSS_MARGIN,
        ),
    ],
)
def test_each_fundamental_rule_vetoes(field, value, config, reason):
    values = _safe_financials()
    values[field] = [value]
    result = evaluate_fundamental_veto(_market(values), config)

    assert result.veto[0, 0]
    assert result.reason_codes_at(0, 0) == [reason]


def test_multiple_veto_reasons_are_retained_in_stable_order():
    values = _safe_financials()
    values.update(
        {
            "roe_latest": [-1.0],
            "revenue_yoy_latest": [-40.0],
            "net_income_yoy_latest": [-60.0],
        }
    )
    result = evaluate_fundamental_veto(_market(values), FundamentalVetoConfig())

    assert result.reason_codes_at(0, 0) == [
        NEGATIVE_ROE,
        SEVERE_PROFIT_DECLINE,
        SEVERE_REVENUE_DECLINE,
    ]


def test_missing_data_policy_reject_and_allow():
    values = _safe_financials()
    values["net_margin_latest"] = [None]
    market = _market(values)

    rejected = evaluate_fundamental_veto(market, FundamentalVetoConfig())
    allowed = evaluate_fundamental_veto(
        market,
        FundamentalVetoConfig(missing_data_policy="allow"),
    )
    relaxed = evaluate_fundamental_veto(
        market,
        FundamentalVetoConfig(minimum_available_fields=3),
    )

    assert rejected.veto[0, 0]
    assert rejected.reason_codes_at(0, 0) == [MISSING_FINANCIAL_DATA]
    assert not allowed.veto[0, 0]
    assert allowed.reason_codes_at(0, 0) == []
    assert not relaxed.veto[0, 0]


def _snapshot(rows: list[dict]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("announce_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("_announce")
        )
        .sort(["symbol", "_announce"])
    )


def test_pit_report_switch_and_future_mutation_isolation():
    dates = (date(2026, 8, 24), date(2026, 8, 25), date(2026, 8, 26))
    panel = _panel(dates=dates)
    base_rows = [
        {
            "symbol": "000001.SZ",
            "announce_date": "2026-04-30",
            "roe": 8.0,
            "revenue_yoy": 5.0,
            "net_income_yoy": 5.0,
            "net_margin": 5.0,
            "gross_margin": 20.0,
            "debt_to_asset_ratio": 40.0,
            "bps": 5.0,
        },
        {
            "symbol": "000001.SZ",
            "announce_date": "2026-08-25",
            "roe": -2.0,
            "revenue_yoy": -40.0,
            "net_income_yoy": -60.0,
            "net_margin": -8.0,
            "gross_margin": 10.0,
            "debt_to_asset_ratio": 45.0,
            "bps": 4.0,
        },
    ]
    names = fundamental_veto_required_fields(V4_TREND_STRATEGY_ID)
    snapshot = _snapshot(base_rows)
    attached = attach_fundamental_factors(panel, snapshot, names)
    market = build_market_data_matrix(panel)
    market = replace(
        market,
        fields={**dict(market.fields), **build_fundamental_matrices(market, snapshot, names)},
    )
    result = evaluate_fundamental_veto(market, FundamentalVetoConfig())

    assert not result.veto[0, 0]  # 8/24 sees the old report
    assert not result.veto[1, 0]  # announcement day still sees the old report
    assert result.veto[2, 0]  # first later trading day sees the new report
    assert attached.sort("date")["roe_latest"].to_list() == [8.0, 8.0, -2.0]

    mutated_rows = [dict(row) for row in base_rows]
    mutated_rows[1]["roe"] = -99.0
    mutated_rows[1]["revenue_yoy"] = -99.0
    mutated = build_fundamental_matrices(market, _snapshot(mutated_rows), names)
    mutated_market = replace(market, fields={**dict(market.fields), **mutated})
    mutated_result = evaluate_fundamental_veto(mutated_market, FundamentalVetoConfig())
    np.testing.assert_array_equal(mutated_result.veto[:2], result.veto[:2])
    np.testing.assert_array_equal(mutated_result.reason_mask[:2], result.reason_mask[:2])


def test_signal_application_preserves_exits_scores_and_read_only_contract():
    values = _safe_financials(2)
    values["roe_latest"] = [8.0, -1.0]
    market = _market(values)
    signals = make_signal_matrix(
        market.shape,
        entry=np.ones(market.shape, dtype=np.uint8),
        exit=np.ones(market.shape, dtype=np.uint8),
        score=np.asarray([[60.0, 90.0]], dtype=np.float32),
        entry_signal_code=np.zeros(market.shape, dtype=np.int16),
        exit_signal_code=np.zeros(market.shape, dtype=np.int16),
        entry_signal_ids=("technical",),
        exit_signal_ids=("exit",),
    )

    filtered = apply_fundamental_veto(
        signals,
        evaluate_fundamental_veto(market, FundamentalVetoConfig()),
    )

    assert filtered.entry.tolist() == [[1, 0]]
    assert filtered.entry_signal_code.tolist() == [[0, -1]]
    np.testing.assert_array_equal(filtered.exit, signals.exit)
    np.testing.assert_array_equal(filtered.score, signals.score)
    assert not filtered.entry.flags.writeable


class _AllEntryStrategy:
    def required_fields(self):
        return frozenset({"open", "high", "low", "close", "volume"})

    def required_warmup_bars(self, params):
        return 1

    def compute_signals(self, market, params):
        return make_signal_matrix(
            market.shape,
            entry=np.ones(market.shape, dtype=np.uint8),
            score=np.full(market.shape, 75.0, dtype=np.float32),
        )


def _strategy_def() -> StrategyDef:
    return StrategyDef(
        meta={
            "id": V4_TREND_STRATEGY_ID,
            "name": "test v4",
            "params": [],
            "scoring": {},
            "limit": 100,
            "descending": True,
            "asset_types": ["stock"],
            "timeframes": ["1d"],
        },
        basic_filter={"enabled": False},
        entry_signals=[],
        exit_signals=[],
        stop_loss=None,
        trailing_stop=None,
        trailing_take_profit_activate=None,
        trailing_take_profit_drawdown=None,
        max_hold_days=None,
        filter_fn=None,
        filter_history_fn=None,
        lookback_days=1,
        source="custom",
        execution_backend="matrix_native",
        matrix_strategy=_AllEntryStrategy(),
    )


class _BacktestEngineStub:
    def __init__(self, panel):
        self.panel = panel
        self.repo = SimpleNamespace(
            store=SimpleNamespace(data_dir=None),
            get_index_daily=lambda *_args, **_kwargs: pl.DataFrame(),
        )
        self.sim_matrix = None

    def load_market_data_matrix_for_backtest(
        self,
        symbols,
        start,
        end,
        feature_plan,
        asset_type="stock",
        **kwargs,
    ):
        fields = set(feature_plan.matrix_columns) | set(feature_plan.fundamental_columns)
        return build_market_data_matrix(self.panel, field_columns=fields)

    def simulate_market_matrix(
        self,
        matrix,
        config,
        progress_cb=None,
        cancel_event=None,
        options=None,
    ):
        self.sim_matrix = matrix
        return SimResult(
            equity_curve=[],
            drawdown_curve=[],
            trades=[],
            per_symbol_stats=[],
            stats={"total_return": 0.0, "n_trades": 0},
        )


def test_realtime_and_backtest_apply_identical_veto_mask():
    dates = (date(2026, 8, 25), date(2026, 8, 26))
    symbols = ("000001.SZ", "000002.SZ")
    financials = _safe_financials(2)
    financials["roe_latest"] = [8.0, -1.0]
    panel = _panel(symbols=symbols, dates=dates, financials=financials)
    strategy = _strategy_def()

    realtime_engine = StrategyEngine(strategy_dirs=[])
    realtime_engine._strategies[V4_TREND_STRATEGY_ID] = strategy
    realtime = realtime_engine.run(
        V4_TREND_STRATEGY_ID,
        StrategyDataContext(
            asset_type="stock",
            timeframe="1d",
            as_of=dates[-1],
            current=panel.filter(pl.col("date") == dates[-1]),
            history=panel,
        ),
    )

    backtest_engine = _BacktestEngineStub(panel)
    strategy_registry = SimpleNamespace(
        get=lambda _strategy_id: strategy,
        strategy_definitions=lambda: (strategy,),
    )
    backtest = StrategyBacktestService(backtest_engine, strategy_registry).run(
        StrategyBacktestConfig(
            strategy_id=V4_TREND_STRATEGY_ID,
            symbols=None,
            start=dates[0],
            end=dates[-1],
            matching="close_t",
            mode="position",
        )
    )

    assert backtest.error is None
    assert [row["symbol"] for row in realtime.rows] == ["000001.SZ"]
    assert realtime.fundamental_vetoes == [
        {
            "symbol": "000002.SZ",
            "fundamental_veto": True,
            "fundamental_veto_reason_codes": [NEGATIVE_ROE],
        }
    ]
    assert backtest_engine.sim_matrix.entry[-1].tolist() == [1, 0]
    assert backtest.stats["selection"]["technical_candidates"] == 4
    assert backtest.stats["selection"]["fundamental_vetoed"] == 2
    assert backtest.stats["selection"]["fundamental_veto_reason_counts"][NEGATIVE_ROE] == 2


def test_dependency_resolver_requests_veto_fields_from_pit_loader():
    strategy = _strategy_def()
    plan = StrategyDependencyResolver().resolve(
        strategy,
        params={},
        basic_filter=strategy.basic_filter,
        entry_signals=[],
        exit_signals=[],
        overrides={},
    )

    assert plan.fundamental_columns == fundamental_veto_required_fields(V4_TREND_STRATEGY_ID)


def test_committed_v4_strategy_loads_without_embedding_fundamental_logic():
    repo_root = Path(__file__).resolve().parents[2]
    strategy = StrategyEngine._load_file(
        repo_root / "data" / "strategies" / "custom" / "v4_trend_strategy.py"
    )
    assert strategy.meta["id"] == V4_TREND_STRATEGY_ID
    assert strategy.matrix_strategy.required_fields() == frozenset(
        {"open", "high", "low", "close", "volume"}
    )

    plan = StrategyDependencyResolver().resolve(
        strategy,
        params=StrategyEngine.resolve_params(strategy),
        basic_filter=strategy.basic_filter,
        entry_signals=strategy.entry_signals,
        exit_signals=strategy.exit_signals,
        overrides={},
    )
    assert plan.fundamental_columns == fundamental_veto_required_fields(V4_TREND_STRATEGY_ID)


def test_screener_context_attaches_financials_with_shared_pit_service(
    tmp_path,
    monkeypatch,
):
    metrics_dir = tmp_path / "financials" / "metrics"
    metrics_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "period_end": ["2026-06-30"],
            "announce_date": ["2026-08-25"],
            "roe": [-2.0],
            "revenue_yoy": [-40.0],
            "net_income_yoy": [-60.0],
            "net_margin": [-8.0],
            "gross_margin": [10.0],
            "debt_to_asset_ratio": [45.0],
            "bps": [4.0],
        }
    ).write_parquet(metrics_dir / "part.parquet")
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    service = ScreenerService(repo)
    history = _panel(dates=(date(2026, 8, 25), date(2026, 8, 26)))
    current = history.filter(pl.col("date") == date(2026, 8, 26))
    monkeypatch.setattr(service, "_load_enriched_for_date", lambda _as_of: current)
    monkeypatch.setattr(
        service,
        "_load_enriched_history",
        lambda _as_of, _bars: history,
    )
    engine = SimpleNamespace(required_history_bars=lambda *_args, **_kwargs: 2)

    context = service.build_strategy_context(
        engine,
        date(2026, 8, 26),
        [V4_TREND_STRATEGY_ID],
    )

    assert context.history.sort("date")["roe_latest"].to_list() == [None, -2.0]
    assert context.current["roe_latest"].item() == -2.0


def test_config_defaults_units_and_validation():
    config = resolve_fundamental_veto_config(V4_TREND_STRATEGY_ID)
    assert config == FundamentalVetoConfig(
        roe_min=0.0,
        net_income_yoy_min=-50.0,
        revenue_yoy_min=-30.0,
        net_margin_min=-5.0,
        debt_ratio_enabled=False,
        debt_ratio_max=85.0,
        gross_margin_enabled=True,
        gross_margin_min=-20.0,
        missing_data_policy="reject",
        minimum_available_fields=4,
    )
    with pytest.raises(ValueError, match="missing_data_policy"):
        resolve_fundamental_veto_config(
            V4_TREND_STRATEGY_ID,
            {"fundamental_veto": {"missing_data_policy": "unsafe"}},
        )

    stricter = resolve_fundamental_veto_config(
        V4_TREND_STRATEGY_ID,
        {"fundamental_veto": {"roe_min": 10.0}},
    )
    assert stricter is not None
    result = evaluate_fundamental_veto(_market(_safe_financials()), stricter)
    assert result.reason_codes_at(0, 0) == [NEGATIVE_ROE]


def test_load_snapshot_used_by_screener_is_strictly_announcement_dated(tmp_path):
    assert load_fundamental_snapshot(tmp_path) is None
