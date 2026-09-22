"""V4.8 orchestration tests; the V4 signal itself remains owned by its strategy."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from app.services import daily_scan


def _row(symbol: str, score: float, amount: float, raw_close: float) -> dict:
    return {
        "symbol": symbol,
        "name": symbol,
        "date": date(2026, 9, 11),
        "score": score,
        "amount": amount,
        "close": raw_close * 1.1,
        "raw_close": raw_close,
        "atr_pct": 0.02,
        "primary_entry_pattern": "BREAKOUT",
        "matched_entry_patterns": ["BREAKOUT"],
        "score_breakdown": {
            "trend": 20.0,
            "momentum": 15.0,
            "breakout": 18.0,
            "volume": 10.0,
            "volatility": 8.0,
        },
    }


class _Screener:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def latest_date(self) -> date:
        return date(2026, 9, 11)

    def build_strategy_context(self, *_args, **_kwargs) -> object:
        return object()


class _Engine:
    def __init__(self) -> None:
        self.strategy = SimpleNamespace(
            meta={"version": "test"}, stop_loss=-0.08, trailing_stop=-0.10, max_hold_days=40
        )

    def get(self, strategy_id: str) -> object:
        assert strategy_id == daily_scan.STRATEGY_ID
        return self.strategy

    def resolve_params(self, _strategy: object, params: dict, _overrides: dict) -> dict:
        return {"max_daily_gain": 0.085, **params}

    def run(self, strategy_id: str, _context: object, *, params: dict, overrides: dict) -> object:
        assert strategy_id == daily_scan.STRATEGY_ID
        assert params["max_daily_gain"] == 0.085
        assert overrides is None
        vetoed = _row("000003.SZ", 95.0, 1.0, 8.0)
        vetoed["fundamental_veto"] = True
        vetoed["fundamental_veto_reason_codes"] = ["MISSING_FINANCIAL_DATA"]
        return SimpleNamespace(
            rows=[_row("000001.SZ", 80.0, 10.0, 10.0), _row("000002.SZ", 90.0, 5.0, 20.0)],
            vetoed_rows=[vetoed],
        )


def test_daily_scan_reuses_engine_result_and_keeps_raw_plan_prices(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_scan, "ScreenerService", _Screener)
    monkeypatch.setattr(daily_scan.strategy_config, "load_override", lambda *_args: {})
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    store = daily_scan.DailyScanRunStore(tmp_path)
    manifest = store.create()
    result = daily_scan.DailyScanService(repo, _Engine(), store).run(
        manifest["run_id"],
        {
            "account_size": 100000,
            "max_position_pct": 0.6,
            "risk_per_trade_pct": 0.1,
            "max_candidates": 1,
        },
    )
    candidate = result["candidates"][0]
    assert candidate["symbol"] == "000002.SZ"  # final score DESC, then amount, then symbol
    assert candidate["signal_reference_price"] == 22.0
    assert candidate["execution_reference_price"] == candidate["latest_raw_price"] == 20.0
    assert candidate["signal_price_basis"] == "FORWARD_ADJUSTED"
    assert candidate["execution_price_basis"] == "RAW"
    assert candidate["plan"]["buy_zone_low"] == 20.0
    assert candidate["plan"]["buy_zone_high"] == 20.4
    assert candidate["plan"]["estimated_shares"] % 100 == 0
    assert candidate["plan"]["estimated_order_value"] <= 100000
    assert result["rejected"][0]["fundamental"]["status"] == "VETO"
    assert result["rejected"][0]["plan"]["fundamental_status"] == "VETO"
    assert result["fundamental_veto_policy"] is not None


def test_run_store_is_immutable_and_retains_only_finished_runs(tmp_path) -> None:
    store = daily_scan.DailyScanRunStore(tmp_path, retention=1)
    first = store.create()
    store.complete(first["run_id"], {"candidates": [], "rejected": []})
    second = store.create()
    store.complete(second["run_id"], {"candidates": [], "rejected": []})
    assert store.result(first["run_id"]) is None
    restored = store.result(second["run_id"])
    assert restored is not None
    assert restored["manifest"]["state"] == "COMPLETED"
    assert restored["result"] == {"candidates": [], "rejected": []}


def test_plan_with_missing_raw_reference_never_invents_nominal_prices() -> None:
    strategy = SimpleNamespace(stop_loss=-0.08, trailing_stop=-0.10, max_hold_days=40)
    plan = daily_scan.DailyScanService._plan(None, None, 0.08, strategy, {"account_size": 100000})
    assert plan["plan_price_status"] == "REFERENCE_ONLY_UNADJUSTED_MAPPING_UNAVAILABLE"
    assert plan["buy_zone_low"] is None
    assert plan["stop_loss_price"] is None
