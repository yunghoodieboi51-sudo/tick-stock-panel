"""东方财富免费财务 provider 契约测试(不访问真实网络)。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from app.api import financials as financial_api
from app.data_providers import custom as custom_sources
from app.plugins.eastmoney_financial import provider as ep
from app.plugins.eastmoney_financial.client import (
    EastmoneyFinancialClient,
    EastmoneyFinancialError,
)
from app.plugins.eastmoney_financial.provider import (
    EastmoneyFinancialProvider,
    _normalize_metrics_row,
    _normalize_metrics_row_with_reason,
    to_eastmoney_symbol,
)
from app.services import financial_sync, preferences
from app.tickflow.capabilities import Cap, CapabilityLimits, CapabilitySet


def _raw(symbol: str, period: str, notice: str, **values) -> dict:
    return {
        "SECUCODE": symbol,
        "REPORT_DATE": f"{period} 00:00:00",
        "NOTICE_DATE": f"{notice} 00:00:00",
        "UPDATE_DATE": f"{notice} 00:00:00",
        "BPS": 200.99,
        "ROEJQ": 16.75,
        "XSMLL": 89.55,
        "XSJLL": 50.75,
        "TOTALOPERATEREVETZ": 1.30,
        "PARENTNETPROFITTZ": -1.95,
        "ZCFZL": 15.19,
        **values,
    }


class _FakeClient:
    def __init__(self, by_batch=None, failures=None):
        self.by_batch = by_batch or {}
        self.failures = set(failures or [])
        self.calls: list[tuple[tuple[str, ...], str]] = []
        self.closed = False

    def fetch_metrics(self, symbols, *, report_start):
        key = tuple(symbols)
        self.calls.append((key, report_start))
        if key in self.failures:
            raise EastmoneyFinancialError("temporary")
        return self.by_batch.get(key, [])

    def close(self):
        self.closed = True


def _prepare_multistage_scheduler(
    tmp_path,
    monkeypatch,
    *,
    universe: list[str],
    existing_symbols: list[str],
    client: _FakeClient,
):
    custom_sources.load_all(tmp_path / "sources")
    monkeypatch.setattr(preferences, "get_financial_provider", lambda: "eastmoney_financial")
    monkeypatch.setattr(ep, "_SYMBOL_BATCH_SIZE", 1)
    monkeypatch.setattr(ep, "_BATCH_INTERVAL_SECONDS", 0)
    provider = custom_sources.get_provider("eastmoney_financial")
    monkeypatch.setattr(provider, "_client", client)

    instruments_path = tmp_path / "instruments" / "instruments.parquet"
    instruments_path.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": universe}).write_parquet(instruments_path)
    metrics_path = tmp_path / "financials" / "metrics" / "part.parquet"
    metrics_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": existing_symbols,
            "period_end": ["2025-12-31"] * len(existing_symbols),
            "announce_date": ["2026-03-31"] * len(existing_symbols),
            "roe": [10.0] * len(existing_symbols),
        }
    ).write_parquet(metrics_path)

    scheduler = financial_sync.FinancialScheduler()
    scheduler._data_dir = tmp_path
    scheduler._capset = CapabilitySet()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                capabilities=CapabilitySet(),
                repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
                financial_scheduler=scheduler,
            )
        )
    )
    return scheduler, request


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("000001.SZ", "000001.SZ"),
        ("600519.SH", "600519.SH"),
        ("688001.SH", "688001.SH"),
        ("300750.SZ", "300750.SZ"),
        ("920002.BJ", "920002.BJ"),
        ("430047.BJ", "430047.BJ"),
        ("510300.SH", None),
        ("159919.SZ", None),
        ("000001.SH", None),
        ("HSI.HK", None),
    ],
)
def test_symbol_conversion_accepts_only_a_share_stocks(source, expected):
    assert to_eastmoney_symbol(source) == expected


def test_metrics_mapping_dates_units_and_nulls():
    raw = _raw(
        "600519.SH",
        "2026-06-30",
        "2026-08-15",
        XSMLL=None,
        XSJLL="50.7515",
        TOTALOPERATEREVETZ=float("nan"),
    )
    row = _normalize_metrics_row(raw)
    assert row == {
        "symbol": "600519.SH",
        "period_end": "2026-06-30",
        "announce_date": "2026-08-15",
        "bps": pytest.approx(200.99),
        "roe": pytest.approx(16.75),
        "gross_margin": None,
        "net_margin": pytest.approx(50.7515),
        "revenue_yoy": None,
        "net_income_yoy": pytest.approx(-1.95),
        "debt_to_asset_ratio": pytest.approx(15.19),
    }
    assert row["roe"] == 16.75  # 百分数值, 不转换为 0.1675
    assert not math.isnan(row["net_margin"])


def test_missing_announcement_or_later_revision_is_rejected():
    missing = _raw("600519.SH", "2026-06-30", "2026-08-15")
    missing["NOTICE_DATE"] = None
    assert _normalize_metrics_row(missing) is None

    revised = _raw("600519.SH", "2025-12-31", "2026-03-31")
    revised["UPDATE_DATE"] = "2026-08-15 00:00:00"
    assert _normalize_metrics_row(revised) is None

    impossible = _raw("600519.SH", "2026-06-30", "2026-06-29")
    assert _normalize_metrics_row_with_reason(impossible) == (None, "missing_or_invalid_date")


def test_history_and_latest_only(monkeypatch):
    monkeypatch.setattr(ep, "_SYMBOL_BATCH_SIZE", 10)
    monkeypatch.setattr(ep, "_BATCH_INTERVAL_SECONDS", 0)
    rows = [
        _raw("600519.SH", "2025-12-31", "2026-03-31"),
        _raw("600519.SH", "2026-06-30", "2026-08-15"),
    ]
    client = _FakeClient({("600519.SH",): rows})
    provider = EastmoneyFinancialProvider(client)

    history = provider.get_financials("metrics", ["600519.SH"], latest_only=False)
    assert history["period_end"].to_list() == ["2025-12-31", "2026-06-30"]
    assert int(client.calls[-1][1][:4]) <= ep.date.today().year - ep._HISTORY_YEARS

    latest = provider.get_financials("metrics", ["600519.SH"], latest_only=True)
    assert latest["period_end"].to_list() == ["2026-06-30"]
    assert int(client.calls[-1][1][:4]) <= ep.date.today().year - ep._LATEST_LOOKBACK_YEARS


def test_batch_failure_isolated_and_non_stocks_not_requested(monkeypatch):
    monkeypatch.setattr(ep, "_SYMBOL_BATCH_SIZE", 1)
    monkeypatch.setattr(ep, "_BATCH_INTERVAL_SECONDS", 0)
    first = ("600519.SH",)
    second = ("000001.SZ",)
    client = _FakeClient(
        {second: [_raw("000001.SZ", "2026-06-30", "2026-08-15")]},
        failures={first},
    )
    provider = EastmoneyFinancialProvider(client)

    frame = provider.get_financials(
        "metrics", ["600519.SH", "510300.SH", "000001.SZ"], latest_only=False
    )

    assert frame["symbol"].to_list() == ["000001.SZ"]
    assert [call[0] for call in client.calls] == [first, second]


def test_all_batch_failures_are_not_reported_as_empty_success(monkeypatch):
    monkeypatch.setattr(ep, "_SYMBOL_BATCH_SIZE", 1)
    monkeypatch.setattr(ep, "_BATCH_INTERVAL_SECONDS", 0)
    client = _FakeClient(failures={("600519.SH",), ("000001.SZ",)})
    provider = EastmoneyFinancialProvider(client)

    with pytest.raises(EastmoneyFinancialError, match="全部 2 个 metrics 批次请求失败"):
        provider.get_financials("metrics", ["600519.SH", "000001.SZ"], latest_only=True)


def test_unsupported_tables_are_explicitly_empty():
    provider = EastmoneyFinancialProvider(_FakeClient())
    for table in ("income", "balance_sheet", "cash_flow", "shares"):
        assert provider.get_financials(table, ["600519.SH"]).is_empty()


def test_client_paginates_batch_and_passes_history_filter(monkeypatch):
    client = object.__new__(EastmoneyFinancialClient)
    calls: list[tuple[str, int]] = []

    def fake_page(filter_value, page):
        calls.append((filter_value, page))
        return {
            "result": {
                "pages": 2,
                "data": [{"page": page}],
            }
        }

    monkeypatch.setattr(client, "_request_page", fake_page)
    rows = client.fetch_metrics(["600519.SH", "000001.SZ"], report_start="2016-01-01")

    assert rows == [{"page": 1}, {"page": 2}]
    assert [page for _, page in calls] == [1, 2]
    assert 'SECUCODE in ("600519.SH","000001.SZ")' in calls[0][0]
    assert "REPORT_DATE>='2016-01-01'" in calls[0][0]


def test_client_rejects_truncated_pagination(monkeypatch):
    client = object.__new__(EastmoneyFinancialClient)

    def fake_page(_filter_value, page):
        return {
            "result": {
                "pages": 2,
                "data": [{"page": 1}] if page == 1 else [],
            }
        }

    monkeypatch.setattr(client, "_request_page", fake_page)
    with pytest.raises(EastmoneyFinancialError, match="第 2 页为空"):
        client.fetch_metrics(["600519.SH"], report_start="2016-01-01")


def test_eastmoney_response_schema_fixture_locks_pit_and_pagination_fields():
    fixture_path = Path(__file__).parent / "fixtures" / "eastmoney_financial_metrics_page.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert payload["success"] is True
    assert {"pages", "count", "data"} <= payload["result"].keys()
    raw = payload["result"]["data"][0]
    assert {
        "SECUCODE",
        "REPORT_DATE",
        "NOTICE_DATE",
        "UPDATE_DATE",
        "BPS",
        "ROEJQ",
        "XSMLL",
        "XSJLL",
        "TOTALOPERATEREVETZ",
        "PARENTNETPROFITTZ",
        "ZCFZL",
    } <= raw.keys()
    row = _normalize_metrics_row(raw)
    assert row["period_end"] == "2026-06-30"
    assert row["announce_date"] == "2026-08-15"


def test_structured_stats_split_drop_reasons_and_failed_symbols(monkeypatch):
    monkeypatch.setattr(ep, "_SYMBOL_BATCH_SIZE", 1)
    monkeypatch.setattr(ep, "_BATCH_INTERVAL_SECONDS", 0)
    failed = ("600519.SH",)
    successful = ("000001.SZ",)
    missing_date = _raw("000001.SZ", "2026-03-31", "2026-04-25")
    missing_date["NOTICE_DATE"] = None
    revision = _raw("000001.SZ", "2025-12-31", "2026-03-21")
    revision["UPDATE_DATE"] = "2026-04-01 00:00:00"
    rows = [
        _raw("000001.SZ", "2026-06-30", "2026-08-15"),
        missing_date,
        revision,
        _raw("510300.SH", "2026-06-30", "2026-08-15"),
        _raw("601988.SH", "2026-06-30", "2026-08-29"),
    ]
    provider = EastmoneyFinancialProvider(_FakeClient({successful: rows}, failures={failed}))

    frame = provider.get_financials(
        "metrics",
        ["600519.SH", "000001.SZ", "510300.SH"],
        latest_only=False,
    )

    assert frame["symbol"].to_list() == ["000001.SZ"]
    stats = provider.last_stats
    assert stats["status"] == "partial"
    assert stats["requested_symbols"] == 3
    assert stats["valid_symbols"] == 2
    assert stats["invalid_input_symbols"] == 1
    assert stats["total_batches"] == 2
    assert stats["failed_batches"] == 1
    assert stats["failed_symbols"] == 1
    assert stats["upstream_rows"] == 5
    assert stats["safe_rows"] == 1
    assert stats["output_rows"] == 1
    assert stats["dropped_rows"] == 4
    assert stats["dropped_revision"] == 1
    assert stats["dropped_missing_or_invalid_date"] == 1
    assert stats["dropped_invalid_symbol"] == 1
    assert stats["dropped_unexpected_symbol"] == 1
    assert stats["pit_acceptance_rate"] == 0.2
    assert stats["pit_drop_rate"] == 0.8
    assert stats["symbol_coverage_rate"] == 0.5


def test_builtin_plugin_is_selectable_by_financial_routing(tmp_path, monkeypatch):
    custom_sources.load_all(tmp_path / "sources")
    monkeypatch.setattr(
        preferences,
        "get_financial_provider",
        lambda: "eastmoney_financial",
    )

    assert custom_sources.provider_has_dataset("eastmoney_financial", "financial")
    assert financial_sync._financial_is_custom() is True


def test_financial_sync_fetches_through_registered_provider(tmp_path, monkeypatch):
    custom_sources.load_all(tmp_path / "sources")
    monkeypatch.setattr(
        preferences,
        "get_financial_provider",
        lambda: "eastmoney_financial",
    )
    provider = custom_sources.get_provider("eastmoney_financial")
    fake_client = _FakeClient({("600519.SH",): [_raw("600519.SH", "2026-06-30", "2026-08-15")]})
    monkeypatch.setattr(provider, "_client", fake_client)
    monkeypatch.setattr(ep, "_BATCH_INTERVAL_SECONDS", 0)

    frame = financial_sync._fetch_table(
        "metrics",
        ["600519.SH"],
        CapabilitySet(),
        latest_only=False,
    )

    assert frame.select("symbol", "period_end", "announce_date").to_dicts() == [
        {
            "symbol": "600519.SH",
            "period_end": "2026-06-30",
            "announce_date": "2026-08-15",
        }
    ]


def test_legacy_custom_provider_untyped_error_keeps_empty_fallback(monkeypatch):
    class LegacyProvider:
        def get_financials(self, *_args, **_kwargs):
            raise RuntimeError("legacy temporary failure")

    monkeypatch.setattr(financial_sync, "_financial_is_custom", lambda: True)
    monkeypatch.setattr(preferences, "get_financial_provider", lambda: "legacy")
    monkeypatch.setattr(custom_sources, "get_provider", lambda _name: LegacyProvider())

    assert financial_sync._fetch_table("metrics", ["600519.SH"], CapabilitySet()).is_empty()


def test_financial_sync_initial_and_incremental_merge(tmp_path, monkeypatch):
    first = pl.DataFrame(
        {
            "symbol": ["600519.SH", "600519.SH"],
            "period_end": ["2025-12-31", "2026-03-31"],
            "announce_date": ["2026-03-31", "2026-04-25"],
            "roe": [30.0, 10.0],
            "revenue_yoy": [5.0, 6.0],
        }
    )
    incremental = pl.DataFrame(
        {
            "symbol": ["600519.SH", "600519.SH"],
            "period_end": ["2026-03-31", "2026-06-30"],
            "announce_date": ["2026-04-26", "2026-08-15"],
            "roe": [None, 16.75],
            "revenue_yoy": [7.0, 1.3],
        }
    )
    calls: list[bool] = []

    def fake_fetch(_table, _symbols, _capset, latest_only=True):
        calls.append(latest_only)
        return first if len(calls) == 1 else incremental

    monkeypatch.setattr(financial_sync, "_fetch_table", fake_fetch)
    capset = CapabilitySet()

    assert (
        financial_sync._sync_history_table_for_symbols("metrics", ["600519.SH"], tmp_path, capset)
        == 2
    )
    assert (
        financial_sync._sync_history_table_for_symbols("metrics", ["600519.SH"], tmp_path, capset)
        == 3
    )

    stored = financial_sync.get_financial_df(tmp_path, "metrics").sort("period_end")
    assert calls == [False, True]
    assert stored["period_end"].to_list() == [
        "2025-12-31",
        "2026-03-31",
        "2026-06-30",
    ]
    q1 = stored.filter(pl.col("period_end") == "2026-03-31").to_dicts()[0]
    assert q1["announce_date"] == "2026-04-26"
    assert q1["roe"] == 10.0  # 新行缺字段时保留旧非空值
    assert q1["revenue_yoy"] == 7.0


def test_empty_sync_response_preserves_existing_history(tmp_path, monkeypatch):
    path = tmp_path / "financials" / "metrics" / "part.parquet"
    path.parent.mkdir(parents=True)
    existing = pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "period_end": ["2026-06-30"],
            "announce_date": ["2026-08-15"],
            "roe": [16.75],
        }
    )
    existing.write_parquet(path)
    monkeypatch.setattr(
        financial_sync,
        "_fetch_table",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )

    rows = financial_sync._sync_history_table_for_symbols(
        "metrics", ["600519.SH"], tmp_path, CapabilitySet()
    )

    assert rows == 1
    assert financial_sync.get_financial_df(tmp_path, "metrics").equals(existing)


def test_multistage_first_partial_then_success_stays_partial(tmp_path, monkeypatch):
    revised = _raw("000002.SZ", "2025-12-31", "2026-03-31")
    revised["UPDATE_DATE"] = "2026-04-01 00:00:00"
    client = _FakeClient(
        {
            ("000002.SZ",): [
                _raw("000002.SZ", "2026-06-30", "2026-08-15"),
                revised,
            ],
            ("600519.SH",): [_raw("600519.SH", "2026-06-30", "2026-08-15")],
        },
        failures={("000001.SZ",)},
    )
    scheduler, request = _prepare_multistage_scheduler(
        tmp_path,
        monkeypatch,
        universe=["600519.SH", "000001.SZ", "000002.SZ"],
        existing_symbols=["600519.SH"],
        client=client,
    )

    scheduler.run_now("metrics")

    result = financial_api.financial_status(request)["sync_results"]["metrics"]
    assert result["status"] == "partial"
    assert result["provider_stats"]["stage_statuses"] == ["partial", "success"]
    assert result["provider_stats"]["requested_symbols"] == 3
    assert result["provider_stats"]["successful_symbols"] == 2
    assert result["provider_stats"]["failed_symbols"] == 1
    assert result["provider_stats"]["failed_batches"] == 1
    assert result["provider_stats"]["received_rows"] == 3
    assert result["provider_stats"]["accepted_rows"] == 2
    assert result["provider_stats"]["dropped_rows"] == 1
    assert result["provider_stats"]["dropped_revision"] == 1


def test_multistage_first_success_then_partial_stays_partial(tmp_path, monkeypatch):
    client = _FakeClient(
        {
            ("000001.SZ",): [_raw("000001.SZ", "2026-06-30", "2026-08-15")],
            ("000002.SZ",): [_raw("000002.SZ", "2026-06-30", "2026-08-15")],
        },
        failures={("600519.SH",)},
    )
    scheduler, request = _prepare_multistage_scheduler(
        tmp_path,
        monkeypatch,
        universe=["600519.SH", "000001.SZ", "000002.SZ"],
        existing_symbols=["600519.SH", "000002.SZ"],
        client=client,
    )

    scheduler.run_now("metrics")

    result = financial_api.financial_status(request)["sync_results"]["metrics"]
    assert result["status"] == "partial"
    assert result["provider_stats"]["stage_statuses"] == ["success", "partial"]
    assert result["provider_stats"]["requested_symbols"] == 3
    assert result["provider_stats"]["successful_symbols"] == 2
    assert result["provider_stats"]["failed_symbols"] == 1
    assert result["provider_stats"]["failed_batches"] == 1


def test_multistage_all_success_is_success(tmp_path, monkeypatch):
    client = _FakeClient(
        {
            ("000001.SZ",): [_raw("000001.SZ", "2026-06-30", "2026-08-15")],
            ("600519.SH",): [_raw("600519.SH", "2026-06-30", "2026-08-15")],
        }
    )
    scheduler, request = _prepare_multistage_scheduler(
        tmp_path,
        monkeypatch,
        universe=["600519.SH", "000001.SZ"],
        existing_symbols=["600519.SH"],
        client=client,
    )

    scheduler.run_now("metrics")

    result = financial_api.financial_status(request)["sync_results"]["metrics"]
    assert result["status"] == "success"
    assert result["provider_stats"]["stage_statuses"] == ["success", "success"]
    assert result["provider_stats"]["requested_symbols"] == 2
    assert result["provider_stats"]["successful_symbols"] == 2
    assert result["provider_stats"]["failed_symbols"] == 0
    assert result["provider_stats"]["failed_batches"] == 0


def test_total_provider_failure_reaches_scheduler_status_and_preserves_parquet(
    tmp_path, monkeypatch
):
    custom_sources.load_all(tmp_path / "sources")
    monkeypatch.setattr(
        preferences,
        "get_financial_provider",
        lambda: "eastmoney_financial",
    )
    provider = custom_sources.get_provider("eastmoney_financial")
    monkeypatch.setattr(
        provider,
        "_client",
        _FakeClient(failures={("600519.SH",)}),
    )
    _write_instruments = pl.DataFrame({"symbol": ["600519.SH"]})
    instruments_path = tmp_path / "instruments" / "instruments.parquet"
    instruments_path.parent.mkdir(parents=True)
    _write_instruments.write_parquet(instruments_path)
    existing = pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "period_end": ["2025-12-31"],
            "announce_date": ["2026-04-17"],
            "roe": [30.0],
        }
    )
    metrics_path = tmp_path / "financials" / "metrics" / "part.parquet"
    metrics_path.parent.mkdir(parents=True)
    existing.write_parquet(metrics_path)
    persisted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        preferences,
        "set_financial_sync_time",
        lambda table, value: persisted.append((table, value)),
    )
    scheduler = financial_sync.FinancialScheduler()
    scheduler._data_dir = tmp_path
    scheduler._capset = CapabilitySet()

    with pytest.raises(financial_sync.FinancialSyncError, match="provider request failed"):
        scheduler.run_now("metrics")

    assert pl.read_parquet(metrics_path).equals(existing)
    assert scheduler.last_sync == {}
    assert persisted == []
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                capabilities=CapabilitySet(),
                repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
                financial_scheduler=scheduler,
            )
        )
    )
    status = financial_api.financial_status(request)
    assert status["sync_results"]["metrics"]["status"] == "failed"
    assert status["sync_results"]["metrics"]["provider_stats"]["failed_batches"] == 1
    assert status["sync_results"]["metrics"]["provider_stats"]["failed_symbols"] == 1
    assert status["sync_results"]["metrics"]["provider_stats"]["successful_symbols"] == 0
    assert "metrics" not in status["last_sync"]


def test_metrics_only_provider_hides_unsupported_old_files_and_switch_restores(
    tmp_path, monkeypatch
):
    custom_sources.load_all(tmp_path / "sources")
    selected = {"provider": "eastmoney_financial"}
    monkeypatch.setattr(
        preferences,
        "get_financial_provider",
        lambda: selected["provider"],
    )
    income_path = tmp_path / "financials" / "income" / "part.parquet"
    income_path.parent.mkdir(parents=True)
    old_income = pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "period_end": ["2025-12-31"],
            "revenue": [100.0],
        }
    )
    old_income.write_parquet(income_path)
    scheduler = financial_sync.FinancialScheduler()
    scheduler._capset = CapabilitySet()
    scheduler._last_sync = {"income": "2026-04-17T00:00:00+00:00"}
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                capabilities=CapabilitySet(),
                repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
                financial_scheduler=scheduler,
            )
        )
    )

    status = financial_api.financial_status(request)
    assert status["supported_tables"] == ["metrics"]
    assert status["tables"]["income"] == {
        "rows": 0,
        "symbols": 0,
        "supported": False,
        "available": False,
        "provider": "eastmoney_financial",
        "reason": "unsupported_by_provider",
        "retained_local_data": True,
    }
    assert "income" not in status["last_sync"]
    assert financial_api.get_income(request)["data"] == []
    assert financial_api.get_income(request)["supported"] is False
    assert financial_sync.get_financial_df(tmp_path, "income").is_empty()
    assert scheduler.trigger("income") == {
        "started": False,
        "reason": "unsupported table",
        "table": "income",
    }
    assert income_path.exists()

    selected["provider"] = "tickflow"
    request.app.state.capabilities = CapabilitySet({Cap.FINANCIAL: CapabilityLimits()})
    restored_status = financial_api.financial_status(request)
    assert restored_status["tables"]["income"]["supported"] is True
    assert restored_status["tables"]["income"]["rows"] == 1
    assert restored_status["last_sync"]["income"] == "2026-04-17T00:00:00+00:00"
    assert financial_api.get_income(request)["data"] == old_income.to_dicts()
