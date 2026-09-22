from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime

import httpx
import pytest

import app.services.historical_master as historical_master
from app.data_providers.historical_master import (
    HistoricalCredentialUnavailable,
    HistoricalSourcePayload,
)
from app.data_providers.tushare_historical import TushareHistoricalMasterProvider
from app.services.historical_master import (
    Coverage,
    HistoricalMasterQuery,
    HistoricalMasterStore,
    TruthValue,
    build_manifest,
    normalize_tushare,
)


def _payload(endpoint: str, rows: list[dict]) -> HistoricalSourcePayload:
    return HistoricalSourcePayload(
        "tushare", None, endpoint, datetime(2026, 9, 22, tzinfo=UTC), {}, tuple(rows)
    )


def _normalized():
    lifecycle = _payload(
        "stock_basic",
        [
            {
                "ts_code": "600000.SH",
                "exchange": "SSE",
                "name": "current",
                "list_date": "20000101",
                "list_status": "L",
            },
            {
                "ts_code": "600145.SH",
                "exchange": "SSE",
                "name": "delisted",
                "list_date": "19990501",
                "delist_date": "20220722",
                "list_status": "D",
            },
            {
                "ts_code": "920000.BJ",
                "exchange": "BSE",
                "name": "bse",
                "list_date": "20211115",
                "list_status": "L",
            },
            {
                "ts_code": "300750.SZ",
                "exchange": "SZSE",
                "name": "later",
                "list_date": "20180611",
                "list_status": "L",
            },
            {
                "ts_code": "600001.SH",
                "exchange": "SSE",
                "list_date": "20200102",
                "delist_date": "20200101",
                "list_status": "D",
            },
            {
                "ts_code": "600000.SH",
                "exchange": "SSE",
                "list_date": "20000101",
                "list_status": "L",
            },
            {"ts_code": "bad", "exchange": "SSE", "list_date": "20200101", "list_status": "L"},
        ],
    )
    st = _payload(
        "stock_st",
        [
            {"ts_code": "600000.SH", "type": "ST", "trade_date": "20200102"},
            {"ts_code": "600000.SH", "type": "ST", "trade_date": "20200103"},
            {"ts_code": "600000.SH", "type": "*ST", "trade_date": "20200106"},
            {"ts_code": "bad", "type": "ST", "trade_date": "20200102"},
        ],
    )
    suspensions = _payload(
        "suspend_d",
        [
            {
                "ts_code": "600000.SH",
                "trade_date": "20200103",
                "suspend_type": "S",
                "suspend_timing": "09:30-10:00",
            },
            {"ts_code": "600000.SH", "trade_date": "20200104", "suspend_type": "S"},
            {
                "ts_code": "600000.SH",
                "trade_date": "20200106",
                "suspend_type": "R",
                "suspend_timing": None,
            },
            {"ts_code": "999999.SH", "trade_date": "20200103", "suspend_type": "S"},
        ],
    )
    return normalize_tushare(
        [("life-1", lifecycle)],
        ("st-1", st),
        ("susp-1", suspensions),
        [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
    )


def test_normalization_keeps_lifecycle_st_type_bse_and_quarantines_bad_rows():
    result = _normalized()
    assert {row.symbol for row in result.lifecycle} == {
        "600000.SH",
        "600145.SH",
        "920000.BJ",
        "300750.SZ",
    }
    assert [
        (row.status_type, row.effective_from, row.effective_to) for row in result.risk_status
    ] == [("ST", "2020-01-02", "2020-01-06"), ("*ST", "2020-01-06", None)]
    assert result.suspensions[0].suspend_time == "09:30"
    assert {row.reason_code for row in result.rejected} >= {
        "INVALID_SYMBOL_OR_EXCHANGE",
        "INVALID_LIFECYCLE_DATE_OR_STATUS",
        "DUPLICATE_INSTRUMENT",
        "INVALID_OR_NON_TRADING_ST_ROW",
        "INVALID_OR_NON_TRADING_SUSPENSION_ROW",
        "NO_LIFECYCLE_MATCH",
    }


def test_query_helpers_are_tri_state_and_preserve_unverified_delisting_boundary():
    result = _normalized()
    coverage = {
        "historical_st": Coverage(
            "historical_st", "2020-01-02", "2020-01-06", ("SH",), complete=True
        ),
        "suspension_status": Coverage(
            "suspension_status", "2020-01-02", "2020-01-06", ("SH",), complete=True
        ),
    }
    query = HistoricalMasterQuery(
        result.lifecycle, result.risk_status, result.suspensions, coverage
    )
    assert query.is_listed_at("300750.SZ", date(2018, 6, 8)) is TruthValue.FALSE
    assert query.is_listed_at("600145.SH", date(2022, 7, 22)) is TruthValue.UNKNOWN
    assert query.is_listed_at("600145.SH", date(2022, 7, 23)) is TruthValue.FALSE
    assert query.risk_status_at("600000.SH", date(2020, 1, 2)) is TruthValue.TRUE
    assert query.risk_status_at("600000.SH", date(2020, 1, 3)) is TruthValue.TRUE
    assert query.risk_status_at("600000.SH", date(2020, 1, 6)) is TruthValue.TRUE
    assert query.risk_status_at("600000.SH", date(2020, 1, 7)) is TruthValue.UNKNOWN
    assert query.trading_status_at("600000.SH", date(2020, 1, 3)) is TruthValue.TRUE
    assert query.trading_status_at("600000.SH", date(2020, 1, 6)) is TruthValue.FALSE
    assert query.trading_status_at("600000.SH", date(2020, 1, 2)) is TruthValue.FALSE
    assert query.trading_status_at("300750.SZ", date(2020, 1, 2)) is TruthValue.UNKNOWN
    assert not Coverage(
        "historical_st",
        "2020-01-02",
        "2020-01-06",
        ("SH",),
        ("etf",),
        complete=True,
    ).covers(date(2020, 1, 2), "SH", "stock")


def test_snapshot_and_generation_are_immutable_and_atomic(tmp_path):
    result = _normalized()
    store = HistoricalMasterStore(tmp_path)
    payload = _payload("stock_basic", [{"ts_code": "600000.SH"}])
    snapshot = store.write_snapshot("life-1", payload)
    manifest_data = json.loads((snapshot / "manifest.json").read_text())
    assert "token" not in json.dumps(manifest_data).lower()
    with pytest.raises(FileExistsError):
        store.write_snapshot("life-1", payload)
    with pytest.raises(ValueError, match="credentials"):
        store.write_snapshot(
            "secret-1",
            HistoricalSourcePayload(
                "tushare",
                None,
                "stock_basic",
                datetime(2026, 9, 22, tzinfo=UTC),
                {"request": {"api_key": "x"}},
                (),
            ),
        )
    manifest = build_manifest("gen-1", "tushare", ["life-1", "st-1", "susp-1"], result, {}, {})
    generation = store.publish_generation("gen-1", result, manifest)
    assert (generation / "instrument_lifecycle.parquet").exists()
    assert not list(generation.parent.glob("*.staging"))
    with pytest.raises(FileExistsError):
        store.publish_generation("gen-1", result, manifest)
    with pytest.raises(ValueError, match="fingerprint"):
        store.publish_generation(
            "gen-2", result, replace(manifest, generation_id="gen-2", fingerprints={})
        )


def test_snapshot_failure_cleans_its_staging_directory(tmp_path, monkeypatch):
    store = HistoricalMasterStore(tmp_path)
    payload = _payload("stock_basic", [{"ts_code": "600000.SH"}])

    def fail_manifest_write(path, value):
        if path.name == "manifest.json":
            raise OSError("simulated write failure")
        return original_atomic_json(path, value)

    original_atomic_json = historical_master._atomic_json
    monkeypatch.setattr(historical_master, "_atomic_json", fail_manifest_write)
    with pytest.raises(OSError, match="simulated"):
        store.write_snapshot("failed-life", payload)
    assert not (store.source_root / "tushare" / "failed-life").exists()
    assert not list((store.source_root / "tushare").glob("*.staging"))


def test_st_intervals_follow_the_trading_calendar_over_a_weekend():
    lifecycle = _payload(
        "stock_basic",
        [{"ts_code": "600000.SH", "exchange": "SSE", "list_date": "20000101", "list_status": "L"}],
    )
    st = _payload(
        "stock_st",
        [
            {"ts_code": "600000.SH", "type": "ST", "trade_date": "20200103"},
            {"ts_code": "600000.SH", "type": "ST", "trade_date": "20200106"},
        ],
    )
    result = normalize_tushare(
        [("life-1", lifecycle)], ("st-1", st), None, [date(2020, 1, 3), date(2020, 1, 6)]
    )
    assert [(row.effective_from, row.effective_to) for row in result.risk_status] == [
        ("2020-01-03", None)
    ]


def test_conflicting_st_statuses_on_one_trade_date_are_quarantined():
    lifecycle = _payload(
        "stock_basic",
        [{"ts_code": "600000.SH", "exchange": "SSE", "list_date": "20000101", "list_status": "L"}],
    )
    st = _payload(
        "stock_st",
        [
            {"ts_code": "600000.SH", "type": "ST", "trade_date": "20200103"},
            {"ts_code": "600000.SH", "type": "*ST", "trade_date": "20200103"},
        ],
    )
    result = normalize_tushare([("life-1", lifecycle)], ("st-1", st), None, [date(2020, 1, 3)])
    assert not result.risk_status
    assert {row.reason_code for row in result.rejected} == {"CONTRADICTORY_RISK_STATUS_ROW"}


def test_tushare_adapter_has_no_credential_network_path():
    provider = TushareHistoricalMasterProvider(token="")
    with pytest.raises(HistoricalCredentialUnavailable):
        provider.fetch_instrument_lifecycle()
    provider.close()


def test_tushare_adapter_maps_schema_and_never_persists_token():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "stock_st"
        assert body["token"] == "secret"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "fields": ["ts_code", "type", "trade_date"],
                    "items": [["600000.SH", "ST", "20200102"]],
                },
            },
        )

    provider = TushareHistoricalMasterProvider(
        "secret", httpx.Client(transport=httpx.MockTransport(handler))
    )
    payload = provider.fetch_historical_st(date(2020, 1, 2), date(2020, 1, 2))
    assert payload.rows[0]["ts_code"] == "600000.SH"
    assert "token" not in payload.request_params
    provider.close()
