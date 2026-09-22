from datetime import date

import polars as pl
import pytest

from app.services.historical_pilot import (
    PILOT_END,
    PILOT_INDEXES,
    PILOT_STOCKS,
    WARMUP_START,
    new_manifest,
    restore_backup,
    run_pilot,
)
from app.tickflow.repository import DataStore, KlineRepository


def _daily(symbols, start=WARMUP_START, end=PILOT_END):
    dates = sorted({start, date(2025, 9, 15), end})
    return pl.DataFrame(
        {
            "symbol": [s for s in symbols for _ in dates],
            "date": dates * len(symbols),
            "open": [10.0] * (len(symbols) * len(dates)),
            "high": [11.0] * (len(symbols) * len(dates)),
            "low": [9.0] * (len(symbols) * len(dates)),
            "close": [10.0] * (len(symbols) * len(dates)),
            "volume": [100.0] * (len(symbols) * len(dates)),
            "amount": [1000.0] * (len(symbols) * len(dates)),
        }
    )


class Provider:
    name = "fake"

    def __init__(self):
        self.calls = []

    def get_daily(self, symbols, start_time, end_time, asset_type):
        self.calls.append((tuple(symbols), asset_type))
        return _daily(symbols)


@pytest.fixture
def repo(tmp_path):
    value = KlineRepository(DataStore(tmp_path))
    value.append_daily(_daily([PILOT_STOCKS[0]], date(2025, 9, 15), PILOT_END))
    value.append_index_daily(_daily([PILOT_INDEXES[0]], date(2025, 9, 15), PILOT_END))
    return value


def test_manifest_has_bounded_batch_plan():
    manifest = new_manifest(Provider())
    assert [len(item.symbols) for item in manifest.batches] == [3, 5, 3]
    assert manifest.adjustment_policy == "RAW_UNADJUSTED"


def test_pilot_writes_only_historical_rows_and_resume_skips(repo, tmp_path, monkeypatch):
    provider = Provider()
    monkeypatch.setattr("app.services.historical_pilot.run_pipeline", lambda **kwargs: 0)
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    backup = tmp_path / "backup"
    before = pl.read_parquet(
        repo.store.data_dir / "kline_daily" / "date=2025-09-15" / "part.parquet"
    )
    result = run_pilot(
        repo, provider, manifest_path=manifest, report_path=report, backup_root=backup
    )
    after = pl.read_parquet(
        repo.store.data_dir / "kline_daily" / "date=2025-09-15" / "part.parquet"
    )
    assert before.equals(after)
    assert result.stock_rows_written > 0 and result.index_rows_written > 0
    report_payload = __import__("json").loads(report.read_text())
    assert report_payload["provider_returned_rows"] == {"stock": 24, "index": 9}
    assert report_payload["summary"]["stock"]["historical_rows_before_existing_range"] > 0
    calls = len(provider.calls)
    second = run_pilot(
        repo, provider, manifest_path=manifest, report_path=report, backup_root=backup
    )
    assert second.skipped_batches == 3 and len(provider.calls) == calls


def test_restore_removes_only_pilot_history(repo, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.historical_pilot.run_pipeline", lambda **kwargs: 0)
    backup = tmp_path / "backup"
    run_pilot(
        repo,
        Provider(),
        manifest_path=tmp_path / "m.json",
        report_path=tmp_path / "r.json",
        backup_root=backup,
    )
    restore_backup(repo.store.data_dir, backup)
    assert not (repo.store.data_dir / "kline_daily" / "date=2020-01-02").exists()
    assert (repo.store.data_dir / "kline_daily" / "date=2025-09-15" / "part.parquet").exists()


def test_completed_manifest_without_report_repairs_derived_stage(repo, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.services.historical_pilot.run_pipeline",
        lambda **kwargs: calls.append(kwargs) or 0,
    )
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    run_pilot(repo, Provider(), manifest_path=manifest, report_path=report)
    report.unlink()
    run_pilot(repo, Provider(), manifest_path=manifest, report_path=report)
    assert len(calls) == 2
    assert report.exists()


def test_invalid_ohlc_blocks_write_and_marks_failed(repo, tmp_path, monkeypatch):
    class Bad(Provider):
        def get_daily(self, symbols, start_time, end_time, asset_type):
            value = _daily(symbols)
            return value.with_columns(pl.lit(-1.0).alias("open"))

    monkeypatch.setattr("app.services.historical_pilot.run_pipeline", lambda **kwargs: 0)
    with pytest.raises(ValueError, match="raw validation"):
        run_pilot(repo, Bad(), manifest_path=tmp_path / "m.json", report_path=tmp_path / "r.json")
    assert not (repo.store.data_dir / "kline_daily" / "date=2020-01-02").exists()


def test_partial_response_is_persisted_not_completed(repo, tmp_path, monkeypatch):
    class Partial(Provider):
        def get_daily(self, symbols, start_time, end_time, asset_type):
            return _daily(symbols[:1])

    monkeypatch.setattr("app.services.historical_pilot.run_pipeline", lambda **kwargs: 0)
    result = run_pilot(
        repo, Partial(), manifest_path=tmp_path / "m.json", report_path=tmp_path / "r.json"
    )
    payload = __import__("json").loads(result.manifest_path.read_text())
    assert payload["batches"][0]["status"] == "PARTIAL"
