"""Small, manifest-driven Level-1 historical-data pilot.

This is deliberately an offline operator tool, not a scheduler or API.  It
uses the normal provider and repository contracts, but is hard-limited by its
caller to a small universe.  Prices are stored as provider RAW_UNADJUSTED
OHLCV: the result must not be used for adjusted-price strategy research.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from app.indicators.pipeline import compute_enriched, run_pipeline
from app.services.historical_research import (
    BackfillBatchStatus,
    HistoricalBackfillBatch,
    HistoricalBackfillManifest,
    is_valid_listing_date,
    unexpected_missing_symbols,
    validate_daily_coverage,
)
from app.tickflow.repository import KlineRepository

PILOT_STOCKS = (
    "600519.SH",
    "600000.SH",
    "600735.SH",
    "000001.SZ",
    "300750.SZ",
    "688981.SH",
    "920000.BJ",
    "688836.SH",
)
PILOT_INDEXES = ("000001.SH", "399001.SZ", "000688.SH")
WARMUP_START = date(2020, 1, 2)
RESEARCH_START = date(2021, 1, 1)
PILOT_END = date(2026, 9, 11)
DATASET_VERSION = "research_daily_v1_pilot"


class DailyProvider(Protocol):
    name: str

    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str,
    ) -> pl.DataFrame: ...


@dataclass(frozen=True)
class PilotResult:
    manifest_path: Path
    report_path: Path
    backup_path: Path | None
    stock_rows_written: int
    index_rows_written: int
    enriched_rows_written: int
    skipped_batches: int


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _logical_fingerprint(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "sha256:empty"
    cols = [
        c
        for c in ("symbol", "date", "open", "high", "low", "close", "volume", "amount")
        if c in frame.columns
    ]
    payload = frame.select(cols).sort([c for c in ("symbol", "date") if c in cols]).write_json()
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _load_manifest(path: Path) -> HistoricalBackfillManifest | None:
    if not path.exists():
        return None
    return HistoricalBackfillManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _save_manifest(path: Path, manifest: HistoricalBackfillManifest) -> None:
    _atomic_json(path, manifest.to_dict())


def _plan_batches() -> tuple[HistoricalBackfillBatch, ...]:
    batches: list[HistoricalBackfillBatch] = []
    # Index first: 000001.SH becomes the real-market calendar used to assess
    # stock coverage, never a Monday-Friday approximation.
    for asset_type, symbols, size in (("index", PILOT_INDEXES, 3), ("stock", PILOT_STOCKS, 5)):
        for number, offset in enumerate(range(0, len(symbols), size), 1):
            batches.append(
                HistoricalBackfillBatch(
                    f"{asset_type}-{number:02d}",
                    asset_type,
                    tuple(symbols[offset : offset + size]),
                    WARMUP_START.isoformat(),
                    PILOT_END.isoformat(),
                )
            )
    return tuple(batches)


def new_manifest(provider: DailyProvider) -> HistoricalBackfillManifest:
    timestamp = _now()
    return HistoricalBackfillManifest(
        schema_version=1,
        dataset_version=DATASET_VERSION,
        provider=provider.name,
        provider_version=None,
        provider_capability="daily",
        requested_start=WARMUP_START.isoformat(),
        requested_end=PILOT_END.isoformat(),
        warmup_start=WARMUP_START.isoformat(),
        research_start=RESEARCH_START.isoformat(),
        universe_version="current_instruments_snapshot",
        survivorship_level="LEVEL 1",
        adjustment_policy="RAW_UNADJUSTED",
        created_at=timestamp,
        updated_at=timestamp,
        batches=_plan_batches(),
    )


def _replace_and_save(
    path: Path, manifest: HistoricalBackfillManifest, batch: HistoricalBackfillBatch
) -> HistoricalBackfillManifest:
    updated = replace(manifest.replace_batch(batch), updated_at=_now())
    _save_manifest(path, updated)
    return updated


def _canonical_columns(frame: pl.DataFrame) -> pl.DataFrame:
    cols = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    absent = sorted(set(cols) - set(frame.columns))
    if absent:
        raise ValueError(f"provider response missing canonical columns: {','.join(absent)}")
    return (
        frame.select(cols)
        .with_columns(
            [
                pl.col("symbol").cast(pl.Utf8),
                pl.col("date").cast(pl.Date, strict=False),
                *[pl.col(col).cast(pl.Float64, strict=False) for col in cols[2:]],
            ]
        )
        .drop_nulls(["date"])
    )


def _read_partitions(
    data_dir: Path, table: str, symbols: tuple[str, ...], start: date, end: date
) -> pl.DataFrame:
    root = data_dir / table
    files = [
        p / "part.parquet"
        for p in root.glob("date=*")
        if p.is_dir()
        and start <= date.fromisoformat(p.name[5:]) <= end
        and (p / "part.parquet").exists()
    ]
    if not files:
        return pl.DataFrame()
    return pl.read_parquet([str(p) for p in files]).filter(pl.col("symbol").is_in(symbols))


def _assert_overlap_matches(existing: pl.DataFrame, incoming: pl.DataFrame) -> None:
    if existing.is_empty() or incoming.is_empty():
        return
    keys = ["symbol", "date"]
    overlap = incoming.join(existing.select(keys), on=keys, how="inner")
    if overlap.is_empty():
        return
    left = overlap.sort(keys).select([*keys, "open", "high", "low", "close", "volume", "amount"])
    right = (
        _canonical_columns(existing)
        .join(overlap.select(keys), on=keys, how="inner")
        .sort(keys)
        .select(left.columns)
    )
    if not left.equals(right):
        raise RuntimeError("provider overlap differs from canonical data; refusing to overwrite")


def _backup_existing_partitions(data_dir: Path, backup_root: Path) -> Path:
    """Copy only currently-existing partitions that this pilot can modify."""
    for table in (
        "kline_daily",
        "kline_index_daily",
        "kline_daily_enriched",
        "kline_index_enriched",
    ):
        source = data_dir / table
        if not source.exists():
            continue
        for part in source.glob("date=*"):
            try:
                value = date.fromisoformat(part.name[5:])
            except ValueError:
                continue
            if WARMUP_START <= value <= PILOT_END:
                target = backup_root / table / part.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(part, target, dirs_exist_ok=True)
    for name in (".matrix_generation_stock.json", ".matrix_generation_index.json"):
        source = data_dir / name
        if source.exists():
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup_root / name)
    return backup_root


def restore_backup(data_dir: Path, backup_root: Path) -> None:
    """Operator rollback helper; restore backups and remove pilot-only prepend rows."""
    for table in (
        "kline_daily",
        "kline_index_daily",
        "kline_daily_enriched",
        "kline_index_enriched",
    ):
        source = backup_root / table
        if not source.exists():
            continue
        for part in source.glob("date=*"):
            target = data_dir / table / part.name
            shutil.copytree(part, target, dirs_exist_ok=True)
    for name in (".matrix_generation_stock.json", ".matrix_generation_index.json"):
        source = backup_root / name
        if source.exists():
            shutil.copy2(source, data_dir / name)
    # The snapshot only contains pre-existing partitions.  Historical dates
    # created by this pilot have no snapshot counterpart, so remove just pilot
    # symbols there and retain any unrelated rows a concurrent operator added.
    for table, symbols in (
        ("kline_daily", PILOT_STOCKS),
        ("kline_index_daily", PILOT_INDEXES),
        ("kline_daily_enriched", PILOT_STOCKS),
        ("kline_index_enriched", PILOT_INDEXES),
    ):
        root = data_dir / table
        for part in root.glob("date=*") if root.exists() else ():
            try:
                value = date.fromisoformat(part.name[5:])
            except ValueError:
                continue
            if value >= date(2025, 9, 15) or (backup_root / table / part.name).exists():
                continue
            parquet = part / "part.parquet"
            if not parquet.exists():
                continue
            retained = pl.read_parquet(parquet).filter(~pl.col("symbol").is_in(symbols))
            if retained.is_empty():
                shutil.rmtree(part)
            else:
                _atomic_parquet(retained.sort(["symbol", "date"]), parquet)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def _latest_pilot_backup(data_dir: Path) -> Path | None:
    root = data_dir / ".pilot_backups"
    candidates = sorted(root.glob("research_daily_v1_pilot_*") if root.exists() else ())
    return candidates[-1] if candidates else None


def _preservation_summary(
    data_dir: Path, backup_root: Path | None
) -> dict[str, dict[str, int | bool]]:
    """Compare logical partition content, not Parquet bytes/mtimes after publication."""
    if backup_root is None:
        return {}
    result: dict[str, dict[str, int | bool]] = {}
    for table in (
        "kline_daily",
        "kline_index_daily",
        "kline_daily_enriched",
        "kline_index_enriched",
    ):
        backup_table = backup_root / table
        checked = mismatches = 0
        for backup_part in backup_table.glob("date=*") if backup_table.exists() else ():
            current = data_dir / table / backup_part.name / "part.parquet"
            if not current.exists():
                mismatches += 1
                continue
            before = pl.read_parquet(backup_part / "part.parquet").sort(["symbol", "date"])
            after = pl.read_parquet(current).sort(["symbol", "date"])
            checked += 1
            if not before.equals(after):
                mismatches += 1
        result[table] = {
            "checked_partitions": checked,
            "mismatched_partitions": mismatches,
            "preserved": mismatches == 0,
        }
    return result


def _listing_dates(data_dir: Path) -> dict[str, str | None]:
    path = data_dir / "instruments" / "instruments.parquet"
    if not path.exists():
        return {}
    frame = pl.read_parquet(path)
    column = next((c for c in ("listing_date", "list_date") if c in frame.columns), None)
    if column is None:
        return {}
    return {row["symbol"]: row[column] for row in frame.select(["symbol", column]).to_dicts()}


def _expected_empty(
    symbols: tuple[str, ...], listing_dates: dict[str, str | None]
) -> tuple[str, ...]:
    result = []
    for symbol in symbols:
        value = listing_dates.get(symbol)
        if (
            value
            and is_valid_listing_date(str(value))
            and date.fromisoformat(str(value)) > PILOT_END
        ):
            result.append(symbol)
    return tuple(result)


def run_pilot(
    repo: KlineRepository,
    provider: DailyProvider,
    *,
    manifest_path: Path,
    report_path: Path,
    backup_root: Path | None = None,
    force: bool = False,
) -> PilotResult:
    """Run or resume the bounded pilot.  Call only after mocked tests pass."""
    data_dir = repo.store.data_dir
    manifest = _load_manifest(manifest_path) or new_manifest(provider)
    if manifest.provider != provider.name or manifest.dataset_version != DATASET_VERSION:
        raise ValueError("manifest does not belong to this provider/pilot dataset")
    _save_manifest(manifest_path, manifest)
    listings = _listing_dates(data_dir)
    backup_path: Path | None = None
    stock_written = index_written = skipped = 0
    calendar: list[date] = []
    for batch in manifest.batches:
        if not batch.can_resume(force=force):
            skipped += 1
            continue
        running = batch.begin_attempt(force=force)
        manifest = _replace_and_save(manifest_path, manifest, running)
        expected_empty = (
            _expected_empty(batch.symbols, listings) if batch.asset_type == "stock" else ()
        )
        try:
            raw = provider.get_daily(
                list(batch.symbols),
                datetime.combine(WARMUP_START, datetime.min.time()),
                datetime.combine(PILOT_END, datetime.min.time()),
                batch.asset_type,
            )
            frame = _canonical_columns(raw)
            returned = (
                tuple(sorted(set(frame.get_column("symbol").to_list())))
                if not frame.is_empty()
                else ()
            )
            missing = unexpected_missing_symbols(batch.symbols, returned, expected_empty)
            expected_dates = (
                calendar
                if batch.asset_type == "stock"
                else sorted(set(frame.get_column("date").to_list()))
            )
            validation = validate_daily_coverage(
                frame,
                requested_start=WARMUP_START,
                requested_end=PILOT_END,
                expected_trading_dates=expected_dates,
            )
            if validation.errors or validation.duplicate_count:
                raise ValueError(
                    "raw validation failed: "
                    + ",".join([*validation.errors, f"duplicates:{validation.duplicate_count}"])
                )
            if batch.asset_type == "index":
                calendar = sorted(
                    set(frame.filter(pl.col("symbol") == "000001.SH")["date"].to_list())
                )
            table = "kline_daily" if batch.asset_type == "stock" else "kline_index_daily"
            existing = _read_partitions(data_dir, table, batch.symbols, WARMUP_START, PILOT_END)
            _assert_overlap_matches(existing, frame)
            historical = frame.filter(pl.col("date") < date(2025, 9, 15))
            if not historical.is_empty() and backup_path is None and backup_root is not None:
                backup_path = _backup_existing_partitions(data_dir, backup_root)
            if batch.asset_type == "stock":
                repo.append_daily(historical)
                stock_written += historical.height
            else:
                repo.append_index_daily(historical)
                index_written += historical.height
            status = BackfillBatchStatus.PARTIAL if missing else BackfillBatchStatus.COMPLETED
            finished = running.finish(
                status,
                rows=frame.height,
                actual_first_date=validation.first_date,
                actual_last_date=validation.last_date,
                checksum=_logical_fingerprint(frame),
                checksum_kind="normalized_logical_rows",
                reason="missing_symbols:" + ",".join(missing) if missing else None,
            )
            manifest = _replace_and_save(manifest_path, manifest, finished)
        except Exception as exc:
            manifest = _replace_and_save(
                manifest_path, manifest, running.finish(BackfillBatchStatus.FAILED, error=str(exc))
            )
            raise
    # Completed-only resumes must be read-only.  On a first/partial run, the
    # stock pipeline calculates only pilot symbols, while its existing local
    # merge preserves all non-pilot rows in shared date partitions.
    if backup_path is None:
        backup_path = _latest_pilot_backup(data_dir)
    # A batch checkpoint means its raw write completed, not that downstream
    # enriched/report publication completed.  A missing report therefore
    # makes a completed-manifest resume repair the bounded derived stage.
    needs_derived_rebuild = bool(stock_written or index_written or not report_path.exists())
    enriched_written = (
        run_pipeline(data_dir=data_dir, symbols=list(PILOT_STOCKS)) if needs_derived_rebuild else 0
    )
    # Index storage has no equivalent selective pipeline entrypoint.  Its
    # three-symbol raw window is bounded, so compute then merge only these
    # symbols through the normal index-enriched repository contract.
    index_raw = _read_partitions(
        data_dir, "kline_index_daily", PILOT_INDEXES, WARMUP_START, PILOT_END
    )
    if needs_derived_rebuild and not index_raw.is_empty():
        repo.append_index_enriched(compute_enriched(index_raw, factors=None, instruments=None))
    repo.refresh_index_views()
    report = _build_report(
        repo, manifest, stock_written, index_written, enriched_written, backup_path
    )
    _atomic_json(report_path, report)
    return PilotResult(
        manifest_path,
        report_path,
        backup_path,
        stock_written,
        index_written,
        enriched_written,
        skipped,
    )


def _build_report(
    repo: KlineRepository,
    manifest: HistoricalBackfillManifest,
    stock_rows: int,
    index_rows: int,
    enriched_rows: int,
    backup_path: Path | None,
) -> dict:
    data_dir = repo.store.data_dir
    summary: dict[str, dict] = {}
    listing_dates = _listing_dates(data_dir)
    for asset_type, table, symbols in (
        ("stock", "kline_daily", PILOT_STOCKS),
        ("index", "kline_index_daily", PILOT_INDEXES),
    ):
        frame = _read_partitions(data_dir, table, tuple(symbols), WARMUP_START, PILOT_END)
        summary[asset_type] = {
            "symbols": list(symbols),
            "rows": frame.height,
            "historical_rows_before_existing_range": frame.filter(
                pl.col("date") < date(2025, 9, 15)
            ).height,
            "first_date": str(frame["date"].min()) if not frame.is_empty() else None,
            "last_date": str(frame["date"].max()) if not frame.is_empty() else None,
            "duplicate_count": int(
                frame.group_by(["symbol", "date"]).len().filter(pl.col("len") > 1).height
            )
            if not frame.is_empty()
            else 0,
            "fingerprint": _logical_fingerprint(frame),
            "per_symbol": {
                symbol: {
                    "rows": symbol_frame.height,
                    "first_date": str(symbol_frame["date"].min())
                    if not symbol_frame.is_empty()
                    else None,
                    "last_date": str(symbol_frame["date"].max())
                    if not symbol_frame.is_empty()
                    else None,
                    "warmup_valid_bars_before_research": symbol_frame.filter(
                        pl.col("date") < RESEARCH_START
                    ).height,
                    "listing_date": listing_dates.get(symbol) if asset_type == "stock" else None,
                    "listing_date_valid": is_valid_listing_date(str(listing_dates.get(symbol)))
                    if asset_type == "stock" and listing_dates.get(symbol) is not None
                    else None,
                }
                for symbol in symbols
                for symbol_frame in [frame.filter(pl.col("symbol") == symbol)]
            },
        }
    limitations = [
        "current-universe LEVEL 1 only",
        "historical ST unavailable",
        "raw unadjusted prices are not strategy-research safe",
    ]
    insufficient = [
        symbol
        for symbol, details in summary["stock"]["per_symbol"].items()
        if details["warmup_valid_bars_before_research"] < 300
    ]
    if insufficient:
        limitations.append("fewer_than_300_pre_research_bars:" + ",".join(insufficient))
    index_calendar = (
        _read_partitions(data_dir, "kline_index_daily", ("000001.SH",), WARMUP_START, PILOT_END)
        .get_column("date")
        .to_list()
    )
    validation = {
        "stock": _coverage_to_dict(
            validate_daily_coverage(
                _read_partitions(data_dir, "kline_daily", PILOT_STOCKS, WARMUP_START, PILOT_END),
                requested_start=WARMUP_START,
                requested_end=PILOT_END,
                expected_trading_dates=index_calendar,
            )
        ),
        "index": _coverage_to_dict(
            validate_daily_coverage(
                _read_partitions(
                    data_dir, "kline_index_daily", PILOT_INDEXES, WARMUP_START, PILOT_END
                ),
                requested_start=WARMUP_START,
                requested_end=PILOT_END,
                expected_trading_dates=index_calendar,
            )
        ),
    }
    return {
        "dataset_version": DATASET_VERSION,
        "scope": "PILOT_ONLY",
        "survivorship_level": "LEVEL 1",
        "adjustment_policy": "RAW_UNADJUSTED",
        "research_start": RESEARCH_START.isoformat(),
        "warmup_start": WARMUP_START.isoformat(),
        "end": PILOT_END.isoformat(),
        "manifest": manifest.to_dict(),
        "stock_rows_written": stock_rows,
        "index_rows_written": index_rows,
        "provider_returned_rows": {
            "stock": sum(batch.rows for batch in manifest.batches if batch.asset_type == "stock"),
            "index": sum(batch.rows for batch in manifest.batches if batch.asset_type == "index"),
        },
        "enriched_rows_written": enriched_rows,
        "backup_path": str(backup_path) if backup_path else None,
        "existing_range_preservation": _preservation_summary(data_dir, backup_path),
        "summary": summary,
        "validation": validation,
        "limitations": limitations,
    }


def _coverage_to_dict(result: object) -> dict:
    return dict(vars(result))
