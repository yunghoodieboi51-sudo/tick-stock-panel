"""东方财富免费 A 股财务主要指标 provider。

数据来自东方财富 F10 ``RPT_F10_FINANCE_MAINFINADATA``。该端点同时提供报告期、
公告日和数据更新时间。为避免把后来修订的当前快照倒灌到原公告日, 本插件只接收
``UPDATE_DATE <= NOTICE_DATE`` 的版本; 无法还原修订历史的行严格丢弃。

百分比字段沿用上游百分数值口径(16.75 表示 16.75%), 与项目现有财务因子契约一致。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from app.plugins.eastmoney_financial.client import (
    EastmoneyFinancialClient,
    EastmoneyFinancialError,
)

logger = logging.getLogger(__name__)

_DATASETS = ("financial",)
_SYMBOL_BATCH_SIZE = 50
_BATCH_INTERVAL_SECONDS = 0.15
_HISTORY_YEARS = 10
_LATEST_LOOKBACK_YEARS = 2
_SYMBOL_RE = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")

_METRICS_FIELD_MAP = {
    "BPS": "bps",
    "ROEJQ": "roe",
    "XSMLL": "gross_margin",
    "XSJLL": "net_margin",
    "TOTALOPERATEREVETZ": "revenue_yoy",
    "PARENTNETPROFITTZ": "net_income_yoy",
    "ZCFZL": "debt_to_asset_ratio",
}
_CANONICAL_COLUMNS = [
    "symbol",
    "period_end",
    "announce_date",
    *_METRICS_FIELD_MAP.values(),
]
_METRICS_SCHEMA = {
    "symbol": pl.String,
    "period_end": pl.String,
    "announce_date": pl.String,
    **{column: pl.Float64 for column in _METRICS_FIELD_MAP.values()},
}


def availability() -> tuple[bool, str]:
    """零依赖、免 Key; 启动时不做网络请求。"""
    return True, "ok"


def to_eastmoney_symbol(symbol: str) -> str | None:
    """把项目 A 股代码校验并转换为东财 SECUCODE。

    东财本接口使用的格式与项目 canonical 格式一致。显式校验股票号段, 避免 ETF、
    指数、基金或其他交易所资产误入财务同步。
    """
    value = str(symbol or "").strip().upper()
    matched = _SYMBOL_RE.fullmatch(value)
    if not matched:
        return None
    code, exchange = matched.groups()
    if exchange == "SH" and not code.startswith(("60", "68")):
        return None
    if exchange == "SZ" and not code.startswith(("00", "30")):
        return None
    if exchange == "BJ" and not code.startswith(("4", "8", "92")):
        return None
    return f"{code}.{exchange}"


def _iso_date(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _report_start(years: int) -> str:
    today = date.today()
    return date(today.year - years, 1, 1).isoformat()


def _normalize_metrics_row_with_reason(
    raw: dict,
    requested_symbols: set[str] | None = None,
) -> tuple[dict | None, str | None]:
    symbol = to_eastmoney_symbol(raw.get("SECUCODE"))
    if not symbol:
        return None, "invalid_symbol"
    if requested_symbols is not None and symbol not in requested_symbols:
        return None, "unexpected_symbol"
    period_end = _iso_date(raw.get("REPORT_DATE"))
    announce_date = _iso_date(raw.get("NOTICE_DATE"))
    update_date = _iso_date(raw.get("UPDATE_DATE"))
    if not period_end or not announce_date or not update_date:
        return None, "missing_or_invalid_date"
    if announce_date < period_end:
        return None, "missing_or_invalid_date"
    # 当前端点只给每报告期的最新快照, 没有修订版本历史。若更新时间晚于原公告日,
    # 无法证明当前数值在原公告日已存在, 故整行拒收而不是伪造修订公告日。
    # UPDATE_DATE == NOTICE_DATE 当前视为同次披露而接受; 因上游没有更细粒度版本链,
    # 这是一项保守假设, 不是对历史修订版本完整性的证明。
    if update_date > announce_date:
        return None, "revision"
    row = {
        "symbol": symbol,
        "period_end": period_end,
        "announce_date": announce_date,
    }
    for source, target in _METRICS_FIELD_MAP.items():
        row[target] = _to_float(raw.get(source))
    return row, None


def _normalize_metrics_row(raw: dict) -> dict | None:
    row, _reason = _normalize_metrics_row_with_reason(raw)
    return row


@dataclass
class _EastmoneyFinancialConfig:
    name: str = "eastmoney_financial"
    display_name: str = "东方财富财务(免费)"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


class EastmoneyFinancialProvider:
    """只实现 P0 metrics 的免费原生财务 provider。"""

    name = "eastmoney_financial"
    builtin = True
    financial_tables = frozenset({"metrics"})

    def __init__(self, client: EastmoneyFinancialClient | None = None) -> None:
        self.config = _EastmoneyFinancialConfig()
        self._client = client or EastmoneyFinancialClient()
        self._last_stats: dict = {}

    def close(self) -> None:
        self._client.close()

    @property
    def last_stats(self) -> dict:
        return dict(self._last_stats)

    def _record_stats(self, stats: dict) -> None:
        self._last_stats = dict(stats)
        message = json.dumps(stats, ensure_ascii=False, sort_keys=True)
        if stats["status"] == "success" and not stats["dropped_rows"]:
            logger.info("eastmoney_financial_metrics_stats=%s", message)
        else:
            logger.warning("eastmoney_financial_metrics_stats=%s", message)

    def get_financials(
        self,
        table: str,
        symbols: list[str],
        latest_only: bool = True,
    ) -> pl.DataFrame:
        if table != "metrics":
            logger.info("东方财富免费财务源暂不支持 %s, 跳过", table)
            return pl.DataFrame()

        converted_symbols = [to_eastmoney_symbol(symbol) for symbol in symbols]
        normalized = [symbol for symbol in converted_symbols if symbol]
        invalid_input_symbols = len(symbols) - len(normalized)
        duplicate_input_symbols = len(normalized) - len(set(normalized))
        normalized = list(dict.fromkeys(normalized))
        report_start = _report_start(_LATEST_LOOKBACK_YEARS if latest_only else _HISTORY_YEARS)
        stats = {
            "provider": self.name,
            "table": table,
            "status": "success",
            "latest_only": latest_only,
            "report_start": report_start,
            "requested_symbols": len(symbols),
            "valid_symbols": len(normalized),
            "successful_symbols": 0,
            "invalid_input_symbols": invalid_input_symbols,
            "duplicate_input_symbols": duplicate_input_symbols,
            "total_batches": 0,
            "failed_batches": 0,
            "failed_symbols": 0,
            "upstream_rows": 0,
            "received_rows": 0,
            "safe_rows": 0,
            "accepted_rows": 0,
            "output_rows": 0,
            "dropped_rows": 0,
            "dropped_revision": 0,
            "dropped_missing_or_invalid_date": 0,
            "dropped_invalid_symbol": 0,
            "dropped_unexpected_symbol": 0,
            "pit_acceptance_rate": None,
            "pit_drop_rate": None,
            "symbol_coverage_rate": None,
            "symbols_with_safe_rows": 0,
            "oldest_safe_period": None,
            "newest_safe_period": None,
        }
        if not normalized:
            self._record_stats(stats)
            return pl.DataFrame(schema=_METRICS_SCHEMA)

        rows: list[dict] = []
        failed_batches = 0
        total_batches = (len(normalized) + _SYMBOL_BATCH_SIZE - 1) // _SYMBOL_BATCH_SIZE
        stats["total_batches"] = total_batches
        for offset in range(0, len(normalized), _SYMBOL_BATCH_SIZE):
            if offset:
                time.sleep(_BATCH_INTERVAL_SECONDS)
            batch = normalized[offset : offset + _SYMBOL_BATCH_SIZE]
            batch_number = offset // _SYMBOL_BATCH_SIZE + 1
            try:
                raw_rows = self._client.fetch_metrics(batch, report_start=report_start)
            except EastmoneyFinancialError as exc:
                failed_batches += 1
                stats["failed_batches"] = failed_batches
                stats["failed_symbols"] += len(batch)
                logger.warning(
                    "东方财富 metrics 批次 %d/%d 失败(%d 只): %s",
                    batch_number,
                    total_batches,
                    len(batch),
                    exc,
                )
                continue
            requested = set(batch)
            stats["upstream_rows"] += len(raw_rows)
            for raw in raw_rows:
                row, reason = _normalize_metrics_row_with_reason(raw, requested)
                if row is None:
                    stats[f"dropped_{reason}"] += 1
                    continue
                rows.append(row)

        stats["successful_symbols"] = len(normalized) - stats["failed_symbols"]
        stats["received_rows"] = stats["upstream_rows"]
        stats["accepted_rows"] = len(rows)
        if failed_batches == total_batches:
            stats["status"] = "failed"
            self._record_stats(stats)
            raise EastmoneyFinancialError(f"全部 {total_batches} 个 metrics 批次请求失败")
        stats["status"] = "partial" if failed_batches else "success"
        stats["safe_rows"] = len(rows)
        stats["dropped_rows"] = sum(
            stats[f"dropped_{reason}"]
            for reason in (
                "revision",
                "missing_or_invalid_date",
                "invalid_symbol",
                "unexpected_symbol",
            )
        )
        if stats["upstream_rows"]:
            stats["pit_acceptance_rate"] = round(len(rows) / stats["upstream_rows"], 6)
            stats["pit_drop_rate"] = round(stats["dropped_rows"] / stats["upstream_rows"], 6)
        symbols_with_rows = {row["symbol"] for row in rows}
        stats["symbols_with_safe_rows"] = len(symbols_with_rows)
        stats["symbol_coverage_rate"] = round(len(symbols_with_rows) / len(normalized), 6)
        if rows:
            periods = [row["period_end"] for row in rows]
            stats["oldest_safe_period"] = min(periods)
            stats["newest_safe_period"] = max(periods)
        if not rows:
            self._record_stats(stats)
            return pl.DataFrame(schema=_METRICS_SCHEMA)
        frame = pl.DataFrame(rows, schema=_METRICS_SCHEMA, strict=False).select(_CANONICAL_COLUMNS)
        frame = frame.unique(subset=["symbol", "period_end"], keep="last")
        if latest_only:
            frame = (
                frame.sort(["symbol", "period_end"]).group_by("symbol", maintain_order=True).tail(1)
            )
        frame = frame.sort(["symbol", "period_end"])
        stats["output_rows"] = frame.height
        self._record_stats(stats)
        return frame

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        if dataset != "financial":
            return {
                "provider": self.name,
                "dataset": dataset,
                "rows": 0,
                "error": f"东方财富免费财务源未接入 {dataset} 数据集",
            }
        sample = (symbols or ["600519.SH"])[:3]
        try:
            frame = self.get_financials("metrics", sample, latest_only=True)
        except EastmoneyFinancialError as exc:
            return {
                "provider": self.name,
                "dataset": dataset,
                "rows": 0,
                "error": str(exc),
            }
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": frame.height,
            "columns": frame.columns,
            "preview": frame.head(5).to_dicts(),
            "stats": self.last_stats,
        }
