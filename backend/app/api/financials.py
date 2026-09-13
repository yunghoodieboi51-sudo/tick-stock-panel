"""财务数据 API — 独立路由, Cap.FINANCIAL 门控。"""

from __future__ import annotations

import logging

import polars as pl
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import ai_reports
from app.services.financial_analyzer import analyze_financials_stream
from app.services.financial_sync import (
    FINANCIAL_TABLES,
    get_financial_df,
    get_financial_provider_name,
    get_supported_financial_tables,
    is_financial_table_supported,
)
from app.tickflow.capabilities import Cap

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/financials", tags=["financials"])


def _financial_allowed(capset) -> bool:
    """是否有财务数据访问权限 (TickFlow FINANCIAL 套餐 或 custom 财务源)。"""
    if capset.has(Cap.FINANCIAL):
        return True
    from app.services.financial_sync import _financial_is_custom

    return _financial_is_custom()


def _require_financial(capset) -> None:
    """_require_financial(capset) 的 custom 感知版本。"""
    if not _financial_allowed(capset):
        from app.tickflow.capabilities import CapabilityDenied

        raise CapabilityDenied(Cap.FINANCIAL)


@router.get("/status")
def financial_status(request: Request):
    """返回各财务表的同步状态。无需 FINANCIAL 权限(前端根据 available 决定是否展示)。"""
    capset = request.app.state.capabilities
    if not _financial_allowed(capset):
        return {"available": False, "tables": {}}

    data_dir = request.app.state.repo.store.data_dir
    provider = get_financial_provider_name()
    supported_tables = set(get_supported_financial_tables())
    tables = {}

    for table in FINANCIAL_TABLES:
        path = data_dir / "financials" / table / "part.parquet"
        if table not in supported_tables:
            tables[table] = {
                "rows": 0,
                "symbols": 0,
                "supported": False,
                "available": False,
                "provider": provider,
                "reason": "unsupported_by_provider",
                "retained_local_data": path.exists(),
            }
            continue
        if path.exists():
            try:
                df = pl.read_parquet(path, columns=["symbol"])
                tables[table] = {
                    "rows": len(df),
                    "symbols": df["symbol"].n_unique() if not df.is_empty() else 0,
                    "supported": True,
                    "available": not df.is_empty(),
                    "provider": provider,
                }
            except Exception:
                tables[table] = {
                    "rows": 0,
                    "symbols": 0,
                    "supported": True,
                    "available": False,
                    "provider": provider,
                }
        else:
            tables[table] = {
                "rows": 0,
                "symbols": 0,
                "supported": True,
                "available": False,
                "provider": provider,
            }

    fs = getattr(request.app.state, "financial_scheduler", None)
    last_sync = {
        table: value
        for table, value in (getattr(fs, "last_sync", {}) if fs else {}).items()
        if table in supported_tables
    }
    sync_results = {
        table: value
        for table, value in (getattr(fs, "last_result", {}) if fs else {}).items()
        if table in supported_tables and value.get("provider") == provider
    }

    return {
        "available": True,
        "provider": provider,
        "supported_tables": [table for table in FINANCIAL_TABLES if table in supported_tables],
        "tables": tables,
        "last_sync": last_sync,
        "sync_results": sync_results,
        # 服务端是否正在同步(手动触发)——前端据此显示"同步中"并防重复点击,
        # 且刷新页面后仍能正确反映服务端状态。
        "syncing": bool(fs and fs.is_syncing),
    }


def _table_payload(request: Request, table: str, symbol: str | None) -> dict:
    provider = get_financial_provider_name()
    if not is_financial_table_supported(table):
        return {
            "data": [],
            "supported": False,
            "available": False,
            "provider": provider,
            "reason": "unsupported_by_provider",
        }
    df = get_financial_df(request.app.state.repo.store.data_dir, table)
    if symbol and not df.is_empty():
        df = df.filter(pl.col("symbol") == symbol)
    return {
        "data": [] if df.is_empty() else df.to_dicts(),
        "supported": True,
        "available": not df.is_empty(),
        "provider": provider,
    }


@router.get("/metrics")
def get_metrics(request: Request, symbol: str | None = None):
    """查询核心财务指标。"""
    capset = request.app.state.capabilities
    _require_financial(capset)

    return _table_payload(request, "metrics", symbol)


@router.get("/income")
def get_income(request: Request, symbol: str | None = None):
    """查询利润表。"""
    capset = request.app.state.capabilities
    _require_financial(capset)

    return _table_payload(request, "income", symbol)


@router.get("/balance-sheet")
def get_balance_sheet(request: Request, symbol: str | None = None):
    """查询资产负债表。"""
    capset = request.app.state.capabilities
    _require_financial(capset)

    return _table_payload(request, "balance_sheet", symbol)


@router.get("/cash-flow")
def get_cash_flow(request: Request, symbol: str | None = None):
    """查询现金流量表。"""
    capset = request.app.state.capabilities
    _require_financial(capset)

    return _table_payload(request, "cash_flow", symbol)


@router.get("/shares")
def get_shares(request: Request, symbol: str | None = None):
    """查询历史股本表。"""
    capset = request.app.state.capabilities
    _require_financial(capset)

    return _table_payload(request, "shares", symbol)


@router.post("/sync/{table}")
def sync_table(request: Request, table: str):
    """手动触发同步(立即返回,后台异步执行)。

    table: metrics / income / balance_sheet / cash_flow / shares / all
    同步在后台线程执行,全量同步需数分钟。本接口立即返回 started 状态,
    前端通过轮询 GET /status 的 syncing 字段观察进度。
    """
    capset = request.app.state.capabilities
    _require_financial(capset)

    valid_tables = {*FINANCIAL_TABLES, "all"}
    if table not in valid_tables:
        raise HTTPException(400, f"invalid table: {table}, expected one of {valid_tables}")

    fs = getattr(request.app.state, "financial_scheduler", None)
    if not fs:
        return {"status": "error", "message": "FinancialScheduler not available"}

    target = None if table == "all" else table
    result = fs.trigger(target)

    return {"status": "ok", "synced": result}


class AnalyzeRequest(BaseModel):
    """AI 财务分析请求。"""

    symbol: str
    focus: str = ""  # 可选:用户追加的分析关注点


@router.post("/analyze")
async def analyze_financials(request: Request, req: AnalyzeRequest):
    """AI 财务分析 — SSE 流式返回。

    后端读取该标的财务报表与股本表 → 注入 CFA 分析师级提示词 → 流式调用 LLM →
    逐 chunk 以 SSE 形式推给前端(JSON per line, 非 text/event-stream,
    以便前端用 ReadableStream 逐行解析,更简单可靠)。
    """
    capset = request.app.state.capabilities
    _require_financial(capset)

    if not req.symbol:
        raise HTTPException(400, "symbol 不能为空")

    data_dir = request.app.state.repo.store.data_dir

    async def stream_gen():
        async for chunk in analyze_financials_stream(data_dir, req.symbol, req.focus):
            yield chunk + "\n"

    return StreamingResponse(
        stream_gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ================================================================
# AI 报告 CRUD(历史报告持久化)
# ================================================================


class SaveReportRequest(BaseModel):
    """保存一条 AI 财务分析报告。"""

    symbol: str
    name: str = ""
    focus: str = ""
    content: str
    periods: int | None = None
    summary: str = ""


@router.get("/reports")
def list_reports(request: Request):
    """获取全部历史报告(按时间降序,后端已裁剪到上限)。无需 FINANCIAL 能力读取列表元信息。"""
    capset = request.app.state.capabilities
    if not _financial_allowed(capset):
        return {"reports": []}
    return {"reports": ai_reports.list_reports()}


@router.post("/reports")
def save_report(request: Request, req: SaveReportRequest):
    """保存一条报告。"""
    capset = request.app.state.capabilities
    _require_financial(capset)
    report = ai_reports.save_report(
        {
            "symbol": req.symbol,
            "name": req.name,
            "focus": req.focus,
            "content": req.content,
            "periods": req.periods,
            "summary": req.summary,
        }
    )
    return {"ok": True, "report": report}


@router.delete("/reports/{report_id}")
def delete_report(request: Request, report_id: str):
    """删除一条报告。"""
    capset = request.app.state.capabilities
    _require_financial(capset)
    ok = ai_reports.delete_report(report_id)
    return {"ok": ok}
