"""V4.8 persisted daily-scan API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.daily_scan import DailyScanJobManager, DailyScanRunStore, DailyScanService

router = APIRouter(prefix="/api/daily-scan", tags=["daily-scan"])


class ScanRequest(BaseModel):
    account_size: float = Field(0, ge=0, le=1_000_000_000)
    max_position_pct: float = Field(0, ge=0, le=1)
    max_candidates: int = Field(10, ge=1, le=100)
    risk_per_trade_pct: float = Field(0, ge=0, le=1)


def _manager(request: Request) -> DailyScanJobManager:
    manager = getattr(request.app.state, "daily_scan_manager", None)
    if manager is None:
        repo = request.app.state.repo
        store = DailyScanRunStore(repo.store.data_dir)
        manager = DailyScanJobManager(
            DailyScanService(repo, request.app.state.strategy_engine, store)
        )
        request.app.state.daily_scan_manager = manager
    return manager


@router.get("/status")
def status(request: Request) -> dict[str, Any]:
    return _manager(request).status()


@router.post("/runs")
def start_run(payload: ScanRequest, request: Request) -> dict[str, Any]:
    return _manager(request).start(payload.model_dump())


@router.get("/runs")
def list_runs(request: Request, limit: int = 30) -> dict[str, Any]:
    return {"items": _manager(request).service.store.list(limit=limit)}


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    try:
        found = _manager(request).service.store.result(run_id)
    except ValueError:
        found = None
    if found is None:
        raise HTTPException(status_code=404, detail="daily scan run not found")
    return found


@router.get("/runs/{run_id}/candidates/{symbol}")
def get_candidate(run_id: str, symbol: str, request: Request) -> dict[str, Any]:
    try:
        found = _manager(request).service.store.result(run_id)
    except ValueError:
        found = None
    if not found or not found.get("result"):
        raise HTTPException(status_code=404, detail="daily scan run not found")
    for section in ("candidates", "rejected"):
        for candidate in found["result"].get(section, []):
            if candidate.get("symbol") == symbol:
                return candidate
    raise HTTPException(status_code=404, detail="daily scan candidate not found")
