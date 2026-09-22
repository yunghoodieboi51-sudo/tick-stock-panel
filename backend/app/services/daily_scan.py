"""V4.8 daily-scan product orchestration.

This module deliberately consumes the established strategy/context/veto path;
it does not implement a second screener or signal engine.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.fs_utils import atomic_write_text
from app.services.screener import ScreenerService
from app.strategy import config as strategy_config
from app.strategy.fundamental_veto import fundamental_veto_public_config

STRATEGY_ID = "v4_trend_strategy"
RETENTION = 60
RESEARCH_STATUS = "RESEARCH_UNVALIDATED"
_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


class DailyScanRunStore:
    def __init__(self, data_dir: Path, retention: int = RETENTION) -> None:
        self.root = data_dir / "user_data" / "daily_scan_runs"
        self.retention = retention
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, run_id: str) -> Path:
        if not run_id.isalnum() or len(run_id) > 64:
            raise ValueError("invalid daily scan run id")
        path = (self.root / run_id).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("daily scan run path escapes root")
        return path

    def create(self) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        manifest = {
            "run_id": run_id,
            "state": "RUNNING",
            "started_at": _now(),
            "completed_at": None,
            "error": None,
            "stage": "validating_data",
            "research_status": RESEARCH_STATUS,
        }
        with _LOCK:
            run_dir = self._dir(run_id)
            run_dir.mkdir()
            atomic_write_text(
                run_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, allow_nan=False)
            )
        return manifest

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with _LOCK:
            manifest = self.get(run_id)
            if manifest is None:
                raise KeyError(run_id)
            manifest.update(_clean(changes))
            atomic_write_text(
                self._dir(run_id) / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, allow_nan=False),
            )
            return manifest

    def complete(self, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
        with _LOCK:
            atomic_write_text(
                self._dir(run_id) / "result.json",
                json.dumps(_clean(result), ensure_ascii=False, allow_nan=False),
            )
            manifest = self.update(
                run_id,
                state="COMPLETED",
                stage="persisting_run",
                completed_at=_now(),
                error=None,
                candidate_count=len(result.get("candidates", [])),
                rejected_count=len(result.get("rejected", [])),
            )
            self._retain()
            return manifest

    def fail(self, run_id: str, error: str) -> dict[str, Any]:
        return self.update(run_id, state="FAILED", completed_at=_now(), error=str(error)[:1000])

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self._dir(run_id) / "manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def result(self, run_id: str) -> dict[str, Any] | None:
        manifest = self.get(run_id)
        if manifest is None:
            return None
        path = self._dir(run_id) / "result.json"
        return {
            "manifest": manifest,
            "result": json.loads(path.read_text(encoding="utf-8")) if path.exists() else None,
        }

    def list(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("*/manifest.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        # Retention needs to inspect more than the public retention window.
        # Keep a bounded filesystem scan, while callers still choose their own
        # presentation limit.
        return sorted(rows, key=lambda row: row.get("started_at") or "", reverse=True)[
            : max(1, min(limit, 10_000))
        ]

    def _retain(self) -> None:
        finished = [row for row in self.list(limit=10_000) if row.get("state") != "RUNNING"]
        for row in finished[self.retention :]:
            run_dir = self._dir(row["run_id"])
            for path in run_dir.glob("*"):
                path.unlink(missing_ok=True)
            run_dir.rmdir()


class DailyScanService:
    def __init__(self, repo: Any, strategy_engine: Any, store: DailyScanRunStore) -> None:
        self.repo, self.strategy_engine, self.store = repo, strategy_engine, store

    def run(self, run_id: str, sizing: dict[str, float] | None = None) -> dict[str, Any]:
        sizing = sizing or {}
        self.store.update(run_id, stage="preparing_market_context")
        svc = ScreenerService(self.repo, asset_type="stock")
        as_of = svc.latest_date()
        if as_of is None:
            raise ValueError("no enriched daily data available; update data first")
        data_dir = self.repo.store.data_dir
        overrides = strategy_config.load_override(data_dir, STRATEGY_ID)
        strategy = self.strategy_engine.get(STRATEGY_ID)
        saved_params = dict(overrides.get("params") or {})
        params = self.strategy_engine.resolve_params(strategy, saved_params, overrides)
        context = svc.build_strategy_context(
            self.strategy_engine,
            as_of,
            [STRATEGY_ID],
            params_map={STRATEGY_ID: params},
            overrides_map={STRATEGY_ID: overrides},
        )
        self.store.update(run_id, stage="running_v4_signal")
        outcome = self.strategy_engine.run(
            STRATEGY_ID, context, params=params, overrides=overrides or None
        )
        self.store.update(run_id, stage="applying_fundamental_veto")
        max_candidates = min(100, max(1, int(sizing.get("max_candidates") or 10)))
        candidates = [
            self._candidate(run_id, i + 1, row, strategy, params, sizing)
            for i, row in enumerate(self._rank(outcome.rows)[:max_candidates])
        ]
        rejected = [
            self._candidate(run_id, None, row, strategy, params, sizing)
            for row in self._rank(outcome.vetoed_rows)
        ]
        for item in rejected:
            item["fundamental"]["status"] = "VETO"
            item["plan"]["fundamental_status"] = "VETO"
            item["plan"]["fundamental_reason_codes"] = item["fundamental"]["reason_codes"]
            item["plan"]["invalidation_conditions"].append("FUNDAMENTAL_VETO")
        self._allocate(candidates, sizing)
        self.store.update(run_id, stage="building_plans")
        return {
            "trade_date": str(as_of),
            "strategy_id": STRATEGY_ID,
            "strategy_version": strategy.meta.get("version", "unknown"),
            "strategy_config": {"params": params, "overrides": overrides},
            "fundamental_veto_policy": fundamental_veto_public_config(STRATEGY_ID, overrides),
            "research_status": RESEARCH_STATUS,
            "survivorship_level": "LEVEL_1",
            "corporate_action_quality": "RAW_ONLY",
            "dataset_generation": None,
            "research_validation_status": RESEARCH_STATUS,
            "candidates": candidates,
            "rejected": rejected,
            "warnings": ["V4 Trend Strategy is research/unvalidated."],
        }

    @staticmethod
    def _rank(rows: list[dict]) -> list[dict]:
        def key(row: dict) -> tuple:
            amount = row.get("amount")
            amount_key = (
                -float(amount)
                if isinstance(amount, (int, float)) and math.isfinite(amount)
                else math.inf
            )
            return (-float(row.get("score") or 0), amount_key, str(row.get("symbol") or ""))

        return sorted(rows, key=key)

    def _candidate(
        self,
        run_id: str,
        rank: int | None,
        row: dict,
        strategy: Any,
        params: dict[str, Any],
        sizing: dict[str, float],
    ) -> dict[str, Any]:
        raw = row.get("raw_close")
        raw = (
            float(raw) if isinstance(raw, (int, float)) and raw > 0 and math.isfinite(raw) else None
        )
        atr = row.get("atr_pct")
        atr = (
            float(atr)
            if isinstance(atr, (int, float)) and math.isfinite(atr) and atr >= 0
            else None
        )
        max_gain = float(params.get("max_daily_gain", 0.085))
        stop_pct = abs(float(strategy.stop_loss or 0.0))
        high = raw * (1 + min(atr, max_gain)) if raw is not None and atr is not None else None
        plan = self._plan(raw, high, stop_pct, strategy, sizing)
        reasons = list(row.get("fundamental_veto_reason_codes") or [])
        breakdown = row.get("score_breakdown") or {}
        pattern = row.get("primary_entry_pattern") or "UNKNOWN"
        why_selected = [
            f"Entry pattern: {pattern}",
            f"Score {float(row.get('score') or 0):.2f}; trend {float(breakdown.get('trend') or 0):.2f}, momentum {float(breakdown.get('momentum') or 0):.2f}",
            f"Breakout {float(breakdown.get('breakout') or 0):.2f}, volume {float(breakdown.get('volume') or 0):.2f}, volatility {float(breakdown.get('volatility') or 0):.2f}",
            "Shared PIT fundamental veto: PASS"
            if not row.get("fundamental_veto")
            else "Shared PIT fundamental veto: VETO",
        ]
        status = "VETO" if row.get("fundamental_veto") else "PASS"
        plan["fundamental_status"] = status
        plan["fundamental_reason_codes"] = reasons
        return _clean(
            {
                "run_id": run_id,
                "rank": rank,
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "trade_date": row.get("date"),
                "final_score": row.get("score"),
                "score_breakdown": breakdown,
                "entry_pattern_primary": row.get("primary_entry_pattern"),
                "entry_patterns_matched": row.get("matched_entry_patterns") or [],
                "signal_reference_price": row.get("close"),
                "signal_price_basis": "FORWARD_ADJUSTED",
                "execution_reference_price": raw,
                "execution_price_basis": "RAW" if raw is not None else None,
                "latest_raw_price": raw,
                "latest_raw_price_basis": "RAW_DATA_AS_OF" if raw is not None else None,
                "fundamental": {"status": status, "reason_codes": reasons},
                "why_selected": why_selected,
                "warnings": []
                if raw is not None
                else ["RAW execution-price mapping unavailable; plan prices are omitted."],
                "research_status": RESEARCH_STATUS,
                "plan": plan,
            }
        )

    @staticmethod
    def _plan(
        raw: float | None,
        high: float | None,
        stop_pct: float,
        strategy: Any,
        sizing: dict[str, float],
    ) -> dict[str, Any]:
        account = float(sizing.get("account_size") or 0)
        max_pct = min(max(float(sizing.get("max_position_pct") or 0), 0), 1)
        risk_pct = min(max(float(sizing.get("risk_per_trade_pct") or 0), 0), 1)
        target = (
            min(account * max_pct, account * risk_pct / stop_pct)
            if account > 0 and stop_pct > 0
            else None
        )
        shares = math.floor(target / high / 100) * 100 if target and high else None
        warnings = []
        if target and high and shares == 0:
            warnings.append("TARGET_CAPITAL_BELOW_A_SHARE_LOT")
        return {
            "buy_zone_low": raw,
            "buy_zone_high": high,
            "chase_limit_price": high,
            "stop_loss_price": raw * (1 - stop_pct) if raw is not None else None,
            "stop_loss_pct": stop_pct or None,
            "trailing_stop_pct": abs(float(strategy.trailing_stop or 0)) or None,
            "max_hold_days": strategy.max_hold_days,
            "target_capital": target,
            "estimated_shares": shares,
            "estimated_order_value": shares * high if shares and high else None,
            "lot_size": 100,
            "lot_rule_status": "ASSUMED_A_SHARE_STOCK",
            "plan_price_status": "RAW_REFERENCE_AVAILABLE"
            if raw is not None
            else "REFERENCE_ONLY_UNADJUSTED_MAPPING_UNAVAILABLE",
            "initial_position_pct": (target / account)
            if target is not None and account > 0
            else None,
            "warnings": warnings,
            "invalidation_conditions": [
                "CHASE_LIMIT_EXCEEDED",
                "REFERENCE_STOP_BREACHED",
                "TREND_EXIT_SIGNAL",
                "MAX_HOLD_DAYS_REACHED",
                "DATA_PRICE_BASIS_WARNING",
            ],
        }

    @staticmethod
    def _allocate(candidates: list[dict[str, Any]], sizing: dict[str, float]) -> None:
        """Apply the account-wide cap after deterministic score ranking."""
        remaining = max(0.0, float(sizing.get("account_size") or 0))
        for candidate in candidates:
            plan = candidate["plan"]
            target = plan.get("target_capital")
            high = plan.get("buy_zone_high")
            if target is None or high is None:
                continue
            target = min(float(target), remaining)
            shares = math.floor(target / float(high) / 100) * 100
            order_value = shares * float(high)
            plan["target_capital"] = target
            plan["estimated_shares"] = shares
            plan["estimated_order_value"] = order_value if shares else 0.0
            if (
                target > 0
                and shares == 0
                and "TARGET_CAPITAL_BELOW_A_SHARE_LOT" not in plan["warnings"]
            ):
                plan["warnings"].append("TARGET_CAPITAL_BELOW_A_SHARE_LOT")
            remaining = max(0.0, remaining - order_value)


class DailyScanJobManager:
    def __init__(self, service: DailyScanService) -> None:
        self.service, self._active_id = service, None
        self._lock = threading.Lock()

    def start(self, sizing: dict[str, float] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._active_id:
                active = self.service.store.get(self._active_id)
                if active and active.get("state") == "RUNNING":
                    return {**active, "reused": True}
            manifest = self.service.store.create()
            self._active_id = manifest["run_id"]

        def work() -> None:
            try:
                self.service.store.complete(
                    manifest["run_id"], self.service.run(manifest["run_id"], sizing)
                )
            except Exception as exc:
                self.service.store.fail(manifest["run_id"], str(exc))
            finally:
                with self._lock:
                    if self._active_id == manifest["run_id"]:
                        self._active_id = None

        threading.Thread(target=work, name="daily-v4-scan", daemon=True).start()
        return {**manifest, "reused": False}

    def status(self) -> dict[str, Any]:
        with self._lock:
            active_id = self._active_id
        latest = self.service.store.list(limit=1)
        return {
            "state": "RUNNING" if active_id else "IDLE",
            "active_run_id": active_id,
            "latest": latest[0] if latest else None,
        }
