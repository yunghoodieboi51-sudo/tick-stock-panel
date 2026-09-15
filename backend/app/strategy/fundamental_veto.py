"""Shared point-in-time fundamental veto policy for A-share strategies.

Financial values must already be attached through ``app.backtest.fundamentals``.
That module is the single source of truth for announcement-date availability;
this module only evaluates the resulting point-in-time matrices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from app.backtest.matrix import MarketDataMatrix, SignalMatrix, make_signal_matrix

V4_TREND_STRATEGY_ID = "v4_trend_strategy"
V4_5_BREAKOUT_STRATEGY_ID = "v4_5_breakout_strategy"

CORE_FUNDAMENTAL_FIELDS = (
    "roe_latest",
    "revenue_yoy_latest",
    "net_income_yoy_latest",
    "net_margin_latest",
)

NEGATIVE_ROE = "NEGATIVE_ROE"
SEVERE_PROFIT_DECLINE = "SEVERE_PROFIT_DECLINE"
SEVERE_REVENUE_DECLINE = "SEVERE_REVENUE_DECLINE"
NEGATIVE_NET_MARGIN = "NEGATIVE_NET_MARGIN"
HIGH_DEBT = "HIGH_DEBT"
ABNORMAL_GROSS_MARGIN = "ABNORMAL_GROSS_MARGIN"
MISSING_FINANCIAL_DATA = "MISSING_FINANCIAL_DATA"

_REASON_BITS = MappingProxyType(
    {
        NEGATIVE_ROE: np.uint16(1 << 0),
        SEVERE_PROFIT_DECLINE: np.uint16(1 << 1),
        SEVERE_REVENUE_DECLINE: np.uint16(1 << 2),
        NEGATIVE_NET_MARGIN: np.uint16(1 << 3),
        HIGH_DEBT: np.uint16(1 << 4),
        ABNORMAL_GROSS_MARGIN: np.uint16(1 << 5),
        MISSING_FINANCIAL_DATA: np.uint16(1 << 6),
    }
)
FUNDAMENTAL_VETO_REASON_CODES = tuple(_REASON_BITS)


@dataclass(frozen=True)
class FundamentalVetoConfig:
    """Effective policy values, using the provider's percentage-point units."""

    enabled: bool = True
    roe_min: float = 0.0
    net_income_yoy_min: float = -50.0
    revenue_yoy_min: float = -30.0
    net_margin_min: float = -5.0
    debt_ratio_enabled: bool = False
    debt_ratio_max: float = 85.0
    gross_margin_enabled: bool = True
    gross_margin_min: float = -20.0
    missing_data_policy: str = "reject"
    minimum_available_fields: int = 4


@dataclass(frozen=True)
class FundamentalVetoResult:
    """Compact matrix result; reasons are decoded only for displayed candidates."""

    veto: np.ndarray
    reason_mask: np.ndarray

    def reason_codes_at(self, time_id: int, asset_id: int) -> list[str]:
        mask = int(self.reason_mask[time_id, asset_id])
        return [code for code, bit in _REASON_BITS.items() if mask & int(bit)]

    def reason_code_counts(self, eligible: np.ndarray | None = None) -> dict[str, int]:
        if eligible is None:
            eligible_mask = np.ones(self.veto.shape, dtype=bool)
        else:
            eligible_mask = np.asarray(eligible, dtype=bool)
            if eligible_mask.shape != self.veto.shape:
                raise ValueError("fundamental veto eligibility shape does not match result")
        return {
            code: int(np.count_nonzero(eligible_mask & ((self.reason_mask & bit) != 0)))
            for code, bit in _REASON_BITS.items()
        }


_DEFAULT_POLICIES = MappingProxyType(
    {
        V4_TREND_STRATEGY_ID: FundamentalVetoConfig(),
        # V4.5c is a separate research strategy, but it is loaded through the
        # normal custom-strategy path.  Give it the same PIT fail-closed policy
        # so an accidental normal Screener/backtest invocation cannot bypass
        # the shared V4.2 financial safety layer.
        V4_5_BREAKOUT_STRATEGY_ID: FundamentalVetoConfig(),
    }
)


def resolve_fundamental_veto_config(
    strategy_id: str,
    overrides: dict | None = None,
) -> FundamentalVetoConfig | None:
    """Resolve a strategy policy and its optional ``fundamental_veto`` override."""
    default = _DEFAULT_POLICIES.get(str(strategy_id))
    if default is None:
        return None
    raw = (overrides or {}).get("fundamental_veto")
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("fundamental_veto override must be an object")

    values = asdict(default)
    unknown = set(raw) - set(values)
    if unknown:
        raise ValueError(f"unknown fundamental_veto options: {sorted(unknown)}")
    values.update(raw)
    config = FundamentalVetoConfig(**values)
    _validate_config(config)
    return config


def fundamental_veto_public_config(
    strategy_id: str,
    overrides: dict | None = None,
) -> dict[str, Any] | None:
    config = resolve_fundamental_veto_config(strategy_id, overrides)
    return asdict(config) if config is not None else None


def fundamental_veto_required_fields(
    strategy_id: str,
    overrides: dict | None = None,
) -> frozenset[str]:
    config = resolve_fundamental_veto_config(strategy_id, overrides)
    if config is None or not config.enabled:
        return frozenset()
    return _required_fields_for_config(config)


def _required_fields_for_config(config: FundamentalVetoConfig) -> frozenset[str]:
    fields = set(CORE_FUNDAMENTAL_FIELDS)
    if config.debt_ratio_enabled:
        fields.add("debt_ratio_latest")
    if config.gross_margin_enabled:
        fields.add("gross_margin_latest")
    return frozenset(fields)


def evaluate_fundamental_veto(
    market: MarketDataMatrix,
    config: FundamentalVetoConfig,
) -> FundamentalVetoResult:
    """Evaluate all veto rules against already point-in-time-aligned fields."""
    _validate_config(config)
    shape = market.shape
    reason_mask = np.zeros(shape, dtype=np.uint16)
    if not config.enabled:
        veto = np.zeros(shape, dtype=bool)
        veto.flags.writeable = False
        reason_mask.flags.writeable = False
        return FundamentalVetoResult(veto=veto, reason_mask=reason_mask)

    fields = {name: _field_or_nan(market, name) for name in _required_fields_for_config(config)}

    _add_reason(reason_mask, fields["roe_latest"] < config.roe_min, NEGATIVE_ROE)
    _add_reason(
        reason_mask,
        fields["net_income_yoy_latest"] < config.net_income_yoy_min,
        SEVERE_PROFIT_DECLINE,
    )
    _add_reason(
        reason_mask,
        fields["revenue_yoy_latest"] < config.revenue_yoy_min,
        SEVERE_REVENUE_DECLINE,
    )
    _add_reason(
        reason_mask,
        fields["net_margin_latest"] < config.net_margin_min,
        NEGATIVE_NET_MARGIN,
    )
    if config.debt_ratio_enabled:
        _add_reason(
            reason_mask,
            fields["debt_ratio_latest"] > config.debt_ratio_max,
            HIGH_DEBT,
        )
    if config.gross_margin_enabled:
        _add_reason(
            reason_mask,
            fields["gross_margin_latest"] < config.gross_margin_min,
            ABNORMAL_GROSS_MARGIN,
        )

    if config.missing_data_policy == "reject":
        available = np.zeros(shape, dtype=np.uint8)
        for name in CORE_FUNDAMENTAL_FIELDS:
            available += np.isfinite(fields[name]).astype(np.uint8)
        _add_reason(
            reason_mask,
            available < config.minimum_available_fields,
            MISSING_FINANCIAL_DATA,
        )

    veto = reason_mask != 0
    veto.flags.writeable = False
    reason_mask.flags.writeable = False
    return FundamentalVetoResult(veto=veto, reason_mask=reason_mask)


def apply_fundamental_veto(
    signals: SignalMatrix,
    result: FundamentalVetoResult,
) -> SignalMatrix:
    """Remove vetoed entries while preserving exits and technical scores."""
    if result.veto.shape != signals.shape or result.reason_mask.shape != signals.shape:
        raise ValueError("fundamental veto shape does not match SignalMatrix")
    entry = (signals.entry.astype(bool) & ~result.veto).astype(np.uint8)
    entry_codes = np.where(entry != 0, signals.entry_signal_code, -1).astype(np.int16)
    return make_signal_matrix(
        signals.shape,
        entry=entry,
        exit=signals.exit,
        score=signals.score,
        entry_signal_code=entry_codes,
        exit_signal_code=signals.exit_signal_code,
        entry_signal_ids=signals.entry_signal_ids,
        exit_signal_ids=signals.exit_signal_ids,
        entry_pattern_mask=(
            np.where(entry != 0, signals.entry_pattern_mask, 0).astype(np.uint8)
            if signals.entry_pattern_mask is not None
            else None
        ),
        entry_pattern_ids=signals.entry_pattern_ids,
    )


def _field_or_nan(market: MarketDataMatrix, name: str) -> np.ndarray:
    values = market.fields.get(name)
    if values is None:
        return np.full(market.shape, np.nan, dtype=np.float32)
    array = np.asarray(values)
    if array.shape != market.shape:
        raise ValueError(f"fundamental field {name} shape does not match market")
    return array


def _add_reason(reason_mask: np.ndarray, condition: np.ndarray, code: str) -> None:
    reason_mask[np.asarray(condition, dtype=bool)] |= _REASON_BITS[code]


def _validate_config(config: FundamentalVetoConfig) -> None:
    for name in ("enabled", "debt_ratio_enabled", "gross_margin_enabled"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be boolean")
    if config.missing_data_policy not in {"allow", "reject"}:
        raise ValueError("missing_data_policy must be 'allow' or 'reject'")
    if isinstance(config.minimum_available_fields, bool) or not isinstance(
        config.minimum_available_fields,
        int,
    ):
        raise ValueError("minimum_available_fields must be an integer")
    if not 0 <= config.minimum_available_fields <= len(CORE_FUNDAMENTAL_FIELDS):
        raise ValueError(
            f"minimum_available_fields must be within [0, {len(CORE_FUNDAMENTAL_FIELDS)}]"
        )
    numeric_names = (
        "roe_min",
        "net_income_yoy_min",
        "revenue_yoy_min",
        "net_margin_min",
        "debt_ratio_max",
        "gross_margin_min",
    )
    for name in numeric_names:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
