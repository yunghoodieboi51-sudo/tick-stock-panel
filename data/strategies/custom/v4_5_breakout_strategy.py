"""V4.5c offline BREAKOUT setup/trigger ablation strategy.

This is intentionally independent of V4.1.  It retains V4.1's default score
and exits so the three entry gates below can be ablated without changing the
matcher, risk controls, or candidate-ordering convention.
"""

# ruff: noqa: RUF001

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
    safe_divide,
    valid_rolling_max,
    valid_rolling_mean,
    valid_rolling_min,
    valid_shift,
)

_TREND_SUBWEIGHTS = (0.40, 0.35, 0.25)
_MOMENTUM_SUBWEIGHTS = (0.45, 0.35, 0.20)
_BREAKOUT_BASE_SCORES = (70.0, 65.0, 70.0)
_PATTERN_ID = "BREAKOUT_SETUP::PIVOT_BREAK_CONFIRM"

META = {
    "id": "v4_5_breakout_strategy",
    "name": "V4.5 BREAKOUT Setup/Trigger（实验）",
    "description": "仅供离线消融：前一日 setup + 当日 pivot 确认突破",
    "tags": ["V4.5", "实验", "BREAKOUT"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "enabled": True,
        "price_min": 3.0,
        "price_max": 300.0,
        "market_cap_min": None,
        "amount_min": 30_000_000.0,
        "exclude_st": True,
        "exclude_new_days": None,
    },
    "params": [
        {"id": "use_contraction", "label": "实验：启用收缩", "type": "bool", "default": True},
        {"id": "use_extension", "label": "实验：启用突破延伸", "type": "bool", "default": True},
    ],
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_v4_5_pivot_break_confirm"]
EXIT_SIGNALS = [
    "signal_v4_ma20_breakdown",
    "signal_v4_ma20_weakening",
    "signal_v4_trend_structure_broken",
]
STOP_LOSS = -0.08
TRAILING_STOP = -0.10
MAX_HOLD_DAYS = 40


def _rising(values: np.ndarray, floor: float, target: float) -> np.ndarray:
    return np.clip((values - floor) / max(target - floor, 1e-6), 0.0, 1.0)


def _falling(values: np.ndarray, full: float, zero: float) -> np.ndarray:
    return np.clip((zero - values) / max(zero - full, 1e-6), 0.0, 1.0)


class V45BreakoutMatrixStrategy:
    entry_pattern_ids = (_PATTERN_ID,)

    def required_fields(self) -> frozenset[str]:
        return frozenset({"open", "high", "low", "close", "volume"})

    def required_warmup_bars(self, _params: dict) -> int:
        # 60-day momentum plus a previous 20-day ATR% mean ending at T-1.
        return 70

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close, low, volume = market.close, market.low, market.volume
        valid_close = np.isfinite(close) & (close > 0)
        valid_volume = np.isfinite(volume) & (volume > 0)
        tradable = np.asarray(market.tradable, dtype=bool)
        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        mom20 = matrix_feature(market, "momentum_20d")
        mom60 = matrix_feature(market, "momentum_60d")
        rsi = matrix_feature(market, "rsi_14")
        atr_pct = matrix_feature(market, "atr_pct")
        vol_ratio = matrix_feature(market, "vol_ratio_5d")
        daily_return = matrix_feature(market, "change_pct")

        ma20_prev = valid_shift(ma20, 5, np.isfinite(ma20))
        ma60_prev = valid_shift(ma60, 10, np.isfinite(ma60))
        ma20_slope = safe_divide(ma20, ma20_prev) - 1.0
        ma60_slope = safe_divide(ma60, ma60_prev) - 1.0
        ma_spread = safe_divide(ma20, ma60) - 1.0
        ma20_bias = safe_divide(close, ma20) - 1.0
        trend = (
            valid_close
            & np.isfinite(ma20)
            & np.isfinite(ma60)
            & (close > ma20)
            & (ma20 > ma60)
            & (ma20_slope >= 0.0)
            & (ma60_slope >= -0.005)
        )
        strength = (
            np.isfinite(mom20)
            & np.isfinite(mom60)
            & np.isfinite(rsi)
            & np.isfinite(atr_pct)
            & np.isfinite(ma20_bias)
            & (mom20 > 0.0)
            & (mom60 >= -0.03)
            & (rsi <= 82.0)
            & (ma20_bias <= 0.12)
            & (atr_pct <= 0.06)
        )

        stage_high = valid_rolling_max(close, valid_close, 20)
        pivot = valid_shift(stage_high, 1, np.isfinite(stage_high))
        price_breakout = np.isfinite(pivot) & (close > pivot)
        atr = atr_pct * close
        extension_atr = safe_divide(close - pivot, atr)

        # All setup inputs are shifted first: no T value enters setup[T].
        prior_close = valid_shift(close, 1, valid_close)
        prior_atr_pct = valid_shift(atr_pct, 1, np.isfinite(atr_pct))
        prior_atr = prior_atr_pct * prior_close
        prior_atr_mean = valid_rolling_mean(prior_atr_pct, np.isfinite(prior_atr_pct), 20)
        contraction = safe_divide(prior_atr_pct, prior_atr_mean)
        prior_trend = valid_shift(trend.astype(np.float32), 1, valid_close) > 0.5
        setup_distance = safe_divide(np.abs(prior_close - pivot), prior_atr)
        setup = prior_trend & np.isfinite(setup_distance) & (setup_distance <= 1.0)

        use_contraction = bool(params.get("use_contraction", True))
        use_extension = bool(params.get("use_extension", True))
        contraction_gate = np.isfinite(contraction) & (contraction <= 0.9)
        extension_gate = np.isfinite(extension_atr) & (extension_atr <= 1.0)

        # Preserve the V4.1 default score formula, including its diagnostic
        # raw pullback/consolidation terms, so only entry eligibility changes.
        touched = (
            valid_close
            & np.isfinite(ma20)
            & np.isfinite(low)
            & (low <= ma20 * 1.02)
            & (close >= ma20 * 0.98)
        )
        touches = np.where(touched, 1.0, 0.0).astype(np.float32)
        touches[~valid_close] = np.nan
        recent_touch = (
            valid_shift(
                valid_rolling_max(touches, np.isfinite(touches), 5), 1, np.isfinite(touches)
            )
            > 0.5
        )
        pullback = recent_touch & (close > ma20) & (daily_return >= 0.0)
        range_high = valid_rolling_max(close, valid_close, 15)
        range_low = valid_rolling_min(close, valid_close, 15)
        prior_range_high = valid_shift(range_high, 1, np.isfinite(range_high))
        prior_range_low = valid_shift(range_low, 1, np.isfinite(range_low))
        range_width = safe_divide(prior_range_high - prior_range_low, prior_range_low)
        consolidation = (
            np.isfinite(range_width)
            & (range_width <= 0.10)
            & (close > prior_range_high)
            & (vol_ratio >= 1.20)
        )
        trend_score = 100.0 * (
            _TREND_SUBWEIGHTS[0] * _rising(ma20_slope, 0.0, 0.025)
            + _TREND_SUBWEIGHTS[1] * _rising(ma60_slope, -0.005, 0.02)
            + _TREND_SUBWEIGHTS[2] * _rising(ma_spread, 0.0, 0.08)
        )
        momentum_score = 100.0 * (
            _MOMENTUM_SUBWEIGHTS[0] * _rising(mom20, 0.0, 0.12)
            + _MOMENTUM_SUBWEIGHTS[1] * _rising(mom60, -0.03, 0.25)
            + _MOMENTUM_SUBWEIGHTS[2] * _falling(rsi, 72.0, 82.0)
        )
        breakout_return = safe_divide(close, pivot) - 1.0
        price_score = np.where(
            price_breakout,
            _BREAKOUT_BASE_SCORES[0] + 30.0 * _rising(breakout_return, 0.0, 0.04),
            0.0,
        )
        pullback_score = np.where(
            pullback, _BREAKOUT_BASE_SCORES[1] + 35.0 * _rising(daily_return, 0.0, 0.03), 0.0
        )
        consolidation_score = np.where(
            consolidation,
            _BREAKOUT_BASE_SCORES[2]
            + 15.0 * _falling(range_width, 0.0, 0.10)
            + 15.0 * _rising(vol_ratio, 1.20, 1.80),
            0.0,
        )
        breakout_score = np.maximum.reduce([price_score, pullback_score, consolidation_score])
        volume_score = np.where(
            vol_ratio <= 1.80,
            100.0 * np.clip(vol_ratio / 1.80, 0.0, 1.0),
            100.0 * _falling(vol_ratio, 1.80, 3.50),
        )
        volatility_score = 100.0 * _falling(atr_pct, 0.02, 0.06)
        total_score = (
            0.30 * trend_score
            + 0.25 * momentum_score
            + 0.20 * breakout_score
            + 0.15 * volume_score
            + 0.10 * volatility_score
        )
        total_score = np.clip(np.nan_to_num(total_score, nan=0.0), 0.0, 100.0).astype(np.float32)

        entry = (
            tradable
            & valid_close
            & valid_volume
            & np.isfinite(vol_ratio)
            & (vol_ratio > 0)
            & trend
            & strength
            & setup
            & price_breakout
            & (daily_return <= 0.085)
            & (vol_ratio <= 5.0)
            & (total_score >= 58.0)
        )
        if use_contraction:
            entry &= contraction_gate
        if use_extension:
            entry &= extension_gate

        exit_ma20_prev = valid_shift(ma20, 5, np.isfinite(ma20))
        exit_slope = safe_divide(ma20, exit_ma20_prev) - 1.0
        previous_close = valid_shift(close, 1, valid_close)
        previous_ma20 = valid_shift(ma20, 1, np.isfinite(ma20))
        ma20_breakdown = (
            valid_close
            & np.isfinite(previous_close)
            & np.isfinite(previous_ma20)
            & (close < ma20)
            & (previous_close >= previous_ma20)
        )
        ma20_weakening = np.isfinite(exit_slope) & (exit_slope <= -0.01)
        structure_broken = valid_close & np.isfinite(ma60) & (close < ma60)
        exit_ = ma20_breakdown | ma20_weakening | structure_broken
        entry_code = np.where(entry, 0, -1).astype(np.int16)
        exit_code = np.where(
            ma20_breakdown, 0, np.where(ma20_weakening, 1, np.where(structure_broken, 2, -1))
        ).astype(np.int16)
        exit_code[~exit_] = -1
        pattern = np.where(entry, 1, 0).astype(np.uint8)
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            score=total_score,
            entry_signal_code=entry_code,
            exit_signal_code=exit_code,
            entry_signal_ids=tuple(ENTRY_SIGNALS),
            exit_signal_ids=tuple(EXIT_SIGNALS),
            entry_pattern_mask=pattern,
            entry_pattern_ids=(_PATTERN_ID,),
        )


MATRIX_STRATEGY = V45BreakoutMatrixStrategy()
