"""V4.1 A股通用改良版趋势交易策略。

本阶段只使用价量、动量和波动率数据。次新股过滤不依赖当前尚未
生效的 exclude_new_days，留待 V4.2 或独立核心修复阶段处理。
"""

# ruff: noqa: RUF001, RUF002, RUF003

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
    safe_divide,
    valid_rolling_max,
    valid_rolling_min,
    valid_shift,
)

# 分项内部权重集中在此，避免散落在公式中。五个主分项权重由策略参数提供。
_TREND_SUBWEIGHTS = (0.40, 0.35, 0.25)
_MOMENTUM_SUBWEIGHTS = (0.45, 0.35, 0.20)
_BREAKOUT_BASE_SCORES = (70.0, 65.0, 70.0)


META = {
    "id": "v4_trend_strategy",
    "name": "V4 A股通用改良趋势",
    "description": "中期多头趋势 + 启动确认 + 价量动量波动率综合评分",
    "tags": ["V4", "A股", "趋势", "突破", "回踩"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "basic_filter": {
        "enabled": True,
        "price_min": 3.0,
        "price_max": 300.0,
        "market_cap_min": None,
        "amount_min": 30_000_000.0,
        "exclude_st": True,
        # 当前核心过滤链不会执行该字段；显式中和，避免产生已过滤次新股的错觉。
        "exclude_new_days": None,
    },
    "params": [
        {"id": "ma20_slope_lookback", "label": "MA20斜率回看", "type": "int", "default": 5, "min": 2, "max": 20, "step": 1},
        {"id": "ma20_slope_min", "label": "MA20最低斜率", "type": "float", "default": 0.0, "min": -0.03, "max": 0.10, "step": 0.001},
        {"id": "ma20_slope_target", "label": "MA20斜率满分值", "type": "float", "default": 0.025, "min": 0.005, "max": 0.15, "step": 0.005},
        {"id": "ma60_slope_lookback", "label": "MA60斜率回看", "type": "int", "default": 10, "min": 3, "max": 30, "step": 1},
        {"id": "ma60_slope_min", "label": "MA60最低斜率", "type": "float", "default": -0.005, "min": -0.05, "max": 0.05, "step": 0.001},
        {"id": "ma60_slope_target", "label": "MA60斜率满分值", "type": "float", "default": 0.02, "min": 0.0, "max": 0.10, "step": 0.005},
        {"id": "ma_spread_target", "label": "MA20/MA60乖离满分值", "type": "float", "default": 0.08, "min": 0.02, "max": 0.30, "step": 0.01},
        {"id": "momentum_20_min", "label": "20日最低动量", "type": "float", "default": 0.0, "min": 0.0, "max": 0.30, "step": 0.01},
        {"id": "momentum_20_target", "label": "20日动量满分值", "type": "float", "default": 0.12, "min": 0.03, "max": 0.50, "step": 0.01},
        {"id": "momentum_60_min", "label": "60日最低动量", "type": "float", "default": -0.03, "min": -0.20, "max": 0.20, "step": 0.01},
        {"id": "momentum_60_target", "label": "60日动量满分值", "type": "float", "default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05},
        {"id": "rsi_soft_max", "label": "RSI柔性降分起点", "type": "float", "default": 72.0, "min": 55.0, "max": 85.0, "step": 1.0},
        {"id": "rsi_max", "label": "RSI上限", "type": "float", "default": 82.0, "min": 65.0, "max": 95.0, "step": 1.0},
        {"id": "max_ma20_bias", "label": "MA20最大正偏离", "type": "float", "default": 0.12, "min": 0.03, "max": 0.30, "step": 0.01},
        {"id": "atr_ideal_pct", "label": "ATR理想波动率", "type": "float", "default": 0.02, "min": 0.005, "max": 0.05, "step": 0.005},
        {"id": "max_atr_pct", "label": "ATR最大波动率", "type": "float", "default": 0.06, "min": 0.02, "max": 0.15, "step": 0.005},
        {"id": "breakout_lookback", "label": "阶段高点回看", "type": "int", "default": 20, "min": 10, "max": 60, "step": 1},
        {"id": "breakout_buffer", "label": "高点突破缓冲", "type": "float", "default": 0.0, "min": 0.0, "max": 0.05, "step": 0.0025},
        {"id": "breakout_strength_target", "label": "突破强度满分值", "type": "float", "default": 0.04, "min": 0.01, "max": 0.15, "step": 0.005},
        {"id": "pullback_lookback", "label": "MA20回踩回看", "type": "int", "default": 5, "min": 2, "max": 15, "step": 1},
        {"id": "pullback_tolerance", "label": "MA20回踩容差", "type": "float", "default": 0.02, "min": 0.005, "max": 0.08, "step": 0.005},
        {"id": "restart_min_return", "label": "回踩转强最低涨幅", "type": "float", "default": 0.0, "min": -0.02, "max": 0.05, "step": 0.005},
        {"id": "restart_target_return", "label": "回踩转强满分涨幅", "type": "float", "default": 0.03, "min": 0.01, "max": 0.10, "step": 0.005},
        {"id": "consolidation_lookback", "label": "整理区间回看", "type": "int", "default": 15, "min": 5, "max": 40, "step": 1},
        {"id": "consolidation_range_max", "label": "整理区间最大振幅", "type": "float", "default": 0.10, "min": 0.03, "max": 0.30, "step": 0.01},
        {"id": "breakout_volume_min", "label": "整理突破最低量比", "type": "float", "default": 1.20, "min": 0.80, "max": 3.0, "step": 0.10},
        {"id": "volume_ideal_ratio", "label": "理想量比", "type": "float", "default": 1.80, "min": 1.0, "max": 4.0, "step": 0.10},
        {"id": "volume_extreme_ratio", "label": "极端量比降分点", "type": "float", "default": 3.50, "min": 2.0, "max": 8.0, "step": 0.25},
        {"id": "volume_hard_max", "label": "量比硬上限", "type": "float", "default": 5.0, "min": 2.5, "max": 12.0, "step": 0.5},
        {"id": "max_daily_gain", "label": "单日最大涨幅", "type": "float", "default": 0.085, "min": 0.03, "max": 0.20, "step": 0.005},
        {"id": "trend_weight", "label": "趋势评分权重", "type": "float", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05},
        {"id": "momentum_weight", "label": "动量评分权重", "type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},
        {"id": "breakout_weight", "label": "启动评分权重", "type": "float", "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05},
        {"id": "volume_weight", "label": "成交量评分权重", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05},
        {"id": "volatility_weight", "label": "波动率评分权重", "type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05},
        {"id": "min_total_score", "label": "最低综合分", "type": "float", "default": 58.0, "min": 0.0, "max": 100.0, "step": 1.0},
        {"id": "exit_slope_lookback", "label": "退出斜率回看", "type": "int", "default": 5, "min": 2, "max": 20, "step": 1},
        {"id": "exit_ma20_slope_max", "label": "MA20转弱阈值", "type": "float", "default": -0.01, "min": -0.10, "max": 0.0, "step": 0.005},
    ],
    # 策略自身返回综合 score；留空可防止框架另外重算一套截面评分。
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = [
    "signal_v4_price_breakout",
    "signal_v4_pullback_restart",
    "signal_v4_consolidation_breakout",
]
ENTRY_PATTERN_IDS = (
    "BREAKOUT",
    "PULLBACK_RESTART",
    "CONSOLIDATION_BREAKOUT",
)
# This is deliberately not a public META parameter.  It is a research-only
# input used by the V4.4.2 experiment runner to isolate existing entry shapes;
# normal screener/backtest calls never provide it and therefore retain V4.1.
_EXPERIMENT_ENABLED_ENTRY_PATTERNS = "__experiment_enabled_entry_patterns"
_EXPERIMENT_CONTEXT_PARAM = "__v4_4_2_experiment_context"
_EXPERIMENT_CONTEXT_VALUE = 0x44
EXIT_SIGNALS = [
    "signal_v4_ma20_breakdown",
    "signal_v4_ma20_weakening",
    "signal_v4_trend_structure_broken",
]

# 成交和风控由 BacktestEngine 执行，策略只声明默认值，可由现有 override 调整。
STOP_LOSS = -0.08
TRAILING_STOP = -0.10
MAX_HOLD_DAYS = 40

RULES = """
1. 仅支持 A 股日线；基础层排除 ST/*ST/退市整理、无效成交和停牌行。
2. 收盘价高于 MA20、MA20 高于 MA60，且 MA20/MA60 斜率不低于参数阈值。
3. 20 日动量为正、60 日动量不明显为负，并限制 RSI、MA20 偏离和 ATR 波动率。
4. 阶段新高、MA20 回踩转强、放量突破整理区三种确认任一成立即可。
5. 成交量是确认和评分项，不是唯一买入理由；极端量比降分或过滤。
6. 总分由趋势、动量、启动、成交量和波动率五项加权，达到最低分才发生买入信号。
7. 退出信号为跌破 MA20、MA20 明显转弱或跌破 MA60；止损/移动止损/最长持有交给回测引擎。
8. 所有突破基准和近期事件都先向后移一根有效 K 线，不读取未来数据。
9. 次新股过滤待 V4.2/独立修复阶段处理，本版不依赖 exclude_new_days。
"""


def _float_param(params: dict, name: str, default: float) -> float:
    return float(params.get(name, default))


def _int_param(params: dict, name: str, default: int) -> int:
    return max(1, int(params.get(name, default)))


def _enabled_entry_patterns(params: dict) -> frozenset[str]:
    """Return the research-only entry-pattern subset, or the V4.1 default."""
    raw = params.get(_EXPERIMENT_ENABLED_ENTRY_PATTERNS)
    context = params.get(_EXPERIMENT_CONTEXT_PARAM)
    valid_context = (
        isinstance(context, np.ndarray)
        and context.shape == (1,)
        and context.dtype == np.dtype(np.uint8)
        and context[0] == _EXPERIMENT_CONTEXT_VALUE
    )
    if raw is None or not valid_context:
        return frozenset(ENTRY_PATTERN_IDS)
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError("experimental entry patterns must be a sequence")
    enabled = frozenset(str(value) for value in raw)
    unknown = enabled.difference(ENTRY_PATTERN_IDS)
    if unknown:
        raise ValueError(f"unknown experimental entry patterns: {sorted(unknown)}")
    if not enabled:
        raise ValueError("experimental entry patterns must not be empty")
    return enabled


def _rising_score(values: np.ndarray, floor: float, target: float) -> np.ndarray:
    span = max(float(target) - float(floor), 1e-6)
    return np.clip((values - float(floor)) / span, 0.0, 1.0)


def _falling_score(values: np.ndarray, full_score_at: float, zero_score_at: float) -> np.ndarray:
    span = max(float(zero_score_at) - float(full_score_at), 1e-6)
    return np.clip((float(zero_score_at) - values) / span, 0.0, 1.0)


class V4TrendMatrixStrategy:
    entry_pattern_ids = ENTRY_PATTERN_IDS

    def required_fields(self) -> frozenset[str]:
        return frozenset({"open", "high", "low", "close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        ma20_slope_bars = _int_param(params, "ma20_slope_lookback", 5)
        ma60_slope_bars = _int_param(params, "ma60_slope_lookback", 10)
        breakout_bars = _int_param(params, "breakout_lookback", 20)
        pullback_bars = _int_param(params, "pullback_lookback", 5)
        consolidation_bars = _int_param(params, "consolidation_lookback", 15)
        exit_slope_bars = _int_param(params, "exit_slope_lookback", 5)
        return max(
            61,  # momentum_60d
            60 + ma60_slope_bars,
            20 + ma20_slope_bars,
            20 + pullback_bars + 1,
            breakout_bars + 1,
            consolidation_bars + 1,
            20 + exit_slope_bars,
        )

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        valid_close = np.isfinite(close) & (close > 0)
        valid_volume = np.isfinite(market.volume) & (market.volume > 0)
        tradable = np.asarray(market.tradable, dtype=bool)

        ma20 = matrix_feature(market, "ma20")
        ma60 = matrix_feature(market, "ma60")
        momentum_20 = matrix_feature(market, "momentum_20d")
        momentum_60 = matrix_feature(market, "momentum_60d")
        rsi_14 = matrix_feature(market, "rsi_14")
        atr_pct = matrix_feature(market, "atr_pct")
        volume_ratio = matrix_feature(market, "vol_ratio_5d")
        daily_return = matrix_feature(market, "change_pct")
        enabled_patterns = _enabled_entry_patterns(params)

        ma20_slope_min = _float_param(params, "ma20_slope_min", 0.0)
        ma20_slope_target = _float_param(params, "ma20_slope_target", 0.025)
        ma60_slope_min = _float_param(params, "ma60_slope_min", -0.005)
        ma60_slope_target = _float_param(params, "ma60_slope_target", 0.02)
        ma_spread_target = _float_param(params, "ma_spread_target", 0.08)
        momentum_20_min = _float_param(params, "momentum_20_min", 0.0)
        momentum_20_target = _float_param(params, "momentum_20_target", 0.12)
        momentum_60_min = _float_param(params, "momentum_60_min", -0.03)
        momentum_60_target = _float_param(params, "momentum_60_target", 0.25)
        rsi_soft_max = _float_param(params, "rsi_soft_max", 72.0)
        rsi_max = _float_param(params, "rsi_max", 82.0)
        max_ma20_bias = _float_param(params, "max_ma20_bias", 0.12)
        atr_ideal_pct = _float_param(params, "atr_ideal_pct", 0.02)
        max_atr_pct = _float_param(params, "max_atr_pct", 0.06)

        ma20_previous = valid_shift(
            ma20,
            _int_param(params, "ma20_slope_lookback", 5),
            np.isfinite(ma20),
        )
        ma60_previous = valid_shift(
            ma60,
            _int_param(params, "ma60_slope_lookback", 10),
            np.isfinite(ma60),
        )
        ma20_slope = safe_divide(ma20, ma20_previous) - 1.0
        ma60_slope = safe_divide(ma60, ma60_previous) - 1.0
        ma_spread = safe_divide(ma20, ma60) - 1.0
        ma20_bias = safe_divide(close, ma20) - 1.0

        trend_ok = (
            valid_close
            & np.isfinite(ma20)
            & np.isfinite(ma60)
            & (close > ma20)
            & (ma20 > ma60)
            & (ma20_slope >= ma20_slope_min)
            & (ma60_slope >= ma60_slope_min)
        )
        strength_ok = (
            np.isfinite(momentum_20)
            & np.isfinite(momentum_60)
            & np.isfinite(rsi_14)
            & np.isfinite(atr_pct)
            & np.isfinite(ma20_bias)
            & (momentum_20 > momentum_20_min)
            & (momentum_60 >= momentum_60_min)
            & (rsi_14 <= rsi_max)
            & (ma20_bias <= max_ma20_bias)
            & (atr_pct <= max_atr_pct)
        )

        breakout_lookback = _int_param(params, "breakout_lookback", 20)
        stage_high = valid_rolling_max(close, valid_close, breakout_lookback)
        prior_stage_high = valid_shift(stage_high, 1, np.isfinite(stage_high))
        breakout_buffer = _float_param(params, "breakout_buffer", 0.0)
        price_breakout = (
            np.isfinite(prior_stage_high)
            & (close > prior_stage_high * (1.0 + breakout_buffer))
        )

        pullback_tolerance = _float_param(params, "pullback_tolerance", 0.02)
        touched_ma20 = (
            valid_close
            & np.isfinite(ma20)
            & np.isfinite(market.low)
            & (market.low <= ma20 * (1.0 + pullback_tolerance))
            & (close >= ma20 * (1.0 - pullback_tolerance))
        )
        touch_values = np.where(touched_ma20, 1.0, 0.0).astype(np.float32)
        touch_values[~valid_close] = np.nan
        recent_touch = valid_rolling_max(
            touch_values,
            np.isfinite(touch_values),
            _int_param(params, "pullback_lookback", 5),
        )
        recent_touch = valid_shift(recent_touch, 1, np.isfinite(recent_touch)) > 0.5
        pullback_restart = (
            recent_touch
            & (close > ma20)
            & (daily_return >= _float_param(params, "restart_min_return", 0.0))
        )

        consolidation_lookback = _int_param(params, "consolidation_lookback", 15)
        range_high = valid_rolling_max(close, valid_close, consolidation_lookback)
        range_low = valid_rolling_min(close, valid_close, consolidation_lookback)
        prior_range_high = valid_shift(range_high, 1, np.isfinite(range_high))
        prior_range_low = valid_shift(range_low, 1, np.isfinite(range_low))
        range_width = safe_divide(prior_range_high - prior_range_low, prior_range_low)
        consolidation_breakout = (
            np.isfinite(range_width)
            & (range_width <= _float_param(params, "consolidation_range_max", 0.10))
            & (close > prior_range_high * (1.0 + breakout_buffer))
            & (volume_ratio >= _float_param(params, "breakout_volume_min", 1.20))
        )
        # Keep the original raw conditions for score calculation.  The
        # research-only selection below changes only entry eligibility and its
        # frozen diagnostic metadata, never the V4.1 score formula.
        enabled_price_breakout = price_breakout & ("BREAKOUT" in enabled_patterns)
        enabled_pullback_restart = pullback_restart & (
            "PULLBACK_RESTART" in enabled_patterns
        )
        enabled_consolidation_breakout = consolidation_breakout & (
            "CONSOLIDATION_BREAKOUT" in enabled_patterns
        )
        launch_confirmed = (
            enabled_price_breakout
            | enabled_pullback_restart
            | enabled_consolidation_breakout
        )

        trend_score = 100.0 * (
            _TREND_SUBWEIGHTS[0]
            * _rising_score(ma20_slope, ma20_slope_min, ma20_slope_target)
            + _TREND_SUBWEIGHTS[1]
            * _rising_score(ma60_slope, ma60_slope_min, ma60_slope_target)
            + _TREND_SUBWEIGHTS[2]
            * _rising_score(ma_spread, 0.0, ma_spread_target)
        )
        momentum_score = 100.0 * (
            _MOMENTUM_SUBWEIGHTS[0]
            * _rising_score(momentum_20, momentum_20_min, momentum_20_target)
            + _MOMENTUM_SUBWEIGHTS[1]
            * _rising_score(momentum_60, momentum_60_min, momentum_60_target)
            + _MOMENTUM_SUBWEIGHTS[2]
            * _falling_score(rsi_14, rsi_soft_max, rsi_max)
        )

        breakout_return = safe_divide(close, prior_stage_high) - 1.0
        price_breakout_score = np.where(
            price_breakout,
            _BREAKOUT_BASE_SCORES[0]
            + 30.0
            * _rising_score(
                breakout_return,
                breakout_buffer,
                _float_param(params, "breakout_strength_target", 0.04),
            ),
            0.0,
        )
        pullback_score = np.where(
            pullback_restart,
            _BREAKOUT_BASE_SCORES[1]
            + 35.0
            * _rising_score(
                daily_return,
                _float_param(params, "restart_min_return", 0.0),
                _float_param(params, "restart_target_return", 0.03),
            ),
            0.0,
        )
        tightness_score = _falling_score(
            range_width,
            0.0,
            _float_param(params, "consolidation_range_max", 0.10),
        )
        consolidation_score = np.where(
            consolidation_breakout,
            _BREAKOUT_BASE_SCORES[2]
            + 15.0 * tightness_score
            + 15.0
            * _rising_score(
                volume_ratio,
                _float_param(params, "breakout_volume_min", 1.20),
                _float_param(params, "volume_ideal_ratio", 1.80),
            ),
            0.0,
        )
        breakout_score = np.maximum.reduce(
            [price_breakout_score, pullback_score, consolidation_score]
        )

        volume_ideal = _float_param(params, "volume_ideal_ratio", 1.80)
        volume_extreme = _float_param(params, "volume_extreme_ratio", 3.50)
        volume_score = np.where(
            volume_ratio <= volume_ideal,
            100.0 * np.clip(volume_ratio / max(volume_ideal, 1e-6), 0.0, 1.0),
            100.0 * _falling_score(volume_ratio, volume_ideal, volume_extreme),
        )
        volatility_score = 100.0 * _falling_score(
            atr_pct,
            atr_ideal_pct,
            max_atr_pct,
        )

        weights = np.asarray(
            [
                _float_param(params, "trend_weight", 0.30),
                _float_param(params, "momentum_weight", 0.25),
                _float_param(params, "breakout_weight", 0.20),
                _float_param(params, "volume_weight", 0.15),
                _float_param(params, "volatility_weight", 0.10),
            ],
            dtype=np.float32,
        )
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            weights = np.full(5, 0.2, dtype=np.float32)
            weight_sum = 1.0
        total_score = (
            weights[0] * trend_score
            + weights[1] * momentum_score
            + weights[2] * breakout_score
            + weights[3] * volume_score
            + weights[4] * volatility_score
        ) / weight_sum
        total_score = np.nan_to_num(
            total_score,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        total_score = np.clip(total_score, 0.0, 100.0).astype(np.float32)

        entry = (
            tradable
            & valid_close
            & valid_volume
            & np.isfinite(volume_ratio)
            & (volume_ratio > 0)
            & trend_ok
            & strength_ok
            & launch_confirmed
            & (daily_return <= _float_param(params, "max_daily_gain", 0.085))
            & (volume_ratio <= _float_param(params, "volume_hard_max", 5.0))
            & (total_score >= _float_param(params, "min_total_score", 58.0))
        )

        exit_slope_previous = valid_shift(
            ma20,
            _int_param(params, "exit_slope_lookback", 5),
            np.isfinite(ma20),
        )
        exit_ma20_slope = safe_divide(ma20, exit_slope_previous) - 1.0
        previous_close = valid_shift(close, 1, valid_close)
        previous_ma20 = valid_shift(ma20, 1, np.isfinite(ma20))
        ma20_breakdown = (
            valid_close
            & np.isfinite(previous_close)
            & np.isfinite(previous_ma20)
            & (close < ma20)
            & (previous_close >= previous_ma20)
        )
        ma20_weakening = (
            np.isfinite(exit_ma20_slope)
            & (exit_ma20_slope <= _float_param(params, "exit_ma20_slope_max", -0.01))
        )
        trend_structure_broken = valid_close & np.isfinite(ma60) & (close < ma60)
        exit_ = ma20_breakdown | ma20_weakening | trend_structure_broken

        entry_code = np.where(
            enabled_price_breakout,
            0,
            np.where(
                enabled_pullback_restart,
                1,
                np.where(enabled_consolidation_breakout, 2, -1),
            ),
        ).astype(np.int16)
        entry_code[~entry] = -1
        entry_pattern_mask = (
            enabled_price_breakout.astype(np.uint8)
            | (enabled_pullback_restart.astype(np.uint8) << 1)
            | (enabled_consolidation_breakout.astype(np.uint8) << 2)
        )
        entry_pattern_mask = np.where(entry, entry_pattern_mask, 0).astype(np.uint8)
        exit_code = np.where(
            ma20_breakdown,
            0,
            np.where(ma20_weakening, 1, np.where(trend_structure_broken, 2, -1)),
        ).astype(np.int16)
        exit_code[~exit_] = -1

        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            score=total_score,
            entry_signal_code=entry_code,
            exit_signal_code=exit_code,
            entry_signal_ids=tuple(ENTRY_SIGNALS),
            exit_signal_ids=tuple(EXIT_SIGNALS),
            entry_pattern_mask=entry_pattern_mask,
            entry_pattern_ids=ENTRY_PATTERN_IDS,
        )


MATRIX_STRATEGY = V4TrendMatrixStrategy()
