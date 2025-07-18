import math
import warnings

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np

from src.tools.api import get_prices, prices_to_df
from src.utils.progress import progress

# 抑制pandas rolling window的RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')


def safe_float(value, default=0.0):
    """
    安全地将值转换为浮点数，处理NaN、None、无穷大等情况

    Args:
        value: 要转换的值（可以是pandas标量、numpy值等）
        default: 如果输入为NaN或无效时返回的默认值

    Returns:
        float: 转换后的值，如果NaN/无效则返回默认值
    """
    if value is None:
        return default

    try:
        # Handle pandas Series/DataFrame values
        if hasattr(value, 'iloc'):
            if len(value) > 0:
                value = value.iloc[-1]  # Get the last value
            else:
                return default

        # Check for NaN using pandas and numpy
        if pd.isna(value):
            return default

        result = float(value)

        # Check for infinity or extremely large values
        if (math.isinf(result) or
            abs(result) > 1e15):  # Avoid extremely large values
            return default

        return result
    except (ValueError, TypeError, OverflowError, AttributeError, IndexError):
        return default


##### 技术分析师 #####
def technical_analyst_agent(state: AgentState, agent_id: str = "technical_analyst_agent"):
    """
    复杂的技术分析系统，结合多个股票代码的多种交易策略：
    1. 趋势跟踪
    2. 均值回归
    3. 动量分析
    4. 波动性分析
    5. 统计套利信号
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # 为每个股票代码初始化分析
    technical_analysis = {}

    def _analyze_long_term_trend(prices_df):
        """
        分析中长期趋势状态
        返回: {
            "is_potential_top": bool,
            "is_downtrend": bool,
            "trend_strength": float
        }
        """
        if len(prices_df) < 60:
            return {"is_potential_top": False, "is_downtrend": False, "trend_strength": 0.5}

        current_price = prices_df["close"].iloc[-1]

        # 1. 检查是否接近历史高点
        recent_high_60d = prices_df["high"].rolling(60).max().iloc[-1]
        recent_high_120d = prices_df["high"].rolling(120).max().iloc[-1] if len(prices_df) >= 120 else recent_high_60d

        # 2. 计算多个时间框架的移动平均线
        sma_20 = prices_df["close"].rolling(20).mean().iloc[-1]
        sma_50 = prices_df["close"].rolling(50).mean().iloc[-1]
        sma_100 = prices_df["close"].rolling(100).mean().iloc[-1] if len(prices_df) >= 100 else sma_50

        # 3. 计算RSI
        rsi_14 = calculate_rsi(prices_df, 14).iloc[-1] if len(prices_df) >= 14 else 50

        # 4. 分析移动平均线排列
        ma_bearish_alignment = (sma_20 < sma_50 < sma_100) if len(prices_df) >= 100 else (sma_20 < sma_50)
        price_below_ma = current_price < sma_20 < sma_50

        # 5. 计算价格相对位置
        high_ratio_60d = current_price / recent_high_60d
        high_ratio_120d = current_price / recent_high_120d

        # 6. 检查连续下跌天数
        returns = prices_df["close"].pct_change().dropna()
        consecutive_down_days = 0
        for ret in reversed(returns.iloc[-10:]):  # 检查最近10天
            if ret < 0:
                consecutive_down_days += 1
            else:
                break

        # 判断是否为潜在顶部
        is_potential_top = (
            high_ratio_60d >= 0.90 and rsi_14 > 55
        ) or (
            high_ratio_120d >= 0.95 and rsi_14 > 50
        )

        # 判断是否为下行趋势（更严格的条件）
        downtrend_signals = 0

        # 信号1: 移动平均线空头排列
        if ma_bearish_alignment:
            downtrend_signals += 1

        # 信号2: 价格低于关键移动平均线
        if price_below_ma:
            downtrend_signals += 1

        # 信号3: 从高点大幅回落
        if high_ratio_60d < 0.85:  # 从60天高点回落超过15%
            downtrend_signals += 1

        # 信号4: 连续下跌天数较多
        if consecutive_down_days >= 3:
            downtrend_signals += 1

        # 信号5: RSI持续偏低
        if rsi_14 < 45:
            downtrend_signals += 1

        is_downtrend = downtrend_signals >= 3  # 需要至少3个下行信号

        # 计算趋势强度
        if is_downtrend:
            trend_strength = min(0.9, downtrend_signals / 5.0)
        elif is_potential_top:
            trend_strength = 0.7
        else:
            trend_strength = 0.5

        return {
            "is_potential_top": is_potential_top,
            "is_downtrend": is_downtrend,
            "trend_strength": trend_strength,
            "analysis": {
                "high_ratio_60d": high_ratio_60d,
                "ma_bearish_alignment": ma_bearish_alignment,
                "price_below_ma": price_below_ma,
                "consecutive_down_days": consecutive_down_days,
                "downtrend_signals": downtrend_signals
            }
        }

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Analyzing price data")

        # 计算需要更多历史数据的开始日期（回测开始日期前18个月）
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta

        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # 为技术分析获取更多历史数据（18个月前开始，以便识别长期趋势）
        extended_start_date = (end_date_dt - relativedelta(months=18)).strftime("%Y-%m-%d")

        progress.update_status(agent_id, ticker, f"Fetching data from {extended_start_date} to {end_date}")

        # Get the historical price data with extended range
        prices = get_prices(
            ticker=ticker,
            start_date=extended_start_date,
            end_date=end_date,
        )

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)

        if prices_df.empty:
            progress.update_status(agent_id, ticker, "Failed: Empty price data")
            continue

        # 重要：只使用截止到当前回测日期的数据，避免未来数据泄露
        current_backtest_date = datetime.strptime(end_date, "%Y-%m-%d")
        prices_df = prices_df[prices_df.index <= current_backtest_date]

        if len(prices_df) < 20:  # 确保有足够的数据进行技术分析
            progress.update_status(agent_id, ticker, f"Warning: Only {len(prices_df)} days of data available")
            # 继续处理，但信心度会较低

        progress.update_status(agent_id, ticker, f"Calculating trend signals with {len(prices_df)} days of data")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status(agent_id, ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status(agent_id, ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status(agent_id, ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        # 动态权重配置 - 根据市场状态调整权重
        # 检查中长期趋势状态
        trend_analysis = _analyze_long_term_trend(prices_df)
        is_potential_top = trend_analysis["is_potential_top"]
        is_downtrend = trend_analysis["is_downtrend"]
        trend_strength = trend_analysis["trend_strength"]

        if is_downtrend:
            # 在明显下行趋势中，极大提高均值回归权重，几乎禁用买入信号
            strategy_weights = {
                "mean_reversion": 0.60,  # 极大提高均值回归权重
                "trend": 0.15,           # 大幅降低趋势权重
                "momentum": 0.10,        # 极大降低动量权重
                "volatility": 0.10,      # 保持波动率权重
                "stat_arb": 0.05,        # 保持统计套利权重
            }
        elif is_potential_top:
            # 在潜在顶部区域，大幅提高均值回归权重
            strategy_weights = {
                "mean_reversion": 0.45,  # 大幅提高均值回归权重
                "trend": 0.20,           # 降低趋势权重
                "momentum": 0.15,        # 大幅降低动量权重
                "volatility": 0.15,      # 提高波动率权重
                "stat_arb": 0.05,        # 保持统计套利权重
            }
        else:
            # 正常市场条件下的权重配置
            strategy_weights = {
                "trend": 0.30,           # 趋势跟踪权重
                "momentum": 0.25,        # 动量权重
                "mean_reversion": 0.30,  # 均值回归权重
                "volatility": 0.10,      # 波动率权重
                "stat_arb": 0.05,        # 统计套利权重
            }

        progress.update_status(agent_id, ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
            trend_analysis,  # 传递趋势分析结果
        )

        # Helper function to safely convert confidence to integer
        def safe_confidence_round(confidence_value):
            """Safely round confidence value, handling NaN cases."""
            import math
            if math.isnan(confidence_value):
                return 0  # Default to 0% confidence if NaN
            return round(confidence_value * 100)

        # Generate detailed analysis report for this ticker
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": safe_confidence_round(combined_signal["confidence"]),
            "reasoning": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": safe_confidence_round(trend_signals["confidence"]),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": safe_confidence_round(mean_reversion_signals["confidence"]),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": safe_confidence_round(momentum_signals["confidence"]),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": safe_confidence_round(volatility_signals["confidence"]),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": safe_confidence_round(stat_arb_signals["confidence"]),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        progress.update_status(agent_id, ticker, "完成", analysis=json.dumps(technical_analysis, indent=4))

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = technical_analysis

    progress.update_status(agent_id, None, "完成")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_trend_signals(prices_df):
    """
    短线交易优化的趋势跟踪策略
    """
    if len(prices_df) < 21:
        return {
            "signal": "neutral",
            "confidence": 0.3,
            "metrics": {"adx": 25, "trend_strength": 0.25, "ema_alignment": 0}
        }

    # 短线EMA组合：5, 13, 21
    ema_5 = calculate_ema(prices_df, 5)   # 超短线
    ema_13 = calculate_ema(prices_df, 13) # 短线
    ema_21 = calculate_ema(prices_df, 21) # 中短线

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # 获取当前价格
    current_price = prices_df["close"].iloc[-1]

    # 趋势方向判定
    ema5_current = ema_5.iloc[-1]
    ema13_current = ema_13.iloc[-1]
    ema21_current = ema_21.iloc[-1]

    # EMA排列分析
    bullish_alignment = ema5_current > ema13_current > ema21_current
    bearish_alignment = ema5_current < ema13_current < ema21_current

    # 价格相对EMA位置
    price_above_ema5 = current_price > ema5_current
    price_above_ema13 = current_price > ema13_current

    # 短期趋势变化检测
    ema5_slope = (ema_5.iloc[-1] - ema_5.iloc[-3]) / ema_5.iloc[-3] if len(ema_5) >= 3 else 0
    ema13_slope = (ema_13.iloc[-1] - ema_13.iloc[-3]) / ema_13.iloc[-3] if len(ema_13) >= 3 else 0

    # ADX趋势强度
    adx_value = adx["adx"].iloc[-1]
    trend_strength = min(adx_value / 100.0, 1.0)

    # 综合信号判定
    bullish_signals = 0
    bearish_signals = 0

    # EMA排列信号
    if bullish_alignment:
        bullish_signals += 2
    elif bearish_alignment:
        bearish_signals += 2

    # 价格位置信号
    if price_above_ema5 and price_above_ema13:
        bullish_signals += 1
    elif not price_above_ema5 and not price_above_ema13:
        bearish_signals += 1

    # 斜率信号
    if ema5_slope > 0.01 and ema13_slope > 0.005:  # 上升趋势
        bullish_signals += 1
    elif ema5_slope < -0.01 and ema13_slope < -0.005:  # 下降趋势
        bearish_signals += 1

    # 趋势强度加成
    if adx_value > 25:  # 强趋势
        if bullish_signals > bearish_signals:
            bullish_signals += 1
        elif bearish_signals > bullish_signals:
            bearish_signals += 1

    # 新增：潜在顶部/底部识别
    # 检查是否接近历史高点（可能是顶部信号）
    if len(prices_df) >= 60:  # 至少需要60天数据
        recent_high = prices_df["high"].rolling(60).max().iloc[-1]
        current_price = prices_df["close"].iloc[-1]

        # 如果当前价格接近60天高点（95%以上），增加谨慎性
        if current_price >= recent_high * 0.95:
            # 检查是否有背离信号（价格新高但指标未新高）
            recent_rsi = calculate_rsi(prices_df, 14).iloc[-20:]  # 最近20天RSI
            if len(recent_rsi) >= 20:
                rsi_current = recent_rsi.iloc[-1]
                rsi_prev_high = recent_rsi.max()

                # RSI背离：价格新高但RSI未新高，增加看跌信号
                if rsi_current < rsi_prev_high * 0.95 and rsi_current > 70:
                    bearish_signals += 2  # 强烈的顶部背离信号
                elif rsi_current > 75:  # 极度超买
                    bearish_signals += 1

    # 最终信号生成 - 增加谨慎性
    total_signals = bullish_signals + bearish_signals
    if total_signals == 0:
        signal = "neutral"
        confidence = 0.3
    elif bullish_signals > bearish_signals:
        signal = "bullish"
        signal_ratio = bullish_signals / max(total_signals, 1)
        base_confidence = signal_ratio * trend_strength + 0.3

        # 在潜在顶部区域降低看涨信心度
        if len(prices_df) >= 60:
            recent_high = prices_df["high"].rolling(60).max().iloc[-1]
            current_price = prices_df["close"].iloc[-1]
            if current_price >= recent_high * 0.95:  # 接近历史高点
                base_confidence *= 0.7  # 降低30%信心度

        confidence = min(0.85, base_confidence)
    else:
        signal = "bearish"
        signal_ratio = bearish_signals / max(total_signals, 1)
        confidence = min(0.85, signal_ratio * trend_strength + 0.3)

    # EMA排列得分（用于metrics）
    ema_alignment_score = 0
    if bullish_alignment:
        ema_alignment_score = 1
    elif bearish_alignment:
        ema_alignment_score = -1

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": safe_float(adx_value),
            "trend_strength": safe_float(trend_strength),
            "ema_alignment": ema_alignment_score,
            "ema5_slope": safe_float(ema5_slope),
            "ema13_slope": safe_float(ema13_slope),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals
        },
    }


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # 降低数据要求，提高信号敏感度
    if len(prices_df) < 20:  # 从50降到20
        return {
            "signal": "neutral",
            "confidence": 0.0,
            "metrics": {
                "z_score": 0.0,
                "price_vs_bb": 0.5,
                "rsi_14": 50.0,
                "rsi_28": 50.0,
            }
        }

    # 使用更短的移动平均线
    ma_20 = prices_df["close"].rolling(window=20, min_periods=20).mean()
    std_20 = prices_df["close"].rolling(window=20, min_periods=20).std()

    # Handle division by zero and NaN values
    with np.errstate(divide='ignore', invalid='ignore'):
        z_score = (prices_df["close"] - ma_20) / std_20
        z_score = z_score.fillna(0)

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    # Safe calculation of price vs Bollinger Bands
    try:
        bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        if pd.isna(bb_range) or bb_range == 0:
            price_vs_bb = 0.5  # Default to middle
        else:
            price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / bb_range
    except (IndexError, TypeError):
        price_vs_bb = 0.5

    # 短线交易优化的均值回归信号
    z_score_val = safe_float(z_score.iloc[-1] if len(z_score) > 0 else 0)
    rsi_14_val = safe_float(rsi_14.iloc[-1] if len(rsi_14) > 0 else 50)
    rsi_28_val = safe_float(rsi_28.iloc[-1] if len(rsi_28) > 0 else 50)

    # 多重确认信号
    oversold_signals = 0
    overbought_signals = 0

    # Z-score信号（降低阈值，提高敏感度）
    if z_score_val < -1.5:  # 从-2降到-1.5
        oversold_signals += 1
    elif z_score_val > 1.5:  # 从2降到1.5
        overbought_signals += 1

    # 布林带位置信号
    if price_vs_bb < 0.25:  # 从0.2提高到0.25
        oversold_signals += 1
    elif price_vs_bb > 0.75:  # 从0.8降到0.75
        overbought_signals += 1

    # RSI信号 - 更严格的超买超卖阈值
    if rsi_14_val < 30:
        oversold_signals += 1
    elif rsi_14_val > 60:  # 进一步降低到60
        overbought_signals += 1

    # 双RSI确认 - 更严格的阈值
    if rsi_14_val < 35 and rsi_28_val < 40:
        oversold_signals += 1
    elif rsi_14_val > 55 and rsi_28_val > 50:  # 进一步降低超买阈值
        overbought_signals += 1

    # 信号生成和信心度计算
    if oversold_signals >= 2:  # 至少2个超卖信号
        signal = "bullish"
        # 信心度基于信号数量和强度
        signal_strength = min(abs(z_score_val) / 2, 1.0)
        signal_count_bonus = (oversold_signals - 1) * 0.15
        confidence = min(0.8, 0.4 + signal_strength * 0.3 + signal_count_bonus)
    elif overbought_signals >= 2:  # 至少2个超买信号
        signal = "bearish"
        signal_strength = min(abs(z_score_val) / 2, 1.0)
        signal_count_bonus = (overbought_signals - 1) * 0.15
        confidence = min(0.8, 0.4 + signal_strength * 0.3 + signal_count_bonus)
    else:
        signal = "neutral"
        # 中性信号也给予合理信心度
        confidence = 0.4 + min(abs(z_score_val) / 4, 0.2)

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": safe_float(z_score.iloc[-1]),
            "price_vs_bb": safe_float(price_vs_bb),
            "rsi_14": safe_float(rsi_14.iloc[-1]),
            "rsi_28": safe_float(rsi_28.iloc[-1]),
            "oversold_signals": oversold_signals,
            "overbought_signals": overbought_signals
        },
    }


def calculate_kdj_signal(prices_df):
    """
    计算KDJ随机指标 - 短线交易核心指标
    """
    if len(prices_df) < 14:
        return {"signal": "neutral", "confidence": 0.3, "k": 50, "d": 50, "j": 50}

    high = prices_df["high"]
    low = prices_df["low"]
    close = prices_df["close"]

    # 计算RSV (Raw Stochastic Value)
    lowest_low = low.rolling(window=9).min()
    highest_high = high.rolling(window=9).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

    # 计算KDJ
    k_values = []
    d_values = []
    j_values = []

    k_prev = 50  # 初始值
    d_prev = 50  # 初始值

    for i, rsv_val in enumerate(rsv):
        if pd.isna(rsv_val):
            k_values.append(k_prev)
            d_values.append(d_prev)
            j_values.append(3 * k_prev - 2 * d_prev)
            continue

        k_curr = (2/3) * k_prev + (1/3) * rsv_val
        d_curr = (2/3) * d_prev + (1/3) * k_curr
        j_curr = 3 * k_curr - 2 * d_curr

        k_values.append(k_curr)
        d_values.append(d_curr)
        j_values.append(j_curr)

        k_prev = k_curr
        d_prev = d_curr

    # 获取最新值
    k = k_values[-1]
    d = d_values[-1]
    j = j_values[-1]

    # KDJ信号判定
    if k > d and k < 20:  # 超卖区域金叉
        signal = "bullish"
        confidence = min(0.85, (20 - k) / 20 * 0.8 + 0.4)
    elif k < d and k > 80:  # 超买区域死叉
        signal = "bearish"
        confidence = min(0.85, (k - 80) / 20 * 0.8 + 0.4)
    elif k > d and j > k:  # 普通金叉
        signal = "bullish"
        confidence = 0.6
    elif k < d and j < k:  # 普通死叉
        signal = "bearish"
        confidence = 0.6
    else:
        signal = "neutral"
        confidence = 0.3

    return {
        "signal": signal,
        "confidence": confidence,
        "k": k,
        "d": d,
        "j": j
    }


def calculate_williams_r_signal(prices_df):
    """
    计算威廉指标 - 短线超买超卖指标
    """
    if len(prices_df) < 14:
        return {"signal": "neutral", "confidence": 0.3, "wr": -50}

    high = prices_df["high"]
    low = prices_df["low"]
    close = prices_df["close"]

    # 计算威廉指标
    highest_high = high.rolling(window=14).max()
    lowest_low = low.rolling(window=14).min()
    wr = (highest_high - close) / (highest_high - lowest_low) * (-100)

    current_wr = wr.iloc[-1]
    prev_wr = wr.iloc[-2] if len(wr) > 1 else current_wr

    # 威廉指标信号判定
    if current_wr < -80 and prev_wr >= -80:  # 从超卖区域向上突破
        signal = "bullish"
        confidence = 0.8
    elif current_wr > -20 and prev_wr <= -20:  # 从超买区域向下突破
        signal = "bearish"
        confidence = 0.8
    elif current_wr < -80:  # 超卖区域
        signal = "bullish"
        confidence = min(0.7, abs(current_wr + 80) / 20 * 0.5 + 0.4)
    elif current_wr > -20:  # 超买区域
        signal = "bearish"
        confidence = min(0.7, abs(current_wr + 20) / 20 * 0.5 + 0.4)
    else:
        signal = "neutral"
        confidence = 0.3

    return {
        "signal": signal,
        "confidence": confidence,
        "wr": current_wr
    }


def calculate_cci_signal(prices_df):
    """
    计算CCI顺势指标 - 识别趋势变化
    """
    if len(prices_df) < 20:
        return {"signal": "neutral", "confidence": 0.3, "cci": 0}

    high = prices_df["high"]
    low = prices_df["low"]
    close = prices_df["close"]

    # 计算典型价格
    tp = (high + low + close) / 3

    # 计算CCI
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)

    current_cci = cci.iloc[-1]
    prev_cci = cci.iloc[-2] if len(cci) > 1 else current_cci

    # CCI信号判定
    if current_cci > 100 and prev_cci <= 100:  # 突破超买线
        signal = "bullish"
        confidence = 0.8
    elif current_cci < -100 and prev_cci >= -100:  # 跌破超卖线
        signal = "bearish"
        confidence = 0.8
    elif current_cci > 100:  # 超买区域
        signal = "bearish"
        confidence = min(0.7, (current_cci - 100) / 100 * 0.4 + 0.4)
    elif current_cci < -100:  # 超卖区域
        signal = "bullish"
        confidence = min(0.7, abs(current_cci + 100) / 100 * 0.4 + 0.4)
    else:
        signal = "neutral"
        confidence = 0.3

    return {
        "signal": signal,
        "confidence": confidence,
        "cci": current_cci
    }


def calculate_macd_signal(prices_df):
    """
    计算MACD信号 - 短线交易重要指标
    """
    if len(prices_df) < 26:
        return {"signal": "neutral", "confidence": 0.0, "macd": 0, "signal_line": 0, "histogram": 0}

    close = prices_df["close"]

    # 计算MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line

    # 获取最新值
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    current_histogram = histogram.iloc[-1]
    prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0

    # MACD信号判定
    if current_macd > current_signal and current_histogram > 0:
        if current_histogram > prev_histogram:  # 柱状图增长
            signal = "bullish"
            confidence = min(0.8, abs(current_histogram) * 100)
        else:
            signal = "bullish"
            confidence = min(0.6, abs(current_histogram) * 80)
    elif current_macd < current_signal and current_histogram < 0:
        if current_histogram < prev_histogram:  # 柱状图下降
            signal = "bearish"
            confidence = min(0.8, abs(current_histogram) * 100)
        else:
            signal = "bearish"
            confidence = min(0.6, abs(current_histogram) * 80)
    else:
        signal = "neutral"
        confidence = 0.3

    return {
        "signal": signal,
        "confidence": confidence,
        "macd": current_macd,
        "signal_line": current_signal,
        "histogram": current_histogram
    }


def calculate_bollinger_bands_signal(prices_df):
    """
    计算布林带突破信号 - 短线突破策略
    """
    if len(prices_df) < 20:
        return {"signal": "neutral", "confidence": 0.3, "bb_position": 0.5, "bb_width": 0}

    close = prices_df["close"]

    # 计算布林带
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)

    # 获取最新值
    current_close = close.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_sma = sma_20.iloc[-1]
    prev_close = close.iloc[-2] if len(close) > 1 else current_close

    # 计算价格在布林带中的位置
    bb_position = (current_close - current_lower) / (current_upper - current_lower)
    bb_width = (current_upper - current_lower) / current_sma  # 布林带宽度

    # 布林带信号判定
    if current_close > current_upper and prev_close <= current_upper:  # 向上突破
        signal = "bullish"
        confidence = min(0.85, bb_width * 10 + 0.5)  # 宽度越大信心度越高
    elif current_close < current_lower and prev_close >= current_lower:  # 向下突破
        signal = "bearish"
        confidence = min(0.85, bb_width * 10 + 0.5)
    elif bb_position > 0.8:  # 接近上轨
        signal = "bearish"  # 可能回调
        confidence = min(0.6, (bb_position - 0.8) / 0.2 * 0.4 + 0.3)
    elif bb_position < 0.2:  # 接近下轨
        signal = "bullish"  # 可能反弹
        confidence = min(0.6, (0.2 - bb_position) / 0.2 * 0.4 + 0.3)
    else:
        signal = "neutral"
        confidence = 0.3

    return {
        "signal": signal,
        "confidence": confidence,
        "bb_position": bb_position,
        "bb_width": bb_width,
        "upper_band": current_upper,
        "lower_band": current_lower,
        "middle_band": current_sma
    }


def calculate_short_term_indicators(prices_df):
    """
    计算短线交易指标组合
    """
    # 计算各个短线指标
    kdj_result = calculate_kdj_signal(prices_df)
    williams_result = calculate_williams_r_signal(prices_df)
    cci_result = calculate_cci_signal(prices_df)
    bb_result = calculate_bollinger_bands_signal(prices_df)
    macd_result = calculate_macd_signal(prices_df)

    # 短线指标权重分配
    weights = {
        "kdj": 0.25,        # KDJ权重最高，散户最常用
        "williams": 0.20,   # 威廉指标，超买超卖
        "cci": 0.20,        # CCI，趋势确认
        "bollinger": 0.20,  # 布林带，突破信号
        "macd": 0.15        # MACD，动量确认
    }

    indicators = [kdj_result, williams_result, cci_result, bb_result, macd_result]
    weight_list = [weights["kdj"], weights["williams"], weights["cci"], weights["bollinger"], weights["macd"]]

    # 计算加权信号
    bullish_score = 0
    bearish_score = 0
    total_confidence = 0

    for indicator, weight in zip(indicators, weight_list):
        confidence = indicator["confidence"]
        if indicator["signal"] == "bullish":
            bullish_score += weight * confidence
        elif indicator["signal"] == "bearish":
            bearish_score += weight * confidence
        total_confidence += weight * confidence

    # 生成最终信号
    if bullish_score > bearish_score and bullish_score > 0.3:
        final_signal = "bullish"
        final_confidence = min(bullish_score * 1.2, 0.9)  # 提高信心度
    elif bearish_score > bullish_score and bearish_score > 0.3:
        final_signal = "bearish"
        final_confidence = min(bearish_score * 1.2, 0.9)
    else:
        final_signal = "neutral"
        final_confidence = max(total_confidence * 0.6, 0.25)

    return {
        "signal": final_signal,
        "confidence": final_confidence,
        "indicators": {
            "kdj": kdj_result,
            "williams_r": williams_result,
            "cci": cci_result,
            "bollinger_bands": bb_result,
            "macd": macd_result
        }
    }


def calculate_momentum_signals(prices_df):
    """
    Multi-factor momentum strategy - 增强短线交易能力
    """
    # 降低数据要求，提高信号生成频率
    if len(prices_df) < 26:  # 进一步降低到26天，适应MACD
        return {
            "signal": "neutral",
            "confidence": 0.0,
            "metrics": {
                "momentum_1m": 0.0,
                "momentum_3m": 0.0,
                "momentum_6m": 0.0,
                "volume_momentum": 1.0,
                "short_term_indicators": {"signal": "neutral", "confidence": 0.0}
            }
        }

    # 改进的价格动量计算 - 使用价格变化率而不是收益率累计
    current_price = prices_df["close"].iloc[-1]

    # 1个月动量（21个交易日）
    if len(prices_df) >= 21:
        price_1m_ago = prices_df["close"].iloc[-21]
        mom_1m = (current_price - price_1m_ago) / price_1m_ago
    else:
        mom_1m = 0.0

    # 3个月动量（63个交易日）
    if len(prices_df) >= 63:
        price_3m_ago = prices_df["close"].iloc[-63]
        mom_3m = (current_price - price_3m_ago) / price_3m_ago
    else:
        mom_3m = 0.0

    # 6个月动量（126个交易日）- 如果数据不足则使用可用数据
    if len(prices_df) >= 126:
        price_6m_ago = prices_df["close"].iloc[-126]
        mom_6m = (current_price - price_6m_ago) / price_6m_ago
    else:
        # 使用可用的最长期间
        price_start = prices_df["close"].iloc[0]
        mom_6m = (current_price - price_start) / price_start

    # 改进的成交量动量
    volume_ma = prices_df["volume"].rolling(21, min_periods=10).mean()
    current_volume = prices_df["volume"].iloc[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_momentum = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
        volume_momentum = safe_float(volume_momentum, 1.0)

    # 计算短线指标组合
    short_term_result = calculate_short_term_indicators(prices_df)

    # 改进的动量评分计算 - 集成短线指标
    # 给予近期动量更高权重，并结合短线指标
    momentum_score = 0.3 * mom_1m + 0.25 * mom_3m + 0.15 * mom_6m

    # 短线指标权重调整（30%权重）
    short_term_weight = 0.3
    if short_term_result["signal"] == "bullish":
        momentum_score += short_term_weight * short_term_result["confidence"]
    elif short_term_result["signal"] == "bearish":
        momentum_score -= short_term_weight * short_term_result["confidence"]

    # 动态阈值 - 根据波动率调整
    returns = prices_df["close"].pct_change().dropna()
    volatility = returns.std() * math.sqrt(252) if len(returns) > 0 else 0.2

    # 根据波动率调整阈值
    threshold = max(0.02, volatility * 0.1)  # 最小2%，最大为年化波动率的10%

    # 成交量确认 - 降低要求
    volume_confirmation = volume_momentum > 0.8  # 从1.0降低到0.8

    # 信号生成逻辑 - 结合短线指标确认
    if momentum_score > threshold and volume_confirmation:
        signal = "bullish"
        # 短线指标确认加强信心度
        base_confidence = min(abs(momentum_score) / threshold * 0.8, 0.9)
        if short_term_result["signal"] == "bullish":
            confidence = min(base_confidence + 0.15, 0.95)  # 短线指标确认，提高信心度
        else:
            confidence = base_confidence
    elif momentum_score < -threshold and volume_confirmation:
        signal = "bearish"
        base_confidence = min(abs(momentum_score) / threshold * 0.8, 0.9)
        if short_term_result["signal"] == "bearish":
            confidence = min(base_confidence + 0.15, 0.95)
        else:
            confidence = base_confidence
    else:
        signal = "neutral"
        # 如果短线指标有明确信号，适当提高信心度
        if short_term_result["signal"] != "neutral" and short_term_result["confidence"] > 0.6:
            signal = short_term_result["signal"]
            confidence = min(short_term_result["confidence"] * 0.8, 0.7)
        else:
            confidence = 0.3 + min(abs(momentum_score) / threshold * 0.2, 0.2)

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": mom_1m,
            "momentum_3m": mom_3m,
            "momentum_6m": mom_6m,
            "volume_momentum": volume_momentum,
            "threshold_used": threshold,
            "volatility": volatility,
            "short_term_indicators": {
                "signal": short_term_result["signal"],
                "confidence": short_term_result["confidence"],
                "kdj": short_term_result["indicators"]["kdj"],
                "williams_r": short_term_result["indicators"]["williams_r"],
                "cci": short_term_result["indicators"]["cci"],
                "bollinger_bands": short_term_result["indicators"]["bollinger_bands"],
                "macd": short_term_result["indicators"]["macd"]
            }
        },
    }


def calculate_volatility_signals(prices_df):
    """
    Volatility-based trading strategy
    """
    # Check if we have enough data for calculations
    if len(prices_df) < 63:
        return {
            "signal": "neutral",
            "confidence": 0.0,
            "metrics": {
                "historical_volatility": 0.2,
                "volatility_regime": 1.0,
                "volatility_z_score": 0.0,
                "atr_ratio": 0.02,
            }
        }

    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21, min_periods=21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63, min_periods=63).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_regime = hist_vol / vol_ma
        vol_regime = vol_regime.fillna(1.0)

    # Volatility mean reversion
    vol_std = hist_vol.rolling(63, min_periods=63).std()
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_z_score = (hist_vol - vol_ma) / vol_std
        vol_z_score = vol_z_score.fillna(0.0)

    # ATR ratio
    atr = calculate_atr(prices_df)
    with np.errstate(divide='ignore', invalid='ignore'):
        atr_ratio = atr / prices_df["close"]
        atr_ratio = atr_ratio.fillna(0.0)

    # Generate signal based on volatility regime with safe NaN handling
    current_vol_regime = safe_float(vol_regime.iloc[-1], 1.0)
    vol_z = safe_float(vol_z_score.iloc[-1], 0.0)

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": safe_float(hist_vol.iloc[-1]),
            "volatility_regime": safe_float(current_vol_regime),
            "volatility_z_score": safe_float(vol_z),
            "atr_ratio": safe_float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df):
    """
    Statistical arbitrage signals based on price action analysis
    """
    # Check if we have enough data for calculations
    if len(prices_df) < 63:
        return {
            "signal": "neutral",
            "confidence": 0.0,
            "metrics": {
                "hurst_exponent": 0.5,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }
        }

    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63, min_periods=63).skew()
    kurt = returns.rolling(63, min_periods=63).kurt()

    # Test for mean reversion using Hurst exponent
    hurst = calculate_hurst_exponent(prices_df["close"])

    # Correlation analysis
    # (would include correlation with related securities in real implementation)

    # Generate signal based on statistical properties with safe NaN handling
    skew_val = safe_float(skew.iloc[-1] if len(skew) > 0 else 0)
    hurst = safe_float(hurst, 0.5)

    if hurst < 0.4 and skew_val > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew_val < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": safe_float(hurst),
            "skewness": safe_float(skew.iloc[-1]),
            "kurtosis": safe_float(kurt.iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights, trend_analysis=None):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_weight = 0
    confidence_sum = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        # 确保信心度在合理范围内
        confidence = max(0.0, min(1.0, confidence))

        weighted_sum += numeric_signal * weight * confidence
        total_weight += weight
        confidence_sum += confidence * weight

    # Normalize the weighted sum
    if total_weight > 0:
        final_score = weighted_sum / total_weight
        avg_confidence = confidence_sum / total_weight
    else:
        final_score = 0
        avg_confidence = 0

    # 短线交易优化的信号阈值 - 更敏感但平衡
    if final_score > 0.12:  # 适度降低看涨阈值
        signal = "bullish"
    elif final_score < -0.12:  # 适度降低看跌阈值
        signal = "bearish"
    else:
        signal = "neutral"

    # 短线交易优化的信心度计算
    signal_strength = abs(final_score)

    # 统计强信号数量（信心度>60%且非中性）
    strong_signals = 0
    momentum_signal_match = False

    for strategy, signal_data in signals.items():
        if signal_data["confidence"] > 0.6 and signal_data["signal"] != "neutral":
            strong_signals += 1
        # 检查动量策略是否与最终信号一致
        if strategy == "momentum" and signal_data["signal"] == signal:
            momentum_signal_match = True

    # 优化的信心度计算 - 降低过度乐观
    if signal != "neutral":
        # 基础信心度计算 - 降低基础信心度和乘数
        base_confidence = max(0.25, signal_strength * 1.5)  # 降低基础信心度和乘数

        # 强信号加成 - 降低加成幅度
        strong_signal_bonus = strong_signals * 0.05

        # 动量策略加成 - 降低动量加成
        momentum_bonus = 0.08 if momentum_signal_match else 0

        # 检查是否有均值回归看跌信号（在潜在顶部时降低信心度）
        mean_reversion_bearish_penalty = 0
        if "mean_reversion" in signals and signals["mean_reversion"]["signal"] == "bearish":
            mean_reversion_bearish_penalty = signals["mean_reversion"]["confidence"] * 0.15

        # 在下行趋势中大幅降低看涨信号的信心度
        downtrend_penalty = 0
        if trend_analysis and signal == "bullish" and trend_analysis.get("is_downtrend", False):
            downtrend_penalty = 0.4  # 在下行趋势中看涨信号信心度降低40%

        final_confidence = min(0.85, base_confidence + strong_signal_bonus + momentum_bonus - mean_reversion_bearish_penalty - downtrend_penalty)
    else:
        # 中性信号的信心度 - 如果有强信号但被平均化，适当提高信心度
        if strong_signals > 0:
            final_confidence = max(0.35, avg_confidence * 0.9)
        else:
            final_confidence = max(0.25, signal_strength * 2)

    return {"signal": signal, "confidence": final_confidence}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate RSI using the standard Wilder's smoothing method (exponential moving average).

    Args:
        prices_df: DataFrame with 'close' column
        period: RSI period (default 14)

    Returns:
        pd.Series: RSI values
    """
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Use exponential moving average with alpha = 1/period (Wilder's smoothing)
    # This is the correct method for RSI calculation
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rs = rs.fillna(0)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Default RSI to 50 when no data
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window, min_periods=window).mean()
    std_dev = prices_df["close"].rolling(window, min_periods=window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Calculate True Range
    df_copy["high_low"] = df_copy["high"] - df_copy["low"]
    df_copy["high_close"] = abs(df_copy["high"] - df_copy["close"].shift())
    df_copy["low_close"] = abs(df_copy["low"] - df_copy["close"].shift())
    df_copy["tr"] = df_copy[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df_copy["up_move"] = df_copy["high"] - df_copy["high"].shift()
    df_copy["down_move"] = df_copy["low"].shift() - df_copy["low"]

    df_copy["plus_dm"] = np.where((df_copy["up_move"] > df_copy["down_move"]) & (df_copy["up_move"] > 0), df_copy["up_move"], 0)
    df_copy["minus_dm"] = np.where((df_copy["down_move"] > df_copy["up_move"]) & (df_copy["down_move"] > 0), df_copy["down_move"], 0)

    # Calculate smoothed True Range and Directional Movement
    tr_smooth = df_copy["tr"].ewm(span=period, adjust=False).mean()
    plus_dm_smooth = df_copy["plus_dm"].ewm(span=period, adjust=False).mean()
    minus_dm_smooth = df_copy["minus_dm"].ewm(span=period, adjust=False).mean()

    # Calculate Directional Indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)

    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)

    # Calculate ADX
    adx = dx.ewm(span=period, adjust=False).mean()

    # Create result DataFrame with original index
    result = pd.DataFrame(index=df.index)
    result["adx"] = adx
    result["+di"] = plus_di
    result["-di"] = minus_di

    return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period, min_periods=period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent using the R/S (Rescaled Range) method.
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    if len(price_series) < max_lag:
        return 0.5

    # Convert to numpy array for easier manipulation
    prices = np.array(price_series.dropna())
    if len(prices) < max_lag:
        return 0.5

    # Calculate log returns
    log_returns = np.diff(np.log(prices))

    lags = range(2, min(max_lag, len(log_returns) // 2))
    rs_values = []

    for lag in lags:
        # Split the series into non-overlapping subseries of length 'lag'
        n_subseries = len(log_returns) // lag
        if n_subseries == 0:
            continue

        rs_subseries = []
        for i in range(n_subseries):
            subseries = log_returns[i*lag:(i+1)*lag]

            # Calculate mean return for this subseries
            mean_return = np.mean(subseries)

            # Calculate cumulative deviations from mean
            deviations = subseries - mean_return
            cumulative_deviations = np.cumsum(deviations)

            # Calculate range (R)
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

            # Calculate standard deviation (S)
            S = np.std(subseries, ddof=1) if len(subseries) > 1 else 1e-8

            # Calculate R/S for this subseries
            if S > 1e-8:
                rs_subseries.append(R / S)

        # Average R/S across all subseries for this lag
        if rs_subseries:
            rs_values.append(np.mean(rs_subseries))
        else:
            rs_values.append(1.0)

    if len(rs_values) < 3:
        return 0.5

    # Add small epsilon to avoid log(0)
    rs_values = [max(1e-8, rs) for rs in rs_values]

    # Return the Hurst exponent from linear fit of log(R/S) vs log(lag)
    try:
        reg = np.polyfit(np.log(list(lags)[:len(rs_values)]), np.log(rs_values), 1)
        hurst = reg[0]  # Hurst exponent is the slope
        # Clamp to reasonable range
        return max(0.0, min(1.0, hurst))
    except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
        # Return 0.5 (random walk) if calculation fails
        return 0.5
