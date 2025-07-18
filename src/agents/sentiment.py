from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 设置日志
logger = logging.getLogger(__name__)

from src.tools.api import get_prices
from src.tools.index_scraper import get_market_index_data


def calculate_real_market_relative_strength(target_ticker: str, start_date: str, end_date: str) -> dict:
    """
    使用真实大盘指数数据计算相对强度
    """
    try:
        logger.info(f"计算 {target_ticker} 相对于真实大盘指数的相对强度")

        # 获取目标股票数据
        target_data = get_prices(target_ticker, start_date, end_date)
        if not target_data or len(target_data) < 2:
            return {
                "signal": "neutral",
                "confidence": 20,
                "metrics": {
                    "error": f"Insufficient data for target stock {target_ticker}",
                    "data_points": len(target_data) if target_data else 0
                }
            }

        # 获取大盘指数数据
        index_data = get_market_index_data()
        if not index_data or not index_data.get('indices'):
            logger.warning("无法获取大盘指数数据，使用备用方案")
            return calculate_market_benchmark_return(target_ticker, start_date, end_date)

        # 计算目标股票收益率
        target_start_price = target_data[0].close
        target_end_price = target_data[-1].close
        target_return = (target_end_price - target_start_price) / target_start_price

        # 计算大盘指数收益率（使用当日表现作为近期表现的代理）
        indices = index_data['indices']
        market_returns = {}

        for index_code, data in indices.items():
            # 使用当日涨跌幅作为近期表现
            daily_return = data['change_percent'] / 100
            market_returns[index_code] = {
                'name': data['name'],
                'return': daily_return,
                'current_price': data['current_price'],
                'change': data['change']
            }

        # 计算综合市场表现（等权重平均）
        market_return = sum(perf['return'] for perf in market_returns.values()) / len(market_returns)

        # 计算相对强度
        relative_strength = target_return - market_return

        # 生成信号和信心度
        if relative_strength > 0.02:  # 跑赢大盘2%以上
            signal = "bullish"
            confidence = min(80, 50 + abs(relative_strength) * 300)
        elif relative_strength < -0.02:  # 跑输大盘2%以上
            signal = "bearish"
            confidence = min(80, 50 + abs(relative_strength) * 300)
        else:  # 表现接近大盘
            signal = "neutral"
            confidence = 45

        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "metrics": {
                "target_return": round(target_return, 4),
                "market_return": round(market_return, 4),
                "relative_strength": round(relative_strength, 4),
                "market_indices": market_returns,
                "data_source": "real_market_indices",
                "data_quality": f"{len(target_data)} data points, real-time market data"
            }
        }

    except Exception as e:
        logger.error(f"真实大盘相对强度计算失败: {e}")
        # 备用方案：使用股票基准
        return calculate_market_benchmark_return(target_ticker, start_date, end_date)


def calculate_market_benchmark_return(target_ticker: str, start_date: str, end_date: str) -> dict:
    """
    使用可用股票构建市场基准，计算相对强度
    使用AAPL、MSFT、GOOGL、NVDA、TSLA构建等权重虚拟市场指数
    """
    # 可用的股票列表（我们的数据源支持的股票）
    available_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

    # 从基准中排除目标股票，避免自我比较
    benchmark_stocks = [stock for stock in available_stocks if stock != target_ticker]

    try:
        # 获取目标股票数据
        target_data = get_prices(target_ticker, start_date, end_date)
        if not target_data or len(target_data) < 2:
            return {
                "signal": "neutral",
                "confidence": 20,
                "metrics": {
                    "error": f"Insufficient data for target stock {target_ticker}",
                    "benchmark_stocks": benchmark_stocks,
                    "data_points": len(target_data) if target_data else 0
                }
            }

        # 获取基准股票数据
        benchmark_data = {}
        successful_benchmarks = []

        for stock in benchmark_stocks:
            try:
                stock_data = get_prices(stock, start_date, end_date)
                if stock_data and len(stock_data) >= 2:
                    benchmark_data[stock] = stock_data
                    successful_benchmarks.append(stock)
            except Exception as e:
                print(f"Failed to get data for benchmark stock {stock}: {e}")
                continue

        if len(successful_benchmarks) < 2:
            return {
                "signal": "neutral",
                "confidence": 25,
                "metrics": {
                    "error": "Insufficient benchmark data",
                    "successful_benchmarks": successful_benchmarks,
                    "attempted_benchmarks": benchmark_stocks
                }
            }

        # 计算目标股票收益率
        target_start_price = target_data[0].close
        target_end_price = target_data[-1].close
        target_return = (target_end_price - target_start_price) / target_start_price

        # 计算基准指数收益率（等权重平均）
        benchmark_returns = []
        benchmark_details = {}

        for stock in successful_benchmarks:
            stock_data = benchmark_data[stock]
            start_price = stock_data[0].close
            end_price = stock_data[-1].close
            stock_return = (end_price - start_price) / start_price
            benchmark_returns.append(stock_return)
            benchmark_details[stock] = {
                "return": round(stock_return, 4),
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2)
            }

        # 计算等权重基准收益率
        benchmark_return = sum(benchmark_returns) / len(benchmark_returns)

        # 计算相对强度
        relative_strength = target_return - benchmark_return

        # 生成信号和信心度
        if relative_strength > 0.02:  # 跑赢基准2%以上
            signal = "bullish"
            confidence = min(70, 40 + abs(relative_strength) * 300)
        elif relative_strength < -0.02:  # 跑输基准2%以上
            signal = "bearish"
            confidence = min(70, 40 + abs(relative_strength) * 300)
        else:  # 表现接近基准
            signal = "neutral"
            confidence = 35

        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "metrics": {
                "target_return": round(target_return, 4),
                "benchmark_return": round(benchmark_return, 4),
                "relative_strength": round(relative_strength, 4),
                "benchmark_composition": successful_benchmarks,
                "benchmark_details": benchmark_details,
                "data_quality": f"{len(target_data)} data points, {len(successful_benchmarks)} benchmark stocks"
            }
        }

    except Exception as e:
        return {
            "signal": "neutral",
            "confidence": 20,
            "metrics": {
                "error": f"Failed to calculate market benchmark: {str(e)}",
                "target_ticker": target_ticker,
                "benchmark_stocks": benchmark_stocks
            }
        }


def calculate_market_momentum_sentiment(ticker: str, end_date: str) -> dict:
    """
    基于市场动量的客观情绪指标
    """
    try:
        # 尝试从状态中获取已有的价格数据
        # 如果API失败，使用技术分析师已获取的数据
        from src.tools.api import get_prices

        # 获取短期价格数据（用于计算动量）
        end_date_dt = pd.Timestamp(end_date)
        start_date = (end_date_dt - pd.Timedelta(days=30)).strftime('%Y-%m-%d')

        try:
            prices_data = get_prices(ticker, start_date, end_date)
        except:
            # 如果API失败，返回基于简单逻辑的默认情绪
            return {
                "signal": "neutral",
                "confidence": 25,
                "metrics": {
                    "fallback_reason": "API unavailable, using neutral sentiment",
                    "price_momentum": 0,
                    "volume_momentum": 0,
                    "volatility": 0.2,
                    "consecutive_days": 0,
                    "momentum_score": 0
                }
            }

        if not prices_data:
            return {"signal": "neutral", "confidence": 25, "metrics": {"error": "No price data"}}

        # 转换为DataFrame格式
        df = pd.DataFrame([{
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'close': p.close,
            'volume': p.volume,
            'time': p.time
        } for p in prices_data])
        if len(df) < 5:
            return {"signal": "neutral", "confidence": 25, "metrics": {"error": "Insufficient data"}}

        # 计算多个动量指标
        close = df['close']
        volume = df['volume']

        # 1. 价格动量（最近5天 vs 前5天）
        recent_avg = close.tail(5).mean()
        previous_avg = close.iloc[-10:-5].mean() if len(close) >= 10 else close.head(5).mean()
        price_momentum = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0

        # 2. 成交量动量（最近5天 vs 前5天）
        recent_vol = volume.tail(5).mean()
        previous_vol = volume.iloc[-10:-5].mean() if len(volume) >= 10 else volume.head(5).mean()
        volume_momentum = (recent_vol - previous_vol) / previous_vol if previous_vol > 0 else 0

        # 3. 价格波动率（反映市场不确定性）
        returns = close.pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0

        # 4. 连续上涨/下跌天数
        price_changes = close.diff().dropna()
        consecutive_days = 0
        if len(price_changes) > 0:
            current_trend = 1 if price_changes.iloc[-1] > 0 else -1
            for i in range(len(price_changes)-1, -1, -1):
                if (price_changes.iloc[i] > 0) == (current_trend > 0):
                    consecutive_days += 1
                else:
                    break
            consecutive_days *= current_trend

        # 综合评分
        momentum_score = 0
        confidence_factors = []

        # 价格动量评分
        if price_momentum > 0.02:  # 2%以上涨幅
            momentum_score += 30
            confidence_factors.append(abs(price_momentum) * 100)
        elif price_momentum < -0.02:  # 2%以上跌幅
            momentum_score -= 30
            confidence_factors.append(abs(price_momentum) * 100)

        # 成交量确认
        if volume_momentum > 0.2:  # 成交量增加20%以上
            if price_momentum > 0:
                momentum_score += 20  # 量价齐升
            else:
                momentum_score -= 15  # 量增价跌（可能是抛售）
            confidence_factors.append(min(volume_momentum * 50, 30))

        # 连续趋势加分
        if abs(consecutive_days) >= 3:
            momentum_score += consecutive_days * 5
            confidence_factors.append(min(abs(consecutive_days) * 10, 25))

        # 波动率调整（高波动率降低信心度）
        volatility_penalty = min(volatility * 100, 20)

        # 生成信号
        if momentum_score > 25:
            signal = "bullish"
        elif momentum_score < -25:
            signal = "bearish"
        else:
            signal = "neutral"

        # 计算信心度
        base_confidence = min(abs(momentum_score), 50)
        if confidence_factors:
            avg_confidence_factor = sum(confidence_factors) / len(confidence_factors)
            final_confidence = min((base_confidence + avg_confidence_factor) / 2 - volatility_penalty, 85)
        else:
            final_confidence = max(base_confidence - volatility_penalty, 15)

        return {
            "signal": signal,
            "confidence": max(final_confidence, 15),
            "metrics": {
                "price_momentum": round(price_momentum, 4),
                "volume_momentum": round(volume_momentum, 4),
                "volatility": round(volatility, 4),
                "consecutive_days": consecutive_days,
                "momentum_score": momentum_score
            }
        }

    except Exception as e:
        return {"signal": "neutral", "confidence": 15, "metrics": {"error": str(e)}}


def calculate_relative_strength_sentiment(ticker: str, end_date: str) -> dict:
    """
    相对强度情绪指标 - 优先使用真实大盘指数数据
    """
    try:
        # 计算分析时间窗口
        end_date_dt = pd.Timestamp(end_date)
        start_date = (end_date_dt - pd.Timedelta(days=20)).strftime('%Y-%m-%d')

        # 优先使用真实大盘指数数据
        result = calculate_real_market_relative_strength(ticker, start_date, end_date)

        return result

    except Exception as e:
        return {
            "signal": "neutral",
            "confidence": 20,
            "metrics": {
                "error": f"Relative strength calculation failed: {str(e)}",
                "ticker": ticker,
                "date_range": f"{start_date} to {end_date}"
            }
        }


def calculate_volume_price_sentiment(ticker: str, end_date: str) -> dict:
    """
    成交量价格关系情绪指标
    """
    try:
        end_date_dt = pd.Timestamp(end_date)
        start_date = (end_date_dt - pd.Timedelta(days=15)).strftime('%Y-%m-%d')

        prices_data = get_prices(ticker, start_date, end_date)
        if not prices_data:
            return {"signal": "neutral", "confidence": 20, "metrics": {"error": "No volume data"}}

        # 转换为DataFrame格式
        df = pd.DataFrame([{
            'close': p.close,
            'volume': p.volume,
            'time': p.time
        } for p in prices_data])
        if len(df) < 5:
            return {"signal": "neutral", "confidence": 20, "metrics": {"error": "Insufficient volume data"}}

        # 成交量趋势分析
        volume = df['volume']
        close = df['close']

        # 最近3天 vs 平均成交量
        recent_volume = volume.tail(3).mean()
        avg_volume = volume.mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        # 价格变化
        price_change = (close.iloc[-1] - close.iloc[-3]) / close.iloc[-3] if len(close) >= 3 else 0

        # 量价关系分析
        if volume_ratio > 1.5 and price_change > 0.02:  # 量价齐升
            signal = "bullish"
            confidence = min((volume_ratio - 1) * 40 + abs(price_change) * 200, 70)
        elif volume_ratio > 1.5 and price_change < -0.02:  # 量增价跌
            signal = "bearish"
            confidence = min((volume_ratio - 1) * 30 + abs(price_change) * 150, 65)
        elif volume_ratio < 0.7:  # 成交量萎缩
            signal = "neutral"  # 缺乏动力
            confidence = 30
        else:
            signal = "neutral"
            confidence = 25

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "volume_ratio": round(volume_ratio, 2),
                "price_change": round(price_change, 4),
                "recent_volume": int(recent_volume),
                "avg_volume": int(avg_volume)
            }
        }

    except Exception as e:
        return {"signal": "neutral", "confidence": 20, "metrics": {"error": str(e)}}


##### 重新设计的情绪分析代理 #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """基于客观市场指标分析股票的市场情绪。"""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")

    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "计算市场动量情绪")

        # 1. 市场动量情绪
        momentum_sentiment = calculate_market_momentum_sentiment(ticker, end_date)

        progress.update_status(agent_id, ticker, "计算相对强度情绪")

        # 2. 相对强度情绪
        relative_sentiment = calculate_relative_strength_sentiment(ticker, end_date)

        progress.update_status(agent_id, ticker, "分析量价关系")

        # 3. 成交量价格关系情绪
        volume_sentiment = calculate_volume_price_sentiment(ticker, end_date)

        progress.update_status(agent_id, ticker, "综合情绪分析")

        # 综合三个情绪指标
        signals = [momentum_sentiment, relative_sentiment, volume_sentiment]

        # 权重分配：动量40%，相对强度35%，量价关系25%
        weights = [0.4, 0.35, 0.25]

        # 计算加权信号
        bullish_score = 0
        bearish_score = 0
        total_confidence = 0

        for signal_data, weight in zip(signals, weights):
            confidence = signal_data["confidence"] / 100
            if signal_data["signal"] == "bullish":
                bullish_score += weight * confidence
            elif signal_data["signal"] == "bearish":
                bearish_score += weight * confidence
            total_confidence += weight * confidence

        # 生成最终信号
        if bullish_score > bearish_score and bullish_score > 0.25:
            final_signal = "bullish"
            final_confidence = min(bullish_score * 120, 80)
        elif bearish_score > bullish_score and bearish_score > 0.25:
            final_signal = "bearish"
            final_confidence = min(bearish_score * 120, 80)
        else:
            final_signal = "neutral"
            final_confidence = max(total_confidence * 60, 20)

        sentiment_analysis[ticker] = {
            "signal": final_signal,
            "confidence": round(final_confidence, 2),
            "reasoning": {
                "market_momentum": momentum_sentiment,
                "relative_strength": relative_sentiment,
                "volume_price_analysis": volume_sentiment,
                "combined_analysis": {
                    "bullish_score": round(bullish_score, 3),
                    "bearish_score": round(bearish_score, 3),
                    "signal_determination": f"{final_signal.capitalize()} based on objective market indicators"
                }
            }
        }

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "完成")

    return {
        "messages": [message],
        "data": data,
    }
