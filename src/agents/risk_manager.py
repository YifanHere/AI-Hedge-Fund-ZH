from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.tools.api import get_prices, prices_to_df
import json
import math


def _assess_technical_risk(df, current_price):
    """
    评估技术风险，识别潜在的顶部和高风险区域
    返回风险评分 (0-50)
    """
    risk_score = 0

    try:
        # 1. 历史高点风险 (0-20分)
        if len(df) >= 60:
            recent_high_60d = df["high"].rolling(60).max().iloc[-1]
            recent_high_120d = df["high"].rolling(120).max().iloc[-1] if len(df) >= 120 else recent_high_60d

            # 接近60天高点 - 更严格的风险评估
            if current_price >= recent_high_60d * 0.95:
                risk_score += 20  # 提高风险评分
            elif current_price >= recent_high_60d * 0.90:
                risk_score += 15
            elif current_price >= recent_high_60d * 0.85:
                risk_score += 10

            # 接近120天高点（更高风险）
            if current_price >= recent_high_120d * 0.98:
                risk_score += 5

        # 2. RSI超买风险 (0-25分) - 更严格的RSI风险评估
        if len(df) >= 14:
            from src.agents.technicals import calculate_rsi
            rsi = calculate_rsi(df, 14)
            current_rsi = rsi.iloc[-1]

            if current_rsi > 70:
                risk_score += 25  # 极度超买
            elif current_rsi > 65:
                risk_score += 20
            elif current_rsi > 60:
                risk_score += 15  # 降低超买阈值
            elif current_rsi > 55:
                risk_score += 10  # 进一步降低阈值
            elif current_rsi > 50:
                risk_score += 5   # 轻度风险

        # 3. 价格偏离均线风险 (0-15分)
        if len(df) >= 50:
            sma_50 = df["close"].rolling(50).mean().iloc[-1]
            price_deviation = (current_price - sma_50) / sma_50

            if price_deviation > 0.3:  # 超过50日均线30%
                risk_score += 15
            elif price_deviation > 0.2:  # 超过50日均线20%
                risk_score += 10
            elif price_deviation > 0.1:  # 超过50日均线10%
                risk_score += 5

    except Exception:
        pass

    return min(risk_score, 50)  # 最大50分技术风险


def calculate_risk_signal(ticker, current_price, portfolio_value, position_value, position_limit, available_cash, market_trend_signal=None):
    """
    计算风险管理信号，考虑市场趋势和技术风险
    返回: (signal, confidence)
    """
    # 获取历史价格数据进行技术风险评估
    from src.tools.api import get_prices
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")  # 4个月历史数据

    try:
        prices = get_prices(ticker, start_date, end_date)
        if prices and len(prices) >= 60:
            import pandas as pd
            df = pd.DataFrame([p.model_dump() for p in prices])
            df["Date"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df.set_index("Date", inplace=True)
            numeric_cols = ["open", "close", "high", "low", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.sort_index(inplace=True)

            # 技术风险评估
            technical_risk_score = _assess_technical_risk(df, current_price)
        else:
            technical_risk_score = 0
    except Exception:
        technical_risk_score = 0
    # 计算关键风险指标
    position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
    cash_ratio = available_cash / portfolio_value if portfolio_value > 0 else 1
    utilization_ratio = position_value / position_limit if position_limit > 0 else 0

    # 基础风险评分 (0-100) + 技术风险评分 (0-50)
    risk_score = technical_risk_score

    # 仓位集中度风险 (0-40分) - 调整为更积极
    if position_ratio > 0.6:  # 超过60%集中度（提高阈值）
        risk_score += 40
    elif position_ratio > 0.4:  # 40-60%集中度（提高阈值）
        risk_score += 25
    elif position_ratio > 0.25:  # 25-40%集中度（提高阈值）
        risk_score += 10

    # 现金比例风险 (0-30分) - 在明确趋势下放宽
    if cash_ratio < 0.05:  # 现金不足5%（降低阈值）
        risk_score += 30
    elif cash_ratio < 0.1:  # 现金不足10%（降低阈值）
        risk_score += 15
    elif cash_ratio < 0.2:  # 现金不足20%（降低阈值）
        risk_score += 5

    # 仓位利用率风险 (0-30分) - 更积极的利用率
    if utilization_ratio > 0.95:  # 超过95%利用率（提高阈值）
        risk_score += 30
    elif utilization_ratio > 0.8:  # 80-95%利用率（提高阈值）
        risk_score += 15
    elif utilization_ratio > 0.6:  # 60-80%利用率（提高阈值）
        risk_score += 5

    # 市场趋势调整 - 在明确趋势下降低风险评分
    if market_trend_signal:
        if market_trend_signal == "bearish" and position_ratio < 0.3:
            # 在看跌趋势下，如果仓位不高，鼓励做空
            risk_score = max(0, risk_score - 20)
        elif market_trend_signal == "bullish" and position_ratio < 0.3:
            # 在看涨趋势下，如果仓位不高，鼓励买入
            risk_score = max(0, risk_score - 15)

    # 生成信号 - 更严格的风险控制，特别是技术风险
    if risk_score >= 60:  # 降低高风险阈值（包含技术风险）
        signal = "bearish"  # 高风险，建议减仓
        confidence = min(risk_score / 100, 0.9)
    elif risk_score >= 30:  # 降低中等风险阈值
        signal = "neutral"  # 中等风险，保持谨慎
        confidence = 0.4 + (risk_score - 30) / 30 * 0.3
    else:
        signal = "bullish"  # 低风险，可以增仓
        confidence = 0.3 + (30 - risk_score) / 30 * 0.4

    return signal, confidence


##### 风险管理代理 #####
def risk_management_agent(state: AgentState, agent_id: str = "risk_management_agent"):
    """基于现实世界风险因素控制多个股票代码的仓位规模。"""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # 为每个股票代码初始化风险分析
    risk_analysis = {}
    current_prices = {}  # 在此存储价格以避免冗余API调用

    # 首先，获取所有相关股票代码的价格
    all_tickers = set(tickers) | set(portfolio.get("positions", {}).keys())

    for ticker in all_tickers:
        progress.update_status(agent_id, ticker, "Fetching price data")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status(agent_id, ticker, "Warning: No price data found")
            continue

        prices_df = prices_to_df(prices)

        if not prices_df.empty:
            current_price = prices_df["close"].iloc[-1]
            current_prices[ticker] = current_price
            progress.update_status(agent_id, ticker, f"Current price: {current_price}")
        else:
            progress.update_status(agent_id, ticker, "Warning: Empty price data")

    # 基于当前市场价格计算总投资组合价值（净清算价值）
    total_portfolio_value = portfolio.get("cash", 0.0)

    for ticker, position in portfolio.get("positions", {}).items():
        if ticker in current_prices:
            current_price = current_prices[ticker]

            # 添加多头仓位的市场价值
            long_value = position.get("long", 0) * current_price
            total_portfolio_value += long_value

            # 空头仓位：减去当前需要归还的股票价值（负债）
            short_shares = position.get("short", 0)
            if short_shares > 0:
                # 当前做空负债 = 需要归还的股票数量 × 当前价格
                short_liability = short_shares * current_price
                total_portfolio_value -= short_liability

    progress.update_status(agent_id, None, f"Total portfolio value: {total_portfolio_value}")

    # 为投资范围内的每个股票代码计算风险限制
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Calculating position limits")

        if ticker not in current_prices:
            progress.update_status(agent_id, ticker, "Failed: No price data available")
            risk_analysis[ticker] = {
                "remaining_position_limit": 0.0,
                "current_price": 0.0,
                "reasoning": {
                    "error": "Missing price data for risk calculation"
                }
            }
            continue

        current_price = current_prices[ticker]

        # 计算此仓位的当前市场价值
        position = portfolio.get("positions", {}).get(ticker, {})
        long_value = position.get("long", 0) * current_price
        short_value = position.get("short", 0) * current_price
        current_position_value = abs(long_value - short_value)  # 使用绝对敞口

        # 动态仓位限制：根据市场条件调整
        base_position_limit = total_portfolio_value * 0.25  # 提高基础限制到25%

        # 根据投资组合表现调整仓位限制
        initial_capital = 100000  # 应该从配置中获取
        portfolio_performance = total_portfolio_value / initial_capital

        if portfolio_performance > 1.1:  # 盈利超过10%时，允许更大仓位
            position_limit = base_position_limit * 1.2
        elif portfolio_performance < 0.9:  # 亏损超过10%时，减少仓位
            position_limit = base_position_limit * 0.8
        else:
            position_limit = base_position_limit

        # 计算此仓位的剩余限制
        remaining_position_limit = position_limit - current_position_value

        # 确保不超过可用现金，但为空头交易预留更多空间
        available_cash = portfolio.get("cash", 0)
        max_position_size = min(remaining_position_limit, available_cash * 1.5)  # 允许使用150%现金（考虑保证金）
        
        # 添加风险信号生成逻辑，考虑市场趋势
        # 尝试从状态中获取市场趋势信号
        market_trend = None
        if "analyst_signals" in state.get("data", {}):
            analyst_signals = state["data"]["analyst_signals"]
            # 从技术分析师获取趋势信号
            if "technical_analyst_agent" in analyst_signals:
                tech_signals = analyst_signals["technical_analyst_agent"]
                if ticker in tech_signals:
                    market_trend = tech_signals[ticker].get("signal")

        risk_signal, risk_confidence = calculate_risk_signal(
            ticker, current_price, total_portfolio_value, current_position_value,
            position_limit, available_cash, market_trend
        )

        risk_analysis[ticker] = {
            "signal": risk_signal,
            "confidence": risk_confidence,
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position_value": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "risk_assessment": f"风险信号: {risk_signal}, 信心度: {risk_confidence:.1%}",
            },
        }
        
        progress.update_status(agent_id, ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # 将信号添加到分析师信号列表
    state["data"]["analyst_signals"][agent_id] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
