import json
import logging
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.status_messages_zh import get_chinese_analyst_name
from .position_manager import position_manager

logger = logging.getLogger(__name__)


def get_dynamic_weights(signals_by_ticker: dict, trading_period: str = "short") -> dict:
    """
    根据交易周期和市场环境动态调整分析师权重
    """
    if trading_period == "short":
        # 短线交易权重配置
        base_weights = {
            "technical_analyst_agent": 0.40,      # 短线重技术
            "sentiment_analyst_agent": 0.30,      # 情绪驱动短线
            "fundamentals_analyst_agent": 0.15,   # 基本面降权
            "valuation_analyst_agent": 0.05,      # 估值最低权重
            "risk_management_agent": 0.10         # 风险管理
        }
    elif trading_period == "medium":
        # 中线交易权重配置 - 降低技术分析权重，提高风险控制
        base_weights = {
            "technical_analyst_agent": 0.25,      # 降低技术分析权重
            "sentiment_analyst_agent": 0.20,      # 情绪分析重要
            "fundamentals_analyst_agent": 0.25,   # 提高基本面权重
            "valuation_analyst_agent": 0.15,      # 提高估值权重
            "risk_management_agent": 0.15         # 大幅提高风险管理权重
        }
    else:  # long
        # 长线交易权重配置
        base_weights = {
            "technical_analyst_agent": 0.20,      # 技术分析降权
            "sentiment_analyst_agent": 0.15,      # 情绪影响较小
            "fundamentals_analyst_agent": 0.35,   # 基本面主导
            "valuation_analyst_agent": 0.25,      # 估值重要
            "risk_management_agent": 0.05         # 风险管理
        }

    # 根据信号质量动态调整权重
    adjusted_weights = base_weights.copy()

    for ticker, signals in signals_by_ticker.items():
        # 检查技术分析师信心度
        tech_confidence = signals.get("technical_analyst_agent", {}).get("confidence", 0)
        if tech_confidence < 30:  # 技术信号弱时
            # 降低技术分析权重，提高其他权重
            reduction = adjusted_weights["technical_analyst_agent"] * 0.3
            adjusted_weights["technical_analyst_agent"] -= reduction
            adjusted_weights["fundamentals_analyst_agent"] += reduction * 0.5
            adjusted_weights["sentiment_analyst_agent"] += reduction * 0.5

        # 检查估值分析师是否发出强烈警告
        val_confidence = signals.get("valuation_analyst_agent", {}).get("confidence", 0)
        val_signal = signals.get("valuation_analyst_agent", {}).get("signal", "neutral")

        # 当估值分析师高信心度发出看跌信号时，增加其权重（可能是泡沫警告）
        if val_confidence > 80 and val_signal == "bearish":
            # 从技术分析师转移权重到估值分析师
            reduction = adjusted_weights["technical_analyst_agent"] * 0.3
            adjusted_weights["technical_analyst_agent"] -= reduction
            adjusted_weights["valuation_analyst_agent"] += reduction * 0.7
            adjusted_weights["risk_management_agent"] += reduction * 0.3

    return adjusted_weights


def aggregate_analyst_signals(signals_by_ticker: dict, trading_period: str = "medium") -> dict:
    """
    聚合分析师信号，应用动态权重并生成综合评分
    """
    # 获取动态调整的权重
    analyst_weights = get_dynamic_weights(signals_by_ticker, trading_period)

    aggregated_signals = {}

    for ticker, signals in signals_by_ticker.items():
        signal_scores = []
        confidence_scores = []
        total_weight = 0

        # 统计各类信号数量
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for agent_id, signal_data in signals.items():
            weight = analyst_weights.get(agent_id, 0.1)  # 默认权重
            signal = signal_data.get("signal", "neutral")
            confidence = signal_data.get("confidence", 0)

            # 统计信号类型
            if signal == "bullish":
                bullish_count += 1
            elif signal == "bearish":
                bearish_count += 1
            else:
                neutral_count += 1

            # 转换信号为数值
            signal_value = {"bullish": 1, "neutral": 0, "bearish": -1}.get(signal, 0)

            # 加权评分
            weighted_score = signal_value * confidence * weight
            signal_scores.append(weighted_score)
            confidence_scores.append(confidence * weight)
            total_weight += weight

        if total_weight > 0:
            avg_signal_score = sum(signal_scores) / total_weight
            avg_confidence = sum(confidence_scores) / total_weight

            # 计算信号比例
            total_signals = bullish_count + bearish_count + neutral_count
            bearish_ratio = bearish_count / total_signals if total_signals > 0 else 0
            bullish_ratio = bullish_count / total_signals if total_signals > 0 else 0

            # 改进的信号判定逻辑 - 重视高信心度的警告信号
            # 检查是否有高信心度的看跌信号（可能是重要警告）
            high_confidence_bearish = False
            for agent_id, signal_data in signals.items():
                if (signal_data.get("signal") == "bearish" and
                    signal_data.get("confidence", 0) > 75 and
                    agent_id in ["valuation_analyst_agent", "risk_management_agent"]):
                    high_confidence_bearish = True
                    break

            # 1. 高信心度看跌警告优先
            if high_confidence_bearish and avg_signal_score > -20:
                aggregated_signal = "neutral"  # 至少保持中性，避免在警告下买入
            elif bearish_ratio >= 0.6:  # 降低看跌阈值
                aggregated_signal = "bearish"
            elif bullish_ratio >= 0.8:  # 提高看涨阈值，需要更强共识
                aggregated_signal = "bullish"
            # 2. 基于加权评分的精细判定（更保守的阈值）
            elif avg_signal_score > 25:  # 提高看涨阈值
                aggregated_signal = "bullish"
            elif avg_signal_score < -10:  # 降低看跌阈值
                aggregated_signal = "bearish"
            else:
                aggregated_signal = "neutral"

            aggregated_signals[ticker] = {
                "signal": aggregated_signal,
                "confidence": avg_confidence,
                "score": avg_signal_score,
                "bearish_ratio": bearish_ratio,
                "bullish_ratio": bullish_ratio,
                "signal_counts": {
                    "bullish": bullish_count,
                    "bearish": bearish_count,
                    "neutral": neutral_count
                }
            }
        else:
            aggregated_signals[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "score": 0,
                "bearish_ratio": 0,
                "bullish_ratio": 0,
                "signal_counts": {"bullish": 0, "bearish": 0, "neutral": 0}
            }

    return aggregated_signals


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="要交易的股份数量")
    confidence: float = Field(description="决策的置信度，介于0.0和100.0之间")
    reasoning: str = Field(description="决策的推理")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="股票代码到交易决策的字典")


##### 投资组合管理代理 #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """为多个股票代码做出最终交易决策并生成订单"""

    # 获取投资组合和分析师信号
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    # 获取每个股票代码的仓位限制、当前价格和信号
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "处理分析师信号")

        # 获取该股票代码的仓位限制和当前价格
        # 为此投资组合管理器找到对应的风险管理器
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split('_')[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # 旧版本的回退选项

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # 根据仓位限制和价格计算允许的最大股份数
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # 获取该股票代码的信号
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            # 跳过所有风险管理代理（它们有不同的信号结构）
            if not agent.startswith("risk_management_agent") and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status(agent_id, None, "聚合分析师信号")

    # 聚合分析师信号（默认中期交易模式，适合散户）
    aggregated_signals = aggregate_analyst_signals(signals_by_ticker, "medium")

    progress.update_status(agent_id, None, "生成交易决策")

    # 生成交易决策
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        aggregated_signals=aggregated_signals,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        agent_id=agent_id,
        state=state,
    )

    # 创建投资组合管理消息
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name=agent_id,
    )

    # 如果设置了标志，打印决策
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "投资组合经理")

    progress.update_status(agent_id, None, "完成")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    aggregated_signals: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    agent_id: str,
    state: AgentState,
) -> PortfolioManagerOutput:
    """尝试使用重试逻辑从LLM获取决策，集成智能持仓管理"""

    # 获取当前日期
    current_date = state["data"].get("current_date", "")
    if not current_date:
        # 如果没有当前日期，使用今天的日期
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

    # 为每个股票检查持仓管理建议
    position_recommendations = {}
    for ticker in tickers:
        current_price = current_prices.get(ticker, 0)
        if current_price > 0:
            # 获取价格历史数据用于异常波动分析
            price_history = None
            try:
                from src.tools.api import get_prices, prices_to_df
                from datetime import datetime, timedelta

                # 获取最近30天的价格数据用于波动分析
                start_date_for_volatility = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
                prices = get_prices(ticker, start_date_for_volatility, current_date)
                if prices:
                    price_history = prices_to_df(prices)
            except Exception as e:
                logger.warning(f"获取{ticker}价格历史失败: {e}")

            should_hold, reason = position_manager.should_hold_position(
                ticker, current_price, current_date, signals_by_ticker.get(ticker, {}), price_history
            )
            position_summary = position_manager.get_position_summary(ticker, current_price)
            position_recommendations[ticker] = {
                "should_hold": should_hold,
                "reason": reason,
                "position_summary": position_summary
            }
    # 创建提示模板
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一名投资组合经理，基于多个股票代码做出最终交易决策。请务必用中文进行分析和推理。

              分析师权重指导（按重要性排序）：
              1. 技术分析师 (35%权重) - 关注短期趋势和入场时机，在波动市场中最重要
              2. 情绪分析师 (25%权重) - 市场情绪和资金流向，特别关注内幕交易信号
              3. 基本面分析师 (25%权重) - 公司财务健康状况和盈利能力
              4. 估值分析师 (15%权重) - 长期价值评估，但在短期交易中权重较低

              交易策略指导：
              - 当64%以上信号看跌时，积极考虑空头策略
              - 技术分析师的看涨信号应给予更高权重，特别是在趋势转换时
              - 估值分析师过度悲观时（如93%折价），应适当降低其影响
              - 强烈看跌共识时，优先考虑做空而非持有

              智能持仓管理规则：
              - 最小持仓期：买入后至少持有3-14天（根据策略类型）
              - 止损机制：亏损8-15%时考虑止损（根据风险偏好）
              - 止盈机制：盈利15-40%时考虑止盈（根据策略类型）
              - 成本意识：优先考虑持仓成本和时间，避免频繁交易

              交易规则：
              - 对于多头仓位：
                * 只有在有可用现金时才能买入
                * 只有在当前持有该股票多头股份时才能卖出
                * 卖出数量必须 ≤ 当前多头仓位股份
                * 买入数量必须 ≤ 该股票的最大股份数
                * 必须考虑持仓管理建议，避免过早抛售

              - 对于空头仓位：
                * 只有在有可用保证金时才能做空（仓位价值 × 保证金要求）
                * 只有在当前持有该股票空头股份时才能平仓
                * 平仓数量必须 ≤ 当前空头仓位股份
                * 做空数量必须符合保证金要求

              - max_shares 值已预先计算以符合仓位限制
              - 基于信号考虑多头和空头机会
              - 通过多头和空头敞口维持适当的风险管理

              可用操作：
              - "buy": 开仓或增加多头仓位
              - "sell": 平仓或减少多头仓位
              - "short": 开仓或增加空头仓位
              - "cover": 平仓或减少空头仓位
              - "hold": 无操作

              输入参数：
              - signals_by_ticker: 股票代码 → 信号的字典
              - max_shares: 每个股票代码允许的最大股份数
              - portfolio_cash: 投资组合中的当前现金
              - portfolio_positions: 当前仓位（多头和空头）
              - current_prices: 每个股票代码的当前价格
              - margin_requirement: 空头仓位的当前保证金要求（例如，0.5 表示 50%）
              - total_margin_used: 当前使用的总保证金
              """,
            ),
            (
                "human",
                """基于团队的分析，为每个股票代码做出交易决策。请用中文进行推理和分析。

              📊 平衡决策原则（适合短中期交易）：
              1. 技术分析优先：短期交易中技术信号权重最高（35%）
              2. 情绪动量重要：市场情绪和资金流向影响短期走势（25%）
              3. 基本面支撑：提供中期方向指引，避免逆势操作（20%）
              4. 估值参考：长期价值锚定，但不主导短期决策（10%）
              5. 风险控制：保持适度仓位管理（10%）

              🎯 交易策略指导：
              - 强烈看涨信号（≥70%）→ 积极做多
              - 中等看涨信号（60-70%）→ 适度做多
              - 中性信号 → 持有或小幅调仓
              - 中等看跌信号（60-70%）→ 适度做空或减仓
              - 强烈看跌信号（≥70%）→ 积极做空

              各股票代码的详细信号：
              {signals_by_ticker}

              聚合信号分析：
              {aggregated_signals}

              当前价格：
              {current_prices}

              购买允许的最大股份数：
              {max_shares}

              投资组合现金：{portfolio_cash}
              当前仓位：{portfolio_positions}
              当前保证金要求：{margin_requirement}
              已使用总保证金：{total_margin_used}

              🎯 持仓管理建议：
              {position_recommendations}

              🔄 智能持仓决策逻辑：
              1. 有持仓时优先考虑持仓管理建议，避免频繁交易
              2. 强烈看涨（≥70%）→ 积极做多（使用60-80%最大仓位）
              3. 中等看涨（60-70%）→ 适度做多（使用30-50%最大仓位）
              4. 中性信号 → 持有现有仓位，避免无意义交易
              5. 中等看跌（60-70%）→ 适度做空或减仓（使用30-50%最大仓位）
              6. 强烈看跌（≥70%）→ 积极做空（使用60-80%最大仓位）

              🎯 持仓优先原则：
              - 如果持仓管理建议"继续持有"，除非有强烈反向信号，否则保持持有
              - 如果持仓亏损但未达止损线，在没有基本面恶化时耐心等待
              - 如果持仓盈利，在趋势未明确转向时避免过早获利了结

              ⚖️ 平衡原则：
              - 技术分析信号强度决定仓位大小
              - 情绪分析提供入场时机
              - 基本面分析避免逆势操作
              - 估值分析提供安全边际参考

              严格按照以下JSON结构输出：
              {{
                "decisions": {{
                  "TICKER1": {{
                    "action": "buy/sell/short/cover/hold",
                    "quantity": 整数,
                    "confidence": 0到100之间的浮点数,
                    "reasoning": "中文推理字符串，必须说明：1)信号比例分析 2)权重调整逻辑 3)具体决策依据"
                  }},
                  "TICKER2": {{
                    ...
                  }},
                  ...
                }}
              }}
              """,
            ),
        ]
    )

    # 在传递给LLM之前将分析师名称转换为中文
    chinese_signals_by_ticker = {}
    for ticker, signals in signals_by_ticker.items():
        chinese_signals = {}
        for agent_id, signal_data in signals.items():
            chinese_name = get_chinese_analyst_name(agent_id)
            chinese_signals[chinese_name] = signal_data
        chinese_signals_by_ticker[ticker] = chinese_signals

    # 生成提示
    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(chinese_signals_by_ticker, indent=2),
            "aggregated_signals": json.dumps(aggregated_signals, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get("positions", {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
            "total_margin_used": f"{portfolio.get('margin_used', 0):.2f}",
            "position_recommendations": json.dumps(position_recommendations, indent=2),
        }
    )

    # 为PortfolioManagerOutput创建默认工厂
    def create_default_portfolio_output():
        return PortfolioManagerOutput(decisions={ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="投资组合管理出错，默认持有") for ticker in tickers})

    # 获取LLM的初始决策
    llm_result = call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )

    # 应用持仓管理逻辑覆盖
    final_decisions = {}
    for ticker in tickers:
        llm_decision = llm_result.decisions.get(ticker)
        if llm_decision and ticker in position_recommendations:
            final_decision = apply_position_management_override(
                ticker, llm_decision, position_recommendations[ticker],
                current_prices.get(ticker, 0)
            )
            final_decisions[ticker] = final_decision
        else:
            final_decisions[ticker] = llm_decision

    return PortfolioManagerOutput(decisions=final_decisions)


def apply_position_management_override(ticker: str, llm_decision: PortfolioDecision,
                                     position_rec: dict, current_price: float) -> PortfolioDecision:
    """
    应用持仓管理逻辑覆盖LLM决策
    """
    should_hold = position_rec["should_hold"]
    reason = position_rec["reason"]
    position_summary = position_rec["position_summary"]

    # 如果没有持仓，直接返回LLM决策
    if not position_summary["has_position"]:
        return llm_decision

    # 如果持仓管理建议继续持有
    if should_hold:
        # 检查LLM是否建议卖出
        if llm_decision.action in ["sell", "cover"]:
            # 检查是否是强烈的卖出信号
            if llm_decision.confidence > 80:
                # 高信心度卖出信号，允许覆盖持仓管理建议
                new_reasoning = f"持仓管理建议持有({reason})，但AI高信心度({llm_decision.confidence:.1f}%)建议卖出，执行卖出。原因：{llm_decision.reasoning}"
                return PortfolioDecision(
                    action=llm_decision.action,
                    quantity=llm_decision.quantity,
                    confidence=llm_decision.confidence * 0.9,  # 略微降低信心度
                    reasoning=new_reasoning
                )
            else:
                # 低信心度卖出信号，持仓管理优先
                new_reasoning = f"持仓管理建议继续持有({reason})，覆盖AI的低信心度卖出建议。持仓情况：{position_summary['unrealized_pnl_pct']:.1f}%盈亏，持有{position_summary['holding_days']}天"
                return PortfolioDecision(
                    action="hold",
                    quantity=0,
                    confidence=70.0,
                    reasoning=new_reasoning
                )
        else:
            # LLM建议持有或买入，与持仓管理一致
            return llm_decision

    else:
        # 持仓管理建议卖出（触发止损/止盈等）
        if llm_decision.action in ["buy", "short"]:
            # LLM建议买入但持仓管理建议卖出，优先卖出现有持仓
            new_reasoning = f"持仓管理建议卖出({reason})，优先处理现有持仓而非新建仓位"
            return PortfolioDecision(
                action="sell",
                quantity=position_summary["shares"],
                confidence=85.0,
                reasoning=new_reasoning
            )
        else:
            # LLM也建议卖出，增强信心度
            new_reasoning = f"持仓管理建议卖出({reason})，与AI决策一致。{llm_decision.reasoning}"
            return PortfolioDecision(
                action=llm_decision.action,
                quantity=max(llm_decision.quantity, position_summary["shares"]),
                confidence=min(95.0, llm_decision.confidence + 10),
                reasoning=new_reasoning
            )
