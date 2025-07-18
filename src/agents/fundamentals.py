from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import json
import math

from src.tools.api import get_financial_metrics


##### Fundamental Agent #####
def fundamentals_analyst_agent(state: AgentState, agent_id: str = "fundamentals_analyst_agent"):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务指标")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status(agent_id, ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status(agent_id, ticker, "分析盈利能力")
        # 1. 客观盈利能力分析 - 适合短中期交易
        return_on_equity = metrics.return_on_equity or 0
        net_margin = metrics.net_margin or 0
        operating_margin = metrics.operating_margin or 0

        # 更合理的盈利能力评分系统
        profitability_score = 0
        profitability_details = []

        # ROE评分 (权重40%)
        if return_on_equity > 0.20:  # 优秀 >20%
            roe_score = 100
            profitability_details.append("ROE优秀")
        elif return_on_equity > 0.12:  # 良好 >12%
            roe_score = 75
            profitability_details.append("ROE良好")
        elif return_on_equity > 0.08:  # 一般 >8%
            roe_score = 50
            profitability_details.append("ROE一般")
        elif return_on_equity > 0:  # 正值
            roe_score = 25
            profitability_details.append("ROE偏低")
        else:  # 负值
            roe_score = 0
            profitability_details.append("ROE为负")

        # 净利润率评分 (权重35%)
        if net_margin > 0.15:  # 优秀 >15%
            margin_score = 100
            profitability_details.append("净利率优秀")
        elif net_margin > 0.10:  # 良好 >10%
            margin_score = 75
            profitability_details.append("净利率良好")
        elif net_margin > 0.05:  # 一般 >5%
            margin_score = 50
            profitability_details.append("净利率一般")
        elif net_margin > 0:  # 正值
            margin_score = 25
            profitability_details.append("净利率偏低")
        else:  # 负值
            margin_score = 0
            profitability_details.append("净利率为负")

        # 营业利润率评分 (权重25%)
        if operating_margin > 0.12:  # 优秀 >12%
            op_score = 100
            profitability_details.append("营业利率优秀")
        elif operating_margin > 0.08:  # 良好 >8%
            op_score = 75
            profitability_details.append("营业利率良好")
        elif operating_margin > 0.04:  # 一般 >4%
            op_score = 50
            profitability_details.append("营业利率一般")
        elif operating_margin > 0:  # 正值
            op_score = 25
            profitability_details.append("营业利率偏低")
        else:  # 负值
            op_score = 0
            profitability_details.append("营业利率为负")

        # 加权综合评分
        profitability_score = (roe_score * 0.4 + margin_score * 0.35 + op_score * 0.25)

        # 生成信号
        if profitability_score >= 75:
            prof_signal = "bullish"
        elif profitability_score >= 45:
            prof_signal = "neutral"
        else:
            prof_signal = "bearish"

        signals.append(prof_signal)
        reasoning["profitability_signal"] = {
            "signal": prof_signal,
            "score": round(profitability_score, 1),
            "details": f"ROE: {return_on_equity:.2%}, 净利率: {net_margin:.2%}, 营业利率: {operating_margin:.2%}",
            "highlights": profitability_details
        }

        progress.update_status(agent_id, ticker, "分析成长性")
        # 2. 客观成长性分析 - 更适合短中期交易
        revenue_growth = metrics.revenue_growth or 0
        earnings_growth = metrics.earnings_growth or 0
        book_value_growth = metrics.book_value_growth or 0

        # 成长性评分系统
        growth_score = 0
        growth_details = []

        # 营收增长评分 (权重50%)
        if revenue_growth > 0.25:  # 优秀 >25%
            rev_score = 100
            growth_details.append("营收高增长")
        elif revenue_growth > 0.15:  # 良好 >15%
            rev_score = 80
            growth_details.append("营收稳增长")
        elif revenue_growth > 0.05:  # 一般 >5%
            rev_score = 60
            growth_details.append("营收缓增长")
        elif revenue_growth > -0.05:  # 微降 >-5%
            rev_score = 40
            growth_details.append("营收微降")
        else:  # 大幅下降
            rev_score = 20
            growth_details.append("营收下降")

        # 盈利增长评分 (权重40%)
        if earnings_growth > 0.30:  # 优秀 >30%
            earn_score = 100
            growth_details.append("盈利高增长")
        elif earnings_growth > 0.15:  # 良好 >15%
            earn_score = 80
            growth_details.append("盈利稳增长")
        elif earnings_growth > 0:  # 正增长
            earn_score = 60
            growth_details.append("盈利正增长")
        elif earnings_growth > -0.15:  # 轻微下降
            earn_score = 30
            growth_details.append("盈利轻微下降")
        else:  # 大幅下降
            earn_score = 10
            growth_details.append("盈利大幅下降")

        # 净资产增长评分 (权重10%)
        if book_value_growth > 0.15:
            bv_score = 100
        elif book_value_growth > 0.05:
            bv_score = 70
        elif book_value_growth > 0:
            bv_score = 50
        else:
            bv_score = 20

        # 加权综合评分
        growth_score = (rev_score * 0.5 + earn_score * 0.4 + bv_score * 0.1)

        # 生成信号
        if growth_score >= 75:
            growth_signal = "bullish"
        elif growth_score >= 50:
            growth_signal = "neutral"
        else:
            growth_signal = "bearish"

        signals.append(growth_signal)
        reasoning["growth_signal"] = {
            "signal": growth_signal,
            "score": round(growth_score, 1),
            "details": f"营收增长: {revenue_growth:.2%}, 盈利增长: {earnings_growth:.2%}",
            "highlights": growth_details
        }

        progress.update_status(agent_id, ticker, "分析财务健康度")
        # 3. 客观财务健康度分析
        current_ratio = metrics.current_ratio or 0
        debt_to_equity = metrics.debt_to_equity or 0
        free_cash_flow_per_share = metrics.free_cash_flow_per_share or 0
        earnings_per_share = metrics.earnings_per_share or 0

        # 财务健康度评分系统
        health_score = 0
        health_details = []

        # 流动性评分 (权重40%)
        if current_ratio > 2.0:  # 优秀流动性
            liquidity_score = 100
            health_details.append("流动性优秀")
        elif current_ratio > 1.5:  # 良好流动性
            liquidity_score = 80
            health_details.append("流动性良好")
        elif current_ratio > 1.2:  # 一般流动性
            liquidity_score = 60
            health_details.append("流动性一般")
        elif current_ratio > 1.0:  # 基本流动性
            liquidity_score = 40
            health_details.append("流动性偏低")
        else:  # 流动性不足
            liquidity_score = 20
            health_details.append("流动性不足")

        # 杠杆评分 (权重35%)
        if debt_to_equity < 0.3:  # 低杠杆
            leverage_score = 100
            health_details.append("杠杆保守")
        elif debt_to_equity < 0.6:  # 适中杠杆
            leverage_score = 80
            health_details.append("杠杆适中")
        elif debt_to_equity < 1.0:  # 较高杠杆
            leverage_score = 60
            health_details.append("杠杆较高")
        elif debt_to_equity < 1.5:  # 高杠杆
            leverage_score = 30
            health_details.append("杠杆偏高")
        else:  # 极高杠杆
            leverage_score = 10
            health_details.append("杠杆过高")

        # 现金流评分 (权重25%)
        if free_cash_flow_per_share > 0 and earnings_per_share > 0:
            fcf_ratio = free_cash_flow_per_share / earnings_per_share
            if fcf_ratio > 1.2:  # 现金流超过盈利
                cashflow_score = 100
                health_details.append("现金流优秀")
            elif fcf_ratio > 0.8:  # 现金流良好
                cashflow_score = 80
                health_details.append("现金流良好")
            elif fcf_ratio > 0.5:  # 现金流一般
                cashflow_score = 60
                health_details.append("现金流一般")
            else:  # 现金流偏低
                cashflow_score = 40
                health_details.append("现金流偏低")
        elif free_cash_flow_per_share > 0:  # 有正现金流但无盈利数据
            cashflow_score = 60
            health_details.append("现金流为正")
        else:  # 负现金流
            cashflow_score = 20
            health_details.append("现金流为负")

        # 加权综合评分
        health_score = (liquidity_score * 0.4 + leverage_score * 0.35 + cashflow_score * 0.25)

        # 生成信号
        if health_score >= 75:
            health_signal = "bullish"
        elif health_score >= 50:
            health_signal = "neutral"
        else:
            health_signal = "bearish"

        signals.append(health_signal)
        reasoning["financial_health_signal"] = {
            "signal": health_signal,
            "score": round(health_score, 1),
            "details": f"流动比率: {current_ratio:.2f}, 负债权益比: {debt_to_equity:.2f}",
            "highlights": health_details
        }

        progress.update_status(agent_id, ticker, "分析估值合理性")
        # 4. 客观估值合理性分析 - 不过度偏向价值投资
        pe_ratio = metrics.price_to_earnings_ratio or 0
        pb_ratio = metrics.price_to_book_ratio or 0
        ps_ratio = metrics.price_to_sales_ratio or 0

        # 估值合理性评分系统（更宽松的标准）
        valuation_score = 0
        valuation_details = []

        # P/E评分 (权重40%) - 考虑成长股特点
        if pe_ratio <= 0:  # 负盈利或无数据
            pe_score = 30
            valuation_details.append("P/E无效")
        elif pe_ratio < 15:  # 低估值
            pe_score = 100
            valuation_details.append("P/E偏低")
        elif pe_ratio < 25:  # 合理估值
            pe_score = 80
            valuation_details.append("P/E合理")
        elif pe_ratio < 40:  # 较高估值但可接受
            pe_score = 60
            valuation_details.append("P/E偏高")
        elif pe_ratio < 60:  # 高估值
            pe_score = 40
            valuation_details.append("P/E较高")
        else:  # 极高估值
            pe_score = 20
            valuation_details.append("P/E过高")

        # P/B评分 (权重30%)
        if pb_ratio < 1.5:  # 低市净率
            pb_score = 100
            valuation_details.append("P/B偏低")
        elif pb_ratio < 3.0:  # 合理市净率
            pb_score = 80
            valuation_details.append("P/B合理")
        elif pb_ratio < 5.0:  # 较高市净率
            pb_score = 60
            valuation_details.append("P/B偏高")
        elif pb_ratio < 8.0:  # 高市净率
            pb_score = 40
            valuation_details.append("P/B较高")
        else:  # 极高市净率
            pb_score = 20
            valuation_details.append("P/B过高")

        # P/S评分 (权重30%)
        if ps_ratio < 2.0:  # 低市销率
            ps_score = 100
            valuation_details.append("P/S偏低")
        elif ps_ratio < 5.0:  # 合理市销率
            ps_score = 80
            valuation_details.append("P/S合理")
        elif ps_ratio < 10.0:  # 较高市销率
            ps_score = 60
            valuation_details.append("P/S偏高")
        elif ps_ratio < 15.0:  # 高市销率
            ps_score = 40
            valuation_details.append("P/S较高")
        else:  # 极高市销率
            ps_score = 20
            valuation_details.append("P/S过高")

        # 加权综合评分
        valuation_score = (pe_score * 0.4 + pb_score * 0.3 + ps_score * 0.3)

        # 生成信号（注意：高分表示估值合理，应该是正面信号）
        if valuation_score >= 75:
            valuation_signal = "bullish"
        elif valuation_score >= 50:
            valuation_signal = "neutral"
        else:
            valuation_signal = "bearish"

        signals.append(valuation_signal)
        reasoning["valuation_signal"] = {
            "signal": valuation_signal,
            "score": round(valuation_score, 1),
            "details": f"P/E: {pe_ratio:.2f}, P/B: {pb_ratio:.2f}, P/S: {ps_ratio:.2f}",
            "highlights": valuation_details
        }

        progress.update_status(agent_id, ticker, "计算综合信号")
        # 基于加权评分的综合信号计算

        # 获取各维度评分
        prof_score = reasoning["profitability_signal"]["score"]
        growth_score = reasoning["growth_signal"]["score"]
        health_score = reasoning["financial_health_signal"]["score"]
        val_score = reasoning["valuation_signal"]["score"]

        # 权重分配（适合短中期交易）
        weights = {
            "profitability": 0.30,  # 盈利能力最重要
            "growth": 0.35,         # 成长性权重最高
            "health": 0.20,         # 财务健康度
            "valuation": 0.15       # 估值合理性权重降低
        }

        # 计算加权综合评分
        overall_score = (
            prof_score * weights["profitability"] +
            growth_score * weights["growth"] +
            health_score * weights["health"] +
            val_score * weights["valuation"]
        )

        # 生成最终信号
        if overall_score >= 70:
            overall_signal = "bullish"
            confidence = min(85, 50 + (overall_score - 70) * 1.2)  # 50-85%
        elif overall_score >= 45:
            overall_signal = "neutral"
            confidence = 35 + (overall_score - 45) * 0.6  # 35-50%
        else:
            overall_signal = "bearish"
            confidence = min(80, 30 + (45 - overall_score) * 1.1)  # 30-80%

        confidence = round(confidence, 0)

        # 添加综合评分到reasoning
        reasoning["overall_analysis"] = {
            "overall_score": round(overall_score, 1),
            "weights_used": weights,
            "signal_logic": f"综合评分{overall_score:.1f}分，权重：成长性35%，盈利能力30%，财务健康20%，估值15%"
        }

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=json.dumps(reasoning, indent=4))

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = fundamental_analysis

    progress.update_status(agent_id, None, "完成")
    
    return {
        "messages": [message],
        "data": data,
    }
