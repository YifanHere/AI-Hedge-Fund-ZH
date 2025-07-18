from __future__ import annotations

"""优化的估值分析师

实现多元化估值方法，更适合短中期交易，减少极端估值偏见
"""

from statistics import median
import json
import math
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress

from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)

def valuation_analyst_agent(state: AgentState, agent_id: str = "valuation_analyst_agent"):
    """Run valuation across tickers and write signals back to `state`."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    valuation_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务数据")

        # --- Historical financial metrics (pull 8 latest TTM snapshots for medians) ---
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=8,
        )
        if not financial_metrics:
            progress.update_status(agent_id, ticker, "Failed: No financial metrics found")
            continue
        most_recent_metrics = financial_metrics[0]

        # --- Fine‑grained line‑items (need two periods to calc WC change) ---
        progress.update_status(agent_id, ticker, "收集财务项目")
        line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
            ],
            end_date=end_date,
            period="ttm",
            limit=2,
        )
        if len(line_items) < 2:
            progress.update_status(agent_id, ticker, "Failed: Insufficient financial line items")
            continue
        li_curr, li_prev = line_items[0], line_items[1]

        # ------------------------------------------------------------------
        # Valuation models
        # ------------------------------------------------------------------
        wc_change = li_curr.working_capital - li_prev.working_capital

        # 优化的所有者收益模型
        earnings_growth = most_recent_metrics.earnings_growth or 0.05
        adjusted_growth_rate = min(max(earnings_growth, 0.03), 0.15)  # 与DCF保持一致

        owner_val = calculate_owner_earnings_value(
            net_income=li_curr.net_income,
            depreciation=li_curr.depreciation_and_amortization,
            capex=li_curr.capital_expenditure,
            working_capital_change=wc_change,
            growth_rate=adjusted_growth_rate,
        )

        # Discounted Cash Flow
        # 优化的DCF估值 - 更适合成长股
        earnings_growth = most_recent_metrics.earnings_growth or 0.05
        # 对于高增长股，使用更合理的增长率
        adjusted_growth_rate = min(max(earnings_growth, 0.03), 0.15)  # 3%-15%之间

        dcf_val = calculate_intrinsic_value(
            free_cash_flow=li_curr.free_cash_flow,
            growth_rate=adjusted_growth_rate,
            discount_rate=0.09,  # 从10%降到9%，更符合当前利率环境
            terminal_growth_rate=0.025,  # 从3%降到2.5%
            num_years=5,
        )

        # Implied Equity Value
        ev_ebitda_val = calculate_ev_ebitda_value(financial_metrics)

        # Residual Income Model
        rim_val = calculate_residual_income_value(
            market_cap=most_recent_metrics.market_cap,
            net_income=li_curr.net_income,
            price_to_book_ratio=most_recent_metrics.price_to_book_ratio,
            book_value_growth=most_recent_metrics.book_value_growth or 0.03,
        )

        # ------------------------------------------------------------------
        # 优化的估值聚合与信号生成 - 适合短中期交易
        # ------------------------------------------------------------------
        market_cap = get_market_cap(ticker, end_date)
        if not market_cap:
            progress.update_status(agent_id, ticker, "Failed: Market cap unavailable")
            continue

        # 重新设计权重分配 - 降低DCF权重，增加相对估值权重
        method_values = {
            "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.40},      # 提高相对估值权重
            "dcf": {"value": dcf_val, "weight": 0.25},                 # 降低DCF权重
            "owner_earnings": {"value": owner_val, "weight": 0.25},    # 降低所有者收益权重
            "residual_income": {"value": rim_val, "weight": 0.10},     # 保持剩余收益权重
        }

        # 计算有效估值方法
        valid_methods = {k: v for k, v in method_values.items() if v["value"] > 0}
        total_weight = sum(v["weight"] for v in valid_methods.values())

        if total_weight == 0:
            progress.update_status(agent_id, ticker, "Failed: All valuation methods zero")
            continue

        # 计算估值差距
        for v in method_values.values():
            v["gap"] = (v["value"] - market_cap) / market_cap if v["value"] > 0 else None

        # 计算加权平均差距
        weighted_gap = sum(
            v["weight"] * v["gap"] for v in valid_methods.values() if v["gap"] is not None
        ) / total_weight

        # 优化的信号生成逻辑 - 更宽松的阈值
        # 考虑到成长股的特点，适当放宽估值容忍度
        if weighted_gap > 0.25:  # 明显低估 >25%
            signal = "bullish"
            confidence_base = 70
        elif weighted_gap > 0.10:  # 轻微低估 >10%
            signal = "bullish"
            confidence_base = 50
        elif weighted_gap > -0.20:  # 合理估值 -20%~10%
            signal = "neutral"
            confidence_base = 40
        elif weighted_gap > -0.40:  # 轻微高估 -40%~-20%
            signal = "bearish"
            confidence_base = 50
        else:  # 明显高估 <-40%
            signal = "bearish"
            confidence_base = 70

        # 信心度计算 - 基于估值差距的绝对值
        if math.isnan(weighted_gap):
            confidence = 40  # 默认中等信心度
        else:
            gap_magnitude = abs(weighted_gap)
            # 差距越大，信心度越高，但设置上限
            confidence_adjustment = min(gap_magnitude * 100, 30)
            confidence = min(confidence_base + confidence_adjustment, 85)
            confidence = round(confidence)

        # 优化的reasoning - 提供更详细的分析
        reasoning = {}
        for m, vals in method_values.items():
            if vals["value"] > 0:
                # 使用新的阈值标准
                if vals["gap"] and vals["gap"] > 0.25:
                    method_signal = "bullish"
                elif vals["gap"] and vals["gap"] < -0.40:
                    method_signal = "bearish"
                else:
                    method_signal = "neutral"

                reasoning[f"{m}_analysis"] = {
                    "signal": method_signal,
                    "details": (
                        f"Value: ${vals['value']:,.2f}, Market Cap: ${market_cap:,.2f}, "
                        f"Gap: {vals['gap']:.1%}, Weight: {vals['weight']*100:.0f}%"
                    ),
                }

        # 添加综合分析
        reasoning["overall_analysis"] = {
            "weighted_gap": f"{weighted_gap:.1%}",
            "signal_logic": f"加权估值差距{weighted_gap:.1%}，阈值：看涨>25%，看跌<-40%",
            "method_weights": {k: f"{v['weight']*100:.0f}%" for k, v in method_values.items()},
            "confidence_explanation": f"基于{len(valid_methods)}个有效估值方法，差距绝对值{abs(weighted_gap):.1%}"
        }

        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        progress.update_status(agent_id, ticker, "完成", analysis=json.dumps(reasoning, indent=4))

    # ---- Emit message (for LLM tool chain) ----
    msg = HumanMessage(content=json.dumps(valuation_analysis), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = valuation_analysis

    progress.update_status(agent_id, None, "完成")
    
    return {"messages": [msg], "data": data}

#############################
# Helper Valuation Functions
#############################

def calculate_owner_earnings_value(
    net_income: float | None,
    depreciation: float | None,
    capex: float | None,
    working_capital_change: float | None,
    growth_rate: float = 0.05,
    required_return: float = 0.12,  # 从15%降到12%
    margin_of_safety: float = 0.15,  # 从25%降到15%
    num_years: int = 5,
) -> float:
    """Buffett owner‑earnings valuation with margin‑of‑safety."""
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
        return 0

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        future = owner_earnings * (1 + growth_rate) ** yr
        pv += future / (1 + required_return) ** yr

    terminal_growth = min(growth_rate, 0.03)
    term_val = (owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)) / (
        required_return - terminal_growth
    )
    pv_term = term_val / (1 + required_return) ** num_years

    intrinsic = pv + pv_term
    return intrinsic * (1 - margin_of_safety)


def calculate_intrinsic_value(
    free_cash_flow: float | None,
    growth_rate: float = 0.05,
    discount_rate: float = 0.09,  # 从10%降到9%
    terminal_growth_rate: float = 0.025,  # 从2%提高到2.5%
    num_years: int = 5,
) -> float:
    """Classic DCF on FCF with constant growth and terminal value."""
    if free_cash_flow is None or free_cash_flow <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        fcft = free_cash_flow * (1 + growth_rate) ** yr
        pv += fcft / (1 + discount_rate) ** yr

    term_val = (
        free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)
    pv_term = term_val / (1 + discount_rate) ** num_years

    return pv + pv_term


def calculate_ev_ebitda_value(financial_metrics: list):
    """Implied equity value via median EV/EBITDA multiple."""
    if not financial_metrics:
        return 0
    m0 = financial_metrics[0]
    if not (m0.enterprise_value and m0.enterprise_value_to_ebitda_ratio):
        return 0
    if m0.enterprise_value_to_ebitda_ratio == 0:
        return 0

    ebitda_now = m0.enterprise_value / m0.enterprise_value_to_ebitda_ratio
    med_mult = median([
        m.enterprise_value_to_ebitda_ratio for m in financial_metrics if m.enterprise_value_to_ebitda_ratio
    ])
    ev_implied = med_mult * ebitda_now
    net_debt = (m0.enterprise_value or 0) - (m0.market_cap or 0)
    return max(ev_implied - net_debt, 0)


def calculate_residual_income_value(
    market_cap: float | None,
    net_income: float | None,
    price_to_book_ratio: float | None,
    book_value_growth: float = 0.03,
    cost_of_equity: float = 0.09,  # 从10%降到9%
    terminal_growth_rate: float = 0.025,  # 从3%降到2.5%
    num_years: int = 5,
):
    """Residual Income Model (Edwards‑Bell‑Ohlson)."""
    if not (market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0):
        return 0

    book_val = market_cap / price_to_book_ratio
    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0

    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        ri_t = ri0 * (1 + book_value_growth) ** yr
        pv_ri += ri_t / (1 + cost_of_equity) ** yr

    term_ri = ri0 * (1 + book_value_growth) ** (num_years + 1) / (
        cost_of_equity - terminal_growth_rate
    )
    pv_term = term_ri / (1 + cost_of_equity) ** num_years

    intrinsic = book_val + pv_ri + pv_term
    return intrinsic * 0.85  # 15% margin of safety (从20%降到15%)
