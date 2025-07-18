"""
增强投资组合管理器

基于新的风险识别能力，优化投资组合管理的决策逻辑
特别是融券时机的选择，能够区分系统性崩盘和短暂回调
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 导入风险识别引擎
from .integrated_risk_engine import IntegratedRiskEngine, IntegratedRiskAssessment, MarketCondition, TradingAction

logger = logging.getLogger(__name__)

class PositionType(Enum):
    """仓位类型"""
    LONG = "long"           # 多头
    SHORT = "short"         # 空头
    NEUTRAL = "neutral"     # 中性

class TradeType(Enum):
    """交易类型"""
    BUY = "buy"             # 买入
    SELL = "sell"           # 卖出
    SHORT_SELL = "short_sell"  # 融券卖出
    COVER = "cover"         # 平仓
    HOLD = "hold"           # 持有

@dataclass
class PositionRecommendation:
    """仓位建议"""
    ticker: str
    trade_type: TradeType
    position_size: float        # 建议仓位大小 (0-1)
    confidence: float          # 信心度
    reasoning: str             # 决策理由
    risk_level: str           # 风险等级
    stop_loss: Optional[float] # 止损位
    take_profit: Optional[float] # 止盈位
    holding_period: int        # 建议持有期（天）
    risk_mitigation: List[str] # 风险缓解措施

@dataclass
class PortfolioDecision:
    """投资组合决策"""
    overall_strategy: str
    position_recommendations: List[PositionRecommendation]
    portfolio_risk_level: str
    cash_allocation: float     # 现金配置比例
    max_position_size: float   # 单个标的最大仓位
    risk_budget: float         # 风险预算
    rebalance_frequency: int   # 再平衡频率（天）
    market_outlook: str        # 市场展望
    key_risks: List[str]       # 关键风险
    monitoring_points: List[str] # 监控要点

class EnhancedPortfolioManager:
    """增强投资组合管理器"""
    
    def __init__(self):
        # 初始化风险识别引擎
        self.risk_engine = IntegratedRiskEngine()
        
        # 仓位管理参数
        self.position_limits = {
            "max_single_position": 0.2,      # 单个标的最大仓位
            "max_total_short": 0.3,          # 最大空头仓位
            "min_cash_reserve": 0.1,         # 最小现金储备
            "max_leverage": 2.0              # 最大杠杆
        }
        
        # 风险管理参数
        self.risk_parameters = {
            "stop_loss_threshold": 0.08,     # 止损阈值
            "take_profit_threshold": 0.15,   # 止盈阈值
            "volatility_adjustment": True,   # 是否根据波动率调整
            "correlation_limit": 0.7         # 相关性限制
        }
        
        # 融券决策参数
        self.short_selling_criteria = {
            "min_systemic_probability": 0.6,    # 最小系统性风险概率
            "min_confidence": 0.7,              # 最小信心度
            "max_recovery_probability": 0.4,    # 最大恢复概率
            "min_cooling_period": 5             # 最小冷却期
        }

    def make_portfolio_decision(self, tickers: List[str], market_data: Dict,
                              price_data: Dict[str, pd.DataFrame],
                              economic_data: Dict, news_data: Dict[str, List[Dict]],
                              current_positions: Dict, end_date: str) -> PortfolioDecision:
        """
        制定投资组合决策
        
        Args:
            tickers: 股票代码列表
            market_data: 市场数据
            price_data: 各股票价格数据
            economic_data: 经济数据
            news_data: 各股票新闻数据
            current_positions: 当前持仓
            end_date: 决策日期
            
        Returns:
            PortfolioDecision: 投资组合决策
        """
        
        # 1. 对每个标的进行风险评估
        risk_assessments = {}
        for ticker in tickers:
            try:
                assessment = self.risk_engine.assess_market_risk(
                    ticker=ticker,
                    price_data=price_data.get(ticker, pd.DataFrame()),
                    market_data=market_data,
                    economic_data=economic_data,
                    news_data=news_data.get(ticker, []),
                    end_date=end_date
                )
                risk_assessments[ticker] = assessment
            except Exception as e:
                logger.error(f"评估{ticker}风险失败: {e}")
                continue
        
        # 2. 生成个股仓位建议
        position_recommendations = []
        for ticker, assessment in risk_assessments.items():
            recommendation = self._generate_position_recommendation(
                ticker, assessment, current_positions.get(ticker, 0), price_data.get(ticker)
            )
            position_recommendations.append(recommendation)
        
        # 3. 制定整体投资组合策略
        portfolio_strategy = self._determine_portfolio_strategy(risk_assessments)
        
        # 4. 计算资产配置
        cash_allocation, max_position_size = self._calculate_asset_allocation(risk_assessments)
        
        # 5. 评估投资组合整体风险
        portfolio_risk_level = self._assess_portfolio_risk(risk_assessments)
        
        # 6. 生成监控要点
        monitoring_points = self._generate_monitoring_points(risk_assessments)
        
        # 7. 识别关键风险
        key_risks = self._identify_portfolio_risks(risk_assessments)
        
        return PortfolioDecision(
            overall_strategy=portfolio_strategy,
            position_recommendations=position_recommendations,
            portfolio_risk_level=portfolio_risk_level,
            cash_allocation=cash_allocation,
            max_position_size=max_position_size,
            risk_budget=self._calculate_risk_budget(risk_assessments),
            rebalance_frequency=self._determine_rebalance_frequency(risk_assessments),
            market_outlook=self._generate_market_outlook(risk_assessments),
            key_risks=key_risks,
            monitoring_points=monitoring_points
        )

    def _generate_position_recommendation(self, ticker: str, assessment: IntegratedRiskAssessment,
                                        current_position: float, price_data: pd.DataFrame) -> PositionRecommendation:
        """生成个股仓位建议"""
        
        # 基础交易建议
        base_action = assessment.recommended_action
        base_position_size = assessment.position_sizing_recommendation
        
        # 融券决策逻辑
        trade_type, adjusted_position_size = self._determine_trade_type(
            assessment, current_position, base_action, base_position_size
        )
        
        # 计算止损止盈位
        stop_loss, take_profit = self._calculate_stop_levels(
            price_data, trade_type, assessment.market_condition
        )
        
        # 确定持有期
        holding_period = self._determine_holding_period(
            assessment.market_condition, assessment.cooling_period_days, trade_type
        )
        
        # 生成决策理由
        reasoning = self._generate_reasoning(assessment, trade_type, adjusted_position_size)
        
        # 风险等级
        risk_level = self._determine_position_risk_level(assessment)
        
        return PositionRecommendation(
            ticker=ticker,
            trade_type=trade_type,
            position_size=adjusted_position_size,
            confidence=assessment.confidence_level,
            reasoning=reasoning,
            risk_level=risk_level,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period=holding_period,
            risk_mitigation=assessment.risk_mitigation_strategies
        )

    def _determine_trade_type(self, assessment: IntegratedRiskAssessment, current_position: float,
                            base_action: TradingAction, base_position_size: float) -> Tuple[TradeType, float]:
        """确定交易类型和调整后的仓位大小"""
        
        # 融券决策逻辑
        should_short = self._should_short_sell(assessment)
        
        if should_short and current_position >= 0:
            # 满足融券条件且当前非空头仓位
            return TradeType.SHORT_SELL, min(base_position_size, self.position_limits["max_single_position"])
        
        # 常规交易逻辑
        if base_action == TradingAction.STRONG_SELL:
            if current_position > 0:
                return TradeType.SELL, 0.0  # 清仓
            elif current_position < 0:
                return TradeType.HOLD, abs(current_position)  # 保持空头
            else:
                return TradeType.SHORT_SELL, base_position_size if should_short else 0.0
        
        elif base_action == TradingAction.SELL:
            if current_position > 0:
                return TradeType.SELL, max(0.0, current_position - base_position_size)
            else:
                return TradeType.HOLD, abs(current_position)
        
        elif base_action == TradingAction.REDUCE_POSITION:
            if current_position > 0:
                return TradeType.SELL, current_position * 0.5
            elif current_position < 0:
                return TradeType.COVER, abs(current_position) * 0.5
            else:
                return TradeType.HOLD, 0.0
        
        elif base_action == TradingAction.HOLD:
            return TradeType.HOLD, abs(current_position)
        
        elif base_action == TradingAction.CAUTIOUS_BUY:
            if current_position < 0:
                return TradeType.COVER, 0.0  # 先平空头
            else:
                return TradeType.BUY, min(base_position_size, self.position_limits["max_single_position"])
        
        elif base_action in [TradingAction.BUY, TradingAction.STRONG_BUY]:
            if current_position < 0:
                return TradeType.COVER, 0.0  # 先平空头
            else:
                return TradeType.BUY, min(base_position_size, self.position_limits["max_single_position"])
        
        else:
            return TradeType.HOLD, abs(current_position)

    def _should_short_sell(self, assessment: IntegratedRiskAssessment) -> bool:
        """判断是否应该融券"""
        
        # 关键条件检查
        conditions_met = 0
        total_conditions = 4
        
        # 1. 系统性风险概率足够高
        if assessment.systemic_risk_probability >= self.short_selling_criteria["min_systemic_probability"]:
            conditions_met += 1
        
        # 2. 分析信心度足够高
        if assessment.confidence_level >= self.short_selling_criteria["min_confidence"]:
            conditions_met += 1
        
        # 3. 短期恢复概率较低
        if assessment.short_term_recovery_probability <= self.short_selling_criteria["max_recovery_probability"]:
            conditions_met += 1
        
        # 4. 冷却期足够长
        if assessment.cooling_period_days >= self.short_selling_criteria["min_cooling_period"]:
            conditions_met += 1
        
        # 需要满足至少3个条件
        basic_criteria_met = conditions_met >= 3
        
        # 额外的市场状况检查
        market_condition_suitable = assessment.market_condition in [
            MarketCondition.SYSTEMIC_CRASH,
            MarketCondition.SEVERE_CORRECTION
        ]
        
        # 避免在短期回调时融券（这是原问题的核心）
        avoid_short_correction = assessment.market_condition != MarketCondition.SHORT_CORRECTION
        
        # 检查是否有明确的系统性风险信号
        systemic_signals = [
            factor for factor in assessment.key_risk_factors
            if any(keyword in factor for keyword in ["系统性", "崩盘", "危机", "恐慌"])
        ]
        has_systemic_signals = len(systemic_signals) > 0
        
        # 综合判断
        should_short = (
            basic_criteria_met and
            market_condition_suitable and
            avoid_short_correction and
            has_systemic_signals
        )
        
        logger.info(f"融券决策分析: 基础条件={conditions_met}/{total_conditions}, "
                   f"市场状况适合={market_condition_suitable}, "
                   f"避免短期回调={avoid_short_correction}, "
                   f"系统性信号={len(systemic_signals)}, "
                   f"最终决策={should_short}")
        
        return should_short

    def _calculate_stop_levels(self, price_data: pd.DataFrame, trade_type: TradeType,
                             market_condition: MarketCondition) -> Tuple[Optional[float], Optional[float]]:
        """计算止损止盈位"""

        if price_data.empty:
            return None, None

        current_price = price_data["close"].iloc[-1]

        # 根据市场状况调整止损止盈幅度
        if market_condition in [MarketCondition.SYSTEMIC_CRASH, MarketCondition.SEVERE_CORRECTION]:
            stop_loss_pct = 0.12  # 高风险环境，放宽止损
            take_profit_pct = 0.20
        elif market_condition == MarketCondition.MODERATE_CORRECTION:
            stop_loss_pct = 0.08
            take_profit_pct = 0.15
        else:
            stop_loss_pct = 0.05
            take_profit_pct = 0.10

        # 根据交易类型设置止损止盈
        if trade_type in [TradeType.BUY]:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        elif trade_type == TradeType.SHORT_SELL:
            stop_loss = current_price * (1 + stop_loss_pct)  # 空头止损在上方
            take_profit = current_price * (1 - take_profit_pct)  # 空头止盈在下方
        else:
            return None, None

        return stop_loss, take_profit

    def _determine_holding_period(self, market_condition: MarketCondition,
                                cooling_period: int, trade_type: TradeType) -> int:
        """确定建议持有期"""

        base_periods = {
            MarketCondition.SYSTEMIC_CRASH: 30,
            MarketCondition.SEVERE_CORRECTION: 21,
            MarketCondition.MODERATE_CORRECTION: 14,
            MarketCondition.SHORT_CORRECTION: 7,
            MarketCondition.NORMAL_VOLATILITY: 5,
            MarketCondition.BULL_MARKET: 10
        }

        base_period = base_periods.get(market_condition, 7)

        # 融券持有期通常较短
        if trade_type == TradeType.SHORT_SELL:
            base_period = min(base_period, 14)

        # 结合冷却期
        return max(base_period, cooling_period)

    def _generate_reasoning(self, assessment: IntegratedRiskAssessment,
                          trade_type: TradeType, position_size: float) -> str:
        """生成决策理由"""

        reasoning_parts = []

        # 市场状况
        reasoning_parts.append(f"市场状况: {assessment.market_condition.value}")

        # 系统性风险
        reasoning_parts.append(f"系统性风险概率: {assessment.systemic_risk_probability:.1%}")

        # 信心度
        reasoning_parts.append(f"分析信心度: {assessment.confidence_level:.1%}")

        # 交易类型特殊说明
        if trade_type == TradeType.SHORT_SELL:
            reasoning_parts.append("满足融券条件: 高系统性风险且非短期回调")
        elif trade_type == TradeType.HOLD:
            reasoning_parts.append("建议观望: 等待更明确的信号")

        # 主要风险因素
        if assessment.key_risk_factors:
            top_risk = assessment.key_risk_factors[0]
            reasoning_parts.append(f"主要风险: {top_risk}")

        return "; ".join(reasoning_parts)

    def _determine_position_risk_level(self, assessment: IntegratedRiskAssessment) -> str:
        """确定仓位风险等级"""

        if assessment.systemic_risk_probability > 0.8:
            return "极高风险"
        elif assessment.systemic_risk_probability > 0.6:
            return "高风险"
        elif assessment.systemic_risk_probability > 0.4:
            return "中等风险"
        elif assessment.systemic_risk_probability > 0.2:
            return "低风险"
        else:
            return "极低风险"

    def _determine_portfolio_strategy(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> str:
        """确定整体投资组合策略"""

        if not risk_assessments:
            return "保守策略: 数据不足"

        # 计算平均系统性风险概率
        avg_systemic_risk = np.mean([
            assessment.systemic_risk_probability for assessment in risk_assessments.values()
        ])

        # 统计各种市场状况
        condition_counts = {}
        for assessment in risk_assessments.values():
            condition = assessment.market_condition
            condition_counts[condition] = condition_counts.get(condition, 0) + 1

        # 找出主导的市场状况
        dominant_condition = max(condition_counts, key=condition_counts.get)

        # 制定策略
        if avg_systemic_risk > 0.7:
            return "防御策略: 高系统性风险，大幅减仓避险"
        elif avg_systemic_risk > 0.5:
            return "谨慎策略: 中高风险，适度减仓观望"
        elif dominant_condition == MarketCondition.SHORT_CORRECTION:
            return "逢低布局策略: 短期回调，寻找优质标的"
        elif dominant_condition == MarketCondition.BULL_MARKET:
            return "积极策略: 牛市环境，适度加仓"
        else:
            return "平衡策略: 保持中性仓位"

    def _calculate_asset_allocation(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> Tuple[float, float]:
        """计算资产配置"""

        if not risk_assessments:
            return 0.5, 0.1  # 默认50%现金，10%最大单仓

        # 基于平均风险水平调整现金比例
        avg_systemic_risk = np.mean([
            assessment.systemic_risk_probability for assessment in risk_assessments.values()
        ])

        # 现金配置
        if avg_systemic_risk > 0.8:
            cash_allocation = 0.8  # 极高风险，80%现金
        elif avg_systemic_risk > 0.6:
            cash_allocation = 0.6  # 高风险，60%现金
        elif avg_systemic_risk > 0.4:
            cash_allocation = 0.4  # 中等风险，40%现金
        elif avg_systemic_risk > 0.2:
            cash_allocation = 0.2  # 低风险，20%现金
        else:
            cash_allocation = 0.1  # 极低风险，10%现金

        # 最大单仓位
        if avg_systemic_risk > 0.7:
            max_position_size = 0.05  # 高风险时降低单仓
        elif avg_systemic_risk > 0.5:
            max_position_size = 0.10
        elif avg_systemic_risk > 0.3:
            max_position_size = 0.15
        else:
            max_position_size = 0.20

        return cash_allocation, max_position_size

    def _assess_portfolio_risk(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> str:
        """评估投资组合整体风险"""

        if not risk_assessments:
            return "中等风险"

        # 计算风险指标
        systemic_risks = [assessment.systemic_risk_probability for assessment in risk_assessments.values()]
        avg_risk = np.mean(systemic_risks)
        max_risk = np.max(systemic_risks)

        # 风险集中度
        high_risk_count = sum(1 for risk in systemic_risks if risk > 0.6)
        risk_concentration = high_risk_count / len(systemic_risks)

        # 综合评估
        if avg_risk > 0.7 or max_risk > 0.9:
            return "极高风险"
        elif avg_risk > 0.5 or risk_concentration > 0.5:
            return "高风险"
        elif avg_risk > 0.3 or risk_concentration > 0.3:
            return "中等风险"
        elif avg_risk > 0.2:
            return "低风险"
        else:
            return "极低风险"

    def _calculate_risk_budget(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> float:
        """计算风险预算"""

        if not risk_assessments:
            return 0.02  # 默认2%

        avg_systemic_risk = np.mean([
            assessment.systemic_risk_probability for assessment in risk_assessments.values()
        ])

        # 风险预算与系统性风险反向相关
        if avg_systemic_risk > 0.7:
            return 0.01  # 1%
        elif avg_systemic_risk > 0.5:
            return 0.015  # 1.5%
        elif avg_systemic_risk > 0.3:
            return 0.02  # 2%
        elif avg_systemic_risk > 0.2:
            return 0.025  # 2.5%
        else:
            return 0.03  # 3%

    def _determine_rebalance_frequency(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> int:
        """确定再平衡频率"""

        if not risk_assessments:
            return 7  # 默认每周

        avg_systemic_risk = np.mean([
            assessment.systemic_risk_probability for assessment in risk_assessments.values()
        ])

        # 高风险时更频繁再平衡
        if avg_systemic_risk > 0.7:
            return 1  # 每日
        elif avg_systemic_risk > 0.5:
            return 3  # 每3天
        elif avg_systemic_risk > 0.3:
            return 7  # 每周
        else:
            return 14  # 每两周

    def _generate_market_outlook(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> str:
        """生成市场展望"""

        if not risk_assessments:
            return "市场前景不明，建议谨慎观望"

        # 统计市场状况分布
        conditions = [assessment.market_condition for assessment in risk_assessments.values()]
        condition_counts = {}
        for condition in conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1

        dominant_condition = max(condition_counts, key=condition_counts.get)

        outlooks = {
            MarketCondition.SYSTEMIC_CRASH: "市场面临系统性崩盘风险，预计大幅下跌",
            MarketCondition.SEVERE_CORRECTION: "市场处于严重调整期，短期承压明显",
            MarketCondition.MODERATE_CORRECTION: "市场中度调整，寻找结构性机会",
            MarketCondition.SHORT_CORRECTION: "短期技术性回调，中长期趋势未变",
            MarketCondition.NORMAL_VOLATILITY: "市场正常波动，保持谨慎乐观",
            MarketCondition.BULL_MARKET: "牛市趋势延续，适度参与上涨"
        }

        return outlooks.get(dominant_condition, "市场前景不明确")

    def _identify_portfolio_risks(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> List[str]:
        """识别投资组合风险"""

        all_risks = []
        for assessment in risk_assessments.values():
            all_risks.extend(assessment.key_risk_factors)

        # 统计风险因素频率
        risk_counts = {}
        for risk in all_risks:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        # 按频率排序，返回最常见的风险
        sorted_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)

        return [risk for risk, count in sorted_risks[:8]]

    def _generate_monitoring_points(self, risk_assessments: Dict[str, IntegratedRiskAssessment]) -> List[str]:
        """生成监控要点"""

        monitoring_points = []

        if not risk_assessments:
            return ["密切关注市场动态"]

        # 基于风险水平生成监控要点
        avg_systemic_risk = np.mean([
            assessment.systemic_risk_probability for assessment in risk_assessments.values()
        ])

        if avg_systemic_risk > 0.7:
            monitoring_points.extend([
                "密切关注系统性风险指标",
                "监控政策救市措施",
                "关注市场流动性状况",
                "跟踪恐慌指数变化"
            ])
        elif avg_systemic_risk > 0.5:
            monitoring_points.extend([
                "关注宏观经济数据",
                "监控市场广度指标",
                "跟踪资金流向变化"
            ])
        else:
            monitoring_points.extend([
                "关注个股基本面变化",
                "监控技术指标信号",
                "跟踪行业轮动情况"
            ])

        # 添加通用监控要点
        monitoring_points.extend([
            "定期评估仓位风险",
            "关注止损止盈执行",
            "监控相关性变化"
        ])

        return monitoring_points[:6]  # 返回最多6个要点
