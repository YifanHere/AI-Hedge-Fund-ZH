"""
集成风险识别决策引擎

将宏观基本面、市场广度、技术分析、消息面分析、情绪指标等模块
集成到一个统一的风险识别引擎中，能够准确区分系统性崩盘和短暂回调
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 导入各个分析模块
from .systemic_risk_analyzer import SystemicRiskAnalyzer, RiskType, RiskAnalysisResult
from .macro_fundamentals_analyzer import MacroFundamentalsAnalyzer, MacroAnalysisResult
from .market_breadth_analyzer import MarketBreadthAnalyzer, MarketBreadthResult
from .enhanced_technical_analyzer import EnhancedTechnicalAnalyzer, TechnicalAnalysisResult
from .news_impact_analyzer import NewsImpactAnalyzer, NewsAnalysisResult
from .sentiment_flow_analyzer import SentimentFlowAnalyzer, SentimentAnalysisResult

logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """市场状况"""
    SYSTEMIC_CRASH = "systemic_crash"          # 系统性崩盘
    SEVERE_CORRECTION = "severe_correction"     # 严重调整
    MODERATE_CORRECTION = "moderate_correction" # 中度调整
    SHORT_CORRECTION = "short_correction"       # 短期回调
    NORMAL_VOLATILITY = "normal_volatility"     # 正常波动
    BULL_MARKET = "bull_market"                # 牛市

class TradingAction(Enum):
    """交易行动"""
    STRONG_SELL = "strong_sell"        # 强烈卖出
    SELL = "sell"                      # 卖出
    REDUCE_POSITION = "reduce_position" # 减仓
    HOLD = "hold"                      # 持有
    CAUTIOUS_BUY = "cautious_buy"      # 谨慎买入
    BUY = "buy"                        # 买入
    STRONG_BUY = "strong_buy"          # 强烈买入

@dataclass
class IntegratedRiskAssessment:
    """集成风险评估结果"""
    market_condition: MarketCondition
    recommended_action: TradingAction
    confidence_level: float
    systemic_risk_probability: float
    short_term_recovery_probability: float
    key_risk_factors: List[str]
    supporting_evidence: Dict[str, any]
    risk_mitigation_strategies: List[str]
    position_sizing_recommendation: float  # 建议仓位比例 (0-1)
    cooling_period_days: int
    detailed_analysis: Dict[str, any]
    interpretation: str

class IntegratedRiskEngine:
    """集成风险识别引擎"""
    
    def __init__(self):
        # 初始化各个分析器
        self.systemic_analyzer = SystemicRiskAnalyzer()
        self.macro_analyzer = MacroFundamentalsAnalyzer()
        self.breadth_analyzer = MarketBreadthAnalyzer()
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.news_analyzer = NewsImpactAnalyzer()
        self.sentiment_analyzer = SentimentFlowAnalyzer()
        
        # 各维度权重配置
        self.dimension_weights = {
            "macro_fundamentals": 0.25,    # 宏观基本面
            "market_breadth": 0.20,        # 市场广度
            "technical_analysis": 0.20,    # 技术分析
            "news_impact": 0.15,           # 消息面影响
            "sentiment_flow": 0.20         # 情绪与资金流
        }
        
        # 系统性风险判断阈值
        self.systemic_thresholds = {
            "high_confidence": 0.8,        # 高信心度阈值
            "medium_confidence": 0.6,      # 中等信心度阈值
            "systemic_probability": 0.7,   # 系统性风险概率阈值
            "severe_correction": 0.6,      # 严重调整阈值
            "moderate_correction": 0.4     # 中度调整阈值
        }

    def assess_market_risk(self, ticker: str, price_data: pd.DataFrame,
                          market_data: Dict, economic_data: Dict,
                          news_data: List[Dict], end_date: str) -> IntegratedRiskAssessment:
        """
        综合评估市场风险
        
        Args:
            ticker: 股票代码
            price_data: 价格数据
            market_data: 市场数据
            economic_data: 经济数据
            news_data: 新闻数据
            end_date: 分析日期
            
        Returns:
            IntegratedRiskAssessment: 集成风险评估结果
        """
        
        try:
            # 1. 宏观基本面分析
            macro_result = self.macro_analyzer.analyze_macro_fundamentals(
                market_data, economic_data, end_date
            )
            
            # 2. 市场广度分析
            sector_data = market_data.get("sector_data", {})
            correlation_data = market_data.get("correlation_data", {})
            breadth_result = self.breadth_analyzer.analyze_market_breadth(
                market_data, sector_data, correlation_data, end_date
            )
            
            # 3. 技术分析
            technical_result = self.technical_analyzer.analyze_technical_risk(
                price_data, market_data, ticker
            )
            
            # 4. 消息面分析
            news_result = self.news_analyzer.analyze_news_impact(
                news_data, ticker, market_data
            )
            
            # 5. 情绪与资金流分析
            options_data = market_data.get("options_data", {})
            flow_data = market_data.get("flow_data", {})
            sentiment_result = self.sentiment_analyzer.analyze_sentiment_flow(
                market_data, options_data, flow_data, end_date
            )
            
            # 6. 综合风险评估
            integrated_assessment = self._synthesize_risk_assessment(
                macro_result, breadth_result, technical_result,
                news_result, sentiment_result, ticker, price_data
            )
            
            return integrated_assessment
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return self._get_default_assessment()

    def _synthesize_risk_assessment(self, macro_result: MacroAnalysisResult,
                                   breadth_result: MarketBreadthResult,
                                   technical_result: TechnicalAnalysisResult,
                                   news_result: NewsAnalysisResult,
                                   sentiment_result: SentimentAnalysisResult,
                                   ticker: str, price_data: pd.DataFrame) -> IntegratedRiskAssessment:
        """综合各维度分析结果"""
        
        # 计算各维度风险评分
        dimension_scores = self._calculate_dimension_scores(
            macro_result, breadth_result, technical_result, news_result, sentiment_result
        )
        
        # 计算加权综合风险评分
        overall_risk_score = sum(
            score * self.dimension_weights[dimension]
            for dimension, score in dimension_scores.items()
        )
        
        # 计算系统性风险概率
        systemic_probability = self._calculate_systemic_probability(
            dimension_scores, macro_result, breadth_result, technical_result
        )
        
        # 确定市场状况
        market_condition = self._determine_market_condition(
            overall_risk_score, systemic_probability, dimension_scores
        )
        
        # 生成交易建议
        trading_action, position_sizing = self._generate_trading_recommendation(
            market_condition, systemic_probability, overall_risk_score
        )
        
        # 计算信心度
        confidence_level = self._calculate_confidence_level(
            dimension_scores, systemic_probability
        )
        
        # 计算短期恢复概率
        recovery_probability = self._calculate_recovery_probability(
            news_result, technical_result, sentiment_result
        )
        
        # 识别关键风险因素
        key_risk_factors = self._identify_key_risk_factors(
            macro_result, breadth_result, technical_result, news_result, sentiment_result
        )
        
        # 生成风险缓解策略
        risk_mitigation_strategies = self._generate_risk_mitigation_strategies(
            market_condition, key_risk_factors
        )
        
        # 计算冷却期
        cooling_period = self._calculate_cooling_period(
            market_condition, news_result, technical_result
        )
        
        # 生成解释
        interpretation = self._generate_comprehensive_interpretation(
            market_condition, systemic_probability, key_risk_factors, confidence_level
        )
        
        # 汇总支撑证据
        supporting_evidence = {
            "macro_analysis": macro_result,
            "breadth_analysis": breadth_result,
            "technical_analysis": technical_result,
            "news_analysis": news_result,
            "sentiment_analysis": sentiment_result,
            "dimension_scores": dimension_scores,
            "overall_risk_score": overall_risk_score
        }
        
        return IntegratedRiskAssessment(
            market_condition=market_condition,
            recommended_action=trading_action,
            confidence_level=confidence_level,
            systemic_risk_probability=systemic_probability,
            short_term_recovery_probability=recovery_probability,
            key_risk_factors=key_risk_factors,
            supporting_evidence=supporting_evidence,
            risk_mitigation_strategies=risk_mitigation_strategies,
            position_sizing_recommendation=position_sizing,
            cooling_period_days=cooling_period,
            detailed_analysis=supporting_evidence,
            interpretation=interpretation
        )

    def _calculate_dimension_scores(self, macro_result: MacroAnalysisResult,
                                  breadth_result: MarketBreadthResult,
                                  technical_result: TechnicalAnalysisResult,
                                  news_result: NewsAnalysisResult,
                                  sentiment_result: SentimentAnalysisResult) -> Dict[str, float]:
        """计算各维度风险评分"""
        
        # 宏观基本面评分 (转换为0-100风险评分)
        macro_score = macro_result.systemic_risk_score
        
        # 市场广度评分
        breadth_score = breadth_result.breadth_score
        
        # 技术分析评分
        technical_score = technical_result.systemic_risk_score
        
        # 消息面评分
        news_score = (
            news_result.systemic_impact_probability * 50 +
            news_result.short_term_impact_score * 0.3 +
            news_result.long_term_impact_score * 0.2
        )
        
        # 情绪评分 (恐慌贪婪指数转换为风险评分)
        sentiment_score = self._convert_sentiment_to_risk_score(
            sentiment_result.fear_greed_index, sentiment_result.sentiment_level
        )
        
        return {
            "macro_fundamentals": macro_score,
            "market_breadth": breadth_score,
            "technical_analysis": technical_score,
            "news_impact": news_score,
            "sentiment_flow": sentiment_score
        }

    def _calculate_systemic_probability(self, dimension_scores: Dict[str, float],
                                      macro_result: MacroAnalysisResult,
                                      breadth_result: MarketBreadthResult,
                                      technical_result: TechnicalAnalysisResult) -> float:
        """计算系统性风险概率"""
        
        # 基础概率基于综合评分
        base_probability = sum(dimension_scores.values()) / 500  # 平均后转换为概率
        
        # 关键指标调整
        adjustments = 0
        
        # 宏观基本面调整
        if macro_result.systemic_risk_score > 80:
            adjustments += 0.2
        elif macro_result.systemic_risk_score > 60:
            adjustments += 0.1
        
        # 市场广度调整
        if breadth_result.systemic_risk_probability > 0.7:
            adjustments += 0.15
        elif breadth_result.systemic_risk_probability > 0.5:
            adjustments += 0.1
        
        # 技术分析调整
        if technical_result.crash_probability > 0.7:
            adjustments += 0.15
        elif technical_result.crash_probability > 0.5:
            adjustments += 0.1
        
        # 多维度同时高风险的额外调整
        high_risk_dimensions = sum(1 for score in dimension_scores.values() if score > 70)
        if high_risk_dimensions >= 4:
            adjustments += 0.2
        elif high_risk_dimensions >= 3:
            adjustments += 0.15
        elif high_risk_dimensions >= 2:
            adjustments += 0.1
        
        final_probability = min(base_probability + adjustments, 0.95)
        return max(final_probability, 0.05)

    def _determine_market_condition(self, overall_risk_score: float,
                                  systemic_probability: float,
                                  dimension_scores: Dict[str, float]) -> MarketCondition:
        """确定市场状况"""
        
        # 系统性崩盘判断
        if (systemic_probability > self.systemic_thresholds["systemic_probability"] and
            overall_risk_score > 75):
            return MarketCondition.SYSTEMIC_CRASH
        
        # 严重调整判断
        elif (overall_risk_score > 70 or
              systemic_probability > self.systemic_thresholds["severe_correction"]):
            return MarketCondition.SEVERE_CORRECTION
        
        # 中度调整判断
        elif (overall_risk_score > 50 or
              systemic_probability > self.systemic_thresholds["moderate_correction"]):
            return MarketCondition.MODERATE_CORRECTION
        
        # 短期回调判断
        elif overall_risk_score > 35:
            return MarketCondition.SHORT_CORRECTION
        
        # 正常波动
        elif overall_risk_score > 20:
            return MarketCondition.NORMAL_VOLATILITY
        
        # 牛市
        else:
            return MarketCondition.BULL_MARKET

    def _generate_trading_recommendation(self, market_condition: MarketCondition,
                                       systemic_probability: float,
                                       overall_risk_score: float) -> Tuple[TradingAction, float]:
        """生成交易建议和仓位建议"""
        
        # 基于市场状况的基础建议
        condition_recommendations = {
            MarketCondition.SYSTEMIC_CRASH: (TradingAction.STRONG_SELL, 0.0),
            MarketCondition.SEVERE_CORRECTION: (TradingAction.SELL, 0.2),
            MarketCondition.MODERATE_CORRECTION: (TradingAction.REDUCE_POSITION, 0.4),
            MarketCondition.SHORT_CORRECTION: (TradingAction.HOLD, 0.6),
            MarketCondition.NORMAL_VOLATILITY: (TradingAction.CAUTIOUS_BUY, 0.8),
            MarketCondition.BULL_MARKET: (TradingAction.BUY, 1.0)
        }
        
        base_action, base_position = condition_recommendations[market_condition]
        
        # 基于系统性风险概率的调整
        if systemic_probability > 0.8:
            # 极高系统性风险，进一步降低仓位
            adjusted_position = max(0.0, base_position - 0.3)
            if base_action in [TradingAction.BUY, TradingAction.CAUTIOUS_BUY]:
                adjusted_action = TradingAction.HOLD
            elif base_action == TradingAction.HOLD:
                adjusted_action = TradingAction.REDUCE_POSITION
            else:
                adjusted_action = base_action
        elif systemic_probability > 0.6:
            # 高系统性风险，适度降低仓位
            adjusted_position = max(0.0, base_position - 0.2)
            if base_action == TradingAction.BUY:
                adjusted_action = TradingAction.CAUTIOUS_BUY
            else:
                adjusted_action = base_action
        else:
            adjusted_action = base_action
            adjusted_position = base_position
        
        return adjusted_action, adjusted_position

    def _calculate_confidence_level(self, dimension_scores: Dict[str, float],
                                   systemic_probability: float) -> float:
        """计算信心度"""

        # 基于各维度评分的一致性
        scores = list(dimension_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # 一致性评分 (标准差越小，一致性越高)
        consistency_score = max(0, 1 - std_score / 50)  # 标准化到0-1

        # 极端值调整
        extreme_adjustment = 0
        if mean_score > 80 or mean_score < 20:
            extreme_adjustment = 0.1  # 极端情况增加信心度

        # 系统性风险概率调整
        systemic_adjustment = 0
        if systemic_probability > 0.8 or systemic_probability < 0.2:
            systemic_adjustment = 0.1

        confidence = min(0.95, consistency_score + extreme_adjustment + systemic_adjustment)
        return max(0.3, confidence)

    def _calculate_recovery_probability(self, news_result: NewsAnalysisResult,
                                      technical_result: TechnicalAnalysisResult,
                                      sentiment_result: SentimentAnalysisResult) -> float:
        """计算短期恢复概率"""

        base_probability = 0.5

        # 消息面因素
        if news_result.event_solvability.value in ["easily_solvable", "moderately_solvable"]:
            base_probability += 0.2
        elif news_result.event_solvability.value == "difficult_to_solve":
            base_probability -= 0.2
        elif news_result.event_solvability.value == "unsolvable":
            base_probability -= 0.3

        if news_result.market_overreaction_probability > 0.6:
            base_probability += 0.15  # 过度反应后容易修复

        # 技术面因素
        if technical_result.correction_probability > 0.7:
            base_probability += 0.15
        elif technical_result.crash_probability > 0.7:
            base_probability -= 0.2

        # 情绪因素
        if sentiment_result.sentiment_level.value in ["extreme_fear", "extreme_greed"]:
            base_probability += 0.1  # 极端情绪后容易反转

        if len(sentiment_result.contrarian_signals) > 0:
            base_probability += 0.1

        return max(0.1, min(0.9, base_probability))

    def _identify_key_risk_factors(self, macro_result: MacroAnalysisResult,
                                 breadth_result: MarketBreadthResult,
                                 technical_result: TechnicalAnalysisResult,
                                 news_result: NewsAnalysisResult,
                                 sentiment_result: SentimentAnalysisResult) -> List[str]:
        """识别关键风险因素"""

        risk_factors = []

        # 宏观风险因素
        risk_factors.extend(macro_result.risk_factors)

        # 市场广度风险因素
        risk_factors.extend(breadth_result.risk_signals)

        # 技术面风险因素
        risk_factors.extend(technical_result.risk_signals)

        # 消息面风险因素
        risk_factors.extend(news_result.risk_factors)

        # 情绪风险因素
        risk_factors.extend(sentiment_result.sentiment_extremes)

        # 去重并按重要性排序
        unique_factors = list(set(risk_factors))

        # 简单排序：包含"系统性"、"极端"、"崩盘"等关键词的排在前面
        priority_keywords = ["系统性", "极端", "崩盘", "恐慌", "危机"]

        def get_priority(factor):
            for i, keyword in enumerate(priority_keywords):
                if keyword in factor:
                    return i
            return len(priority_keywords)

        sorted_factors = sorted(unique_factors, key=get_priority)

        return sorted_factors[:10]  # 返回最重要的10个风险因素

    def _generate_risk_mitigation_strategies(self, market_condition: MarketCondition,
                                           key_risk_factors: List[str]) -> List[str]:
        """生成风险缓解策略"""

        strategies = []

        # 基于市场状况的基础策略
        if market_condition == MarketCondition.SYSTEMIC_CRASH:
            strategies.extend([
                "立即大幅减仓或清仓，保留现金",
                "考虑做空对冲或购买看跌期权",
                "避免抄底，等待明确的反转信号",
                "关注政策救市措施和央行行动"
            ])
        elif market_condition == MarketCondition.SEVERE_CORRECTION:
            strategies.extend([
                "大幅减少风险敞口，提高现金比例",
                "分批减仓，避免一次性操作",
                "重点关注防御性板块",
                "设置严格的止损位"
            ])
        elif market_condition == MarketCondition.MODERATE_CORRECTION:
            strategies.extend([
                "适度减仓，降低杠杆",
                "增加债券等避险资产配置",
                "关注优质股票的买入机会",
                "保持充足的流动性"
            ])
        elif market_condition == MarketCondition.SHORT_CORRECTION:
            strategies.extend([
                "保持现有仓位，避免频繁交易",
                "可考虑逢低适度加仓优质标的",
                "设置合理的止损和止盈位",
                "关注回调结束信号"
            ])

        # 基于具体风险因素的针对性策略
        for factor in key_risk_factors[:5]:  # 针对前5个主要风险因素
            if "流动性" in factor:
                strategies.append("避免小盘股和流动性差的标的")
            elif "估值" in factor:
                strategies.append("重点关注估值合理的价值股")
            elif "政策" in factor or "监管" in factor:
                strategies.append("密切关注政策动向和监管变化")
            elif "消息" in factor or "新闻" in factor:
                strategies.append("理性分析消息面影响，避免情绪化交易")
            elif "技术" in factor:
                strategies.append("关注关键技术位的支撑和阻力")

        # 去重
        unique_strategies = list(set(strategies))

        return unique_strategies[:8]  # 返回最多8个策略

    def _calculate_cooling_period(self, market_condition: MarketCondition,
                                news_result: NewsAnalysisResult,
                                technical_result: TechnicalAnalysisResult) -> int:
        """计算冷却期（天）"""

        # 基础冷却期
        base_periods = {
            MarketCondition.SYSTEMIC_CRASH: 30,
            MarketCondition.SEVERE_CORRECTION: 21,
            MarketCondition.MODERATE_CORRECTION: 14,
            MarketCondition.SHORT_CORRECTION: 7,
            MarketCondition.NORMAL_VOLATILITY: 3,
            MarketCondition.BULL_MARKET: 1
        }

        base_period = base_periods[market_condition]

        # 基于消息面的调整
        news_adjustment = news_result.recovery_time_estimate * 0.3

        # 基于技术面的调整
        technical_adjustment = 0
        if technical_result.crash_probability > 0.7:
            technical_adjustment = 7
        elif technical_result.correction_probability > 0.7:
            technical_adjustment = 3

        total_period = base_period + news_adjustment + technical_adjustment

        return max(1, min(int(total_period), 60))  # 限制在1-60天之间

    def _generate_comprehensive_interpretation(self, market_condition: MarketCondition,
                                             systemic_probability: float,
                                             key_risk_factors: List[str],
                                             confidence_level: float) -> str:
        """生成综合解释"""

        # 市场状况描述
        condition_descriptions = {
            MarketCondition.SYSTEMIC_CRASH: "系统性崩盘风险极高",
            MarketCondition.SEVERE_CORRECTION: "面临严重市场调整",
            MarketCondition.MODERATE_CORRECTION: "处于中度市场调整",
            MarketCondition.SHORT_CORRECTION: "短期技术性回调",
            MarketCondition.NORMAL_VOLATILITY: "正常市场波动",
            MarketCondition.BULL_MARKET: "牛市上涨趋势"
        }

        interpretation = f"综合分析显示，当前市场{condition_descriptions[market_condition]}。"

        # 系统性风险概率描述
        if systemic_probability > 0.8:
            interpretation += f"系统性风险概率高达{systemic_probability:.1%}，需要极度谨慎。"
        elif systemic_probability > 0.6:
            interpretation += f"系统性风险概率为{systemic_probability:.1%}，风险较高。"
        elif systemic_probability > 0.4:
            interpretation += f"系统性风险概率为{systemic_probability:.1%}，需要适度警惕。"
        else:
            interpretation += f"系统性风险概率相对较低({systemic_probability:.1%})。"

        # 主要风险因素
        if key_risk_factors:
            top_factors = key_risk_factors[:3]
            interpretation += f" 主要风险因素包括：{'; '.join(top_factors)}。"

        # 信心度描述
        if confidence_level > 0.8:
            interpretation += f"分析信心度较高({confidence_level:.1%})。"
        elif confidence_level > 0.6:
            interpretation += f"分析信心度中等({confidence_level:.1%})。"
        else:
            interpretation += f"分析信心度偏低({confidence_level:.1%})，建议谨慎决策。"

        return interpretation

    def _convert_sentiment_to_risk_score(self, fear_greed_index: float, sentiment_level) -> float:
        """将情绪指标转换为风险评分"""

        # 恐慌贪婪指数转换 (0=极度恐慌, 100=极度贪婪)
        if fear_greed_index <= 10:
            return 80  # 极度恐慌，高风险
        elif fear_greed_index <= 25:
            return 70  # 高度恐慌，较高风险
        elif fear_greed_index <= 40:
            return 50  # 中度恐慌，中等风险
        elif fear_greed_index <= 60:
            return 30  # 中性，低风险
        elif fear_greed_index <= 75:
            return 40  # 中度贪婪，中等风险
        elif fear_greed_index <= 90:
            return 60  # 高度贪婪，较高风险
        else:
            return 70  # 极度贪婪，高风险

    def _get_default_assessment(self) -> IntegratedRiskAssessment:
        """获取默认评估结果"""
        return IntegratedRiskAssessment(
            market_condition=MarketCondition.NORMAL_VOLATILITY,
            recommended_action=TradingAction.HOLD,
            confidence_level=0.5,
            systemic_risk_probability=0.3,
            short_term_recovery_probability=0.6,
            key_risk_factors=["数据不足，无法进行完整分析"],
            supporting_evidence={},
            risk_mitigation_strategies=["保持谨慎，等待更多数据"],
            position_sizing_recommendation=0.5,
            cooling_period_days=7,
            detailed_analysis={},
            interpretation="由于数据不足，无法进行完整的风险评估，建议保持谨慎态度"
        )
