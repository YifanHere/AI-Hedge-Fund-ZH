"""
系统性风险vs短暂回调识别框架

基于用户提供的多维度分析方法，实现对股价波动性质的准确判断
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RiskType(Enum):
    """风险类型枚举"""
    SYSTEMIC_CRASH = "systemic_crash"      # 系统性崩盘
    SHORT_CORRECTION = "short_correction"   # 短暂回调
    TREND_REVERSAL = "trend_reversal"      # 趋势反转
    UNKNOWN = "unknown"                     # 无法判断

class ConfidenceLevel(Enum):
    """信心度等级"""
    VERY_HIGH = "very_high"    # 90%+
    HIGH = "high"              # 75-90%
    MEDIUM = "medium"          # 50-75%
    LOW = "low"                # 25-50%
    VERY_LOW = "very_low"      # <25%

@dataclass
class RiskAnalysisResult:
    """风险分析结果"""
    risk_type: RiskType
    confidence: float
    confidence_level: ConfidenceLevel
    contributing_factors: Dict[str, float]
    analysis_details: Dict[str, any]
    recommendation: str
    cooling_period_days: int

class SystemicRiskAnalyzer:
    """系统性风险分析器"""
    
    def __init__(self):
        # 各维度权重配置
        self.dimension_weights = {
            "macro_fundamentals": 0.30,      # 宏观基本面
            "market_breadth": 0.25,          # 市场广度
            "technical_analysis": 0.20,      # 技术分析
            "news_analysis": 0.15,           # 消息面分析
            "sentiment_indicators": 0.10     # 情绪指标
        }
        
        # 系统性风险阈值
        self.systemic_thresholds = {
            "macro_score": 60,               # 宏观风险评分阈值
            "market_breadth_score": 70,      # 市场广度风险阈值
            "technical_score": 65,           # 技术风险阈值
            "news_severity": 0.7,            # 消息严重性阈值
            "sentiment_panic": 0.8           # 情绪恐慌阈值
        }

    def analyze_risk_type(self, 
                         ticker: str,
                         current_price: float,
                         price_history: pd.DataFrame,
                         market_data: Dict,
                         news_data: List[Dict],
                         end_date: str) -> RiskAnalysisResult:
        """
        综合分析风险类型
        
        Args:
            ticker: 股票代码
            current_price: 当前价格
            price_history: 价格历史数据
            market_data: 市场数据
            news_data: 新闻数据
            end_date: 分析日期
            
        Returns:
            RiskAnalysisResult: 风险分析结果
        """
        
        # 1. 宏观基本面分析
        macro_analysis = self._analyze_macro_fundamentals(market_data, end_date)
        
        # 2. 市场广度分析
        breadth_analysis = self._analyze_market_breadth(market_data, ticker, end_date)
        
        # 3. 技术分析
        technical_analysis = self._analyze_technical_indicators(price_history, current_price)
        
        # 4. 消息面分析
        news_analysis = self._analyze_news_impact(news_data, ticker)
        
        # 5. 情绪指标分析
        sentiment_analysis = self._analyze_sentiment_indicators(market_data, price_history)
        
        # 综合评分
        total_score = (
            macro_analysis["score"] * self.dimension_weights["macro_fundamentals"] +
            breadth_analysis["score"] * self.dimension_weights["market_breadth"] +
            technical_analysis["score"] * self.dimension_weights["technical_analysis"] +
            news_analysis["score"] * self.dimension_weights["news_analysis"] +
            sentiment_analysis["score"] * self.dimension_weights["sentiment_indicators"]
        )
        
        # 判断风险类型
        risk_type, confidence = self._determine_risk_type(
            total_score, macro_analysis, breadth_analysis, 
            technical_analysis, news_analysis, sentiment_analysis
        )
        
        # 生成建议和冷却期
        recommendation, cooling_period = self._generate_recommendation(
            risk_type, confidence, technical_analysis
        )
        
        return RiskAnalysisResult(
            risk_type=risk_type,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            contributing_factors={
                "macro_fundamentals": macro_analysis["score"],
                "market_breadth": breadth_analysis["score"],
                "technical_analysis": technical_analysis["score"],
                "news_analysis": news_analysis["score"],
                "sentiment_indicators": sentiment_analysis["score"]
            },
            analysis_details={
                "macro_analysis": macro_analysis,
                "breadth_analysis": breadth_analysis,
                "technical_analysis": technical_analysis,
                "news_analysis": news_analysis,
                "sentiment_analysis": sentiment_analysis,
                "total_score": total_score
            },
            recommendation=recommendation,
            cooling_period_days=cooling_period
        )

    def _analyze_macro_fundamentals(self, market_data: Dict, end_date: str) -> Dict:
        """
        宏观基本面分析
        
        分析要点：
        1. 经济周期位置
        2. 金融体系健康度
        3. 政策空间
        4. 整体估值水平
        5. 系统性风险指标
        """
        score = 0
        details = {}
        
        # 经济周期分析 (0-25分)
        economic_cycle_score = self._assess_economic_cycle(market_data)
        score += economic_cycle_score
        details["economic_cycle"] = economic_cycle_score
        
        # 金融体系健康度 (0-25分)
        financial_health_score = self._assess_financial_system_health(market_data)
        score += financial_health_score
        details["financial_health"] = financial_health_score
        
        # 政策空间 (0-20分)
        policy_space_score = self._assess_policy_space(market_data)
        score += policy_space_score
        details["policy_space"] = policy_space_score
        
        # 估值水平 (0-20分)
        valuation_score = self._assess_valuation_levels(market_data)
        score += valuation_score
        details["valuation"] = valuation_score
        
        # 系统性风险指标 (0-10分)
        systemic_risk_score = self._assess_systemic_risk_indicators(market_data)
        score += systemic_risk_score
        details["systemic_risk"] = systemic_risk_score
        
        return {
            "score": score,
            "details": details,
            "interpretation": self._interpret_macro_score(score)
        }

    def _analyze_market_breadth(self, market_data: Dict, ticker: str, end_date: str) -> Dict:
        """
        市场广度与内部结构分析
        
        分析要点：
        1. 涨跌家数比
        2. 新高新低指数
        3. 板块表现
        4. 相关性分析
        """
        score = 0
        details = {}
        
        # 涨跌家数比分析 (0-30分)
        advance_decline_score = self._analyze_advance_decline_ratio(market_data)
        score += advance_decline_score
        details["advance_decline"] = advance_decline_score
        
        # 新高新低分析 (0-25分)
        new_highs_lows_score = self._analyze_new_highs_lows(market_data)
        score += new_highs_lows_score
        details["new_highs_lows"] = new_highs_lows_score
        
        # 板块表现分析 (0-25分)
        sector_performance_score = self._analyze_sector_performance(market_data, ticker)
        score += sector_performance_score
        details["sector_performance"] = sector_performance_score
        
        # 相关性分析 (0-20分)
        correlation_score = self._analyze_correlation_patterns(market_data, ticker)
        score += correlation_score
        details["correlation"] = correlation_score
        
        return {
            "score": score,
            "details": details,
            "interpretation": self._interpret_breadth_score(score)
        }

    def _analyze_technical_indicators(self, price_history: pd.DataFrame, current_price: float) -> Dict:
        """
        技术分析 (辅助判断趋势和强度)
        
        分析要点：
        1. 趋势指标
        2. 动能指标
        3. 成交量分析
        4. 形态识别
        """
        score = 0
        details = {}
        
        # 趋势指标分析 (0-30分)
        trend_score = self._analyze_trend_indicators(price_history, current_price)
        score += trend_score
        details["trend"] = trend_score
        
        # 动能指标分析 (0-25分)
        momentum_score = self._analyze_momentum_indicators(price_history)
        score += momentum_score
        details["momentum"] = momentum_score
        
        # 成交量分析 (0-25分)
        volume_score = self._analyze_volume_patterns(price_history)
        score += volume_score
        details["volume"] = volume_score
        
        # 形态识别 (0-20分)
        pattern_score = self._analyze_price_patterns(price_history)
        score += pattern_score
        details["patterns"] = pattern_score
        
        return {
            "score": score,
            "details": details,
            "interpretation": self._interpret_technical_score(score)
        }

    def _analyze_news_impact(self, news_data: List[Dict], ticker: str) -> Dict:
        """
        消息面分析 (判断是否局部性)
        
        分析要点：
        1. 事件性质判断
        2. 可预测性与可解决性
        3. 市场反应对比
        """
        score = 0
        details = {}
        
        if not news_data:
            return {"score": 50, "details": {"no_news": True}, "interpretation": "无重大消息"}
        
        # 事件性质分析 (0-40分)
        event_nature_score = self._analyze_event_nature(news_data, ticker)
        score += event_nature_score
        details["event_nature"] = event_nature_score
        
        # 可解决性分析 (0-35分)
        solvability_score = self._analyze_event_solvability(news_data)
        score += solvability_score
        details["solvability"] = solvability_score
        
        # 市场反应对比 (0-25分)
        market_reaction_score = self._analyze_market_reaction_comparison(news_data, ticker)
        score += market_reaction_score
        details["market_reaction"] = market_reaction_score
        
        return {
            "score": score,
            "details": details,
            "interpretation": self._interpret_news_score(score)
        }

    def _analyze_sentiment_indicators(self, market_data: Dict, price_history: pd.DataFrame) -> Dict:
        """
        情绪与资金流指标分析
        
        分析要点：
        1. 恐慌指数 (VIX)
        2. 期权偏度
        3. 资金流向
        """
        score = 0
        details = {}
        
        # VIX恐慌指数分析 (0-40分)
        vix_score = self._analyze_vix_levels(market_data)
        score += vix_score
        details["vix"] = vix_score
        
        # 期权偏度分析 (0-30分)
        options_skew_score = self._analyze_options_skew(market_data)
        score += options_skew_score
        details["options_skew"] = options_skew_score
        
        # 资金流向分析 (0-30分)
        money_flow_score = self._analyze_money_flow(market_data, price_history)
        score += money_flow_score
        details["money_flow"] = money_flow_score
        
        return {
            "score": score,
            "details": details,
            "interpretation": self._interpret_sentiment_score(score)
        }

    def _determine_risk_type(self, total_score: float, macro_analysis: Dict,
                           breadth_analysis: Dict, technical_analysis: Dict,
                           news_analysis: Dict, sentiment_analysis: Dict) -> Tuple[RiskType, float]:
        """
        基于综合评分判断风险类型

        判断逻辑：
        - 系统性崩盘：多个维度同时显示高风险
        - 短暂回调：主要由单一因素驱动，其他维度相对健康
        - 趋势反转：技术面主导，基本面支撑
        """

        # 获取各维度评分
        macro_score = macro_analysis["score"]
        breadth_score = breadth_analysis["score"]
        technical_score = technical_analysis["score"]
        news_score = news_analysis["score"]
        sentiment_score = sentiment_analysis["score"]

        # 高风险维度计数
        high_risk_dimensions = 0
        if macro_score >= self.systemic_thresholds["macro_score"]:
            high_risk_dimensions += 1
        if breadth_score >= self.systemic_thresholds["market_breadth_score"]:
            high_risk_dimensions += 1
        if technical_score >= self.systemic_thresholds["technical_score"]:
            high_risk_dimensions += 1
        if news_score >= 70:  # 消息面高风险阈值
            high_risk_dimensions += 1
        if sentiment_score >= 80:  # 情绪高风险阈值
            high_risk_dimensions += 1

        # 判断逻辑
        if high_risk_dimensions >= 3:
            # 系统性崩盘：多个维度同时高风险
            confidence = min(0.95, 0.6 + (high_risk_dimensions - 3) * 0.1 + total_score / 500)
            return RiskType.SYSTEMIC_CRASH, confidence

        elif high_risk_dimensions == 2:
            # 需要进一步判断
            if macro_score >= 60 and breadth_score >= 70:
                # 宏观+市场广度 = 系统性风险
                confidence = 0.7 + total_score / 600
                return RiskType.SYSTEMIC_CRASH, confidence
            elif news_score >= 70 and technical_score < 60:
                # 消息面主导，技术面不太差 = 短暂回调
                confidence = 0.6 + (100 - technical_score) / 200
                return RiskType.SHORT_CORRECTION, confidence
            else:
                # 趋势反转
                confidence = 0.5 + technical_score / 200
                return RiskType.TREND_REVERSAL, confidence

        elif high_risk_dimensions == 1:
            # 单一维度风险
            if news_score >= 70:
                # 纯消息面驱动 = 短暂回调
                confidence = 0.7 + (100 - max(macro_score, breadth_score)) / 150
                return RiskType.SHORT_CORRECTION, confidence
            elif technical_score >= 65:
                # 技术面主导 = 趋势反转
                confidence = 0.6 + technical_score / 200
                return RiskType.TREND_REVERSAL, confidence
            else:
                # 其他单一风险
                confidence = 0.4 + total_score / 300
                return RiskType.SHORT_CORRECTION, confidence
        else:
            # 无明显风险
            confidence = max(0.2, 0.8 - total_score / 200)
            return RiskType.SHORT_CORRECTION, confidence

    def _generate_recommendation(self, risk_type: RiskType, confidence: float,
                               technical_analysis: Dict) -> Tuple[str, int]:
        """生成交易建议和冷却期"""

        if risk_type == RiskType.SYSTEMIC_CRASH:
            if confidence >= 0.8:
                recommendation = "强烈建议避险：减仓或做空，等待系统性风险释放"
                cooling_period = 15  # 2-3周
            else:
                recommendation = "建议谨慎：减少风险敞口，密切关注宏观指标"
                cooling_period = 10  # 1-2周

        elif risk_type == RiskType.SHORT_CORRECTION:
            if confidence >= 0.7:
                recommendation = "短期回调：可考虑逢低布局，但需设置止损"
                cooling_period = 3   # 3-5天
            else:
                recommendation = "观望为主：等待回调结束信号"
                cooling_period = 5   # 5-7天

        elif risk_type == RiskType.TREND_REVERSAL:
            recommendation = "趋势转换：根据新趋势调整策略"
            cooling_period = 7   # 1周

        else:
            recommendation = "市场不明朗：保持中性仓位"
            cooling_period = 5

        return recommendation, cooling_period

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取信心度等级"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    # 宏观基本面分析的具体实现方法
    def _assess_economic_cycle(self, market_data: Dict) -> float:
        """评估经济周期位置 (0-25分)"""
        score = 0

        # 这里需要根据实际可获得的数据来实现
        # 示例：基于PMI、失业率、通胀率等指标
        try:
            # 假设从market_data中获取经济指标
            pmi = market_data.get("pmi", 50)  # 制造业PMI
            unemployment = market_data.get("unemployment_rate", 5)  # 失业率
            inflation = market_data.get("inflation_rate", 2)  # 通胀率

            # PMI评分 (0-10分)
            if pmi < 45:  # 严重收缩
                score += 10
            elif pmi < 48:  # 轻度收缩
                score += 7
            elif pmi < 52:  # 中性
                score += 3
            # PMI > 52 不加分（经济扩张）

            # 失业率评分 (0-8分)
            if unemployment > 8:  # 高失业
                score += 8
            elif unemployment > 6:  # 中等失业
                score += 5
            elif unemployment > 4:  # 轻度失业
                score += 2

            # 通胀评分 (0-7分)
            if inflation > 5:  # 高通胀
                score += 7
            elif inflation > 3:  # 中等通胀
                score += 3
            elif inflation < 0:  # 通缩
                score += 5

        except Exception as e:
            logger.warning(f"经济周期评估失败: {e}")
            score = 12  # 默认中等风险

        return min(score, 25)

    def _assess_financial_system_health(self, market_data: Dict) -> float:
        """评估金融体系健康度 (0-25分)"""
        score = 0

        try:
            # 银行体系指标
            bank_capital_ratio = market_data.get("bank_capital_ratio", 12)  # 银行资本充足率
            npl_ratio = market_data.get("npl_ratio", 2)  # 不良贷款率

            # 流动性指标
            libor_ois_spread = market_data.get("libor_ois_spread", 0.1)  # Libor-OIS利差
            ted_spread = market_data.get("ted_spread", 0.2)  # TED利差

            # 银行资本充足率评分 (0-8分)
            if bank_capital_ratio < 8:  # 资本不足
                score += 8
            elif bank_capital_ratio < 10:  # 资本偏低
                score += 5
            elif bank_capital_ratio < 12:  # 资本一般
                score += 2

            # 不良贷款率评分 (0-7分)
            if npl_ratio > 5:  # 高不良率
                score += 7
            elif npl_ratio > 3:  # 中等不良率
                score += 4
            elif npl_ratio > 2:  # 轻度不良率
                score += 1

            # 流动性紧张评分 (0-10分)
            if libor_ois_spread > 0.5 or ted_spread > 1.0:  # 严重流动性紧张
                score += 10
            elif libor_ois_spread > 0.3 or ted_spread > 0.5:  # 中度流动性紧张
                score += 6
            elif libor_ois_spread > 0.2 or ted_spread > 0.3:  # 轻度流动性紧张
                score += 3

        except Exception as e:
            logger.warning(f"金融体系健康度评估失败: {e}")
            score = 10  # 默认中等风险

        return min(score, 25)

    def _assess_policy_space(self, market_data: Dict) -> float:
        """评估政策空间 (0-20分)"""
        score = 0

        try:
            # 货币政策空间
            interest_rate = market_data.get("interest_rate", 2.5)  # 基准利率
            inflation_rate = market_data.get("inflation_rate", 2)  # 通胀率

            # 财政政策空间
            debt_to_gdp = market_data.get("debt_to_gdp", 60)  # 债务/GDP比率
            fiscal_deficit = market_data.get("fiscal_deficit", 3)  # 财政赤字率

            # 货币政策空间评分 (0-12分)
            real_rate = interest_rate - inflation_rate
            if real_rate < 0:  # 负实际利率，空间有限
                score += 8
            elif real_rate < 1:  # 低实际利率
                score += 5
            elif real_rate < 2:  # 中等实际利率
                score += 2

            # 财政政策空间评分 (0-8分)
            if debt_to_gdp > 100 or fiscal_deficit > 6:  # 财政空间有限
                score += 8
            elif debt_to_gdp > 80 or fiscal_deficit > 4:  # 财政空间偏紧
                score += 5
            elif debt_to_gdp > 60 or fiscal_deficit > 3:  # 财政空间一般
                score += 2

        except Exception as e:
            logger.warning(f"政策空间评估失败: {e}")
            score = 8  # 默认中等风险

        return min(score, 20)

    def _assess_valuation_levels(self, market_data: Dict) -> float:
        """评估整体估值水平 (0-20分)"""
        score = 0

        try:
            # 市场整体估值指标
            market_pe = market_data.get("market_pe", 20)  # 市场平均P/E
            market_pb = market_data.get("market_pb", 2.5)  # 市场平均P/B
            cape_ratio = market_data.get("cape_ratio", 25)  # 周期调整P/E

            # 历史分位数
            pe_percentile = market_data.get("pe_percentile", 50)  # P/E历史分位数
            pb_percentile = market_data.get("pb_percentile", 50)  # P/B历史分位数

            # P/E估值评分 (0-8分)
            if pe_percentile > 90:  # 极度高估
                score += 8
            elif pe_percentile > 80:  # 高估
                score += 6
            elif pe_percentile > 70:  # 偏高
                score += 3

            # P/B估值评分 (0-6分)
            if pb_percentile > 90:  # 极度高估
                score += 6
            elif pb_percentile > 80:  # 高估
                score += 4
            elif pb_percentile > 70:  # 偏高
                score += 2

            # CAPE评分 (0-6分)
            if cape_ratio > 30:  # 历史高位
                score += 6
            elif cape_ratio > 25:  # 偏高
                score += 4
            elif cape_ratio > 20:  # 中等
                score += 2

        except Exception as e:
            logger.warning(f"估值水平评估失败: {e}")
            score = 8  # 默认中等风险

        return min(score, 20)

    def _assess_systemic_risk_indicators(self, market_data: Dict) -> float:
        """评估系统性风险指标 (0-10分)"""
        score = 0

        try:
            # 金融稳定指标
            financial_stress_index = market_data.get("financial_stress_index", 0)
            credit_spread = market_data.get("credit_spread", 1.5)  # 信用利差

            # 金融压力指数评分 (0-6分)
            if financial_stress_index > 2:  # 高压力
                score += 6
            elif financial_stress_index > 1:  # 中等压力
                score += 4
            elif financial_stress_index > 0.5:  # 轻度压力
                score += 2

            # 信用利差评分 (0-4分)
            if credit_spread > 4:  # 高信用风险
                score += 4
            elif credit_spread > 2.5:  # 中等信用风险
                score += 2
            elif credit_spread > 2:  # 轻度信用风险
                score += 1

        except Exception as e:
            logger.warning(f"系统性风险指标评估失败: {e}")
            score = 3  # 默认低风险

        return min(score, 10)

    # 市场广度分析的具体实现方法
    def _analyze_advance_decline_ratio(self, market_data: Dict) -> float:
        """分析涨跌家数比 (0-30分)"""
        score = 0

        try:
            # 获取涨跌家数数据
            advancing_stocks = market_data.get("advancing_stocks", 1500)
            declining_stocks = market_data.get("declining_stocks", 1500)
            unchanged_stocks = market_data.get("unchanged_stocks", 100)

            total_stocks = advancing_stocks + declining_stocks + unchanged_stocks
            if total_stocks == 0:
                return 15  # 默认中等风险

            decline_ratio = declining_stocks / total_stocks
            advance_ratio = advancing_stocks / total_stocks

            # 下跌家数占比评分
            if decline_ratio > 0.8:  # 80%以上下跌
                score += 30
            elif decline_ratio > 0.7:  # 70-80%下跌
                score += 25
            elif decline_ratio > 0.6:  # 60-70%下跌
                score += 20
            elif decline_ratio > 0.55:  # 55-60%下跌
                score += 15
            elif decline_ratio > 0.5:  # 50-55%下跌
                score += 10
            elif decline_ratio > 0.45:  # 45-50%下跌
                score += 5
            # decline_ratio <= 0.45 不加分

        except Exception as e:
            logger.warning(f"涨跌家数比分析失败: {e}")
            score = 15  # 默认中等风险

        return min(score, 30)

    def _analyze_new_highs_lows(self, market_data: Dict) -> float:
        """分析新高新低指数 (0-25分)"""
        score = 0

        try:
            # 获取新高新低数据
            new_highs = market_data.get("new_highs_52w", 100)
            new_lows = market_data.get("new_lows_52w", 100)

            total_new_extremes = new_highs + new_lows
            if total_new_extremes == 0:
                return 10  # 默认低风险

            new_lows_ratio = new_lows / total_new_extremes

            # 新低占比评分
            if new_lows_ratio > 0.9:  # 90%以上为新低
                score += 25
            elif new_lows_ratio > 0.8:  # 80-90%为新低
                score += 20
            elif new_lows_ratio > 0.7:  # 70-80%为新低
                score += 15
            elif new_lows_ratio > 0.6:  # 60-70%为新低
                score += 10
            elif new_lows_ratio > 0.5:  # 50-60%为新低
                score += 5
            # new_lows_ratio <= 0.5 不加分

        except Exception as e:
            logger.warning(f"新高新低分析失败: {e}")
            score = 10  # 默认低风险

        return min(score, 25)

    def _analyze_sector_performance(self, market_data: Dict, ticker: str) -> float:
        """分析板块表现 (0-25分)"""
        score = 0

        try:
            # 获取各板块表现数据
            sector_performance = market_data.get("sector_performance", {})
            defensive_sectors = ["utilities", "consumer_staples", "healthcare"]
            cyclical_sectors = ["technology", "financials", "industrials", "materials"]

            # 计算防御性板块和周期性板块的平均表现
            defensive_avg = 0
            cyclical_avg = 0
            defensive_count = 0
            cyclical_count = 0

            for sector, performance in sector_performance.items():
                if sector in defensive_sectors:
                    defensive_avg += performance
                    defensive_count += 1
                elif sector in cyclical_sectors:
                    cyclical_avg += performance
                    cyclical_count += 1

            if defensive_count > 0:
                defensive_avg /= defensive_count
            if cyclical_count > 0:
                cyclical_avg /= cyclical_count

            # 防御性板块表现评分 (0-15分)
            if defensive_avg < -5:  # 防御性板块大跌
                score += 15
            elif defensive_avg < -2:  # 防御性板块下跌
                score += 10
            elif defensive_avg < 0:  # 防御性板块微跌
                score += 5
            # defensive_avg >= 0 不加分

            # 周期性板块表现评分 (0-10分)
            if cyclical_avg < -10:  # 周期性板块暴跌
                score += 10
            elif cyclical_avg < -5:  # 周期性板块大跌
                score += 7
            elif cyclical_avg < -2:  # 周期性板块下跌
                score += 4
            elif cyclical_avg < 0:  # 周期性板块微跌
                score += 2

        except Exception as e:
            logger.warning(f"板块表现分析失败: {e}")
            score = 10  # 默认中等风险

        return min(score, 25)

    def _analyze_correlation_patterns(self, market_data: Dict, ticker: str) -> float:
        """分析相关性模式 (0-20分)"""
        score = 0

        try:
            # 获取相关性数据
            avg_correlation = market_data.get("avg_stock_correlation", 0.3)
            sector_correlation = market_data.get("sector_correlation", 0.5)

            # 整体相关性评分 (0-12分)
            if avg_correlation > 0.8:  # 极高相关性
                score += 12
            elif avg_correlation > 0.7:  # 高相关性
                score += 9
            elif avg_correlation > 0.6:  # 中高相关性
                score += 6
            elif avg_correlation > 0.5:  # 中等相关性
                score += 3
            # avg_correlation <= 0.5 不加分

            # 板块相关性评分 (0-8分)
            if sector_correlation > 0.9:  # 板块间极高相关性
                score += 8
            elif sector_correlation > 0.8:  # 板块间高相关性
                score += 6
            elif sector_correlation > 0.7:  # 板块间中高相关性
                score += 4
            elif sector_correlation > 0.6:  # 板块间中等相关性
                score += 2

        except Exception as e:
            logger.warning(f"相关性分析失败: {e}")
            score = 8  # 默认中等风险

        return min(score, 20)

    # 技术分析的具体实现方法
    def _analyze_trend_indicators(self, price_history: pd.DataFrame, current_price: float) -> float:
        """分析趋势指标 (0-30分)"""
        score = 0

        try:
            if len(price_history) < 200:
                return 15  # 数据不足，默认中等风险

            # 计算移动平均线
            sma_50 = price_history["close"].rolling(50).mean().iloc[-1]
            sma_200 = price_history["close"].rolling(200).mean().iloc[-1]

            # 价格相对于移动平均线的位置
            price_vs_sma50 = (current_price - sma_50) / sma_50
            price_vs_sma200 = (current_price - sma_200) / sma_200

            # 移动平均线排列
            ma_bearish = sma_50 < sma_200

            # 趋势强度评分 (0-15分)
            if price_vs_sma200 < -0.2:  # 远低于200日均线
                score += 15
            elif price_vs_sma200 < -0.1:  # 低于200日均线
                score += 10
            elif price_vs_sma200 < -0.05:  # 略低于200日均线
                score += 5
            elif price_vs_sma200 < 0:  # 微低于200日均线
                score += 2

            # 移动平均线排列评分 (0-10分)
            if ma_bearish:
                score += 10

            # 短期趋势评分 (0-5分)
            if price_vs_sma50 < -0.1:  # 远低于50日均线
                score += 5
            elif price_vs_sma50 < -0.05:  # 低于50日均线
                score += 3
            elif price_vs_sma50 < 0:  # 微低于50日均线
                score += 1

        except Exception as e:
            logger.warning(f"趋势指标分析失败: {e}")
            score = 15  # 默认中等风险

        return min(score, 30)

    def _analyze_momentum_indicators(self, price_history: pd.DataFrame) -> float:
        """分析动能指标 (0-25分)"""
        score = 0

        try:
            if len(price_history) < 28:
                return 12  # 数据不足，默认中等风险

            # 计算RSI
            delta = price_history["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # 计算MACD
            ema_12 = price_history["close"].ewm(span=12).mean()
            ema_26 = price_history["close"].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal

            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_histogram = histogram.iloc[-1]

            # RSI评分 (0-12分)
            if current_rsi < 20:  # 极度超卖，可能反弹
                score += 3  # 低风险
            elif current_rsi < 30:  # 超卖
                score += 1
            elif current_rsi > 80:  # 极度超买
                score += 12
            elif current_rsi > 70:  # 超买
                score += 8
            elif current_rsi > 60:  # 偏强
                score += 4

            # MACD评分 (0-8分)
            if current_macd < current_signal and current_histogram < 0:  # 空头信号
                score += 8
            elif current_macd < current_signal:  # 弱势
                score += 5
            elif current_histogram < 0:  # 动能减弱
                score += 3

            # 动量持续性评分 (0-5分)
            recent_returns = price_history["close"].pct_change().tail(5)
            negative_days = (recent_returns < 0).sum()
            if negative_days >= 4:  # 连续下跌
                score += 5
            elif negative_days >= 3:  # 多数下跌
                score += 3
            elif negative_days >= 2:  # 部分下跌
                score += 1

        except Exception as e:
            logger.warning(f"动能指标分析失败: {e}")
            score = 12  # 默认中等风险

        return min(score, 25)

    def _analyze_volume_patterns(self, price_history: pd.DataFrame) -> float:
        """分析成交量模式 (0-25分)"""
        score = 0

        try:
            if len(price_history) < 20:
                return 12  # 数据不足，默认中等风险

            # 计算成交量指标
            volume = price_history["volume"]
            avg_volume_20 = volume.rolling(20).mean()
            current_volume = volume.iloc[-1]
            avg_volume = avg_volume_20.iloc[-1]

            # 价格变化
            price_change = price_history["close"].pct_change().iloc[-1]

            # 成交量放大评分 (0-15分)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 3 and price_change < -0.05:  # 大量下跌
                score += 15
            elif volume_ratio > 2 and price_change < -0.03:  # 放量下跌
                score += 10
            elif volume_ratio > 1.5 and price_change < -0.02:  # 温和放量下跌
                score += 6
            elif volume_ratio > 1.2 and price_change < 0:  # 轻微放量下跌
                score += 3

            # 成交量趋势评分 (0-10分)
            volume_trend = volume.tail(5).mean() / volume.tail(20).mean()
            if volume_trend > 1.5:  # 成交量持续放大
                score += 10
            elif volume_trend > 1.3:  # 成交量明显增加
                score += 7
            elif volume_trend > 1.1:  # 成交量略有增加
                score += 4

        except Exception as e:
            logger.warning(f"成交量分析失败: {e}")
            score = 12  # 默认中等风险

        return min(score, 25)

    def _analyze_price_patterns(self, price_history: pd.DataFrame) -> float:
        """分析价格形态 (0-20分)"""
        score = 0

        try:
            if len(price_history) < 20:
                return 10  # 数据不足，默认中等风险

            # 计算价格变化
            returns = price_history["close"].pct_change()

            # 连续下跌形态评分 (0-10分)
            recent_returns = returns.tail(10)
            negative_count = (recent_returns < 0).sum()
            if negative_count >= 8:  # 10天内8天下跌
                score += 10
            elif negative_count >= 6:  # 10天内6天下跌
                score += 7
            elif negative_count >= 5:  # 10天内5天下跌
                score += 4

            # 跳空缺口评分 (0-5分)
            gaps = []
            for i in range(1, min(10, len(price_history))):
                prev_close = price_history["close"].iloc[-i-1]
                curr_open = price_history["open"].iloc[-i] if "open" in price_history.columns else prev_close
                gap = (curr_open - prev_close) / prev_close
                if gap < -0.02:  # 向下跳空超过2%
                    gaps.append(gap)

            if len(gaps) >= 2:  # 多个向下跳空
                score += 5
            elif len(gaps) >= 1:  # 单个向下跳空
                score += 3

            # 波动率评分 (0-5分)
            volatility = returns.tail(20).std() * np.sqrt(252)  # 年化波动率
            if volatility > 0.6:  # 高波动率
                score += 5
            elif volatility > 0.4:  # 中高波动率
                score += 3
            elif volatility > 0.3:  # 中等波动率
                score += 1

        except Exception as e:
            logger.warning(f"价格形态分析失败: {e}")
            score = 10  # 默认中等风险

        return min(score, 20)

    # 消息面分析的具体实现方法
    def _analyze_event_nature(self, news_data: List[Dict], ticker: str) -> float:
        """分析事件性质 (0-40分)"""
        score = 0

        try:
            # 分析新闻的影响范围和严重性
            company_specific_count = 0
            industry_wide_count = 0
            systemic_count = 0

            for news in news_data:
                content = news.get("content", "").lower()
                title = news.get("title", "").lower()
                text = content + " " + title

                # 判断影响范围
                if any(word in text for word in ["bankruptcy", "fraud", "scandal", "lawsuit"]):
                    if ticker.lower() in text:
                        company_specific_count += 1
                    else:
                        industry_wide_count += 1
                elif any(word in text for word in ["recession", "crisis", "crash", "collapse", "pandemic"]):
                    systemic_count += 1
                elif any(word in text for word in ["regulation", "policy", "interest rate", "inflation"]):
                    systemic_count += 0.5
                else:
                    company_specific_count += 0.5

            # 系统性事件评分 (0-25分)
            if systemic_count >= 3:  # 多个系统性事件
                score += 25
            elif systemic_count >= 2:  # 较多系统性事件
                score += 20
            elif systemic_count >= 1:  # 部分系统性事件
                score += 15
            elif systemic_count >= 0.5:  # 轻微系统性事件
                score += 10

            # 行业事件评分 (0-10分)
            if industry_wide_count >= 2:  # 多个行业事件
                score += 10
            elif industry_wide_count >= 1:  # 单个行业事件
                score += 7
            elif industry_wide_count >= 0.5:  # 轻微行业事件
                score += 4

            # 公司特定事件评分 (0-5分，反向评分)
            if company_specific_count >= 3:  # 主要是公司特定事件
                score += 5  # 低系统性风险
            elif company_specific_count >= 2:
                score += 3
            elif company_specific_count >= 1:
                score += 1

        except Exception as e:
            logger.warning(f"事件性质分析失败: {e}")
            score = 20  # 默认中等风险

        return min(score, 40)

    def _analyze_event_solvability(self, news_data: List[Dict]) -> float:
        """分析事件可解决性 (0-35分)"""
        score = 0

        try:
            solvable_events = 0
            unsolvable_events = 0

            for news in news_data:
                content = news.get("content", "").lower()
                title = news.get("title", "").lower()
                text = content + " " + title

                # 可解决的事件
                if any(word in text for word in ["settlement", "agreement", "resolution", "fix", "solution"]):
                    solvable_events += 1
                elif any(word in text for word in ["temporary", "short-term", "adjustment"]):
                    solvable_events += 0.5

                # 难以解决的事件
                elif any(word in text for word in ["structural", "fundamental", "permanent", "irreversible"]):
                    unsolvable_events += 1
                elif any(word in text for word in ["crisis", "collapse", "bankruptcy"]):
                    unsolvable_events += 0.8

            # 不可解决事件评分 (0-25分)
            if unsolvable_events >= 2:  # 多个不可解决事件
                score += 25
            elif unsolvable_events >= 1:  # 单个不可解决事件
                score += 20
            elif unsolvable_events >= 0.5:  # 部分不可解决事件
                score += 15

            # 可解决事件评分 (0-10分，反向评分)
            if solvable_events >= 2:  # 多个可解决事件
                score += 5  # 低风险
            elif solvable_events >= 1:  # 单个可解决事件
                score += 3
            elif solvable_events >= 0.5:  # 部分可解决事件
                score += 1

        except Exception as e:
            logger.warning(f"事件可解决性分析失败: {e}")
            score = 15  # 默认中等风险

        return min(score, 35)

    def _analyze_market_reaction_comparison(self, news_data: List[Dict], ticker: str) -> float:
        """分析市场反应对比 (0-25分)"""
        score = 0

        try:
            # 这里需要获取其他股票的反应数据进行对比
            # 由于数据限制，使用简化的逻辑

            # 基于新闻数量和严重性评估
            severe_news_count = 0
            for news in news_data:
                content = news.get("content", "").lower()
                title = news.get("title", "").lower()
                text = content + " " + title

                if any(word in text for word in ["plunge", "crash", "collapse", "crisis"]):
                    severe_news_count += 1
                elif any(word in text for word in ["drop", "fall", "decline", "down"]):
                    severe_news_count += 0.5

            # 严重新闻评分
            if severe_news_count >= 3:  # 多个严重新闻
                score += 25
            elif severe_news_count >= 2:  # 较多严重新闻
                score += 20
            elif severe_news_count >= 1:  # 部分严重新闻
                score += 15
            elif severe_news_count >= 0.5:  # 轻微负面新闻
                score += 10

        except Exception as e:
            logger.warning(f"市场反应对比分析失败: {e}")
            score = 12  # 默认中等风险

        return min(score, 25)

    # 情绪指标分析的具体实现方法
    def _analyze_vix_levels(self, market_data: Dict) -> float:
        """分析VIX恐慌指数 (0-40分)"""
        score = 0

        try:
            vix = market_data.get("vix", 20)  # VIX指数
            vix_percentile = market_data.get("vix_percentile", 50)  # VIX历史分位数

            # VIX绝对水平评分 (0-25分)
            if vix > 40:  # 极度恐慌
                score += 25
            elif vix > 30:  # 高度恐慌
                score += 20
            elif vix > 25:  # 中度恐慌
                score += 15
            elif vix > 20:  # 轻度恐慌
                score += 10
            elif vix > 15:  # 正常偏高
                score += 5

            # VIX相对水平评分 (0-15分)
            if vix_percentile > 95:  # 历史极高位
                score += 15
            elif vix_percentile > 90:  # 历史高位
                score += 12
            elif vix_percentile > 80:  # 历史偏高
                score += 8
            elif vix_percentile > 70:  # 历史中高
                score += 5

        except Exception as e:
            logger.warning(f"VIX分析失败: {e}")
            score = 15  # 默认中等风险

        return min(score, 40)

    def _analyze_options_skew(self, market_data: Dict) -> float:
        """分析期权偏度 (0-30分)"""
        score = 0

        try:
            put_call_ratio = market_data.get("put_call_ratio", 1.0)  # 看跌/看涨期权比率
            skew_index = market_data.get("skew_index", 100)  # 偏度指数

            # 看跌看涨比率评分 (0-20分)
            if put_call_ratio > 1.5:  # 极度看跌情绪
                score += 20
            elif put_call_ratio > 1.3:  # 高度看跌情绪
                score += 15
            elif put_call_ratio > 1.1:  # 中度看跌情绪
                score += 10
            elif put_call_ratio > 1.0:  # 轻度看跌情绪
                score += 5

            # 偏度指数评分 (0-10分)
            if skew_index > 130:  # 极高偏度
                score += 10
            elif skew_index > 120:  # 高偏度
                score += 7
            elif skew_index > 110:  # 中等偏度
                score += 4
            elif skew_index > 100:  # 轻微偏度
                score += 2

        except Exception as e:
            logger.warning(f"期权偏度分析失败: {e}")
            score = 10  # 默认中等风险

        return min(score, 30)

    def _analyze_money_flow(self, market_data: Dict, price_history: pd.DataFrame) -> float:
        """分析资金流向 (0-30分)"""
        score = 0

        try:
            # 资金流向指标
            money_flow_index = market_data.get("money_flow_index", 50)  # 资金流量指数
            institutional_flow = market_data.get("institutional_flow", 0)  # 机构资金流向

            # 资金流量指数评分 (0-20分)
            if money_flow_index < 20:  # 极度资金流出
                score += 20
            elif money_flow_index < 30:  # 大量资金流出
                score += 15
            elif money_flow_index < 40:  # 中等资金流出
                score += 10
            elif money_flow_index < 50:  # 轻微资金流出
                score += 5

            # 机构资金流向评分 (0-10分)
            if institutional_flow < -1000:  # 大量机构资金流出
                score += 10
            elif institutional_flow < -500:  # 中等机构资金流出
                score += 7
            elif institutional_flow < -100:  # 轻微机构资金流出
                score += 4
            elif institutional_flow < 0:  # 微量机构资金流出
                score += 2

        except Exception as e:
            logger.warning(f"资金流向分析失败: {e}")
            score = 10  # 默认中等风险

        return min(score, 30)

    # 解释函数
    def _interpret_macro_score(self, score: float) -> str:
        """解释宏观评分"""
        if score >= 70:
            return "宏观环境极度恶化，系统性风险很高"
        elif score >= 50:
            return "宏观环境明显恶化，系统性风险较高"
        elif score >= 30:
            return "宏观环境有所恶化，存在一定系统性风险"
        elif score >= 15:
            return "宏观环境基本稳定，系统性风险较低"
        else:
            return "宏观环境良好，系统性风险很低"

    def _interpret_breadth_score(self, score: float) -> str:
        """解释市场广度评分"""
        if score >= 70:
            return "市场广度极度恶化，普跌格局明显"
        elif score >= 50:
            return "市场广度明显恶化，下跌面较广"
        elif score >= 30:
            return "市场广度有所恶化，部分板块承压"
        elif score >= 15:
            return "市场广度基本健康，结构性调整"
        else:
            return "市场广度良好，上涨面较广"

    def _interpret_technical_score(self, score: float) -> str:
        """解释技术评分"""
        if score >= 70:
            return "技术面极度恶化，趋势明确向下"
        elif score >= 50:
            return "技术面明显恶化，下行压力较大"
        elif score >= 30:
            return "技术面有所恶化，存在调整压力"
        elif score >= 15:
            return "技术面基本稳定，震荡整理"
        else:
            return "技术面良好，上行趋势明确"

    def _interpret_news_score(self, score: float) -> str:
        """解释消息面评分"""
        if score >= 70:
            return "重大负面消息，影响深远且难以解决"
        elif score >= 50:
            return "明显负面消息，影响较大"
        elif score >= 30:
            return "一般负面消息，影响有限"
        elif score >= 15:
            return "轻微负面消息，影响较小"
        else:
            return "消息面中性或偏正面"

    def _interpret_sentiment_score(self, score: float) -> str:
        """解释情绪评分"""
        if score >= 70:
            return "市场情绪极度恐慌，资金大量流出"
        elif score >= 50:
            return "市场情绪明显恐慌，避险情绪浓厚"
        elif score >= 30:
            return "市场情绪有所恐慌，谨慎情绪上升"
        elif score >= 15:
            return "市场情绪基本稳定，略显谨慎"
        else:
            return "市场情绪良好，风险偏好较高"
