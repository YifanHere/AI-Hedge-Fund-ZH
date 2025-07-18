"""
情绪与资金流指标分析模块

实现恐慌指数(VIX)、期权偏度、资金流向等指标的分析功能
用于判断市场情绪极端程度和资金流向变化
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SentimentLevel(Enum):
    """情绪水平"""
    EXTREME_FEAR = "extreme_fear"        # 极度恐慌
    HIGH_FEAR = "high_fear"             # 高度恐慌
    MODERATE_FEAR = "moderate_fear"     # 中度恐慌
    NEUTRAL = "neutral"                 # 中性
    MODERATE_GREED = "moderate_greed"   # 中度贪婪
    HIGH_GREED = "high_greed"           # 高度贪婪
    EXTREME_GREED = "extreme_greed"     # 极度贪婪

class MoneyFlowDirection(Enum):
    """资金流向"""
    MASSIVE_OUTFLOW = "massive_outflow"     # 大量流出
    MODERATE_OUTFLOW = "moderate_outflow"   # 中等流出
    SLIGHT_OUTFLOW = "slight_outflow"       # 轻微流出
    BALANCED = "balanced"                   # 平衡
    SLIGHT_INFLOW = "slight_inflow"         # 轻微流入
    MODERATE_INFLOW = "moderate_inflow"     # 中等流入
    MASSIVE_INFLOW = "massive_inflow"       # 大量流入

class RiskAppetite(Enum):
    """风险偏好"""
    RISK_OFF = "risk_off"               # 避险
    CAUTIOUS = "cautious"               # 谨慎
    NEUTRAL = "neutral"                 # 中性
    RISK_ON = "risk_on"                 # 风险偏好
    AGGRESSIVE = "aggressive"           # 激进

@dataclass
class SentimentAnalysisResult:
    """情绪分析结果"""
    sentiment_level: SentimentLevel
    money_flow_direction: MoneyFlowDirection
    risk_appetite: RiskAppetite
    fear_greed_index: float
    vix_analysis: Dict[str, float]
    options_analysis: Dict[str, float]
    money_flow_analysis: Dict[str, float]
    contrarian_signals: List[str]
    sentiment_extremes: List[str]
    interpretation: str

class SentimentFlowAnalyzer:
    """情绪与资金流分析器"""
    
    def __init__(self):
        # VIX水平阈值
        self.vix_thresholds = {
            "extreme_fear": 40,      # 极度恐慌
            "high_fear": 30,         # 高度恐慌
            "moderate_fear": 25,     # 中度恐慌
            "normal_high": 20,       # 正常偏高
            "normal": 15,            # 正常
            "complacent": 12         # 自满
        }
        
        # 期权指标阈值
        self.options_thresholds = {
            "put_call_extreme": 1.5,     # 看跌看涨比极端值
            "put_call_high": 1.2,        # 看跌看涨比高值
            "skew_extreme": 130,         # 偏度指数极端值
            "skew_high": 120,            # 偏度指数高值
            "iv_spike": 1.5              # 隐含波动率激增倍数
        }
        
        # 资金流指标阈值
        self.flow_thresholds = {
            "massive_outflow": -2000,    # 大量资金流出（百万美元）
            "moderate_outflow": -500,    # 中等资金流出
            "slight_outflow": -100,      # 轻微资金流出
            "slight_inflow": 100,        # 轻微资金流入
            "moderate_inflow": 500,      # 中等资金流入
            "massive_inflow": 2000       # 大量资金流入
        }

    def analyze_sentiment_flow(self, market_data: Dict, options_data: Dict,
                             flow_data: Dict, end_date: str) -> SentimentAnalysisResult:
        """
        综合分析市场情绪和资金流
        
        Args:
            market_data: 市场数据（包含VIX等）
            options_data: 期权数据
            flow_data: 资金流数据
            end_date: 分析日期
            
        Returns:
            SentimentAnalysisResult: 情绪分析结果
        """
        
        # 1. VIX恐慌指数分析
        vix_analysis = self._analyze_vix_indicators(market_data)
        
        # 2. 期权情绪分析
        options_analysis = self._analyze_options_sentiment(options_data)
        
        # 3. 资金流向分析
        money_flow_analysis = self._analyze_money_flow(flow_data)
        
        # 4. 综合情绪评估
        sentiment_assessment = self._assess_overall_sentiment(
            vix_analysis, options_analysis, money_flow_analysis
        )
        
        return sentiment_assessment

    def _analyze_vix_indicators(self, market_data: Dict) -> Dict:
        """
        分析VIX相关指标
        
        关键指标：
        - VIX绝对水平
        - VIX相对历史分位数
        - VIX期限结构
        - VIX变化趋势
        """
        
        # 获取VIX数据
        vix_current = market_data.get("vix", 20)
        vix_percentile = market_data.get("vix_percentile", 50)
        vix_9d = market_data.get("vix_9d", vix_current)  # 9日VIX
        vix_30d = market_data.get("vix_30d", vix_current)  # 30日VIX
        vix_change = market_data.get("vix_change", 0)  # VIX变化
        
        # VIX期限结构分析
        term_structure = self._analyze_vix_term_structure(vix_9d, vix_current, vix_30d)
        
        # VIX水平评估
        vix_level_assessment = self._assess_vix_level(vix_current, vix_percentile)
        
        # VIX趋势分析
        vix_trend = self._analyze_vix_trend(vix_change, vix_current)
        
        # 计算VIX风险评分
        vix_risk_score = self._calculate_vix_risk_score(
            vix_current, vix_percentile, vix_change, term_structure
        )
        
        return {
            "vix_current": vix_current,
            "vix_percentile": vix_percentile,
            "vix_9d": vix_9d,
            "vix_30d": vix_30d,
            "vix_change": vix_change,
            "term_structure": term_structure,
            "level_assessment": vix_level_assessment,
            "trend": vix_trend,
            "risk_score": vix_risk_score
        }

    def _analyze_options_sentiment(self, options_data: Dict) -> Dict:
        """
        分析期权情绪指标
        
        关键指标：
        - 看跌看涨期权比率
        - 期权偏度指数
        - 隐含波动率
        - 期权交易量
        """
        
        # 获取期权数据
        put_call_ratio = options_data.get("put_call_ratio", 1.0)
        skew_index = options_data.get("skew_index", 100)
        implied_volatility = options_data.get("implied_volatility", 0.2)
        iv_percentile = options_data.get("iv_percentile", 50)
        options_volume = options_data.get("options_volume", 0)
        
        # 看跌看涨比率分析
        put_call_analysis = self._analyze_put_call_ratio(put_call_ratio)
        
        # 偏度指数分析
        skew_analysis = self._analyze_skew_index(skew_index)
        
        # 隐含波动率分析
        iv_analysis = self._analyze_implied_volatility(implied_volatility, iv_percentile)
        
        # 期权交易量分析
        volume_analysis = self._analyze_options_volume(options_volume)
        
        # 计算期权情绪评分
        options_sentiment_score = self._calculate_options_sentiment_score(
            put_call_ratio, skew_index, iv_percentile, options_volume
        )
        
        return {
            "put_call_ratio": put_call_ratio,
            "skew_index": skew_index,
            "implied_volatility": implied_volatility,
            "iv_percentile": iv_percentile,
            "options_volume": options_volume,
            "put_call_analysis": put_call_analysis,
            "skew_analysis": skew_analysis,
            "iv_analysis": iv_analysis,
            "volume_analysis": volume_analysis,
            "sentiment_score": options_sentiment_score
        }

    def _analyze_money_flow(self, flow_data: Dict) -> Dict:
        """
        分析资金流向
        
        关键指标：
        - 机构资金流向
        - 散户资金流向
        - 板块资金流向
        - 避险资产流向
        """
        
        # 获取资金流数据
        institutional_flow = flow_data.get("institutional_flow", 0)  # 机构资金流（百万美元）
        retail_flow = flow_data.get("retail_flow", 0)  # 散户资金流
        equity_flow = flow_data.get("equity_flow", 0)  # 股票资金流
        bond_flow = flow_data.get("bond_flow", 0)  # 债券资金流
        gold_flow = flow_data.get("gold_flow", 0)  # 黄金资金流
        cash_flow = flow_data.get("cash_flow", 0)  # 现金流
        
        # 总体资金流向
        total_equity_flow = institutional_flow + retail_flow + equity_flow
        
        # 避险资产流向
        safe_haven_flow = bond_flow + gold_flow + cash_flow
        
        # 资金流向方向判断
        flow_direction = self._determine_flow_direction(total_equity_flow)
        
        # 风险偏好分析
        risk_appetite = self._analyze_risk_appetite(total_equity_flow, safe_haven_flow)
        
        # 计算资金流风险评分
        flow_risk_score = self._calculate_flow_risk_score(
            total_equity_flow, safe_haven_flow, institutional_flow
        )
        
        return {
            "institutional_flow": institutional_flow,
            "retail_flow": retail_flow,
            "total_equity_flow": total_equity_flow,
            "safe_haven_flow": safe_haven_flow,
            "flow_direction": flow_direction,
            "risk_appetite": risk_appetite,
            "risk_score": flow_risk_score
        }

    def _assess_overall_sentiment(self, vix_analysis: Dict, options_analysis: Dict,
                                money_flow_analysis: Dict) -> SentimentAnalysisResult:
        """综合评估市场情绪"""
        
        # 计算综合恐慌贪婪指数
        fear_greed_index = self._calculate_fear_greed_index(
            vix_analysis, options_analysis, money_flow_analysis
        )
        
        # 确定情绪水平
        sentiment_level = self._determine_sentiment_level(fear_greed_index)
        
        # 识别逆向信号
        contrarian_signals = self._identify_contrarian_signals(
            vix_analysis, options_analysis, money_flow_analysis
        )
        
        # 识别情绪极端
        sentiment_extremes = self._identify_sentiment_extremes(
            vix_analysis, options_analysis, money_flow_analysis
        )
        
        # 生成解释
        interpretation = self._generate_sentiment_interpretation(
            sentiment_level, money_flow_analysis["flow_direction"],
            money_flow_analysis["risk_appetite"], contrarian_signals
        )
        
        return SentimentAnalysisResult(
            sentiment_level=sentiment_level,
            money_flow_direction=money_flow_analysis["flow_direction"],
            risk_appetite=money_flow_analysis["risk_appetite"],
            fear_greed_index=fear_greed_index,
            vix_analysis=vix_analysis,
            options_analysis=options_analysis,
            money_flow_analysis=money_flow_analysis,
            contrarian_signals=contrarian_signals,
            sentiment_extremes=sentiment_extremes,
            interpretation=interpretation
        )

    # VIX分析方法
    def _analyze_vix_term_structure(self, vix_9d: float, vix_current: float, vix_30d: float) -> str:
        """分析VIX期限结构"""
        if vix_9d > vix_current > vix_30d:
            return "backwardation"  # 期货贴水，短期恐慌
        elif vix_9d < vix_current < vix_30d:
            return "contango"       # 期货升水，正常状态
        elif vix_current > max(vix_9d, vix_30d):
            return "spike"          # VIX激增
        else:
            return "mixed"          # 混合状态

    def _assess_vix_level(self, vix_current: float, vix_percentile: float) -> str:
        """评估VIX水平"""
        if vix_current >= self.vix_thresholds["extreme_fear"]:
            return "extreme_fear"
        elif vix_current >= self.vix_thresholds["high_fear"]:
            return "high_fear"
        elif vix_current >= self.vix_thresholds["moderate_fear"]:
            return "moderate_fear"
        elif vix_current >= self.vix_thresholds["normal_high"]:
            return "normal_high"
        elif vix_current >= self.vix_thresholds["normal"]:
            return "normal"
        else:
            return "complacent"

    def _analyze_vix_trend(self, vix_change: float, vix_current: float) -> str:
        """分析VIX趋势"""
        change_ratio = vix_change / vix_current if vix_current > 0 else 0
        
        if change_ratio > 0.2:
            return "spiking"        # 激增
        elif change_ratio > 0.1:
            return "rising"         # 上升
        elif change_ratio > -0.1:
            return "stable"         # 稳定
        elif change_ratio > -0.2:
            return "declining"      # 下降
        else:
            return "collapsing"     # 暴跌

    def _calculate_vix_risk_score(self, vix_current: float, vix_percentile: float,
                                vix_change: float, term_structure: str) -> float:
        """计算VIX风险评分"""
        risk_score = 0
        
        # VIX绝对水平评分
        if vix_current >= 40:
            risk_score += 40
        elif vix_current >= 30:
            risk_score += 30
        elif vix_current >= 25:
            risk_score += 20
        elif vix_current >= 20:
            risk_score += 10
        elif vix_current < 12:
            risk_score += 15  # 过度自满也是风险
        
        # VIX相对水平评分
        if vix_percentile > 90:
            risk_score += 25
        elif vix_percentile > 80:
            risk_score += 20
        elif vix_percentile > 70:
            risk_score += 15
        elif vix_percentile < 10:
            risk_score += 10  # 过低也有风险
        
        # VIX变化评分
        if abs(vix_change) > 5:
            risk_score += 15
        elif abs(vix_change) > 3:
            risk_score += 10
        elif abs(vix_change) > 2:
            risk_score += 5
        
        # 期限结构评分
        if term_structure == "backwardation":
            risk_score += 20
        elif term_structure == "spike":
            risk_score += 15
        
        return min(risk_score, 100)

    # 期权分析方法
    def _analyze_put_call_ratio(self, put_call_ratio: float) -> str:
        """分析看跌看涨比率"""
        if put_call_ratio >= self.options_thresholds["put_call_extreme"]:
            return "extreme_bearish"
        elif put_call_ratio >= self.options_thresholds["put_call_high"]:
            return "high_bearish"
        elif put_call_ratio >= 1.1:
            return "moderate_bearish"
        elif put_call_ratio >= 0.9:
            return "neutral"
        elif put_call_ratio >= 0.7:
            return "moderate_bullish"
        elif put_call_ratio >= 0.5:
            return "high_bullish"
        else:
            return "extreme_bullish"

    def _analyze_skew_index(self, skew_index: float) -> str:
        """分析偏度指数"""
        if skew_index >= self.options_thresholds["skew_extreme"]:
            return "extreme_tail_risk"
        elif skew_index >= self.options_thresholds["skew_high"]:
            return "high_tail_risk"
        elif skew_index >= 110:
            return "moderate_tail_risk"
        elif skew_index >= 90:
            return "normal"
        else:
            return "low_tail_risk"

    def _analyze_implied_volatility(self, implied_volatility: float, iv_percentile: float) -> str:
        """分析隐含波动率"""
        if iv_percentile > 90:
            return "extremely_high"
        elif iv_percentile > 80:
            return "very_high"
        elif iv_percentile > 70:
            return "high"
        elif iv_percentile > 30:
            return "normal"
        elif iv_percentile > 20:
            return "low"
        else:
            return "extremely_low"

    def _analyze_options_volume(self, options_volume: float) -> str:
        """分析期权交易量"""
        # 这里需要与历史平均值比较，简化处理
        if options_volume > 2000000:  # 200万手
            return "extremely_high"
        elif options_volume > 1500000:
            return "very_high"
        elif options_volume > 1000000:
            return "high"
        elif options_volume > 500000:
            return "normal"
        elif options_volume > 100000:
            return "low"
        else:
            return "very_low"

    def _calculate_options_sentiment_score(self, put_call_ratio: float, skew_index: float,
                                         iv_percentile: float, options_volume: float) -> float:
        """计算期权情绪评分"""
        sentiment_score = 0

        # 看跌看涨比率评分
        if put_call_ratio >= 1.5:
            sentiment_score += 40  # 极度看跌
        elif put_call_ratio >= 1.2:
            sentiment_score += 30
        elif put_call_ratio >= 1.1:
            sentiment_score += 20
        elif put_call_ratio >= 1.0:
            sentiment_score += 10
        elif put_call_ratio <= 0.5:
            sentiment_score += 30  # 极度看涨也是风险
        elif put_call_ratio <= 0.7:
            sentiment_score += 20

        # 偏度指数评分
        if skew_index >= 130:
            sentiment_score += 25
        elif skew_index >= 120:
            sentiment_score += 20
        elif skew_index >= 110:
            sentiment_score += 15

        # 隐含波动率评分
        if iv_percentile > 90:
            sentiment_score += 25
        elif iv_percentile > 80:
            sentiment_score += 20
        elif iv_percentile > 70:
            sentiment_score += 15
        elif iv_percentile < 10:
            sentiment_score += 15  # 过低也有风险

        # 期权交易量调整
        if options_volume > 2000000:
            sentiment_score += 10  # 异常高交易量

        return min(sentiment_score, 100)

    # 资金流分析方法
    def _determine_flow_direction(self, total_equity_flow: float) -> MoneyFlowDirection:
        """确定资金流向"""
        if total_equity_flow <= self.flow_thresholds["massive_outflow"]:
            return MoneyFlowDirection.MASSIVE_OUTFLOW
        elif total_equity_flow <= self.flow_thresholds["moderate_outflow"]:
            return MoneyFlowDirection.MODERATE_OUTFLOW
        elif total_equity_flow <= self.flow_thresholds["slight_outflow"]:
            return MoneyFlowDirection.SLIGHT_OUTFLOW
        elif total_equity_flow >= self.flow_thresholds["massive_inflow"]:
            return MoneyFlowDirection.MASSIVE_INFLOW
        elif total_equity_flow >= self.flow_thresholds["moderate_inflow"]:
            return MoneyFlowDirection.MODERATE_INFLOW
        elif total_equity_flow >= self.flow_thresholds["slight_inflow"]:
            return MoneyFlowDirection.SLIGHT_INFLOW
        else:
            return MoneyFlowDirection.BALANCED

    def _analyze_risk_appetite(self, equity_flow: float, safe_haven_flow: float) -> RiskAppetite:
        """分析风险偏好"""
        # 计算风险资产vs避险资产的流向比
        if safe_haven_flow == 0:
            risk_ratio = equity_flow / 100 if equity_flow != 0 else 0
        else:
            risk_ratio = equity_flow / abs(safe_haven_flow)

        if risk_ratio <= -2:
            return RiskAppetite.RISK_OFF
        elif risk_ratio <= -0.5:
            return RiskAppetite.CAUTIOUS
        elif risk_ratio >= 2:
            return RiskAppetite.AGGRESSIVE
        elif risk_ratio >= 0.5:
            return RiskAppetite.RISK_ON
        else:
            return RiskAppetite.NEUTRAL

    def _calculate_flow_risk_score(self, equity_flow: float, safe_haven_flow: float,
                                 institutional_flow: float) -> float:
        """计算资金流风险评分"""
        risk_score = 0

        # 股票资金流出评分
        if equity_flow <= -2000:
            risk_score += 40
        elif equity_flow <= -1000:
            risk_score += 30
        elif equity_flow <= -500:
            risk_score += 20
        elif equity_flow <= -100:
            risk_score += 10

        # 避险资产流入评分
        if safe_haven_flow >= 2000:
            risk_score += 25
        elif safe_haven_flow >= 1000:
            risk_score += 20
        elif safe_haven_flow >= 500:
            risk_score += 15

        # 机构资金流向评分
        if institutional_flow <= -1000:
            risk_score += 25  # 机构大量撤离
        elif institutional_flow <= -500:
            risk_score += 20
        elif institutional_flow <= -100:
            risk_score += 15

        return min(risk_score, 100)

    # 综合评估方法
    def _calculate_fear_greed_index(self, vix_analysis: Dict, options_analysis: Dict,
                                  money_flow_analysis: Dict) -> float:
        """计算恐慌贪婪指数 (0-100, 0=极度恐慌, 100=极度贪婪)"""

        # VIX贡献 (权重40%)
        vix_score = 100 - vix_analysis["risk_score"]  # 转换为贪婪指数

        # 期权贡献 (权重30%)
        options_score = 100 - options_analysis["sentiment_score"]

        # 资金流贡献 (权重30%)
        flow_score = 100 - money_flow_analysis["risk_score"]

        # 加权平均
        fear_greed_index = (
            vix_score * 0.4 +
            options_score * 0.3 +
            flow_score * 0.3
        )

        return max(0, min(fear_greed_index, 100))

    def _determine_sentiment_level(self, fear_greed_index: float) -> SentimentLevel:
        """确定情绪水平"""
        if fear_greed_index <= 10:
            return SentimentLevel.EXTREME_FEAR
        elif fear_greed_index <= 25:
            return SentimentLevel.HIGH_FEAR
        elif fear_greed_index <= 40:
            return SentimentLevel.MODERATE_FEAR
        elif fear_greed_index <= 60:
            return SentimentLevel.NEUTRAL
        elif fear_greed_index <= 75:
            return SentimentLevel.MODERATE_GREED
        elif fear_greed_index <= 90:
            return SentimentLevel.HIGH_GREED
        else:
            return SentimentLevel.EXTREME_GREED

    def _identify_contrarian_signals(self, vix_analysis: Dict, options_analysis: Dict,
                                   money_flow_analysis: Dict) -> List[str]:
        """识别逆向信号"""
        signals = []

        # VIX逆向信号
        if vix_analysis["vix_current"] >= 40:
            signals.append("VIX极度恐慌，可能接近底部")
        elif vix_analysis["vix_current"] <= 12:
            signals.append("VIX过度自满，警惕回调风险")

        # 期权逆向信号
        if options_analysis["put_call_ratio"] >= 1.5:
            signals.append("看跌期权比例极高，可能过度悲观")
        elif options_analysis["put_call_ratio"] <= 0.5:
            signals.append("看涨期权比例极高，可能过度乐观")

        # 资金流逆向信号
        if money_flow_analysis["total_equity_flow"] <= -2000:
            signals.append("资金大量流出，可能接近恐慌性抛售尾声")
        elif money_flow_analysis["safe_haven_flow"] >= 2000:
            signals.append("避险资产大量流入，恐慌情绪浓厚")

        return signals

    def _identify_sentiment_extremes(self, vix_analysis: Dict, options_analysis: Dict,
                                   money_flow_analysis: Dict) -> List[str]:
        """识别情绪极端"""
        extremes = []

        # VIX极端
        if vix_analysis["vix_percentile"] > 95:
            extremes.append("VIX处于历史极高位")
        elif vix_analysis["vix_percentile"] < 5:
            extremes.append("VIX处于历史极低位")

        # 期权极端
        if options_analysis["iv_percentile"] > 95:
            extremes.append("隐含波动率处于极高水平")
        elif options_analysis["skew_index"] >= 130:
            extremes.append("期权偏度指数极高，尾部风险担忧严重")

        # 资金流极端
        flow_direction = money_flow_analysis["flow_direction"]
        if flow_direction == MoneyFlowDirection.MASSIVE_OUTFLOW:
            extremes.append("股票资金大量流出")
        elif flow_direction == MoneyFlowDirection.MASSIVE_INFLOW:
            extremes.append("股票资金大量流入")

        return extremes

    def _generate_sentiment_interpretation(self, sentiment_level: SentimentLevel,
                                         flow_direction: MoneyFlowDirection,
                                         risk_appetite: RiskAppetite,
                                         contrarian_signals: List[str]) -> str:
        """生成情绪解释"""

        # 情绪水平描述
        sentiment_descriptions = {
            SentimentLevel.EXTREME_FEAR: "市场情绪极度恐慌",
            SentimentLevel.HIGH_FEAR: "市场情绪高度恐慌",
            SentimentLevel.MODERATE_FEAR: "市场情绪中度恐慌",
            SentimentLevel.NEUTRAL: "市场情绪相对中性",
            SentimentLevel.MODERATE_GREED: "市场情绪中度贪婪",
            SentimentLevel.HIGH_GREED: "市场情绪高度贪婪",
            SentimentLevel.EXTREME_GREED: "市场情绪极度贪婪"
        }

        # 资金流向描述
        flow_descriptions = {
            MoneyFlowDirection.MASSIVE_OUTFLOW: "资金大量流出",
            MoneyFlowDirection.MODERATE_OUTFLOW: "资金中等流出",
            MoneyFlowDirection.SLIGHT_OUTFLOW: "资金轻微流出",
            MoneyFlowDirection.BALANCED: "资金流向平衡",
            MoneyFlowDirection.SLIGHT_INFLOW: "资金轻微流入",
            MoneyFlowDirection.MODERATE_INFLOW: "资金中等流入",
            MoneyFlowDirection.MASSIVE_INFLOW: "资金大量流入"
        }

        # 风险偏好描述
        appetite_descriptions = {
            RiskAppetite.RISK_OFF: "避险情绪浓厚",
            RiskAppetite.CAUTIOUS: "投资者较为谨慎",
            RiskAppetite.NEUTRAL: "风险偏好中性",
            RiskAppetite.RISK_ON: "风险偏好较高",
            RiskAppetite.AGGRESSIVE: "投资者情绪激进"
        }

        interpretation = f"{sentiment_descriptions[sentiment_level]}，{flow_descriptions[flow_direction]}，{appetite_descriptions[risk_appetite]}。"

        # 添加逆向信号提示
        if contrarian_signals:
            interpretation += f" 逆向指标显示：{'; '.join(contrarian_signals[:2])}。"

        # 根据情绪极端程度给出建议
        if sentiment_level in [SentimentLevel.EXTREME_FEAR, SentimentLevel.EXTREME_GREED]:
            interpretation += "情绪处于极端状态，建议关注反转机会。"
        elif sentiment_level in [SentimentLevel.HIGH_FEAR, SentimentLevel.HIGH_GREED]:
            interpretation += "情绪偏向极端，需要谨慎应对。"

        return interpretation
