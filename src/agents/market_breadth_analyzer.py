"""
市场广度与内部结构分析模块

实现涨跌家数比、新高新低指数、板块表现、相关性分析等功能
用于区分系统性风险和局部调整的关键指标
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MarketBreadthCondition(Enum):
    """市场广度状况"""
    EXTREMELY_WEAK = "extremely_weak"      # 极度疲弱
    WEAK = "weak"                         # 疲弱
    NEUTRAL = "neutral"                   # 中性
    STRONG = "strong"                     # 强劲
    EXTREMELY_STRONG = "extremely_strong"  # 极度强劲

class SectorRotation(Enum):
    """板块轮动状态"""
    DEFENSIVE_ROTATION = "defensive_rotation"    # 防御性轮动
    CYCLICAL_ROTATION = "cyclical_rotation"      # 周期性轮动
    GROWTH_ROTATION = "growth_rotation"          # 成长性轮动
    VALUE_ROTATION = "value_rotation"            # 价值轮动
    NO_CLEAR_ROTATION = "no_clear_rotation"      # 无明确轮动

@dataclass
class MarketBreadthResult:
    """市场广度分析结果"""
    breadth_condition: MarketBreadthCondition
    sector_rotation: SectorRotation
    systemic_risk_probability: float
    breadth_score: float
    advance_decline_ratio: float
    new_highs_lows_ratio: float
    sector_dispersion: float
    correlation_level: float
    key_metrics: Dict[str, float]
    risk_signals: List[str]
    interpretation: str

class MarketBreadthAnalyzer:
    """市场广度分析器"""
    
    def __init__(self):
        # 风险阈值配置
        self.risk_thresholds = {
            "decline_ratio_high": 0.7,         # 下跌股票比例高风险阈值
            "decline_ratio_extreme": 0.8,      # 下跌股票比例极端阈值
            "new_lows_ratio_high": 0.7,        # 新低比例高风险阈值
            "new_lows_ratio_extreme": 0.85,    # 新低比例极端阈值
            "correlation_high": 0.7,           # 高相关性阈值
            "correlation_extreme": 0.85,       # 极端相关性阈值
            "defensive_outperform": 0.05,      # 防御性板块跑赢阈值
            "sector_dispersion_low": 0.02      # 板块离散度低阈值
        }
        
        # 板块分类
        self.sector_categories = {
            "defensive": ["utilities", "consumer_staples", "healthcare", "real_estate"],
            "cyclical": ["financials", "industrials", "materials", "energy"],
            "growth": ["technology", "communication_services", "consumer_discretionary"],
            "value": ["financials", "energy", "materials", "utilities"]
        }

    def analyze_market_breadth(self, market_data: Dict, sector_data: Dict, 
                             correlation_data: Dict, end_date: str) -> MarketBreadthResult:
        """
        综合分析市场广度
        
        Args:
            market_data: 市场数据（涨跌家数、新高新低等）
            sector_data: 板块数据
            correlation_data: 相关性数据
            end_date: 分析日期
            
        Returns:
            MarketBreadthResult: 市场广度分析结果
        """
        
        # 1. 涨跌家数比分析
        advance_decline_analysis = self._analyze_advance_decline_ratio(market_data)
        
        # 2. 新高新低指数分析
        new_highs_lows_analysis = self._analyze_new_highs_lows(market_data)
        
        # 3. 板块表现分析
        sector_analysis = self._analyze_sector_performance(sector_data)
        
        # 4. 相关性分析
        correlation_analysis = self._analyze_correlation_patterns(correlation_data)
        
        # 综合评估
        comprehensive_assessment = self._synthesize_breadth_analysis(
            advance_decline_analysis, new_highs_lows_analysis,
            sector_analysis, correlation_analysis
        )
        
        return comprehensive_assessment

    def _analyze_advance_decline_ratio(self, market_data: Dict) -> Dict:
        """
        分析涨跌家数比
        
        关键指标：
        - 当日涨跌家数比
        - 涨跌家数趋势
        - 累积涨跌线
        """
        
        # 获取涨跌家数数据
        advancing_stocks = market_data.get("advancing_stocks", 1500)
        declining_stocks = market_data.get("declining_stocks", 1500)
        unchanged_stocks = market_data.get("unchanged_stocks", 100)
        
        # 历史涨跌家数数据（用于趋势分析）
        advance_decline_history = market_data.get("advance_decline_history", [])
        
        total_stocks = advancing_stocks + declining_stocks + unchanged_stocks
        if total_stocks == 0:
            return self._get_default_advance_decline_result()
        
        # 计算基本比率
        advance_ratio = advancing_stocks / total_stocks
        decline_ratio = declining_stocks / total_stocks
        
        # 计算涨跌家数比
        ad_ratio = advancing_stocks / declining_stocks if declining_stocks > 0 else 10
        
        # 分析趋势
        trend_score = self._calculate_ad_trend_score(advance_decline_history)
        
        # 计算风险评分 (0-100分，分数越高风险越大)
        risk_score = 0
        
        # 当日涨跌比评分 (0-40分)
        if decline_ratio >= self.risk_thresholds["decline_ratio_extreme"]:
            risk_score += 40
        elif decline_ratio >= self.risk_thresholds["decline_ratio_high"]:
            risk_score += 30
        elif decline_ratio >= 0.6:
            risk_score += 20
        elif decline_ratio >= 0.55:
            risk_score += 10
        elif decline_ratio >= 0.5:
            risk_score += 5
        
        # 趋势评分 (0-30分)
        risk_score += trend_score
        
        # 极端情况加分 (0-30分)
        if ad_ratio < 0.2:  # 涨跌比小于1:5
            risk_score += 30
        elif ad_ratio < 0.3:  # 涨跌比小于1:3
            risk_score += 25
        elif ad_ratio < 0.5:  # 涨跌比小于1:2
            risk_score += 20
        elif ad_ratio < 0.7:  # 涨跌比小于7:10
            risk_score += 15
        elif ad_ratio < 1.0:  # 涨跌比小于1:1
            risk_score += 10
        elif ad_ratio < 1.3:  # 涨跌比小于1.3:1
            risk_score += 5
        
        return {
            "advance_ratio": advance_ratio,
            "decline_ratio": decline_ratio,
            "ad_ratio": ad_ratio,
            "risk_score": min(risk_score, 100),
            "trend_score": trend_score,
            "advancing_stocks": advancing_stocks,
            "declining_stocks": declining_stocks,
            "total_stocks": total_stocks
        }

    def _analyze_new_highs_lows(self, market_data: Dict) -> Dict:
        """
        分析新高新低指数
        
        关键指标：
        - 52周新高新低数量
        - 新高新低比率
        - 新高新低趋势
        """
        
        # 获取新高新低数据
        new_highs_52w = market_data.get("new_highs_52w", 100)
        new_lows_52w = market_data.get("new_lows_52w", 100)
        new_highs_20d = market_data.get("new_highs_20d", 50)
        new_lows_20d = market_data.get("new_lows_20d", 50)
        
        # 历史新高新低数据
        new_highs_lows_history = market_data.get("new_highs_lows_history", [])
        
        # 计算基本比率
        total_new_extremes_52w = new_highs_52w + new_lows_52w
        total_new_extremes_20d = new_highs_20d + new_lows_20d
        
        if total_new_extremes_52w == 0:
            new_lows_ratio_52w = 0.5
        else:
            new_lows_ratio_52w = new_lows_52w / total_new_extremes_52w
        
        if total_new_extremes_20d == 0:
            new_lows_ratio_20d = 0.5
        else:
            new_lows_ratio_20d = new_lows_20d / total_new_extremes_20d
        
        # 新高新低比
        hl_ratio_52w = new_highs_52w / new_lows_52w if new_lows_52w > 0 else 10
        hl_ratio_20d = new_highs_20d / new_lows_20d if new_lows_20d > 0 else 10
        
        # 分析趋势
        trend_score = self._calculate_hl_trend_score(new_highs_lows_history)
        
        # 计算风险评分 (0-100分)
        risk_score = 0
        
        # 52周新低比例评分 (0-40分)
        if new_lows_ratio_52w >= self.risk_thresholds["new_lows_ratio_extreme"]:
            risk_score += 40
        elif new_lows_ratio_52w >= self.risk_thresholds["new_lows_ratio_high"]:
            risk_score += 30
        elif new_lows_ratio_52w >= 0.6:
            risk_score += 20
        elif new_lows_ratio_52w >= 0.55:
            risk_score += 10
        elif new_lows_ratio_52w >= 0.5:
            risk_score += 5
        
        # 20日新低比例评分 (0-30分)
        if new_lows_ratio_20d >= 0.8:
            risk_score += 30
        elif new_lows_ratio_20d >= 0.7:
            risk_score += 25
        elif new_lows_ratio_20d >= 0.6:
            risk_score += 20
        elif new_lows_ratio_20d >= 0.55:
            risk_score += 15
        elif new_lows_ratio_20d >= 0.5:
            risk_score += 10
        
        # 趋势评分 (0-30分)
        risk_score += trend_score
        
        return {
            "new_highs_52w": new_highs_52w,
            "new_lows_52w": new_lows_52w,
            "new_highs_20d": new_highs_20d,
            "new_lows_20d": new_lows_20d,
            "new_lows_ratio_52w": new_lows_ratio_52w,
            "new_lows_ratio_20d": new_lows_ratio_20d,
            "hl_ratio_52w": hl_ratio_52w,
            "hl_ratio_20d": hl_ratio_20d,
            "risk_score": min(risk_score, 100),
            "trend_score": trend_score
        }

    def _analyze_sector_performance(self, sector_data: Dict) -> Dict:
        """
        分析板块表现
        
        关键指标：
        - 各板块相对表现
        - 防御性vs周期性板块表现
        - 板块轮动模式
        - 板块离散度
        """
        
        # 获取板块表现数据
        sector_returns = sector_data.get("sector_returns", {})
        sector_relative_strength = sector_data.get("sector_relative_strength", {})
        
        if not sector_returns:
            return self._get_default_sector_result()
        
        # 计算各类板块平均表现
        defensive_performance = self._calculate_category_performance(
            sector_returns, self.sector_categories["defensive"]
        )
        cyclical_performance = self._calculate_category_performance(
            sector_returns, self.sector_categories["cyclical"]
        )
        growth_performance = self._calculate_category_performance(
            sector_returns, self.sector_categories["growth"]
        )
        value_performance = self._calculate_category_performance(
            sector_returns, self.sector_categories["value"]
        )
        
        # 计算板块离散度
        sector_dispersion = self._calculate_sector_dispersion(sector_returns)
        
        # 判断板块轮动模式
        rotation_pattern = self._identify_sector_rotation(
            defensive_performance, cyclical_performance, 
            growth_performance, value_performance
        )
        
        # 计算风险评分
        risk_score = self._calculate_sector_risk_score(
            defensive_performance, cyclical_performance,
            sector_dispersion, rotation_pattern
        )
        
        return {
            "defensive_performance": defensive_performance,
            "cyclical_performance": cyclical_performance,
            "growth_performance": growth_performance,
            "value_performance": value_performance,
            "sector_dispersion": sector_dispersion,
            "rotation_pattern": rotation_pattern,
            "risk_score": risk_score,
            "sector_returns": sector_returns
        }

    def _analyze_correlation_patterns(self, correlation_data: Dict) -> Dict:
        """
        分析相关性模式

        关键指标：
        - 股票间平均相关性
        - 板块间相关性
        - 相关性趋势
        - 尾部风险相关性
        """

        # 获取相关性数据
        avg_stock_correlation = correlation_data.get("avg_stock_correlation", 0.3)
        sector_correlation = correlation_data.get("sector_correlation", 0.5)
        correlation_trend = correlation_data.get("correlation_trend", 0)  # 相关性变化趋势
        tail_correlation = correlation_data.get("tail_correlation", 0.4)  # 尾部相关性

        # 历史相关性数据
        correlation_history = correlation_data.get("correlation_history", [])

        # 计算相关性风险评分 (0-100分)
        risk_score = 0

        # 平均股票相关性评分 (0-35分)
        if avg_stock_correlation >= self.risk_thresholds["correlation_extreme"]:
            risk_score += 35
        elif avg_stock_correlation >= self.risk_thresholds["correlation_high"]:
            risk_score += 28
        elif avg_stock_correlation >= 0.6:
            risk_score += 21
        elif avg_stock_correlation >= 0.5:
            risk_score += 14
        elif avg_stock_correlation >= 0.4:
            risk_score += 7

        # 板块相关性评分 (0-25分)
        if sector_correlation >= 0.9:
            risk_score += 25
        elif sector_correlation >= 0.8:
            risk_score += 20
        elif sector_correlation >= 0.7:
            risk_score += 15
        elif sector_correlation >= 0.6:
            risk_score += 10
        elif sector_correlation >= 0.5:
            risk_score += 5

        # 尾部相关性评分 (0-25分)
        if tail_correlation >= 0.8:
            risk_score += 25
        elif tail_correlation >= 0.7:
            risk_score += 20
        elif tail_correlation >= 0.6:
            risk_score += 15
        elif tail_correlation >= 0.5:
            risk_score += 10
        elif tail_correlation >= 0.4:
            risk_score += 5

        # 相关性趋势评分 (0-15分)
        if correlation_trend > 0.2:  # 相关性快速上升
            risk_score += 15
        elif correlation_trend > 0.1:
            risk_score += 10
        elif correlation_trend > 0.05:
            risk_score += 5

        return {
            "avg_stock_correlation": avg_stock_correlation,
            "sector_correlation": sector_correlation,
            "correlation_trend": correlation_trend,
            "tail_correlation": tail_correlation,
            "risk_score": min(risk_score, 100)
        }

    def _synthesize_breadth_analysis(self, advance_decline_analysis: Dict,
                                   new_highs_lows_analysis: Dict,
                                   sector_analysis: Dict,
                                   correlation_analysis: Dict) -> MarketBreadthResult:
        """
        综合分析市场广度结果
        """

        # 计算加权综合评分
        weights = {
            "advance_decline": 0.35,
            "new_highs_lows": 0.25,
            "sector_performance": 0.25,
            "correlation": 0.15
        }

        breadth_score = (
            advance_decline_analysis["risk_score"] * weights["advance_decline"] +
            new_highs_lows_analysis["risk_score"] * weights["new_highs_lows"] +
            sector_analysis["risk_score"] * weights["sector_performance"] +
            correlation_analysis["risk_score"] * weights["correlation"]
        )

        # 确定市场广度状况
        if breadth_score >= 80:
            breadth_condition = MarketBreadthCondition.EXTREMELY_WEAK
        elif breadth_score >= 65:
            breadth_condition = MarketBreadthCondition.WEAK
        elif breadth_score >= 35:
            breadth_condition = MarketBreadthCondition.NEUTRAL
        elif breadth_score >= 20:
            breadth_condition = MarketBreadthCondition.STRONG
        else:
            breadth_condition = MarketBreadthCondition.EXTREMELY_STRONG

        # 计算系统性风险概率
        systemic_risk_probability = self._calculate_systemic_risk_probability(
            breadth_score, advance_decline_analysis, correlation_analysis
        )

        # 识别风险信号
        risk_signals = self._identify_risk_signals(
            advance_decline_analysis, new_highs_lows_analysis,
            sector_analysis, correlation_analysis
        )

        # 生成解释
        interpretation = self._generate_interpretation(
            breadth_condition, sector_analysis["rotation_pattern"],
            systemic_risk_probability, risk_signals
        )

        # 汇总关键指标
        key_metrics = {
            "advance_decline_ratio": advance_decline_analysis["ad_ratio"],
            "decline_ratio": advance_decline_analysis["decline_ratio"],
            "new_lows_ratio_52w": new_highs_lows_analysis["new_lows_ratio_52w"],
            "avg_stock_correlation": correlation_analysis["avg_stock_correlation"],
            "sector_correlation": correlation_analysis["sector_correlation"],
            "sector_dispersion": sector_analysis["sector_dispersion"],
            "defensive_performance": sector_analysis["defensive_performance"],
            "cyclical_performance": sector_analysis["cyclical_performance"]
        }

        return MarketBreadthResult(
            breadth_condition=breadth_condition,
            sector_rotation=sector_analysis["rotation_pattern"],
            systemic_risk_probability=systemic_risk_probability,
            breadth_score=breadth_score,
            advance_decline_ratio=advance_decline_analysis["ad_ratio"],
            new_highs_lows_ratio=new_highs_lows_analysis["hl_ratio_52w"],
            sector_dispersion=sector_analysis["sector_dispersion"],
            correlation_level=correlation_analysis["avg_stock_correlation"],
            key_metrics=key_metrics,
            risk_signals=risk_signals,
            interpretation=interpretation
        )

    # 辅助方法
    def _calculate_ad_trend_score(self, history: List) -> float:
        """计算涨跌家数趋势评分"""
        if len(history) < 5:
            return 15  # 默认中等风险

        # 分析最近5天的趋势
        recent_decline_ratios = [day.get("decline_ratio", 0.5) for day in history[-5:]]

        # 计算趋势
        trend_score = 0
        deteriorating_days = sum(1 for ratio in recent_decline_ratios if ratio > 0.6)

        if deteriorating_days >= 4:
            trend_score = 30
        elif deteriorating_days >= 3:
            trend_score = 25
        elif deteriorating_days >= 2:
            trend_score = 20
        elif deteriorating_days >= 1:
            trend_score = 15
        else:
            trend_score = 10

        return trend_score

    def _calculate_hl_trend_score(self, history: List) -> float:
        """计算新高新低趋势评分"""
        if len(history) < 5:
            return 15  # 默认中等风险

        # 分析最近5天的趋势
        recent_new_lows_ratios = [day.get("new_lows_ratio", 0.5) for day in history[-5:]]

        # 计算趋势
        trend_score = 0
        deteriorating_days = sum(1 for ratio in recent_new_lows_ratios if ratio > 0.6)

        if deteriorating_days >= 4:
            trend_score = 30
        elif deteriorating_days >= 3:
            trend_score = 25
        elif deteriorating_days >= 2:
            trend_score = 20
        elif deteriorating_days >= 1:
            trend_score = 15
        else:
            trend_score = 10

        return trend_score

    def _calculate_category_performance(self, sector_returns: Dict, category_sectors: List) -> float:
        """计算板块类别平均表现"""
        if not sector_returns or not category_sectors:
            return 0.0

        valid_returns = []
        for sector in category_sectors:
            if sector in sector_returns:
                valid_returns.append(sector_returns[sector])

        return sum(valid_returns) / len(valid_returns) if valid_returns else 0.0

    def _calculate_sector_dispersion(self, sector_returns: Dict) -> float:
        """计算板块离散度"""
        if not sector_returns:
            return 0.05  # 默认值

        returns = list(sector_returns.values())
        if len(returns) < 2:
            return 0.05

        return np.std(returns)

    def _identify_sector_rotation(self, defensive: float, cyclical: float,
                                growth: float, value: float) -> SectorRotation:
        """识别板块轮动模式"""

        # 计算相对表现
        performances = {
            "defensive": defensive,
            "cyclical": cyclical,
            "growth": growth,
            "value": value
        }

        # 找出表现最好的类别
        best_performer = max(performances, key=performances.get)
        best_performance = performances[best_performer]

        # 判断是否有明显的轮动
        other_performances = [v for k, v in performances.items() if k != best_performer]
        avg_others = sum(other_performances) / len(other_performances)

        if best_performance - avg_others > 0.02:  # 2%的显著差异
            if best_performer == "defensive":
                return SectorRotation.DEFENSIVE_ROTATION
            elif best_performer == "cyclical":
                return SectorRotation.CYCLICAL_ROTATION
            elif best_performer == "growth":
                return SectorRotation.GROWTH_ROTATION
            elif best_performer == "value":
                return SectorRotation.VALUE_ROTATION

        return SectorRotation.NO_CLEAR_ROTATION

    def _calculate_sector_risk_score(self, defensive: float, cyclical: float,
                                   dispersion: float, rotation: SectorRotation) -> float:
        """计算板块风险评分"""
        risk_score = 0

        # 防御性板块表现评分 (0-40分)
        if defensive < -0.05:  # 防御性板块大跌
            risk_score += 40
        elif defensive < -0.03:  # 防御性板块下跌
            risk_score += 30
        elif defensive < -0.01:  # 防御性板块微跌
            risk_score += 20
        elif defensive < 0:  # 防御性板块略跌
            risk_score += 10

        # 周期性板块表现评分 (0-30分)
        if cyclical < -0.10:  # 周期性板块暴跌
            risk_score += 30
        elif cyclical < -0.05:  # 周期性板块大跌
            risk_score += 25
        elif cyclical < -0.03:  # 周期性板块下跌
            risk_score += 20
        elif cyclical < -0.01:  # 周期性板块微跌
            risk_score += 15
        elif cyclical < 0:  # 周期性板块略跌
            risk_score += 10

        # 板块离散度评分 (0-20分)
        if dispersion < self.risk_thresholds["sector_dispersion_low"]:
            risk_score += 20  # 低离散度表明同涨同跌
        elif dispersion < 0.03:
            risk_score += 15
        elif dispersion < 0.04:
            risk_score += 10
        elif dispersion < 0.05:
            risk_score += 5

        # 轮动模式评分 (0-10分)
        if rotation == SectorRotation.DEFENSIVE_ROTATION:
            risk_score += 10  # 防御性轮动表明避险情绪
        elif rotation == SectorRotation.NO_CLEAR_ROTATION and dispersion < 0.03:
            risk_score += 5  # 无明确轮动且低离散度

        return min(risk_score, 100)

    def _calculate_systemic_risk_probability(self, breadth_score: float,
                                           advance_decline_analysis: Dict,
                                           correlation_analysis: Dict) -> float:
        """计算系统性风险概率"""

        # 基础概率基于广度评分
        base_probability = breadth_score / 100

        # 调整因子
        adjustments = 0

        # 涨跌家数极端情况
        if advance_decline_analysis["decline_ratio"] > 0.8:
            adjustments += 0.2
        elif advance_decline_analysis["decline_ratio"] > 0.7:
            adjustments += 0.1

        # 相关性极端情况
        if correlation_analysis["avg_stock_correlation"] > 0.8:
            adjustments += 0.15
        elif correlation_analysis["avg_stock_correlation"] > 0.7:
            adjustments += 0.1

        # 综合概率
        probability = min(base_probability + adjustments, 0.95)
        return max(probability, 0.05)

    def _identify_risk_signals(self, advance_decline_analysis: Dict,
                             new_highs_lows_analysis: Dict,
                             sector_analysis: Dict,
                             correlation_analysis: Dict) -> List[str]:
        """识别风险信号"""
        signals = []

        # 涨跌家数风险信号
        if advance_decline_analysis["decline_ratio"] > 0.8:
            signals.append("超过80%股票下跌，市场广度极度恶化")
        elif advance_decline_analysis["decline_ratio"] > 0.7:
            signals.append("超过70%股票下跌，市场广度明显恶化")

        # 新高新低风险信号
        if new_highs_lows_analysis["new_lows_ratio_52w"] > 0.8:
            signals.append("新低股票占比超过80%，技术面极度恶化")
        elif new_highs_lows_analysis["new_lows_ratio_52w"] > 0.7:
            signals.append("新低股票占比超过70%，技术面明显恶化")

        # 板块表现风险信号
        if sector_analysis["defensive_performance"] < -0.03:
            signals.append("防御性板块大幅下跌，避险功能失效")

        if sector_analysis["rotation_pattern"] == SectorRotation.DEFENSIVE_ROTATION:
            signals.append("资金流向防御性板块，避险情绪浓厚")

        if sector_analysis["sector_dispersion"] < 0.02:
            signals.append("板块间离散度极低，呈现普跌格局")

        # 相关性风险信号
        if correlation_analysis["avg_stock_correlation"] > 0.8:
            signals.append("股票间相关性极高，系统性风险上升")

        if correlation_analysis["sector_correlation"] > 0.85:
            signals.append("板块间相关性极高，分散化效果失效")

        return signals

    def _generate_interpretation(self, breadth_condition: MarketBreadthCondition,
                               rotation: SectorRotation, systemic_risk_prob: float,
                               risk_signals: List[str]) -> str:
        """生成分析解释"""

        # 基础解释
        condition_descriptions = {
            MarketBreadthCondition.EXTREMELY_WEAK: "市场广度极度疲弱，普跌格局明显",
            MarketBreadthCondition.WEAK: "市场广度疲弱，下跌面较广",
            MarketBreadthCondition.NEUTRAL: "市场广度中性，涨跌参半",
            MarketBreadthCondition.STRONG: "市场广度强劲，上涨面较广",
            MarketBreadthCondition.EXTREMELY_STRONG: "市场广度极度强劲，普涨格局明显"
        }

        rotation_descriptions = {
            SectorRotation.DEFENSIVE_ROTATION: "资金流向防御性板块",
            SectorRotation.CYCLICAL_ROTATION: "资金流向周期性板块",
            SectorRotation.GROWTH_ROTATION: "资金流向成长性板块",
            SectorRotation.VALUE_ROTATION: "资金流向价值型板块",
            SectorRotation.NO_CLEAR_ROTATION: "无明确板块轮动"
        }

        interpretation = f"{condition_descriptions[breadth_condition]}，{rotation_descriptions[rotation]}。"

        # 系统性风险评估
        if systemic_risk_prob > 0.7:
            interpretation += f"系统性风险概率高达{systemic_risk_prob:.1%}，建议高度警惕。"
        elif systemic_risk_prob > 0.5:
            interpretation += f"系统性风险概率为{systemic_risk_prob:.1%}，需要谨慎应对。"
        elif systemic_risk_prob > 0.3:
            interpretation += f"系统性风险概率为{systemic_risk_prob:.1%}，保持适度警惕。"
        else:
            interpretation += f"系统性风险概率较低({systemic_risk_prob:.1%})，市场调整可能是局部性的。"

        # 添加主要风险信号
        if risk_signals:
            interpretation += f" 主要风险信号包括：{'; '.join(risk_signals[:3])}。"

        return interpretation

    def _get_default_advance_decline_result(self) -> Dict:
        """获取默认涨跌家数分析结果"""
        return {
            "advance_ratio": 0.5,
            "decline_ratio": 0.5,
            "ad_ratio": 1.0,
            "risk_score": 50,
            "trend_score": 15,
            "advancing_stocks": 1500,
            "declining_stocks": 1500,
            "total_stocks": 3000
        }

    def _get_default_sector_result(self) -> Dict:
        """获取默认板块分析结果"""
        return {
            "defensive_performance": 0.0,
            "cyclical_performance": 0.0,
            "growth_performance": 0.0,
            "value_performance": 0.0,
            "sector_dispersion": 0.05,
            "rotation_pattern": SectorRotation.NO_CLEAR_ROTATION,
            "risk_score": 50,
            "sector_returns": {}
        }
