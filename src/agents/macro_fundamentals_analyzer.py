"""
宏观基本面分析模块

实现经济周期、金融体系健康度、政策空间、估值水平、系统性风险指标的分析功能
用于预测系统性风险的关键指标
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EconomicCyclePhase(Enum):
    """经济周期阶段"""
    EXPANSION = "expansion"          # 扩张期
    PEAK = "peak"                   # 峰值期
    CONTRACTION = "contraction"     # 收缩期
    TROUGH = "trough"              # 谷底期

class PolicySpace(Enum):
    """政策空间"""
    AMPLE = "ample"                # 充足
    MODERATE = "moderate"          # 适中
    LIMITED = "limited"            # 有限
    EXHAUSTED = "exhausted"        # 耗尽

@dataclass
class MacroAnalysisResult:
    """宏观分析结果"""
    economic_cycle_phase: EconomicCyclePhase
    policy_space: PolicySpace
    systemic_risk_score: float
    financial_stability_score: float
    valuation_risk_score: float
    overall_risk_level: str
    key_indicators: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]

class MacroFundamentalsAnalyzer:
    """宏观基本面分析器"""
    
    def __init__(self):
        # 风险阈值配置
        self.risk_thresholds = {
            "recession_probability": 0.3,      # 衰退概率阈值
            "financial_stress": 1.0,           # 金融压力指数阈值
            "debt_to_gdp": 90,                 # 债务/GDP比率阈值
            "unemployment_spike": 1.5,          # 失业率上升阈值
            "inflation_extreme": 5.0,           # 极端通胀阈值
            "yield_curve_inversion": -0.5,     # 收益率曲线倒挂阈值
            "market_valuation_extreme": 90     # 市场估值极端分位数
        }
        
        # 指标权重
        self.indicator_weights = {
            "leading_indicators": 0.35,        # 领先指标
            "financial_conditions": 0.25,      # 金融条件
            "policy_effectiveness": 0.20,      # 政策有效性
            "valuation_metrics": 0.20         # 估值指标
        }

    def analyze_macro_fundamentals(self, market_data: Dict, economic_data: Dict, 
                                 end_date: str) -> MacroAnalysisResult:
        """
        综合分析宏观基本面
        
        Args:
            market_data: 市场数据
            economic_data: 经济数据
            end_date: 分析日期
            
        Returns:
            MacroAnalysisResult: 宏观分析结果
        """
        
        # 1. 经济周期分析
        cycle_analysis = self._analyze_economic_cycle(economic_data)
        
        # 2. 金融体系健康度分析
        financial_analysis = self._analyze_financial_system_health(market_data, economic_data)
        
        # 3. 政策空间分析
        policy_analysis = self._analyze_policy_space(economic_data)
        
        # 4. 估值水平分析
        valuation_analysis = self._analyze_valuation_levels(market_data)
        
        # 5. 系统性风险指标分析
        systemic_risk_analysis = self._analyze_systemic_risk_indicators(market_data, economic_data)
        
        # 综合评估
        overall_assessment = self._synthesize_analysis(
            cycle_analysis, financial_analysis, policy_analysis,
            valuation_analysis, systemic_risk_analysis
        )
        
        return overall_assessment

    def _analyze_economic_cycle(self, economic_data: Dict) -> Dict:
        """
        分析经济周期位置
        
        关键指标：
        - PMI（制造业采购经理指数）
        - 失业率趋势
        - GDP增长率
        - 消费者信心指数
        - 领先经济指标
        """
        
        # 获取经济指标
        pmi = economic_data.get("pmi", 50)
        unemployment_rate = economic_data.get("unemployment_rate", 5)
        unemployment_trend = economic_data.get("unemployment_trend", 0)  # 3个月变化
        gdp_growth = economic_data.get("gdp_growth", 2)
        consumer_confidence = economic_data.get("consumer_confidence", 100)
        leading_index = economic_data.get("leading_economic_index", 100)
        leading_index_trend = economic_data.get("leading_index_trend", 0)  # 6个月变化
        
        # 判断经济周期阶段
        expansion_signals = 0
        contraction_signals = 0
        
        # PMI信号
        if pmi > 52:
            expansion_signals += 1
        elif pmi < 48:
            contraction_signals += 1
            
        # 失业率信号
        if unemployment_trend < -0.2:  # 失业率下降
            expansion_signals += 1
        elif unemployment_trend > 0.5:  # 失业率上升
            contraction_signals += 1
            
        # GDP增长信号
        if gdp_growth > 2.5:
            expansion_signals += 1
        elif gdp_growth < 1:
            contraction_signals += 1
            
        # 消费者信心信号
        if consumer_confidence > 110:
            expansion_signals += 1
        elif consumer_confidence < 90:
            contraction_signals += 1
            
        # 领先指标信号
        if leading_index_trend > 1:
            expansion_signals += 1
        elif leading_index_trend < -1:
            contraction_signals += 1
        
        # 确定周期阶段
        if expansion_signals >= 3 and contraction_signals <= 1:
            if pmi > 55 and unemployment_rate < 4:
                phase = EconomicCyclePhase.PEAK
            else:
                phase = EconomicCyclePhase.EXPANSION
        elif contraction_signals >= 3 and expansion_signals <= 1:
            if unemployment_rate > 7 and gdp_growth < 0:
                phase = EconomicCyclePhase.TROUGH
            else:
                phase = EconomicCyclePhase.CONTRACTION
        else:
            # 根据主导信号判断
            if expansion_signals > contraction_signals:
                phase = EconomicCyclePhase.EXPANSION
            else:
                phase = EconomicCyclePhase.CONTRACTION
        
        # 计算风险评分
        risk_score = 0
        if phase == EconomicCyclePhase.PEAK:
            risk_score = 75  # 峰值期风险较高
        elif phase == EconomicCyclePhase.CONTRACTION:
            risk_score = 85  # 收缩期风险最高
        elif phase == EconomicCyclePhase.TROUGH:
            risk_score = 60  # 谷底期风险中等
        else:
            risk_score = 30  # 扩张期风险较低
        
        return {
            "phase": phase,
            "risk_score": risk_score,
            "expansion_signals": expansion_signals,
            "contraction_signals": contraction_signals,
            "key_indicators": {
                "pmi": pmi,
                "unemployment_rate": unemployment_rate,
                "unemployment_trend": unemployment_trend,
                "gdp_growth": gdp_growth,
                "consumer_confidence": consumer_confidence,
                "leading_index_trend": leading_index_trend
            }
        }

    def _analyze_financial_system_health(self, market_data: Dict, economic_data: Dict) -> Dict:
        """
        分析金融体系健康度
        
        关键指标：
        - 银行体系资本充足率
        - 不良贷款率
        - 流动性指标（Libor-OIS利差、TED利差）
        - 信用利差
        - 金融压力指数
        """
        
        # 银行体系指标
        bank_capital_ratio = economic_data.get("bank_capital_ratio", 12)
        npl_ratio = economic_data.get("npl_ratio", 2)
        
        # 流动性指标
        libor_ois_spread = market_data.get("libor_ois_spread", 0.1)
        ted_spread = market_data.get("ted_spread", 0.2)
        
        # 信用指标
        credit_spread = market_data.get("credit_spread", 1.5)
        high_yield_spread = market_data.get("high_yield_spread", 3.0)
        
        # 金融压力指数
        financial_stress_index = market_data.get("financial_stress_index", 0)
        
        # 计算各维度风险评分
        banking_risk = 0
        liquidity_risk = 0
        credit_risk = 0
        
        # 银行体系风险 (0-30分)
        if bank_capital_ratio < 8:
            banking_risk += 15
        elif bank_capital_ratio < 10:
            banking_risk += 10
        elif bank_capital_ratio < 12:
            banking_risk += 5
            
        if npl_ratio > 5:
            banking_risk += 15
        elif npl_ratio > 3:
            banking_risk += 10
        elif npl_ratio > 2:
            banking_risk += 5
        
        # 流动性风险 (0-35分)
        if libor_ois_spread > 0.5:
            liquidity_risk += 20
        elif libor_ois_spread > 0.3:
            liquidity_risk += 15
        elif libor_ois_spread > 0.2:
            liquidity_risk += 10
        elif libor_ois_spread > 0.15:
            liquidity_risk += 5
            
        if ted_spread > 1.0:
            liquidity_risk += 15
        elif ted_spread > 0.5:
            liquidity_risk += 10
        elif ted_spread > 0.3:
            liquidity_risk += 5
        
        # 信用风险 (0-35分)
        if credit_spread > 4:
            credit_risk += 20
        elif credit_spread > 2.5:
            credit_risk += 15
        elif credit_spread > 2:
            credit_risk += 10
        elif credit_spread > 1.5:
            credit_risk += 5
            
        if high_yield_spread > 8:
            credit_risk += 15
        elif high_yield_spread > 5:
            credit_risk += 10
        elif high_yield_spread > 3:
            credit_risk += 5
        
        # 综合金融稳定评分
        total_risk = banking_risk + liquidity_risk + credit_risk
        
        # 金融压力指数调整
        if financial_stress_index > 2:
            total_risk += 20
        elif financial_stress_index > 1:
            total_risk += 15
        elif financial_stress_index > 0.5:
            total_risk += 10
        elif financial_stress_index > 0:
            total_risk += 5
        
        stability_score = max(0, 100 - total_risk)
        
        return {
            "stability_score": stability_score,
            "banking_risk": banking_risk,
            "liquidity_risk": liquidity_risk,
            "credit_risk": credit_risk,
            "financial_stress_index": financial_stress_index,
            "key_indicators": {
                "bank_capital_ratio": bank_capital_ratio,
                "npl_ratio": npl_ratio,
                "libor_ois_spread": libor_ois_spread,
                "ted_spread": ted_spread,
                "credit_spread": credit_spread,
                "high_yield_spread": high_yield_spread
            }
        }

    def _analyze_policy_space(self, economic_data: Dict) -> Dict:
        """
        分析政策空间

        关键指标：
        - 央行利率水平和空间
        - 政府债务水平
        - 财政赤字情况
        - 通胀水平
        - 政策工具有效性
        """

        # 货币政策空间
        interest_rate = economic_data.get("interest_rate", 2.5)
        inflation_rate = economic_data.get("inflation_rate", 2)
        real_interest_rate = interest_rate - inflation_rate

        # 财政政策空间
        debt_to_gdp = economic_data.get("debt_to_gdp", 60)
        fiscal_deficit = economic_data.get("fiscal_deficit", 3)
        debt_service_ratio = economic_data.get("debt_service_ratio", 15)  # 债务偿付比率

        # 政策有效性指标
        monetary_transmission = economic_data.get("monetary_transmission", 0.7)  # 货币政策传导效率
        fiscal_multiplier = economic_data.get("fiscal_multiplier", 1.2)  # 财政乘数

        # 评估货币政策空间
        monetary_space_score = 0
        if real_interest_rate < 0:
            monetary_space_score = 20  # 负实际利率，空间很有限
        elif real_interest_rate < 1:
            monetary_space_score = 40  # 低实际利率，空间有限
        elif real_interest_rate < 2:
            monetary_space_score = 60  # 中等实际利率，空间适中
        elif real_interest_rate < 3:
            monetary_space_score = 80  # 较高实际利率，空间较大
        else:
            monetary_space_score = 100  # 高实际利率，空间充足

        # 考虑通胀约束
        if inflation_rate > 4:
            monetary_space_score *= 0.5  # 高通胀限制降息空间
        elif inflation_rate > 3:
            monetary_space_score *= 0.7
        elif inflation_rate < 0:
            monetary_space_score *= 1.2  # 通缩增加降息空间

        # 评估财政政策空间
        fiscal_space_score = 100

        # 债务水平约束
        if debt_to_gdp > 120:
            fiscal_space_score -= 50
        elif debt_to_gdp > 100:
            fiscal_space_score -= 40
        elif debt_to_gdp > 80:
            fiscal_space_score -= 30
        elif debt_to_gdp > 60:
            fiscal_space_score -= 20

        # 财政赤字约束
        if fiscal_deficit > 8:
            fiscal_space_score -= 30
        elif fiscal_deficit > 6:
            fiscal_space_score -= 25
        elif fiscal_deficit > 4:
            fiscal_space_score -= 20
        elif fiscal_deficit > 3:
            fiscal_space_score -= 10

        # 债务偿付约束
        if debt_service_ratio > 25:
            fiscal_space_score -= 20
        elif debt_service_ratio > 20:
            fiscal_space_score -= 15
        elif debt_service_ratio > 15:
            fiscal_space_score -= 10

        fiscal_space_score = max(0, fiscal_space_score)

        # 综合政策空间评估
        overall_space_score = (monetary_space_score * 0.6 + fiscal_space_score * 0.4)

        # 确定政策空间等级
        if overall_space_score >= 80:
            policy_space = PolicySpace.AMPLE
        elif overall_space_score >= 60:
            policy_space = PolicySpace.MODERATE
        elif overall_space_score >= 40:
            policy_space = PolicySpace.LIMITED
        else:
            policy_space = PolicySpace.EXHAUSTED

        return {
            "policy_space": policy_space,
            "overall_space_score": overall_space_score,
            "monetary_space_score": monetary_space_score,
            "fiscal_space_score": fiscal_space_score,
            "key_indicators": {
                "interest_rate": interest_rate,
                "real_interest_rate": real_interest_rate,
                "inflation_rate": inflation_rate,
                "debt_to_gdp": debt_to_gdp,
                "fiscal_deficit": fiscal_deficit,
                "debt_service_ratio": debt_service_ratio,
                "monetary_transmission": monetary_transmission,
                "fiscal_multiplier": fiscal_multiplier
            }
        }

    def _analyze_valuation_levels(self, market_data: Dict) -> Dict:
        """
        分析整体估值水平

        关键指标：
        - 市场整体P/E、P/B比率
        - CAPE比率（周期调整市盈率）
        - 估值分位数
        - 风险溢价
        """

        # 估值指标
        market_pe = market_data.get("market_pe", 20)
        market_pb = market_data.get("market_pb", 2.5)
        cape_ratio = market_data.get("cape_ratio", 25)

        # 历史分位数
        pe_percentile = market_data.get("pe_percentile", 50)
        pb_percentile = market_data.get("pb_percentile", 50)
        cape_percentile = market_data.get("cape_percentile", 50)

        # 风险溢价
        equity_risk_premium = market_data.get("equity_risk_premium", 5)
        earnings_yield = market_data.get("earnings_yield", 5)  # 盈利收益率
        bond_yield = market_data.get("bond_yield_10y", 3)  # 10年期国债收益率

        # 计算估值风险评分
        valuation_risk = 0

        # P/E估值风险 (0-25分)
        if pe_percentile > 95:
            valuation_risk += 25
        elif pe_percentile > 90:
            valuation_risk += 20
        elif pe_percentile > 80:
            valuation_risk += 15
        elif pe_percentile > 70:
            valuation_risk += 10
        elif pe_percentile > 60:
            valuation_risk += 5

        # P/B估值风险 (0-20分)
        if pb_percentile > 95:
            valuation_risk += 20
        elif pb_percentile > 90:
            valuation_risk += 16
        elif pb_percentile > 80:
            valuation_risk += 12
        elif pb_percentile > 70:
            valuation_risk += 8
        elif pb_percentile > 60:
            valuation_risk += 4

        # CAPE估值风险 (0-25分)
        if cape_percentile > 95:
            valuation_risk += 25
        elif cape_percentile > 90:
            valuation_risk += 20
        elif cape_percentile > 80:
            valuation_risk += 15
        elif cape_percentile > 70:
            valuation_risk += 10
        elif cape_percentile > 60:
            valuation_risk += 5

        # 风险溢价风险 (0-30分)
        if equity_risk_premium < 2:
            valuation_risk += 30  # 风险溢价过低
        elif equity_risk_premium < 3:
            valuation_risk += 25
        elif equity_risk_premium < 4:
            valuation_risk += 20
        elif equity_risk_premium < 5:
            valuation_risk += 15
        elif equity_risk_premium < 6:
            valuation_risk += 10
        elif equity_risk_premium < 7:
            valuation_risk += 5

        valuation_risk_score = min(valuation_risk, 100)

        return {
            "valuation_risk_score": valuation_risk_score,
            "key_indicators": {
                "market_pe": market_pe,
                "market_pb": market_pb,
                "cape_ratio": cape_ratio,
                "pe_percentile": pe_percentile,
                "pb_percentile": pb_percentile,
                "cape_percentile": cape_percentile,
                "equity_risk_premium": equity_risk_premium,
                "earnings_yield": earnings_yield,
                "bond_yield": bond_yield
            }
        }

    def _analyze_systemic_risk_indicators(self, market_data: Dict, economic_data: Dict) -> Dict:
        """
        分析系统性风险指标

        关键指标：
        - 收益率曲线形状（特别关注倒挂）
        - 金融稳定报告指标
        - 系统重要性机构风险
        - 跨市场传染风险
        """

        # 收益率曲线指标
        yield_2y = market_data.get("yield_2y", 2.5)
        yield_10y = market_data.get("yield_10y", 3.0)
        yield_curve_slope = yield_10y - yield_2y

        # 金融稳定指标
        financial_stress_index = market_data.get("financial_stress_index", 0)
        systemic_risk_index = market_data.get("systemic_risk_index", 0)

        # 银行间市场指标
        interbank_rate = market_data.get("interbank_rate", 2.5)
        repo_rate = market_data.get("repo_rate", 2.3)

        # 跨市场相关性
        cross_market_correlation = market_data.get("cross_market_correlation", 0.5)

        # 计算系统性风险评分
        systemic_risk = 0

        # 收益率曲线风险 (0-30分)
        if yield_curve_slope < -0.5:  # 严重倒挂
            systemic_risk += 30
        elif yield_curve_slope < -0.2:  # 中度倒挂
            systemic_risk += 25
        elif yield_curve_slope < 0:  # 轻度倒挂
            systemic_risk += 20
        elif yield_curve_slope < 0.5:  # 平坦
            systemic_risk += 15
        elif yield_curve_slope < 1:  # 略陡
            systemic_risk += 10
        elif yield_curve_slope < 1.5:  # 正常
            systemic_risk += 5
        # yield_curve_slope >= 1.5 不加分（正常陡峭）

        # 金融压力指数风险 (0-25分)
        if financial_stress_index > 3:
            systemic_risk += 25
        elif financial_stress_index > 2:
            systemic_risk += 20
        elif financial_stress_index > 1:
            systemic_risk += 15
        elif financial_stress_index > 0.5:
            systemic_risk += 10
        elif financial_stress_index > 0:
            systemic_risk += 5

        # 系统性风险指数 (0-25分)
        if systemic_risk_index > 2:
            systemic_risk += 25
        elif systemic_risk_index > 1.5:
            systemic_risk += 20
        elif systemic_risk_index > 1:
            systemic_risk += 15
        elif systemic_risk_index > 0.5:
            systemic_risk += 10
        elif systemic_risk_index > 0:
            systemic_risk += 5

        # 跨市场传染风险 (0-20分)
        if cross_market_correlation > 0.9:
            systemic_risk += 20
        elif cross_market_correlation > 0.8:
            systemic_risk += 16
        elif cross_market_correlation > 0.7:
            systemic_risk += 12
        elif cross_market_correlation > 0.6:
            systemic_risk += 8
        elif cross_market_correlation > 0.5:
            systemic_risk += 4

        systemic_risk_score = min(systemic_risk, 100)

        return {
            "systemic_risk_score": systemic_risk_score,
            "key_indicators": {
                "yield_curve_slope": yield_curve_slope,
                "yield_2y": yield_2y,
                "yield_10y": yield_10y,
                "financial_stress_index": financial_stress_index,
                "systemic_risk_index": systemic_risk_index,
                "interbank_rate": interbank_rate,
                "repo_rate": repo_rate,
                "cross_market_correlation": cross_market_correlation
            }
        }

    def _synthesize_analysis(self, cycle_analysis: Dict, financial_analysis: Dict,
                           policy_analysis: Dict, valuation_analysis: Dict,
                           systemic_risk_analysis: Dict) -> MacroAnalysisResult:
        """
        综合分析各维度结果
        """

        # 计算加权综合风险评分
        cycle_risk = 100 - cycle_analysis["risk_score"]  # 转换为风险评分
        financial_risk = 100 - financial_analysis["stability_score"]
        policy_risk = 100 - policy_analysis["overall_space_score"]
        valuation_risk = valuation_analysis["valuation_risk_score"]
        systemic_risk = systemic_risk_analysis["systemic_risk_score"]

        overall_risk_score = (
            cycle_risk * self.indicator_weights["leading_indicators"] +
            financial_risk * self.indicator_weights["financial_conditions"] +
            policy_risk * self.indicator_weights["policy_effectiveness"] +
            (valuation_risk + systemic_risk) / 2 * self.indicator_weights["valuation_metrics"]
        )

        # 确定整体风险等级
        if overall_risk_score >= 80:
            risk_level = "极高风险"
        elif overall_risk_score >= 65:
            risk_level = "高风险"
        elif overall_risk_score >= 50:
            risk_level = "中等风险"
        elif overall_risk_score >= 35:
            risk_level = "低风险"
        else:
            risk_level = "极低风险"

        # 识别主要风险因素
        risk_factors = []
        if cycle_analysis["phase"] in [EconomicCyclePhase.PEAK, EconomicCyclePhase.CONTRACTION]:
            risk_factors.append(f"经济周期处于{cycle_analysis['phase'].value}阶段")

        if financial_analysis["stability_score"] < 60:
            risk_factors.append("金融体系稳定性较差")

        if policy_analysis["policy_space"] in [PolicySpace.LIMITED, PolicySpace.EXHAUSTED]:
            risk_factors.append(f"政策空间{policy_analysis['policy_space'].value}")

        if valuation_analysis["valuation_risk_score"] > 70:
            risk_factors.append("市场估值过高")

        if systemic_risk_analysis["systemic_risk_score"] > 60:
            risk_factors.append("系统性风险指标恶化")

        # 生成建议
        recommendations = []
        if overall_risk_score >= 70:
            recommendations.extend([
                "建议大幅降低风险敞口",
                "增加防御性资产配置",
                "密切关注系统性风险指标",
                "准备应对市场大幅波动"
            ])
        elif overall_risk_score >= 50:
            recommendations.extend([
                "建议适度降低风险敞口",
                "增加现金和债券配置",
                "关注政策动向和经济数据"
            ])
        else:
            recommendations.extend([
                "可维持正常风险敞口",
                "关注市场机会",
                "保持适度谨慎"
            ])

        # 汇总关键指标
        key_indicators = {}
        key_indicators.update(cycle_analysis["key_indicators"])
        key_indicators.update(financial_analysis["key_indicators"])
        key_indicators.update(policy_analysis["key_indicators"])
        key_indicators.update(valuation_analysis["key_indicators"])
        key_indicators.update(systemic_risk_analysis["key_indicators"])

        return MacroAnalysisResult(
            economic_cycle_phase=cycle_analysis["phase"],
            policy_space=policy_analysis["policy_space"],
            systemic_risk_score=overall_risk_score,
            financial_stability_score=financial_analysis["stability_score"],
            valuation_risk_score=valuation_analysis["valuation_risk_score"],
            overall_risk_level=risk_level,
            key_indicators=key_indicators,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
