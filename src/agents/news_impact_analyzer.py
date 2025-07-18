"""
消息面分析模块

实现事件性质判断、可预测性与可解决性分析、市场反应对比等功能
用于判断价格波动是否为消息面驱动的短期冲击
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re

logger = logging.getLogger(__name__)

class EventNature(Enum):
    """事件性质"""
    COMPANY_SPECIFIC = "company_specific"      # 公司特定事件
    INDUSTRY_WIDE = "industry_wide"           # 行业性事件
    SECTOR_IMPACT = "sector_impact"           # 板块影响
    SYSTEMIC_RISK = "systemic_risk"           # 系统性风险
    REGULATORY = "regulatory"                 # 监管事件
    MACROECONOMIC = "macroeconomic"          # 宏观经济事件

class EventSeverity(Enum):
    """事件严重性"""
    CRITICAL = "critical"        # 严重
    HIGH = "high"               # 高
    MODERATE = "moderate"       # 中等
    LOW = "low"                 # 低
    MINIMAL = "minimal"         # 轻微

class EventSolvability(Enum):
    """事件可解决性"""
    EASILY_SOLVABLE = "easily_solvable"        # 容易解决
    MODERATELY_SOLVABLE = "moderately_solvable" # 中等难度解决
    DIFFICULT_TO_SOLVE = "difficult_to_solve"   # 难以解决
    UNSOLVABLE = "unsolvable"                   # 无法解决
    UNKNOWN = "unknown"                         # 未知

@dataclass
class NewsAnalysisResult:
    """消息面分析结果"""
    event_nature: EventNature
    event_severity: EventSeverity
    event_solvability: EventSolvability
    systemic_impact_probability: float
    short_term_impact_score: float
    long_term_impact_score: float
    market_overreaction_probability: float
    recovery_time_estimate: int  # 预估恢复时间（天）
    key_themes: List[str]
    risk_factors: List[str]
    interpretation: str

class NewsImpactAnalyzer:
    """消息面影响分析器"""
    
    def __init__(self):
        # 关键词分类
        self.keyword_categories = {
            "company_specific": [
                "earnings", "revenue", "profit", "loss", "ceo", "management", 
                "acquisition", "merger", "lawsuit", "fraud", "scandal", "bankruptcy",
                "product", "recall", "patent", "contract", "partnership"
            ],
            "industry_wide": [
                "industry", "sector", "competition", "market share", "technology",
                "disruption", "innovation", "regulation", "compliance", "standards"
            ],
            "systemic_risk": [
                "recession", "crisis", "crash", "collapse", "panic", "contagion",
                "systemic", "financial system", "banking crisis", "credit crunch",
                "liquidity crisis", "market crash", "economic downturn"
            ],
            "regulatory": [
                "regulation", "policy", "government", "federal", "sec", "fda",
                "antitrust", "investigation", "fine", "penalty", "compliance",
                "law", "legal", "court", "ruling", "decision"
            ],
            "macroeconomic": [
                "interest rate", "inflation", "gdp", "unemployment", "fed",
                "monetary policy", "fiscal policy", "trade war", "tariff",
                "economic data", "central bank", "currency", "bond yield"
            ]
        }
        
        # 严重性关键词
        self.severity_keywords = {
            "critical": [
                "bankruptcy", "collapse", "crisis", "emergency", "disaster",
                "catastrophic", "devastating", "severe", "critical", "urgent"
            ],
            "high": [
                "major", "significant", "substantial", "serious", "important",
                "large", "big", "huge", "massive", "dramatic"
            ],
            "moderate": [
                "moderate", "medium", "average", "typical", "normal",
                "standard", "regular", "usual", "expected"
            ],
            "low": [
                "minor", "small", "slight", "limited", "minimal", "low",
                "little", "modest", "marginal", "negligible"
            ]
        }
        
        # 可解决性关键词
        self.solvability_keywords = {
            "easily_solvable": [
                "temporary", "short-term", "fix", "solution", "resolve",
                "address", "correct", "adjust", "modify", "improve"
            ],
            "moderately_solvable": [
                "restructure", "reorganize", "negotiate", "settle", "agreement",
                "compromise", "plan", "strategy", "timeline", "process"
            ],
            "difficult_to_solve": [
                "complex", "complicated", "challenging", "difficult", "long-term",
                "structural", "fundamental", "deep-rooted", "systemic"
            ],
            "unsolvable": [
                "permanent", "irreversible", "impossible", "hopeless", "terminal",
                "final", "definitive", "unchangeable", "fixed"
            ]
        }

    def analyze_news_impact(self, news_data: List[Dict], ticker: str, 
                          market_context: Dict) -> NewsAnalysisResult:
        """
        分析消息面影响
        
        Args:
            news_data: 新闻数据列表
            ticker: 股票代码
            market_context: 市场环境数据
            
        Returns:
            NewsAnalysisResult: 消息面分析结果
        """
        
        if not news_data:
            return self._get_default_result()
        
        # 1. 事件性质分析
        event_nature = self._analyze_event_nature(news_data, ticker)
        
        # 2. 事件严重性分析
        event_severity = self._analyze_event_severity(news_data)
        
        # 3. 事件可解决性分析
        event_solvability = self._analyze_event_solvability(news_data)
        
        # 4. 影响范围分析
        impact_analysis = self._analyze_impact_scope(news_data, ticker, market_context)
        
        # 5. 市场反应分析
        market_reaction_analysis = self._analyze_market_reaction(news_data, market_context)
        
        # 6. 主题提取
        key_themes = self._extract_key_themes(news_data)
        
        # 综合评估
        comprehensive_assessment = self._synthesize_news_analysis(
            event_nature, event_severity, event_solvability,
            impact_analysis, market_reaction_analysis, key_themes
        )
        
        return comprehensive_assessment

    def _analyze_event_nature(self, news_data: List[Dict], ticker: str) -> EventNature:
        """分析事件性质"""
        
        nature_scores = {nature: 0 for nature in EventNature}
        
        for news in news_data:
            content = self._extract_text_content(news)
            content_lower = content.lower()
            
            # 检查是否提及特定公司
            ticker_mentioned = ticker.lower() in content_lower
            
            # 分析关键词
            for nature, keywords in self.keyword_categories.items():
                keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
                
                if nature == "company_specific" and ticker_mentioned:
                    nature_scores[EventNature.COMPANY_SPECIFIC] += keyword_count * 2
                elif nature == "industry_wide":
                    nature_scores[EventNature.INDUSTRY_WIDE] += keyword_count
                elif nature == "systemic_risk":
                    nature_scores[EventNature.SYSTEMIC_RISK] += keyword_count * 1.5
                elif nature == "regulatory":
                    nature_scores[EventNature.REGULATORY] += keyword_count
                elif nature == "macroeconomic":
                    nature_scores[EventNature.MACROECONOMIC] += keyword_count * 1.2
        
        # 特殊逻辑：如果没有明确提及公司但有行业关键词，可能是板块影响
        if (nature_scores[EventNature.COMPANY_SPECIFIC] == 0 and 
            nature_scores[EventNature.INDUSTRY_WIDE] > 0):
            nature_scores[EventNature.SECTOR_IMPACT] = nature_scores[EventNature.INDUSTRY_WIDE]
        
        # 返回得分最高的事件性质
        return max(nature_scores, key=nature_scores.get)

    def _analyze_event_severity(self, news_data: List[Dict]) -> EventSeverity:
        """分析事件严重性"""
        
        severity_scores = {severity: 0 for severity in EventSeverity}
        
        for news in news_data:
            content = self._extract_text_content(news)
            content_lower = content.lower()
            
            # 分析严重性关键词
            for severity, keywords in self.severity_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
                severity_enum = EventSeverity(severity)
                severity_scores[severity_enum] += keyword_count
            
            # 特殊模式检测
            if any(word in content_lower for word in ["plunge", "crash", "collapse"]):
                severity_scores[EventSeverity.CRITICAL] += 3
            elif any(word in content_lower for word in ["surge", "spike", "soar"]):
                severity_scores[EventSeverity.HIGH] += 2
        
        # 如果没有明确的严重性指标，默认为中等
        if all(score == 0 for score in severity_scores.values()):
            return EventSeverity.MODERATE
        
        return max(severity_scores, key=severity_scores.get)

    def _analyze_event_solvability(self, news_data: List[Dict]) -> EventSolvability:
        """分析事件可解决性"""
        
        solvability_scores = {solvability: 0 for solvability in EventSolvability}
        
        for news in news_data:
            content = self._extract_text_content(news)
            content_lower = content.lower()
            
            # 分析可解决性关键词
            for solvability, keywords in self.solvability_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
                solvability_enum = EventSolvability(solvability)
                solvability_scores[solvability_enum] += keyword_count
        
        # 如果没有明确的可解决性指标，返回未知
        if all(score == 0 for score in solvability_scores.values()):
            return EventSolvability.UNKNOWN
        
        return max(solvability_scores, key=solvability_scores.get)

    def _analyze_impact_scope(self, news_data: List[Dict], ticker: str, 
                            market_context: Dict) -> Dict:
        """分析影响范围"""
        
        # 计算短期和长期影响评分
        short_term_impact = 0
        long_term_impact = 0
        systemic_impact_prob = 0
        
        for news in news_data:
            content = self._extract_text_content(news)
            content_lower = content.lower()
            
            # 短期影响因子
            if any(word in content_lower for word in ["immediate", "instant", "sudden", "shock"]):
                short_term_impact += 20
            if any(word in content_lower for word in ["earnings", "quarterly", "guidance"]):
                short_term_impact += 15
            if any(word in content_lower for word in ["announcement", "news", "report"]):
                short_term_impact += 10
            
            # 长期影响因子
            if any(word in content_lower for word in ["strategic", "long-term", "future", "outlook"]):
                long_term_impact += 20
            if any(word in content_lower for word in ["restructure", "transformation", "change"]):
                long_term_impact += 15
            if any(word in content_lower for word in ["regulation", "policy", "law"]):
                long_term_impact += 10
            
            # 系统性影响因子
            if any(word in content_lower for word in ["systemic", "contagion", "spillover"]):
                systemic_impact_prob += 30
            if any(word in content_lower for word in ["financial system", "banking", "credit"]):
                systemic_impact_prob += 20
            if any(word in content_lower for word in ["recession", "crisis", "crash"]):
                systemic_impact_prob += 25
        
        return {
            "short_term_impact_score": min(short_term_impact, 100),
            "long_term_impact_score": min(long_term_impact, 100),
            "systemic_impact_probability": min(systemic_impact_prob / 100, 1.0)
        }

    def _analyze_market_reaction(self, news_data: List[Dict], market_context: Dict) -> Dict:
        """分析市场反应"""
        
        # 分析市场是否过度反应
        overreaction_signals = 0
        
        # 获取市场环境数据
        market_volatility = market_context.get("volatility", 0.2)
        market_sentiment = market_context.get("sentiment", 0.5)
        
        for news in news_data:
            content = self._extract_text_content(news)
            content_lower = content.lower()
            
            # 过度反应信号
            if any(word in content_lower for word in ["overreaction", "excessive", "extreme"]):
                overreaction_signals += 2
            if any(word in content_lower for word in ["panic", "fear", "hysteria"]):
                overreaction_signals += 3
            if any(word in content_lower for word in ["irrational", "emotional", "sentiment"]):
                overreaction_signals += 1
        
        # 基于市场环境调整过度反应概率
        base_overreaction_prob = overreaction_signals * 0.15
        
        # 高波动环境更容易过度反应
        if market_volatility > 0.3:
            base_overreaction_prob += 0.2
        elif market_volatility > 0.25:
            base_overreaction_prob += 0.1
        
        # 极端情绪环境更容易过度反应
        if market_sentiment < 0.3 or market_sentiment > 0.7:
            base_overreaction_prob += 0.15
        
        overreaction_probability = min(base_overreaction_prob, 0.9)
        
        return {
            "overreaction_probability": overreaction_probability,
            "overreaction_signals": overreaction_signals
        }

    def _extract_key_themes(self, news_data: List[Dict]) -> List[str]:
        """提取关键主题"""
        
        themes = []
        theme_counts = {}
        
        # 预定义主题关键词
        theme_keywords = {
            "财报业绩": ["earnings", "revenue", "profit", "loss", "guidance"],
            "监管政策": ["regulation", "policy", "sec", "fda", "government"],
            "并购重组": ["merger", "acquisition", "takeover", "deal"],
            "管理层变动": ["ceo", "management", "executive", "leadership"],
            "产品创新": ["product", "innovation", "technology", "patent"],
            "法律诉讼": ["lawsuit", "legal", "court", "settlement"],
            "市场竞争": ["competition", "market share", "competitor"],
            "宏观经济": ["economy", "gdp", "inflation", "interest rate"],
            "行业趋势": ["industry", "sector", "trend", "outlook"]
        }
        
        for news in news_data:
            content = self._extract_text_content(news)
            content_lower = content.lower()
            
            for theme, keywords in theme_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
                if keyword_count > 0:
                    theme_counts[theme] = theme_counts.get(theme, 0) + keyword_count
        
        # 返回出现频率最高的主题
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        themes = [theme for theme, count in sorted_themes[:5]]  # 最多返回5个主题
        
        return themes

    def _synthesize_news_analysis(self, event_nature: EventNature, event_severity: EventSeverity,
                                event_solvability: EventSolvability, impact_analysis: Dict,
                                market_reaction_analysis: Dict, key_themes: List[str]) -> NewsAnalysisResult:
        """综合消息面分析结果"""

        # 计算系统性影响概率
        systemic_impact_probability = self._calculate_systemic_impact_probability(
            event_nature, event_severity, impact_analysis
        )

        # 估算恢复时间
        recovery_time_estimate = self._estimate_recovery_time(
            event_nature, event_severity, event_solvability
        )

        # 识别风险因素
        risk_factors = self._identify_news_risk_factors(
            event_nature, event_severity, event_solvability, key_themes
        )

        # 生成解释
        interpretation = self._generate_news_interpretation(
            event_nature, event_severity, event_solvability,
            systemic_impact_probability, market_reaction_analysis["overreaction_probability"]
        )

        return NewsAnalysisResult(
            event_nature=event_nature,
            event_severity=event_severity,
            event_solvability=event_solvability,
            systemic_impact_probability=systemic_impact_probability,
            short_term_impact_score=impact_analysis["short_term_impact_score"],
            long_term_impact_score=impact_analysis["long_term_impact_score"],
            market_overreaction_probability=market_reaction_analysis["overreaction_probability"],
            recovery_time_estimate=recovery_time_estimate,
            key_themes=key_themes,
            risk_factors=risk_factors,
            interpretation=interpretation
        )

    def _calculate_systemic_impact_probability(self, event_nature: EventNature,
                                             event_severity: EventSeverity,
                                             impact_analysis: Dict) -> float:
        """计算系统性影响概率"""

        base_probability = impact_analysis["systemic_impact_probability"]

        # 事件性质调整
        nature_multipliers = {
            EventNature.SYSTEMIC_RISK: 1.5,
            EventNature.MACROECONOMIC: 1.3,
            EventNature.REGULATORY: 1.1,
            EventNature.SECTOR_IMPACT: 0.8,
            EventNature.INDUSTRY_WIDE: 0.6,
            EventNature.COMPANY_SPECIFIC: 0.3
        }

        # 事件严重性调整
        severity_multipliers = {
            EventSeverity.CRITICAL: 1.4,
            EventSeverity.HIGH: 1.2,
            EventSeverity.MODERATE: 1.0,
            EventSeverity.LOW: 0.8,
            EventSeverity.MINIMAL: 0.6
        }

        adjusted_probability = (
            base_probability *
            nature_multipliers.get(event_nature, 1.0) *
            severity_multipliers.get(event_severity, 1.0)
        )

        return min(adjusted_probability, 0.95)

    def _estimate_recovery_time(self, event_nature: EventNature, event_severity: EventSeverity,
                              event_solvability: EventSolvability) -> int:
        """估算恢复时间（天）"""

        # 基础恢复时间
        base_times = {
            EventSeverity.CRITICAL: 60,
            EventSeverity.HIGH: 30,
            EventSeverity.MODERATE: 14,
            EventSeverity.LOW: 7,
            EventSeverity.MINIMAL: 3
        }

        base_time = base_times.get(event_severity, 14)

        # 事件性质调整
        nature_adjustments = {
            EventNature.COMPANY_SPECIFIC: 0.7,
            EventNature.INDUSTRY_WIDE: 1.0,
            EventNature.SECTOR_IMPACT: 1.2,
            EventNature.REGULATORY: 1.5,
            EventNature.MACROECONOMIC: 2.0,
            EventNature.SYSTEMIC_RISK: 3.0
        }

        # 可解决性调整
        solvability_adjustments = {
            EventSolvability.EASILY_SOLVABLE: 0.5,
            EventSolvability.MODERATELY_SOLVABLE: 1.0,
            EventSolvability.DIFFICULT_TO_SOLVE: 2.0,
            EventSolvability.UNSOLVABLE: 5.0,
            EventSolvability.UNKNOWN: 1.5
        }

        adjusted_time = (
            base_time *
            nature_adjustments.get(event_nature, 1.0) *
            solvability_adjustments.get(event_solvability, 1.0)
        )

        return max(1, int(adjusted_time))

    def _identify_news_risk_factors(self, event_nature: EventNature, event_severity: EventSeverity,
                                  event_solvability: EventSolvability, key_themes: List[str]) -> List[str]:
        """识别新闻风险因素"""

        risk_factors = []

        # 基于事件性质的风险
        if event_nature == EventNature.SYSTEMIC_RISK:
            risk_factors.append("事件具有系统性风险特征，可能引发连锁反应")
        elif event_nature == EventNature.MACROECONOMIC:
            risk_factors.append("宏观经济事件，影响范围广泛")
        elif event_nature == EventNature.REGULATORY:
            risk_factors.append("监管政策变化，可能影响整个行业")

        # 基于事件严重性的风险
        if event_severity in [EventSeverity.CRITICAL, EventSeverity.HIGH]:
            risk_factors.append("事件严重性较高，市场冲击较大")

        # 基于可解决性的风险
        if event_solvability in [EventSolvability.DIFFICULT_TO_SOLVE, EventSolvability.UNSOLVABLE]:
            risk_factors.append("事件难以快速解决，影响可能持续")
        elif event_solvability == EventSolvability.UNKNOWN:
            risk_factors.append("事件解决方案不明确，增加不确定性")

        # 基于主题的风险
        high_risk_themes = ["法律诉讼", "监管政策", "财报业绩", "管理层变动"]
        for theme in key_themes:
            if theme in high_risk_themes:
                risk_factors.append(f"涉及{theme}，可能带来持续影响")

        return risk_factors

    def _generate_news_interpretation(self, event_nature: EventNature, event_severity: EventSeverity,
                                    event_solvability: EventSolvability, systemic_impact_prob: float,
                                    overreaction_prob: float) -> str:
        """生成消息面解释"""

        # 事件性质描述
        nature_descriptions = {
            EventNature.COMPANY_SPECIFIC: "公司特定事件",
            EventNature.INDUSTRY_WIDE: "行业性事件",
            EventNature.SECTOR_IMPACT: "板块影响事件",
            EventNature.REGULATORY: "监管政策事件",
            EventNature.MACROECONOMIC: "宏观经济事件",
            EventNature.SYSTEMIC_RISK: "系统性风险事件"
        }

        # 严重性描述
        severity_descriptions = {
            EventSeverity.CRITICAL: "严重",
            EventSeverity.HIGH: "较高",
            EventSeverity.MODERATE: "中等",
            EventSeverity.LOW: "较低",
            EventSeverity.MINIMAL: "轻微"
        }

        # 可解决性描述
        solvability_descriptions = {
            EventSolvability.EASILY_SOLVABLE: "容易解决",
            EventSolvability.MODERATELY_SOLVABLE: "中等难度解决",
            EventSolvability.DIFFICULT_TO_SOLVE: "难以解决",
            EventSolvability.UNSOLVABLE: "无法解决",
            EventSolvability.UNKNOWN: "解决方案不明"
        }

        interpretation = f"这是一个{nature_descriptions[event_nature]}，严重性{severity_descriptions[event_severity]}，{solvability_descriptions[event_solvability]}。"

        # 系统性影响评估
        if systemic_impact_prob > 0.7:
            interpretation += f"系统性影响概率高达{systemic_impact_prob:.1%}，需要高度警惕市场传染风险。"
        elif systemic_impact_prob > 0.4:
            interpretation += f"系统性影响概率为{systemic_impact_prob:.1%}，需要关注事件扩散情况。"
        elif systemic_impact_prob > 0.2:
            interpretation += f"系统性影响概率为{systemic_impact_prob:.1%}，影响相对局限。"
        else:
            interpretation += f"系统性影响概率较低({systemic_impact_prob:.1%})，主要为局部性影响。"

        # 市场反应评估
        if overreaction_prob > 0.6:
            interpretation += "市场可能存在过度反应，后续有修复空间。"
        elif overreaction_prob > 0.4:
            interpretation += "市场反应可能略显过度，需观察后续走势。"
        else:
            interpretation += "市场反应相对理性。"

        return interpretation

    def _extract_text_content(self, news: Dict) -> str:
        """提取新闻文本内容"""
        content = ""

        # 提取标题
        if "title" in news:
            content += news["title"] + " "

        # 提取正文
        if "content" in news:
            content += news["content"] + " "
        elif "summary" in news:
            content += news["summary"] + " "
        elif "description" in news:
            content += news["description"] + " "

        return content.strip()

    def _get_default_result(self) -> NewsAnalysisResult:
        """获取默认结果"""
        return NewsAnalysisResult(
            event_nature=EventNature.COMPANY_SPECIFIC,
            event_severity=EventSeverity.MODERATE,
            event_solvability=EventSolvability.UNKNOWN,
            systemic_impact_probability=0.2,
            short_term_impact_score=30.0,
            long_term_impact_score=20.0,
            market_overreaction_probability=0.3,
            recovery_time_estimate=14,
            key_themes=[],
            risk_factors=["缺乏足够的新闻数据进行分析"],
            interpretation="无重大消息面影响，市场波动可能由其他因素驱动"
        )
