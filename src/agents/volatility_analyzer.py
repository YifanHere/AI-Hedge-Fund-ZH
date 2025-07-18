"""
异常波动分析器 - 识别突发性大跌幅，区分消息面冲击和趋势转变
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """异常波动分析器"""
    
    def __init__(self):
        self.shock_thresholds = {
            "minor": 0.05,      # 5%以上为轻微冲击
            "moderate": 0.08,   # 8%以上为中等冲击
            "severe": 0.12,     # 12%以上为严重冲击
            "extreme": 0.20     # 20%以上为极端冲击
        }
        
        self.volume_spike_threshold = 2.0  # 成交量是平均值的2倍以上
        
    def analyze_price_shock(self, prices_df: pd.DataFrame, current_date: str = None) -> Dict:
        """
        分析价格冲击类型
        返回: {
            "shock_type": "none/news_driven/trend_reversal/systematic",
            "shock_severity": "minor/moderate/severe/extreme",
            "confidence": float,
            "cooling_period_days": int,
            "analysis": str
        }
        """
        if len(prices_df) < 10:
            return self._create_default_analysis()
        
        try:
            # 计算日收益率
            prices_df = prices_df.copy()
            prices_df['returns'] = prices_df['close'].pct_change()
            
            # 获取最近的收益率
            recent_return = prices_df['returns'].iloc[-1]
            
            # 如果没有显著波动，返回正常状态
            if abs(recent_return) < self.shock_thresholds["minor"]:
                return {
                    "shock_type": "none",
                    "shock_severity": "none",
                    "confidence": 0.9,
                    "cooling_period_days": 0,
                    "analysis": "价格波动正常，无异常冲击"
                }
            
            # 分析冲击严重程度
            shock_severity = self._classify_shock_severity(abs(recent_return))
            
            # 分析冲击类型
            shock_analysis = self._analyze_shock_type(prices_df, recent_return)
            
            # 确定冷静期
            cooling_period = self._calculate_cooling_period(shock_severity, shock_analysis["shock_type"])
            
            return {
                "shock_type": shock_analysis["shock_type"],
                "shock_severity": shock_severity,
                "confidence": shock_analysis["confidence"],
                "cooling_period_days": cooling_period,
                "analysis": shock_analysis["analysis"]
            }
            
        except Exception as e:
            logger.error(f"分析价格冲击时出错: {e}")
            return self._create_default_analysis()
    
    def _classify_shock_severity(self, abs_return: float) -> str:
        """分类冲击严重程度"""
        if abs_return >= self.shock_thresholds["extreme"]:
            return "extreme"
        elif abs_return >= self.shock_thresholds["severe"]:
            return "severe"
        elif abs_return >= self.shock_thresholds["moderate"]:
            return "moderate"
        else:
            return "minor"
    
    def _analyze_shock_type(self, prices_df: pd.DataFrame, recent_return: float) -> Dict:
        """分析冲击类型"""
        
        # 1. 检查成交量异常
        volume_analysis = self._analyze_volume_spike(prices_df)
        
        # 2. 检查技术指标背离
        technical_analysis = self._analyze_technical_divergence(prices_df)
        
        # 3. 检查趋势连续性
        trend_analysis = self._analyze_trend_continuity(prices_df, recent_return)
        
        # 4. 综合判断冲击类型
        return self._synthesize_shock_type(volume_analysis, technical_analysis, trend_analysis, recent_return)
    
    def _analyze_volume_spike(self, prices_df: pd.DataFrame) -> Dict:
        """分析成交量异常"""
        if 'volume' not in prices_df.columns or len(prices_df) < 20:
            return {"volume_spike": False, "volume_ratio": 1.0}
        
        # 计算最近成交量与20日平均的比值
        recent_volume = prices_df['volume'].iloc[-1]
        avg_volume_20d = prices_df['volume'].rolling(20).mean().iloc[-1]
        
        if pd.isna(avg_volume_20d) or avg_volume_20d == 0:
            return {"volume_spike": False, "volume_ratio": 1.0}
        
        volume_ratio = recent_volume / avg_volume_20d
        volume_spike = volume_ratio >= self.volume_spike_threshold
        
        return {
            "volume_spike": volume_spike,
            "volume_ratio": volume_ratio
        }
    
    def _analyze_technical_divergence(self, prices_df: pd.DataFrame) -> Dict:
        """分析技术指标背离"""
        if len(prices_df) < 14:
            return {"rsi_divergence": False, "trend_divergence": False}
        
        # 计算RSI
        from .technicals import calculate_rsi
        rsi = calculate_rsi(prices_df, 14)
        
        # 检查RSI背离（价格新低但RSI未新低，或价格新高但RSI未新高）
        rsi_divergence = False
        if len(rsi) >= 10:
            recent_rsi = rsi.iloc[-5:]
            recent_prices = prices_df['close'].iloc[-5:]
            
            # 简单的背离检测
            price_trend = recent_prices.iloc[-1] - recent_prices.iloc[0]
            rsi_trend = recent_rsi.iloc[-1] - recent_rsi.iloc[0]
            
            # 如果价格和RSI趋势方向相反，可能存在背离
            if (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0):
                rsi_divergence = True
        
        return {
            "rsi_divergence": rsi_divergence,
            "trend_divergence": False  # 可以后续添加更多技术指标
        }
    
    def _analyze_trend_continuity(self, prices_df: pd.DataFrame, recent_return: float) -> Dict:
        """分析趋势连续性"""
        if len(prices_df) < 10:
            return {"trend_break": False, "consecutive_direction": 0}
        
        # 计算连续同方向天数
        returns = prices_df['returns'].dropna()
        if len(returns) < 5:
            return {"trend_break": False, "consecutive_direction": 0}
        
        # 检查最近几天的趋势方向
        recent_returns = returns.iloc[-5:]
        
        # 计算连续同方向天数
        consecutive_direction = 0
        direction = 1 if recent_return > 0 else -1
        
        for ret in reversed(recent_returns):
            if (ret > 0 and direction > 0) or (ret < 0 and direction < 0):
                consecutive_direction += 1
            else:
                break
        
        # 检查是否是趋势突破
        trend_break = consecutive_direction <= 1 and abs(recent_return) > 0.08
        
        return {
            "trend_break": trend_break,
            "consecutive_direction": consecutive_direction
        }
    
    def _synthesize_shock_type(self, volume_analysis: Dict, technical_analysis: Dict, 
                              trend_analysis: Dict, recent_return: float) -> Dict:
        """综合判断冲击类型"""
        
        # 消息面驱动的特征：
        # 1. 成交量大幅放大
        # 2. 单日大幅波动
        # 3. 没有明显的技术指标背离
        # 4. 趋势突破（之前趋势被打断）
        
        news_driven_score = 0
        trend_reversal_score = 0
        systematic_score = 0
        
        # 成交量分析
        if volume_analysis["volume_spike"]:
            news_driven_score += 30
            if volume_analysis["volume_ratio"] > 3.0:
                news_driven_score += 20
        else:
            trend_reversal_score += 20
        
        # 技术背离分析
        if technical_analysis["rsi_divergence"]:
            trend_reversal_score += 25
        else:
            news_driven_score += 15
        
        # 趋势连续性分析
        if trend_analysis["trend_break"]:
            news_driven_score += 25
        else:
            if trend_analysis["consecutive_direction"] >= 3:
                trend_reversal_score += 30
            else:
                systematic_score += 20
        
        # 波动幅度分析
        abs_return = abs(recent_return)
        if abs_return > 0.15:  # 极端波动更可能是消息面
            news_driven_score += 20
        elif abs_return > 0.10:
            news_driven_score += 10
        
        # 确定最终类型
        max_score = max(news_driven_score, trend_reversal_score, systematic_score)
        
        if max_score < 40:
            shock_type = "systematic"
            confidence = 0.6
            analysis = "市场系统性调整，建议谨慎观察"
        elif news_driven_score == max_score:
            shock_type = "news_driven"
            confidence = min(0.9, max_score / 100)
            analysis = f"疑似消息面驱动的冲击（成交量{volume_analysis['volume_ratio']:.1f}倍，{'趋势突破' if trend_analysis['trend_break'] else '延续趋势'}）"
        elif trend_reversal_score == max_score:
            shock_type = "trend_reversal"
            confidence = min(0.9, max_score / 100)
            analysis = f"可能的趋势转折点（{'技术背离' if technical_analysis['rsi_divergence'] else '趋势延续'}，连续{trend_analysis['consecutive_direction']}天同向）"
        else:
            shock_type = "systematic"
            confidence = min(0.9, max_score / 100)
            analysis = "系统性市场调整"
        
        return {
            "shock_type": shock_type,
            "confidence": confidence,
            "analysis": analysis
        }
    
    def _calculate_cooling_period(self, shock_severity: str, shock_type: str) -> int:
        """计算建议的冷静期天数"""
        base_days = {
            "minor": 1,
            "moderate": 2,
            "severe": 3,
            "extreme": 5
        }
        
        days = base_days.get(shock_severity, 2)
        
        # 根据冲击类型调整
        if shock_type == "news_driven":
            days = max(1, days - 1)  # 消息面冲击恢复较快
        elif shock_type == "trend_reversal":
            days = days + 2  # 趋势转折需要更多观察时间
        elif shock_type == "systematic":
            days = days + 1  # 系统性风险需要适中观察时间
        
        return min(days, 7)  # 最多7天冷静期
    
    def _create_default_analysis(self) -> Dict:
        """创建默认分析结果"""
        return {
            "shock_type": "none",
            "shock_severity": "none",
            "confidence": 0.5,
            "cooling_period_days": 0,
            "analysis": "数据不足，无法分析"
        }

# 全局实例
volatility_analyzer = VolatilityAnalyzer()
