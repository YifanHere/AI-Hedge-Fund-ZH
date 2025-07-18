"""
增强技术分析模块

在现有技术分析基础上，增加趋势指标、动能指标、成交量分析、形态识别等功能
特别关注系统性风险的技术特征，用于区分系统性崩盘和短暂回调
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TechnicalRiskLevel(Enum):
    """技术风险等级"""
    EXTREME = "extreme"          # 极端风险
    HIGH = "high"               # 高风险
    MODERATE = "moderate"       # 中等风险
    LOW = "low"                 # 低风险
    MINIMAL = "minimal"         # 极低风险

class TrendDirection(Enum):
    """趋势方向"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class VolumePattern(Enum):
    """成交量模式"""
    PANIC_SELLING = "panic_selling"        # 恐慌性抛售
    DISTRIBUTION = "distribution"          # 派发
    ACCUMULATION = "accumulation"          # 吸筹
    NORMAL = "normal"                      # 正常
    LOW_VOLUME = "low_volume"              # 缩量

@dataclass
class TechnicalAnalysisResult:
    """技术分析结果"""
    risk_level: TechnicalRiskLevel
    trend_direction: TrendDirection
    volume_pattern: VolumePattern
    systemic_risk_score: float
    crash_probability: float
    correction_probability: float
    key_indicators: Dict[str, float]
    risk_signals: List[str]
    support_resistance: Dict[str, float]
    interpretation: str

class EnhancedTechnicalAnalyzer:
    """增强技术分析器"""
    
    def __init__(self):
        # 系统性风险技术特征阈值
        self.systemic_risk_thresholds = {
            "rsi_extreme_oversold": 20,        # RSI极度超卖
            "rsi_failed_bounce": 30,           # RSI反弹失败
            "volume_spike_threshold": 3.0,     # 成交量激增倍数
            "price_gap_down": -0.05,           # 向下跳空阈值
            "consecutive_down_days": 5,        # 连续下跌天数
            "volatility_spike": 2.0,           # 波动率激增倍数
            "support_break_threshold": -0.03,  # 支撑位跌破阈值
            "correlation_breakdown": 0.8       # 相关性崩溃阈值
        }
        
        # 指标权重
        self.indicator_weights = {
            "trend_analysis": 0.30,
            "momentum_analysis": 0.25,
            "volume_analysis": 0.25,
            "volatility_analysis": 0.20
        }

    def analyze_technical_risk(self, price_data: pd.DataFrame, 
                             market_data: Dict, ticker: str) -> TechnicalAnalysisResult:
        """
        综合技术分析，识别系统性风险特征
        
        Args:
            price_data: 价格数据
            market_data: 市场数据
            ticker: 股票代码
            
        Returns:
            TechnicalAnalysisResult: 技术分析结果
        """
        
        if len(price_data) < 50:
            return self._get_default_result()
        
        # 1. 趋势分析
        trend_analysis = self._analyze_trend_characteristics(price_data)
        
        # 2. 动量分析
        momentum_analysis = self._analyze_momentum_indicators(price_data)
        
        # 3. 成交量分析
        volume_analysis = self._analyze_volume_patterns(price_data)
        
        # 4. 波动率分析
        volatility_analysis = self._analyze_volatility_patterns(price_data)
        
        # 5. 支撑阻力分析
        support_resistance = self._analyze_support_resistance(price_data)
        
        # 6. 形态识别
        pattern_analysis = self._analyze_price_patterns(price_data)
        
        # 综合评估
        comprehensive_assessment = self._synthesize_technical_analysis(
            trend_analysis, momentum_analysis, volume_analysis,
            volatility_analysis, support_resistance, pattern_analysis
        )
        
        return comprehensive_assessment

    def _analyze_trend_characteristics(self, price_data: pd.DataFrame) -> Dict:
        """
        分析趋势特征
        
        关键指标：
        - 多时间框架移动平均线
        - 趋势强度和方向
        - 趋势线突破
        - 价格结构分析
        """
        
        current_price = price_data["close"].iloc[-1]
        
        # 计算多个时间框架的移动平均线
        sma_10 = price_data["close"].rolling(10).mean().iloc[-1]
        sma_20 = price_data["close"].rolling(20).mean().iloc[-1]
        sma_50 = price_data["close"].rolling(50).mean().iloc[-1]
        sma_200 = price_data["close"].rolling(200).mean().iloc[-1] if len(price_data) >= 200 else sma_50
        
        # 计算EMA
        ema_12 = price_data["close"].ewm(span=12).mean().iloc[-1]
        ema_26 = price_data["close"].ewm(span=26).mean().iloc[-1]
        
        # 趋势强度分析
        trend_strength = self._calculate_trend_strength(price_data)
        
        # 移动平均线排列分析
        ma_alignment = self._analyze_ma_alignment(sma_10, sma_20, sma_50, sma_200)
        
        # 价格相对位置
        price_vs_sma20 = (current_price - sma_20) / sma_20
        price_vs_sma50 = (current_price - sma_50) / sma_50
        price_vs_sma200 = (current_price - sma_200) / sma_200
        
        # 趋势方向判断
        trend_direction = self._determine_trend_direction(
            ma_alignment, price_vs_sma20, price_vs_sma50, price_vs_sma200
        )
        
        # 计算趋势风险评分
        trend_risk_score = self._calculate_trend_risk_score(
            trend_direction, price_vs_sma200, ma_alignment, trend_strength
        )
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "ma_alignment": ma_alignment,
            "price_vs_sma20": price_vs_sma20,
            "price_vs_sma50": price_vs_sma50,
            "price_vs_sma200": price_vs_sma200,
            "risk_score": trend_risk_score,
            "key_levels": {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200
            }
        }

    def _analyze_momentum_indicators(self, price_data: pd.DataFrame) -> Dict:
        """
        分析动量指标
        
        关键指标：
        - RSI多时间框架分析
        - MACD信号
        - 动量背离
        - 超买超卖极端情况
        """
        
        # 计算RSI
        rsi_14 = self._calculate_rsi(price_data, 14)
        rsi_28 = self._calculate_rsi(price_data, 28)
        
        # 计算MACD
        macd_line, signal_line, histogram = self._calculate_macd(price_data)
        
        # 计算Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(price_data)
        
        # 计算Williams %R
        williams_r = self._calculate_williams_r(price_data)
        
        # 动量背离分析
        momentum_divergence = self._analyze_momentum_divergence(price_data, rsi_14)
        
        # 极端超卖分析
        extreme_oversold = self._analyze_extreme_oversold(rsi_14, rsi_28, williams_r)
        
        # 计算动量风险评分
        momentum_risk_score = self._calculate_momentum_risk_score(
            rsi_14, rsi_28, macd_line, signal_line, extreme_oversold, momentum_divergence
        )
        
        return {
            "rsi_14": rsi_14,
            "rsi_28": rsi_28,
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "williams_r": williams_r,
            "momentum_divergence": momentum_divergence,
            "extreme_oversold": extreme_oversold,
            "risk_score": momentum_risk_score
        }

    def _analyze_volume_patterns(self, price_data: pd.DataFrame) -> Dict:
        """
        分析成交量模式
        
        关键指标：
        - 成交量相对强度
        - 价量关系
        - 成交量模式识别
        - 恐慌性抛售信号
        """
        
        if "volume" not in price_data.columns:
            return self._get_default_volume_result()
        
        volume = price_data["volume"]
        current_volume = volume.iloc[-1]
        
        # 计算成交量移动平均
        volume_sma_20 = volume.rolling(20).mean().iloc[-1]
        volume_sma_50 = volume.rolling(50).mean().iloc[-1]
        
        # 成交量相对强度
        volume_ratio_20 = current_volume / volume_sma_20 if volume_sma_20 > 0 else 1
        volume_ratio_50 = current_volume / volume_sma_50 if volume_sma_50 > 0 else 1
        
        # 价量关系分析
        price_volume_relationship = self._analyze_price_volume_relationship(price_data)
        
        # 成交量模式识别
        volume_pattern = self._identify_volume_pattern(
            volume_ratio_20, volume_ratio_50, price_volume_relationship
        )
        
        # 恐慌性抛售检测
        panic_selling_signals = self._detect_panic_selling(price_data)
        
        # 计算成交量风险评分
        volume_risk_score = self._calculate_volume_risk_score(
            volume_pattern, panic_selling_signals, volume_ratio_20
        )
        
        return {
            "volume_pattern": volume_pattern,
            "volume_ratio_20": volume_ratio_20,
            "volume_ratio_50": volume_ratio_50,
            "price_volume_relationship": price_volume_relationship,
            "panic_selling_signals": panic_selling_signals,
            "risk_score": volume_risk_score,
            "current_volume": current_volume,
            "avg_volume_20": volume_sma_20
        }

    def _analyze_volatility_patterns(self, price_data: pd.DataFrame) -> Dict:
        """
        分析波动率模式
        
        关键指标：
        - 历史波动率
        - 波动率激增
        - ATR分析
        - 价格跳空
        """
        
        # 计算收益率
        returns = price_data["close"].pct_change().dropna()
        
        # 计算历史波动率
        volatility_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        volatility_60 = returns.rolling(60).std().iloc[-1] * np.sqrt(252) if len(returns) >= 60 else volatility_20
        
        # 波动率比率
        volatility_ratio = volatility_20 / volatility_60 if volatility_60 > 0 else 1
        
        # 计算ATR
        atr = self._calculate_atr(price_data)
        
        # 价格跳空分析
        gaps = self._analyze_price_gaps(price_data)
        
        # 极端波动检测
        extreme_volatility = self._detect_extreme_volatility(returns, volatility_ratio)
        
        # 计算波动率风险评分
        volatility_risk_score = self._calculate_volatility_risk_score(
            volatility_ratio, extreme_volatility, gaps, atr
        )
        
        return {
            "volatility_20": volatility_20,
            "volatility_60": volatility_60,
            "volatility_ratio": volatility_ratio,
            "atr": atr,
            "gaps": gaps,
            "extreme_volatility": extreme_volatility,
            "risk_score": volatility_risk_score
        }

    def _analyze_support_resistance(self, price_data: pd.DataFrame) -> Dict:
        """
        分析支撑阻力位

        关键指标：
        - 关键支撑位
        - 阻力位
        - 支撑位跌破情况
        - 价格结构分析
        """

        # 计算关键价位
        recent_high = price_data["high"].rolling(20).max().iloc[-1]
        recent_low = price_data["low"].rolling(20).min().iloc[-1]

        # 更长期的支撑阻力
        long_term_high = price_data["high"].rolling(60).max().iloc[-1] if len(price_data) >= 60 else recent_high
        long_term_low = price_data["low"].rolling(60).min().iloc[-1] if len(price_data) >= 60 else recent_low

        current_price = price_data["close"].iloc[-1]

        # 支撑位强度分析
        support_strength = self._calculate_support_strength(price_data, recent_low)
        resistance_strength = self._calculate_resistance_strength(price_data, recent_high)

        # 支撑位跌破分析
        support_break = self._analyze_support_break(price_data, recent_low, long_term_low)

        return {
            "recent_support": recent_low,
            "recent_resistance": recent_high,
            "long_term_support": long_term_low,
            "long_term_resistance": long_term_high,
            "support_strength": support_strength,
            "resistance_strength": resistance_strength,
            "support_break": support_break,
            "distance_to_support": (current_price - recent_low) / recent_low,
            "distance_to_resistance": (recent_high - current_price) / current_price
        }

    def _analyze_price_patterns(self, price_data: pd.DataFrame) -> Dict:
        """
        分析价格形态

        关键指标：
        - 连续下跌形态
        - 跳空缺口
        - 反转形态
        - 持续形态
        """

        # 连续下跌分析
        consecutive_down = self._analyze_consecutive_down_days(price_data)

        # 跳空分析
        gap_analysis = self._analyze_gaps_detailed(price_data)

        # 反转形态识别
        reversal_patterns = self._identify_reversal_patterns(price_data)

        # 持续形态识别
        continuation_patterns = self._identify_continuation_patterns(price_data)

        return {
            "consecutive_down": consecutive_down,
            "gap_analysis": gap_analysis,
            "reversal_patterns": reversal_patterns,
            "continuation_patterns": continuation_patterns
        }

    # 具体计算方法
    def _calculate_rsi(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """计算RSI"""
        if len(price_data) < period + 1:
            return 50.0

        delta = price_data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calculate_macd(self, price_data: pd.DataFrame) -> Tuple[float, float, float]:
        """计算MACD"""
        if len(price_data) < 26:
            return 0.0, 0.0, 0.0

        ema_12 = price_data["close"].ewm(span=12).mean()
        ema_26 = price_data["close"].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        return (
            macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0,
            signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0.0,
            histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
        )

    def _calculate_stochastic(self, price_data: pd.DataFrame, k_period: int = 14) -> Tuple[float, float]:
        """计算随机指标"""
        if len(price_data) < k_period:
            return 50.0, 50.0

        lowest_low = price_data["low"].rolling(k_period).min()
        highest_high = price_data["high"].rolling(k_period).max()

        k_percent = 100 * ((price_data["close"] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(3).mean()

        return (
            k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50.0,
            d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50.0
        )

    def _calculate_williams_r(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """计算Williams %R"""
        if len(price_data) < period:
            return -50.0

        highest_high = price_data["high"].rolling(period).max()
        lowest_low = price_data["low"].rolling(period).min()

        williams_r = -100 * ((highest_high - price_data["close"]) / (highest_high - lowest_low))

        return williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50.0

    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        if len(price_data) < period + 1:
            return 0.0

        high_low = price_data["high"] - price_data["low"]
        high_close = np.abs(price_data["high"] - price_data["close"].shift())
        low_close = np.abs(price_data["low"] - price_data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()

        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """计算趋势强度"""
        if len(price_data) < 20:
            return 0.5

        # 使用ADX的简化版本
        high_low = price_data["high"] - price_data["low"]
        high_close = np.abs(price_data["high"] - price_data["close"].shift())
        low_close = np.abs(price_data["low"] - price_data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        plus_dm = np.where(
            (price_data["high"].diff() > price_data["low"].diff().abs()) &
            (price_data["high"].diff() > 0),
            price_data["high"].diff(),
            0
        )

        minus_dm = np.where(
            (price_data["low"].diff().abs() > price_data["high"].diff()) &
            (price_data["low"].diff() < 0),
            price_data["low"].diff().abs(),
            0
        )

        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / true_range.rolling(14).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / true_range.rolling(14).mean())

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()

        return adx.iloc[-1] / 100 if not pd.isna(adx.iloc[-1]) else 0.5

    def _analyze_ma_alignment(self, sma_10: float, sma_20: float,
                            sma_50: float, sma_200: float) -> str:
        """分析移动平均线排列"""
        if sma_10 > sma_20 > sma_50 > sma_200:
            return "bullish_alignment"
        elif sma_10 < sma_20 < sma_50 < sma_200:
            return "bearish_alignment"
        elif sma_10 > sma_20 > sma_50 and sma_50 < sma_200:
            return "mixed_bullish"
        elif sma_10 < sma_20 < sma_50 and sma_50 > sma_200:
            return "mixed_bearish"
        else:
            return "chaotic"

    def _determine_trend_direction(self, ma_alignment: str, price_vs_sma20: float,
                                 price_vs_sma50: float, price_vs_sma200: float) -> TrendDirection:
        """确定趋势方向"""
        if ma_alignment == "bullish_alignment" and price_vs_sma200 > 0.1:
            return TrendDirection.STRONG_UPTREND
        elif ma_alignment == "bullish_alignment" and price_vs_sma200 > 0:
            return TrendDirection.WEAK_UPTREND
        elif ma_alignment == "bearish_alignment" and price_vs_sma200 < -0.1:
            return TrendDirection.STRONG_DOWNTREND
        elif ma_alignment == "bearish_alignment" and price_vs_sma200 < 0:
            return TrendDirection.WEAK_DOWNTREND
        else:
            return TrendDirection.SIDEWAYS

    def _calculate_trend_risk_score(self, trend_direction: TrendDirection,
                                  price_vs_sma200: float, ma_alignment: str,
                                  trend_strength: float) -> float:
        """计算趋势风险评分"""
        risk_score = 0

        # 趋势方向风险
        if trend_direction == TrendDirection.STRONG_DOWNTREND:
            risk_score += 40
        elif trend_direction == TrendDirection.WEAK_DOWNTREND:
            risk_score += 30
        elif trend_direction == TrendDirection.SIDEWAYS:
            risk_score += 20
        elif trend_direction == TrendDirection.WEAK_UPTREND:
            risk_score += 10
        # STRONG_UPTREND 不加分

        # 价格相对200日均线风险
        if price_vs_sma200 < -0.2:
            risk_score += 30
        elif price_vs_sma200 < -0.1:
            risk_score += 25
        elif price_vs_sma200 < -0.05:
            risk_score += 20
        elif price_vs_sma200 < 0:
            risk_score += 15

        # 移动平均线排列风险
        if ma_alignment == "bearish_alignment":
            risk_score += 20
        elif ma_alignment == "mixed_bearish":
            risk_score += 15
        elif ma_alignment == "chaotic":
            risk_score += 10

        # 趋势强度调整
        if trend_strength < 0.2:
            risk_score += 10  # 弱趋势增加不确定性

        return min(risk_score, 100)

    def _analyze_momentum_divergence(self, price_data: pd.DataFrame, rsi: pd.Series) -> bool:
        """分析动量背离"""
        if len(price_data) < 20:
            return False

        # 简化的背离检测
        recent_price_high = price_data["close"].rolling(10).max().iloc[-1]
        recent_rsi_high = rsi.rolling(10).max().iloc[-1] if hasattr(rsi, 'rolling') else rsi

        # 检查是否价格创新高但RSI未创新高
        price_trend = price_data["close"].iloc[-1] / price_data["close"].iloc[-10] - 1

        return price_trend > 0.05 and recent_rsi_high < 70

    def _analyze_extreme_oversold(self, rsi_14: float, rsi_28: float, williams_r: float) -> Dict:
        """分析极端超卖情况"""
        extreme_signals = []

        if rsi_14 < self.systemic_risk_thresholds["rsi_extreme_oversold"]:
            extreme_signals.append("RSI14极度超卖")

        if rsi_28 < 25:
            extreme_signals.append("RSI28极度超卖")

        if williams_r < -90:
            extreme_signals.append("Williams%R极度超卖")

        # 检查是否反弹失败
        failed_bounce = rsi_14 < self.systemic_risk_thresholds["rsi_failed_bounce"] and len(extreme_signals) > 0

        return {
            "is_extreme": len(extreme_signals) >= 2,
            "failed_bounce": failed_bounce,
            "signals": extreme_signals
        }

    def _calculate_momentum_risk_score(self, rsi_14: float, rsi_28: float,
                                     macd_line: float, signal_line: float,
                                     extreme_oversold: Dict, momentum_divergence: bool) -> float:
        """计算动量风险评分"""
        risk_score = 0

        # RSI风险评分
        if rsi_14 > 80:
            risk_score += 25  # 极度超买
        elif rsi_14 > 70:
            risk_score += 15  # 超买
        elif rsi_14 < 20:
            risk_score += 30  # 极度超卖，可能继续下跌
        elif rsi_14 < 30:
            risk_score += 20  # 超卖

        # MACD风险评分
        if macd_line < signal_line and macd_line < 0:
            risk_score += 20  # 空头信号
        elif macd_line < signal_line:
            risk_score += 15  # 弱势

        # 极端超卖风险
        if extreme_oversold["is_extreme"]:
            risk_score += 25
        if extreme_oversold["failed_bounce"]:
            risk_score += 15

        # 动量背离风险
        if momentum_divergence:
            risk_score += 10

        return min(risk_score, 100)

    def _analyze_price_volume_relationship(self, price_data: pd.DataFrame) -> str:
        """分析价量关系"""
        if "volume" not in price_data.columns or len(price_data) < 5:
            return "unknown"

        recent_price_change = price_data["close"].pct_change().iloc[-1]
        recent_volume_change = price_data["volume"].pct_change().iloc[-1]

        if recent_price_change < -0.02 and recent_volume_change > 0.5:
            return "volume_selling"  # 放量下跌
        elif recent_price_change > 0.02 and recent_volume_change > 0.5:
            return "volume_buying"   # 放量上涨
        elif recent_price_change < -0.02 and recent_volume_change < -0.2:
            return "no_volume_selling"  # 缩量下跌
        elif recent_price_change > 0.02 and recent_volume_change < -0.2:
            return "no_volume_buying"   # 缩量上涨
        else:
            return "normal"

    def _identify_volume_pattern(self, volume_ratio_20: float, volume_ratio_50: float,
                               price_volume_relationship: str) -> VolumePattern:
        """识别成交量模式"""
        if volume_ratio_20 > 3 and price_volume_relationship == "volume_selling":
            return VolumePattern.PANIC_SELLING
        elif volume_ratio_20 > 2 and price_volume_relationship == "volume_selling":
            return VolumePattern.DISTRIBUTION
        elif volume_ratio_20 > 2 and price_volume_relationship == "volume_buying":
            return VolumePattern.ACCUMULATION
        elif volume_ratio_20 < 0.5:
            return VolumePattern.LOW_VOLUME
        else:
            return VolumePattern.NORMAL

    def _detect_panic_selling(self, price_data: pd.DataFrame) -> List[str]:
        """检测恐慌性抛售信号"""
        signals = []

        if "volume" not in price_data.columns or len(price_data) < 5:
            return signals

        # 检查最近几天的价量关系
        for i in range(1, min(6, len(price_data))):
            price_change = price_data["close"].iloc[-i] / price_data["close"].iloc[-i-1] - 1
            volume_ratio = price_data["volume"].iloc[-i] / price_data["volume"].rolling(20).mean().iloc[-i]

            if price_change < -0.05 and volume_ratio > 3:
                signals.append(f"第{i}天出现恐慌性抛售")
            elif price_change < -0.03 and volume_ratio > 2:
                signals.append(f"第{i}天出现大量抛售")

        return signals

    def _calculate_volume_risk_score(self, volume_pattern: VolumePattern,
                                   panic_selling_signals: List[str], volume_ratio: float) -> float:
        """计算成交量风险评分"""
        risk_score = 0

        # 成交量模式风险
        if volume_pattern == VolumePattern.PANIC_SELLING:
            risk_score += 40
        elif volume_pattern == VolumePattern.DISTRIBUTION:
            risk_score += 30
        elif volume_pattern == VolumePattern.LOW_VOLUME:
            risk_score += 15  # 缩量下跌也有风险
        elif volume_pattern == VolumePattern.ACCUMULATION:
            risk_score += 5   # 低风险

        # 恐慌性抛售信号
        risk_score += len(panic_selling_signals) * 15

        # 成交量异常放大
        if volume_ratio > 5:
            risk_score += 25
        elif volume_ratio > 3:
            risk_score += 20
        elif volume_ratio > 2:
            risk_score += 15

        return min(risk_score, 100)

    def _detect_extreme_volatility(self, returns: pd.Series, volatility_ratio: float) -> Dict:
        """检测极端波动"""
        extreme_signals = []

        # 检查单日极端收益
        if len(returns) > 0:
            latest_return = abs(returns.iloc[-1])
            if latest_return > 0.1:
                extreme_signals.append("单日波动超过10%")
            elif latest_return > 0.05:
                extreme_signals.append("单日波动超过5%")

        # 检查波动率激增
        if volatility_ratio > self.systemic_risk_thresholds["volatility_spike"]:
            extreme_signals.append("波动率激增")

        # 检查连续大幅波动
        if len(returns) >= 5:
            recent_volatility = returns.tail(5).abs().mean()
            if recent_volatility > 0.03:
                extreme_signals.append("连续高波动")

        return {
            "is_extreme": len(extreme_signals) > 0,
            "signals": extreme_signals
        }

    def _analyze_price_gaps(self, price_data: pd.DataFrame) -> Dict:
        """分析价格跳空"""
        gaps = []

        if len(price_data) < 10:
            return {"gaps": gaps, "gap_count": 0}

        for i in range(1, min(10, len(price_data))):
            if "open" in price_data.columns:
                prev_close = price_data["close"].iloc[-i-1]
                curr_open = price_data["open"].iloc[-i]
                gap = (curr_open - prev_close) / prev_close

                if gap < self.systemic_risk_thresholds["price_gap_down"]:
                    gaps.append({
                        "day": i,
                        "gap_size": gap,
                        "type": "gap_down"
                    })
                elif gap > 0.03:
                    gaps.append({
                        "day": i,
                        "gap_size": gap,
                        "type": "gap_up"
                    })

        return {
            "gaps": gaps,
            "gap_count": len(gaps),
            "down_gaps": len([g for g in gaps if g["type"] == "gap_down"])
        }

    def _calculate_volatility_risk_score(self, volatility_ratio: float,
                                       extreme_volatility: Dict, gaps: Dict, atr: float) -> float:
        """计算波动率风险评分"""
        risk_score = 0

        # 波动率比率风险
        if volatility_ratio > 3:
            risk_score += 30
        elif volatility_ratio > 2:
            risk_score += 25
        elif volatility_ratio > 1.5:
            risk_score += 20
        elif volatility_ratio > 1.2:
            risk_score += 15

        # 极端波动风险
        if extreme_volatility["is_extreme"]:
            risk_score += 25

        # 跳空风险
        down_gaps = gaps.get("down_gaps", 0)
        if down_gaps >= 3:
            risk_score += 25
        elif down_gaps >= 2:
            risk_score += 20
        elif down_gaps >= 1:
            risk_score += 15

        return min(risk_score, 100)

    def _synthesize_technical_analysis(self, trend_analysis: Dict, momentum_analysis: Dict,
                                     volume_analysis: Dict, volatility_analysis: Dict,
                                     support_resistance: Dict, pattern_analysis: Dict) -> TechnicalAnalysisResult:
        """综合技术分析结果"""

        # 计算加权综合风险评分
        total_risk_score = (
            trend_analysis["risk_score"] * self.indicator_weights["trend_analysis"] +
            momentum_analysis["risk_score"] * self.indicator_weights["momentum_analysis"] +
            volume_analysis["risk_score"] * self.indicator_weights["volume_analysis"] +
            volatility_analysis["risk_score"] * self.indicator_weights["volatility_analysis"]
        )

        # 确定风险等级
        if total_risk_score >= 80:
            risk_level = TechnicalRiskLevel.EXTREME
        elif total_risk_score >= 65:
            risk_level = TechnicalRiskLevel.HIGH
        elif total_risk_score >= 45:
            risk_level = TechnicalRiskLevel.MODERATE
        elif total_risk_score >= 25:
            risk_level = TechnicalRiskLevel.LOW
        else:
            risk_level = TechnicalRiskLevel.MINIMAL

        # 计算系统性风险概率
        systemic_risk_score = self._calculate_systemic_risk_score(
            trend_analysis, momentum_analysis, volume_analysis, volatility_analysis
        )

        # 计算崩盘vs回调概率
        crash_probability, correction_probability = self._calculate_crash_correction_probabilities(
            systemic_risk_score, trend_analysis, volume_analysis
        )

        # 识别风险信号
        risk_signals = self._identify_technical_risk_signals(
            trend_analysis, momentum_analysis, volume_analysis, volatility_analysis
        )

        # 生成解释
        interpretation = self._generate_technical_interpretation(
            risk_level, trend_analysis["trend_direction"], volume_analysis["volume_pattern"],
            systemic_risk_score, risk_signals
        )

        # 汇总关键指标
        key_indicators = {
            "rsi_14": momentum_analysis["rsi_14"],
            "macd_line": momentum_analysis["macd_line"],
            "volume_ratio": volume_analysis["volume_ratio_20"],
            "volatility_ratio": volatility_analysis["volatility_ratio"],
            "price_vs_sma200": trend_analysis["price_vs_sma200"],
            "trend_strength": trend_analysis["trend_strength"],
            "atr": volatility_analysis["atr"]
        }

        return TechnicalAnalysisResult(
            risk_level=risk_level,
            trend_direction=trend_analysis["trend_direction"],
            volume_pattern=volume_analysis["volume_pattern"],
            systemic_risk_score=systemic_risk_score,
            crash_probability=crash_probability,
            correction_probability=correction_probability,
            key_indicators=key_indicators,
            risk_signals=risk_signals,
            support_resistance=support_resistance,
            interpretation=interpretation
        )

    def _calculate_systemic_risk_score(self, trend_analysis: Dict, momentum_analysis: Dict,
                                     volume_analysis: Dict, volatility_analysis: Dict) -> float:
        """计算系统性风险评分"""

        systemic_signals = 0

        # 趋势信号
        if trend_analysis["trend_direction"] == TrendDirection.STRONG_DOWNTREND:
            systemic_signals += 2
        elif trend_analysis["trend_direction"] == TrendDirection.WEAK_DOWNTREND:
            systemic_signals += 1

        # 动量信号
        if momentum_analysis["extreme_oversold"]["is_extreme"]:
            systemic_signals += 2
        if momentum_analysis["extreme_oversold"]["failed_bounce"]:
            systemic_signals += 1

        # 成交量信号
        if volume_analysis["volume_pattern"] == VolumePattern.PANIC_SELLING:
            systemic_signals += 2
        elif volume_analysis["volume_pattern"] == VolumePattern.DISTRIBUTION:
            systemic_signals += 1

        # 波动率信号
        if volatility_analysis["extreme_volatility"]["is_extreme"]:
            systemic_signals += 1
        if volatility_analysis["volatility_ratio"] > 2:
            systemic_signals += 1

        # 转换为0-100评分
        return min(systemic_signals * 12.5, 100)

    def _calculate_crash_correction_probabilities(self, systemic_risk_score: float,
                                                trend_analysis: Dict, volume_analysis: Dict) -> Tuple[float, float]:
        """计算崩盘vs回调概率"""

        # 基础概率
        base_crash_prob = systemic_risk_score / 100 * 0.7  # 最高70%

        # 调整因子
        if volume_analysis["volume_pattern"] == VolumePattern.PANIC_SELLING:
            base_crash_prob += 0.2
        elif volume_analysis["volume_pattern"] == VolumePattern.DISTRIBUTION:
            base_crash_prob += 0.1

        if trend_analysis["trend_direction"] == TrendDirection.STRONG_DOWNTREND:
            base_crash_prob += 0.15

        crash_probability = min(base_crash_prob, 0.85)
        correction_probability = 1 - crash_probability

        return crash_probability, correction_probability

    def _get_default_result(self) -> TechnicalAnalysisResult:
        """获取默认结果"""
        return TechnicalAnalysisResult(
            risk_level=TechnicalRiskLevel.MODERATE,
            trend_direction=TrendDirection.SIDEWAYS,
            volume_pattern=VolumePattern.NORMAL,
            systemic_risk_score=50.0,
            crash_probability=0.3,
            correction_probability=0.7,
            key_indicators={},
            risk_signals=[],
            support_resistance={},
            interpretation="数据不足，无法进行完整技术分析"
        )

    def _get_default_volume_result(self) -> Dict:
        """获取默认成交量结果"""
        return {
            "volume_pattern": VolumePattern.NORMAL,
            "volume_ratio_20": 1.0,
            "volume_ratio_50": 1.0,
            "price_volume_relationship": "unknown",
            "panic_selling_signals": [],
            "risk_score": 50,
            "current_volume": 0,
            "avg_volume_20": 0
        }
