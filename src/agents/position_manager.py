"""
持仓管理器 - 为AI添加时间观念和智能持仓策略
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from .volatility_analyzer import volatility_analyzer

logger = logging.getLogger(__name__)

class Position:
    """单个持仓记录"""
    def __init__(self, ticker: str, shares: int, cost_price: float, 
                 entry_date: str, entry_reason: str):
        self.ticker = ticker
        self.shares = shares
        self.cost_price = cost_price
        self.entry_date = datetime.strptime(entry_date, "%Y-%m-%d")
        self.entry_reason = entry_reason
        self.holding_days = 0
        
    def update_holding_days(self, current_date: str):
        """更新持仓天数"""
        current = datetime.strptime(current_date, "%Y-%m-%d")
        self.holding_days = (current - self.entry_date).days
        
    def get_unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        return (current_price - self.cost_price) * self.shares
        
    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """计算未实现盈亏百分比"""
        return (current_price - self.cost_price) / self.cost_price

class PositionManager:
    """持仓管理器"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.trading_rules = {
            # 最小持仓期（天）- 更合理的持仓期
            "min_holding_period": {
                "short_term": 2,    # 短线至少持有2天
                "medium_term": 5,   # 中线至少持有5天
                "long_term": 10     # 长线至少持有10天
            },
            # 止损线 - 更宽松的止损，避免被震出
            "stop_loss": {
                "aggressive": -0.12,  # 激进止损 -12%
                "moderate": -0.18,    # 适中止损 -18%
                "conservative": -0.25 # 保守止损 -25%
            },
            # 止盈线 - 更耐心的止盈策略
            "take_profit": {
                "quick": 0.20,      # 快速止盈 +20%
                "moderate": 0.35,   # 适中止盈 +35%
                "patient": 0.50     # 耐心止盈 +50%
            }
        }
    
    def add_position(self, ticker: str, shares: int, cost_price: float, 
                    entry_date: str, entry_reason: str) -> None:
        """添加新持仓"""
        if ticker in self.positions:
            # 如果已有持仓，计算平均成本
            existing = self.positions[ticker]
            total_shares = existing.shares + shares
            total_cost = existing.shares * existing.cost_price + shares * cost_price
            avg_cost = total_cost / total_shares
            
            existing.shares = total_shares
            existing.cost_price = avg_cost
            logger.info(f"增加持仓 {ticker}: {shares}股 @ ${cost_price:.2f}, 平均成本: ${avg_cost:.2f}")
        else:
            self.positions[ticker] = Position(ticker, shares, cost_price, entry_date, entry_reason)
            logger.info(f"新建持仓 {ticker}: {shares}股 @ ${cost_price:.2f}")
    
    def should_hold_position(self, ticker: str, current_price: float,
                           current_date: str, market_signals: Dict,
                           price_history: Optional[any] = None) -> Tuple[bool, str]:
        """
        判断是否应该继续持有仓位
        返回: (是否持有, 原因)
        """
        if ticker not in self.positions:
            return False, "无持仓"

        position = self.positions[ticker]
        position.update_holding_days(current_date)

        # 计算盈亏
        pnl_pct = position.get_unrealized_pnl_pct(current_price)

        # 异常波动分析
        volatility_analysis = None
        if price_history is not None:
            try:
                volatility_analysis = volatility_analyzer.analyze_price_shock(price_history, current_date)
            except Exception as e:
                logger.warning(f"异常波动分析失败: {e}")
                volatility_analysis = None
        
        # 1. 异常波动保护机制
        if volatility_analysis and volatility_analysis["shock_type"] == "news_driven":
            shock_severity = volatility_analysis["shock_severity"]
            cooling_period = volatility_analysis["cooling_period_days"]

            # 如果是消息面冲击且在冷静期内，避免恐慌性抛售
            if position.holding_days <= cooling_period and shock_severity in ["moderate", "severe", "extreme"]:
                if pnl_pct > -0.15:  # 亏损不超过15%时，给消息面冲击恢复时间
                    return True, f"消息面冲击冷静期({volatility_analysis['analysis']})，避免恐慌性抛售"

        # 2. 检查最小持仓期
        min_days = self._get_min_holding_period(position.entry_reason)
        if position.holding_days < min_days:
            # 但如果是极端波动且严重亏损，可以提前止损
            if (volatility_analysis and
                volatility_analysis["shock_severity"] == "extreme" and
                pnl_pct < -0.20):
                return False, f"极端波动且严重亏损({pnl_pct:.1%})，紧急止损"
            return True, f"未达最小持仓期({min_days}天，已持有{position.holding_days}天)"
        
        # 2. 智能止损检查
        stop_loss_result = self._intelligent_stop_loss_check(
            position, current_price, pnl_pct, volatility_analysis, market_signals
        )
        if not stop_loss_result["should_hold"]:
            return False, stop_loss_result["reason"]
        
        # 3. 检查止盈线
        take_profit = self._get_take_profit_level(position.entry_reason)
        if pnl_pct >= take_profit:
            return False, f"达到止盈目标({take_profit:.1%}，当前{pnl_pct:.1%})"
        
        # 4. 基于市场信号的持仓决策
        return self._evaluate_market_signals(position, current_price, market_signals)
    
    def _get_min_holding_period(self, entry_reason: str) -> int:
        """根据入场原因确定最小持仓期"""
        if "短线" in entry_reason or "momentum" in entry_reason.lower():
            return self.trading_rules["min_holding_period"]["short_term"]
        elif "长线" in entry_reason or "fundamental" in entry_reason.lower():
            return self.trading_rules["min_holding_period"]["long_term"]
        else:
            return self.trading_rules["min_holding_period"]["medium_term"]
    
    def _get_stop_loss_level(self, entry_reason: str) -> float:
        """根据入场原因确定止损水平"""
        if "激进" in entry_reason or "momentum" in entry_reason.lower():
            return self.trading_rules["stop_loss"]["aggressive"]
        elif "保守" in entry_reason or "value" in entry_reason.lower():
            return self.trading_rules["stop_loss"]["conservative"]
        else:
            return self.trading_rules["stop_loss"]["moderate"]
    
    def _get_take_profit_level(self, entry_reason: str) -> float:
        """根据入场原因确定止盈水平"""
        if "短线" in entry_reason or "momentum" in entry_reason.lower():
            return self.trading_rules["take_profit"]["quick"]
        elif "长线" in entry_reason or "value" in entry_reason.lower():
            return self.trading_rules["take_profit"]["patient"]
        else:
            return self.trading_rules["take_profit"]["moderate"]
    
    def _evaluate_market_signals(self, position: Position, current_price: float,
                               market_signals: Dict) -> Tuple[bool, str]:
        """基于市场信号评估是否继续持有"""
        pnl_pct = position.get_unrealized_pnl_pct(current_price)

        # 分析市场信号强度
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        high_confidence_bearish = 0

        for agent, signal_data in market_signals.items():
            if isinstance(signal_data, dict) and "signal" in signal_data:
                total_signals += 1
                confidence = signal_data.get("confidence", 0)
                signal = signal_data["signal"]

                if signal == "bullish":
                    bullish_signals += 1
                elif signal == "bearish":
                    bearish_signals += 1
                    if confidence > 75:  # 高信心度看跌
                        high_confidence_bearish += 1

        if total_signals == 0:
            return True, f"无市场信号，继续持有(盈亏{pnl_pct:.1%})"

        # 1. 严重亏损时的保护机制
        if pnl_pct < -0.10:  # 亏损超过10%
            if high_confidence_bearish >= 2:  # 多个高信心度看跌信号
                return False, f"严重亏损{pnl_pct:.1%}且多个高信心度看跌信号，止损保护"
            elif bearish_signals >= total_signals * 0.7:  # 70%以上看跌信号
                return False, f"严重亏损{pnl_pct:.1%}且市场强烈看跌，止损保护"

        # 2. 轻微亏损时更加耐心
        elif pnl_pct < -0.03:  # 轻微亏损3-10%
            if high_confidence_bearish >= 3:  # 需要更多看跌信号才卖出
                return False, f"轻微亏损{pnl_pct:.1%}但多个强烈看跌信号，谨慎止损"
            else:
                return True, f"轻微亏损{pnl_pct:.1%}，市场信号不够强烈，耐心持有"

        # 3. 盈利时的获利了结策略
        elif pnl_pct > 0.10 and position.holding_days >= 5:  # 盈利10%以上且持有5天以上
            if bearish_signals >= total_signals * 0.6:  # 60%以上看跌信号
                return False, f"盈利{pnl_pct:.1%}且市场信号转弱，适时获利了结"

        # 4. 小幅盈利时保持耐心
        elif pnl_pct > 0.03:  # 小幅盈利
            if high_confidence_bearish >= 3:  # 需要强烈看跌信号才卖出
                return False, f"小幅盈利{pnl_pct:.1%}但强烈看跌信号，获利了结"
            else:
                return True, f"小幅盈利{pnl_pct:.1%}，继续持有等待更大收益"

        return True, f"继续持有(盈亏{pnl_pct:.1%}，持有{position.holding_days}天，看涨{bullish_signals}/看跌{bearish_signals})"

    def _intelligent_stop_loss_check(self, position: Position, current_price: float,
                                   pnl_pct: float, volatility_analysis: Optional[Dict],
                                   market_signals: Dict) -> Dict:
        """
        智能止损检查，区分正常回调和系统性风险
        """
        # 获取基础止损线
        base_stop_loss = self._get_stop_loss_level(position.entry_reason)

        # 如果没有达到基础止损线，继续持有
        if pnl_pct > base_stop_loss:
            return {"should_hold": True, "reason": f"未达止损线({base_stop_loss:.1%})"}

        # 达到止损线，进行智能分析

        # 1. 检查是否是消息面冲击导致的暂时下跌
        if volatility_analysis and volatility_analysis["shock_type"] == "news_driven":
            shock_severity = volatility_analysis["shock_severity"]

            # 如果是轻微到中等的消息面冲击，且亏损不太严重，给予恢复时间
            if shock_severity in ["minor", "moderate"] and pnl_pct > -0.20:
                return {
                    "should_hold": True,
                    "reason": f"消息面冲击导致的暂时下跌({volatility_analysis['analysis']})，给予恢复时间"
                }

            # 如果是严重冲击但持仓时间很短，可能是过度反应
            elif shock_severity == "severe" and position.holding_days <= 2 and pnl_pct > -0.25:
                return {
                    "should_hold": True,
                    "reason": f"严重消息面冲击但持仓时间短，可能是过度反应"
                }

        # 2. 检查技术支撑位
        support_analysis = self._check_technical_support(current_price, pnl_pct)
        if support_analysis["near_support"]:
            return {
                "should_hold": True,
                "reason": f"接近技术支撑位({support_analysis['reason']})，暂缓止损"
            }

        # 3. 检查市场整体情况
        market_analysis = self._analyze_market_context(market_signals, pnl_pct)
        if market_analysis["market_oversold"]:
            return {
                "should_hold": True,
                "reason": f"市场整体超卖({market_analysis['reason']})，等待反弹"
            }

        # 4. 如果以上条件都不满足，执行止损
        return {
            "should_hold": False,
            "reason": f"智能止损：亏损{pnl_pct:.1%}超过止损线{base_stop_loss:.1%}，且无明显支撑因素"
        }

    def _check_technical_support(self, current_price: float, pnl_pct: float) -> Dict:
        """检查是否接近技术支撑位"""
        # 简化的支撑位检查
        # 如果亏损接近整数百分比（如-10%, -15%, -20%），可能接近心理支撑位

        psychological_levels = [-0.10, -0.15, -0.20, -0.25]

        for level in psychological_levels:
            if abs(pnl_pct - level) < 0.02:  # 在心理支撑位附近2%范围内
                return {
                    "near_support": True,
                    "reason": f"接近心理支撑位{level:.0%}"
                }

        return {"near_support": False, "reason": "无明显技术支撑"}

    def _analyze_market_context(self, market_signals: Dict, pnl_pct: float) -> Dict:
        """分析市场整体环境"""
        if not market_signals:
            return {"market_oversold": False, "reason": "无市场信号"}

        # 检查是否有多个分析师认为市场超卖
        oversold_signals = 0
        total_signals = 0

        for agent, signal_data in market_signals.items():
            if isinstance(signal_data, dict) and "signal" in signal_data:
                total_signals += 1
                signal = signal_data["signal"]
                confidence = signal_data.get("confidence", 0)

                # 如果是高信心度的看涨信号（可能表示超卖反弹）
                if signal == "bullish" and confidence > 60:
                    oversold_signals += 1

        # 如果有足够的超卖信号且个股亏损不太严重
        if oversold_signals >= 2 and total_signals >= 3 and pnl_pct > -0.25:
            return {
                "market_oversold": True,
                "reason": f"{oversold_signals}个分析师认为超卖"
            }

        return {"market_oversold": False, "reason": "市场无明显超卖信号"}
    
    def get_position_summary(self, ticker: str, current_price: float) -> Dict:
        """获取持仓摘要"""
        if ticker not in self.positions:
            return {"has_position": False}
            
        position = self.positions[ticker]
        pnl = position.get_unrealized_pnl(current_price)
        pnl_pct = position.get_unrealized_pnl_pct(current_price)
        
        return {
            "has_position": True,
            "shares": position.shares,
            "cost_price": position.cost_price,
            "current_price": current_price,
            "holding_days": position.holding_days,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": pnl_pct,
            "entry_reason": position.entry_reason,
            "position_value": position.shares * current_price
        }
    
    def remove_position(self, ticker: str, shares_to_sell: int = None) -> bool:
        """移除或减少持仓"""
        if ticker not in self.positions:
            return False
            
        position = self.positions[ticker]
        
        if shares_to_sell is None or shares_to_sell >= position.shares:
            # 全部卖出
            del self.positions[ticker]
            logger.info(f"清空持仓 {ticker}")
        else:
            # 部分卖出
            position.shares -= shares_to_sell
            logger.info(f"减少持仓 {ticker}: 卖出{shares_to_sell}股，剩余{position.shares}股")
        
        return True

# 全局持仓管理器实例
position_manager = PositionManager()
