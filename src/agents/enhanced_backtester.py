"""
增强回测验证器

使用历史数据验证新的风险识别框架的有效性
特别是在2025-06-06前后的表现，验证是否能正确识别短暂回调vs系统性崩盘
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 导入增强的分析模块
from .integrated_risk_engine import IntegratedRiskEngine, MarketCondition
from .enhanced_portfolio_manager import EnhancedPortfolioManager, TradeType

logger = logging.getLogger(__name__)

class BacktestResult(Enum):
    """回测结果类型"""
    CORRECT_PREDICTION = "correct_prediction"      # 预测正确
    INCORRECT_PREDICTION = "incorrect_prediction"  # 预测错误
    PARTIAL_CORRECT = "partial_correct"           # 部分正确
    INSUFFICIENT_DATA = "insufficient_data"       # 数据不足

@dataclass
class TradeAnalysis:
    """交易分析结果"""
    date: str
    ticker: str
    trade_type: TradeType
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    holding_days: int
    pnl: float
    pnl_percentage: float
    market_condition_predicted: MarketCondition
    market_condition_actual: MarketCondition
    prediction_accuracy: str
    risk_assessment_score: float
    notes: str

@dataclass
class BacktestSummary:
    """回测总结"""
    total_trades: int
    profitable_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    prediction_accuracy: float
    short_selling_accuracy: float
    key_insights: List[str]
    performance_by_period: Dict[str, Dict]
    trade_analyses: List[TradeAnalysis]

class EnhancedBacktester:
    """增强回测验证器"""
    
    def __init__(self):
        self.risk_engine = IntegratedRiskEngine()
        self.portfolio_manager = EnhancedPortfolioManager()
        
        # 回测参数
        self.initial_capital = 1000000  # 100万初始资金
        self.transaction_cost = 0.001   # 0.1%交易成本
        self.short_borrow_cost = 0.02   # 2%年化融券成本
        
        # 关键测试期间
        self.test_periods = {
            "2025_06_correction": {
                "start": "2025-06-01",
                "end": "2025-06-15",
                "description": "2025年6月回调期间",
                "expected_behavior": "应识别为短期回调，避免融券"
            },
            "pre_correction": {
                "start": "2025-05-20",
                "end": "2025-06-05",
                "description": "回调前期",
                "expected_behavior": "正常市场波动"
            },
            "post_correction": {
                "start": "2025-06-08",
                "end": "2025-06-20",
                "description": "回调后恢复期",
                "expected_behavior": "识别恢复信号"
            }
        }

    def run_enhanced_backtest(self, tickers: List[str], start_date: str, end_date: str,
                            price_data: Dict[str, pd.DataFrame],
                            market_data_history: Dict[str, Dict],
                            economic_data_history: Dict[str, Dict],
                            news_data_history: Dict[str, Dict]) -> BacktestSummary:
        """
        运行增强回测
        
        Args:
            tickers: 股票代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            price_data: 价格数据
            market_data_history: 历史市场数据
            economic_data_history: 历史经济数据
            news_data_history: 历史新闻数据
            
        Returns:
            BacktestSummary: 回测总结
        """
        
        # 初始化回测状态
        portfolio = {ticker: 0.0 for ticker in tickers}  # 持仓
        cash = self.initial_capital
        portfolio_values = []
        trade_analyses = []
        
        # 生成交易日期序列
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_dates = [date.strftime('%Y-%m-%d') for date in date_range]
        
        for current_date in trading_dates:
            try:
                # 获取当日数据
                daily_market_data = market_data_history.get(current_date, {})
                daily_economic_data = economic_data_history.get(current_date, {})
                daily_news_data = news_data_history.get(current_date, {})
                
                # 获取当日价格
                daily_prices = {}
                for ticker in tickers:
                    if ticker in price_data and current_date in price_data[ticker].index:
                        daily_prices[ticker] = price_data[ticker].loc[current_date]
                
                if not daily_prices:
                    continue
                
                # 进行风险评估和决策
                portfolio_decision = self.portfolio_manager.make_portfolio_decision(
                    tickers=tickers,
                    market_data=daily_market_data,
                    price_data={ticker: price_data[ticker].loc[:current_date] 
                              for ticker in tickers if ticker in price_data},
                    economic_data=daily_economic_data,
                    news_data=daily_news_data,
                    current_positions=portfolio.copy(),
                    end_date=current_date
                )
                
                # 执行交易并记录分析
                daily_trades = self._execute_trades(
                    portfolio_decision, portfolio, cash, daily_prices, current_date
                )
                trade_analyses.extend(daily_trades)
                
                # 更新投资组合价值
                portfolio_value = self._calculate_portfolio_value(portfolio, daily_prices, cash)
                portfolio_values.append({
                    'date': current_date,
                    'value': portfolio_value,
                    'cash': cash,
                    'positions': portfolio.copy()
                })
                
            except Exception as e:
                logger.error(f"回测日期{current_date}处理失败: {e}")
                continue
        
        # 生成回测总结
        backtest_summary = self._generate_backtest_summary(
            trade_analyses, portfolio_values, start_date, end_date
        )
        
        return backtest_summary

    def _execute_trades(self, portfolio_decision, portfolio: Dict, cash: float,
                       daily_prices: Dict, current_date: str) -> List[TradeAnalysis]:
        """执行交易并生成分析"""
        
        trade_analyses = []
        
        for recommendation in portfolio_decision.position_recommendations:
            ticker = recommendation.ticker
            trade_type = recommendation.trade_type
            target_position = recommendation.position_size
            current_position = portfolio.get(ticker, 0.0)
            
            if ticker not in daily_prices:
                continue
            
            current_price = daily_prices[ticker]['close']
            
            # 计算交易量
            position_change = target_position - current_position
            
            if abs(position_change) < 0.01:  # 忽略微小变化
                continue
            
            # 执行交易
            trade_value = abs(position_change) * current_price
            transaction_cost = trade_value * self.transaction_cost
            
            if position_change > 0:  # 买入
                if cash >= trade_value + transaction_cost:
                    portfolio[ticker] = target_position
                    cash -= (trade_value + transaction_cost)
                    
                    trade_analysis = TradeAnalysis(
                        date=current_date,
                        ticker=ticker,
                        trade_type=trade_type,
                        entry_price=current_price,
                        exit_price=None,
                        position_size=position_change,
                        holding_days=0,
                        pnl=0.0,
                        pnl_percentage=0.0,
                        market_condition_predicted=MarketCondition.NORMAL_VOLATILITY,  # 需要从决策中获取
                        market_condition_actual=MarketCondition.NORMAL_VOLATILITY,    # 需要后续验证
                        prediction_accuracy="pending",
                        risk_assessment_score=recommendation.confidence,
                        notes=recommendation.reasoning
                    )
                    trade_analyses.append(trade_analysis)
                    
            elif position_change < 0:  # 卖出或融券
                portfolio[ticker] = target_position
                cash += (trade_value - transaction_cost)
                
                # 特别记录融券交易
                if trade_type == TradeType.SHORT_SELL:
                    trade_analysis = TradeAnalysis(
                        date=current_date,
                        ticker=ticker,
                        trade_type=trade_type,
                        entry_price=current_price,
                        exit_price=None,
                        position_size=position_change,
                        holding_days=0,
                        pnl=0.0,
                        pnl_percentage=0.0,
                        market_condition_predicted=MarketCondition.SEVERE_CORRECTION,  # 融券通常预期下跌
                        market_condition_actual=MarketCondition.NORMAL_VOLATILITY,
                        prediction_accuracy="pending",
                        risk_assessment_score=recommendation.confidence,
                        notes=f"融券决策: {recommendation.reasoning}"
                    )
                    trade_analyses.append(trade_analysis)
        
        return trade_analyses

    def _calculate_portfolio_value(self, portfolio: Dict, daily_prices: Dict, cash: float) -> float:
        """计算投资组合价值"""
        
        total_value = cash
        
        for ticker, position in portfolio.items():
            if ticker in daily_prices and position != 0:
                current_price = daily_prices[ticker]['close']
                position_value = position * current_price
                total_value += position_value
        
        return total_value

    def _generate_backtest_summary(self, trade_analyses: List[TradeAnalysis],
                                 portfolio_values: List[Dict],
                                 start_date: str, end_date: str) -> BacktestSummary:
        """生成回测总结"""
        
        if not portfolio_values:
            return self._get_empty_summary()
        
        # 计算基本统计
        initial_value = portfolio_values[0]['value']
        final_value = portfolio_values[-1]['value']
        total_return = (final_value - initial_value) / initial_value
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values)
        
        # 分析交易表现
        profitable_trades = len([t for t in trade_analyses if t.pnl > 0])
        losing_trades = len([t for t in trade_analyses if t.pnl < 0])
        total_trades = len(trade_analyses)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # 分析预测准确性
        prediction_accuracy = self._calculate_prediction_accuracy(trade_analyses)
        
        # 分析融券准确性
        short_selling_accuracy = self._calculate_short_selling_accuracy(trade_analyses)
        
        # 按期间分析表现
        performance_by_period = self._analyze_performance_by_period(
            portfolio_values, trade_analyses
        )
        
        # 生成关键洞察
        key_insights = self._generate_key_insights(
            trade_analyses, performance_by_period, short_selling_accuracy
        )
        
        return BacktestSummary(
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            prediction_accuracy=prediction_accuracy,
            short_selling_accuracy=short_selling_accuracy,
            key_insights=key_insights,
            performance_by_period=performance_by_period,
            trade_analyses=trade_analyses
        )

    def _calculate_max_drawdown(self, portfolio_values: List[Dict]) -> float:
        """计算最大回撤"""
        
        values = [pv['value'] for pv in portfolio_values]
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

    def _calculate_sharpe_ratio(self, portfolio_values: List[Dict]) -> float:
        """计算夏普比率"""
        
        if len(portfolio_values) < 2:
            return 0.0
        
        values = [pv['value'] for pv in portfolio_values]
        returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # 年化夏普比率 (假设252个交易日)
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
        
        return sharpe

    def _calculate_prediction_accuracy(self, trade_analyses: List[TradeAnalysis]) -> float:
        """计算预测准确性"""
        
        if not trade_analyses:
            return 0.0
        
        # 这里需要实际的市场状况数据来验证预测
        # 简化处理：基于交易盈亏来评估
        correct_predictions = len([t for t in trade_analyses if t.pnl > 0])
        total_predictions = len(trade_analyses)
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    def _calculate_short_selling_accuracy(self, trade_analyses: List[TradeAnalysis]) -> float:
        """计算融券准确性"""
        
        short_trades = [t for t in trade_analyses if t.trade_type == TradeType.SHORT_SELL]
        
        if not short_trades:
            return 0.0
        
        # 融券成功的标准：价格下跌
        successful_shorts = len([t for t in short_trades if t.pnl > 0])
        
        return successful_shorts / len(short_trades)

    def _analyze_performance_by_period(self, portfolio_values: List[Dict],
                                     trade_analyses: List[TradeAnalysis]) -> Dict[str, Dict]:
        """按期间分析表现"""
        
        performance_by_period = {}
        
        for period_name, period_info in self.test_periods.items():
            period_start = period_info["start"]
            period_end = period_info["end"]
            
            # 筛选期间内的数据
            period_values = [
                pv for pv in portfolio_values
                if period_start <= pv['date'] <= period_end
            ]
            
            period_trades = [
                t for t in trade_analyses
                if period_start <= t.date <= period_end
            ]
            
            if period_values:
                period_return = (period_values[-1]['value'] - period_values[0]['value']) / period_values[0]['value']
                period_trades_count = len(period_trades)
                period_short_trades = len([t for t in period_trades if t.trade_type == TradeType.SHORT_SELL])
                
                performance_by_period[period_name] = {
                    "return": period_return,
                    "trades_count": period_trades_count,
                    "short_trades_count": period_short_trades,
                    "description": period_info["description"],
                    "expected_behavior": period_info["expected_behavior"]
                }
        
        return performance_by_period

    def _generate_key_insights(self, trade_analyses: List[TradeAnalysis],
                             performance_by_period: Dict[str, Dict],
                             short_selling_accuracy: float) -> List[str]:
        """生成关键洞察"""
        
        insights = []
        
        # 融券表现分析
        if short_selling_accuracy > 0.7:
            insights.append(f"融券策略表现优秀，准确率达到{short_selling_accuracy:.1%}")
        elif short_selling_accuracy > 0.5:
            insights.append(f"融券策略表现一般，准确率为{short_selling_accuracy:.1%}")
        else:
            insights.append(f"融券策略需要改进，准确率仅为{short_selling_accuracy:.1%}")
        
        # 2025年6月期间表现分析
        june_performance = performance_by_period.get("2025_06_correction")
        if june_performance:
            short_count = june_performance["short_trades_count"]
            if short_count == 0:
                insights.append("✓ 在2025年6月回调期间正确避免了融券，识别出这是短期回调")
            elif short_count <= 2:
                insights.append("⚠ 在2025年6月回调期间有少量融券，基本正确识别了市场性质")
            else:
                insights.append("✗ 在2025年6月回调期间过度融券，未能正确识别短期回调性质")
        
        # 整体策略评估
        total_short_trades = len([t for t in trade_analyses if t.trade_type == TradeType.SHORT_SELL])
        if total_short_trades == 0:
            insights.append("策略过于保守，完全避免融券可能错失机会")
        elif total_short_trades > len(trade_analyses) * 0.3:
            insights.append("融券频率较高，需要验证是否过度激进")
        else:
            insights.append("融券频率适中，策略相对平衡")
        
        return insights

    def _get_empty_summary(self) -> BacktestSummary:
        """获取空的回测总结"""
        return BacktestSummary(
            total_trades=0,
            profitable_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            prediction_accuracy=0.0,
            short_selling_accuracy=0.0,
            key_insights=["回测数据不足"],
            performance_by_period={},
            trade_analyses=[]
        )
