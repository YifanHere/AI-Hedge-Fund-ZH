"""
AI分析师思考过程追踪器
用于记录、存储和导出每次决策前所有AI分析师的详细思考过程
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalystReasoning:
    """单个分析师的思考记录"""
    agent_id: str
    agent_name: str
    ticker: str
    signal: str
    confidence: float
    reasoning: str
    timestamp: str
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class DecisionPoint:
    """单个决策点的记录"""
    date: str
    timestamp: str
    tickers: List[str]
    analyst_reasonings: List[AnalystReasoning]
    portfolio_state: Dict[str, Any]
    market_data: Optional[Dict[str, Any]] = None


class AnalystReasoningTracker:
    """AI分析师思考过程追踪器"""
    
    def __init__(self):
        self.decision_points: List[DecisionPoint] = []
        self.current_decision_point: Optional[DecisionPoint] = None
        
    def start_decision_point(self, date: str, tickers: List[str], portfolio_state: Dict[str, Any], 
                           market_data: Optional[Dict[str, Any]] = None):
        """开始一个新的决策点记录"""
        timestamp = datetime.now().isoformat()
        self.current_decision_point = DecisionPoint(
            date=date,
            timestamp=timestamp,
            tickers=tickers,
            analyst_reasonings=[],
            portfolio_state=portfolio_state.copy(),
            market_data=market_data
        )
    
    def add_analyst_reasoning(self, agent_id: str, agent_name: str, ticker: str, 
                            signal: str, confidence: float, reasoning: str,
                            additional_data: Optional[Dict[str, Any]] = None):
        """添加单个分析师的思考记录"""
        if self.current_decision_point is None:
            raise ValueError("必须先调用start_decision_point()开始决策点记录")
        
        timestamp = datetime.now().isoformat()
        analyst_reasoning = AnalystReasoning(
            agent_id=agent_id,
            agent_name=agent_name,
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=timestamp,
            additional_data=additional_data
        )
        
        self.current_decision_point.analyst_reasonings.append(analyst_reasoning)
    
    def finish_decision_point(self):
        """完成当前决策点记录"""
        if self.current_decision_point is None:
            raise ValueError("没有活跃的决策点可以完成")
        
        self.decision_points.append(self.current_decision_point)
        self.current_decision_point = None
    
    def get_decision_points_count(self) -> int:
        """获取决策点数量"""
        return len(self.decision_points)
    
    def get_latest_decision_point(self) -> Optional[DecisionPoint]:
        """获取最新的决策点"""
        return self.decision_points[-1] if self.decision_points else None
    
    def get_decision_point_by_date(self, date: str) -> Optional[DecisionPoint]:
        """根据日期获取决策点"""
        for dp in self.decision_points:
            if dp.date == date:
                return dp
        return None
    
    def get_analyst_reasoning_by_agent(self, agent_id: str) -> List[AnalystReasoning]:
        """获取特定分析师的所有思考记录"""
        reasonings = []
        for dp in self.decision_points:
            for reasoning in dp.analyst_reasonings:
                if reasoning.agent_id == agent_id:
                    reasonings.append(reasoning)
        return reasonings
    
    def get_analyst_reasoning_by_ticker(self, ticker: str) -> List[AnalystReasoning]:
        """获取特定股票的所有分析师思考记录"""
        reasonings = []
        for dp in self.decision_points:
            for reasoning in dp.analyst_reasonings:
                if reasoning.ticker == ticker:
                    reasonings.append(reasoning)
        return reasonings
    
    def clear(self):
        """清空所有记录"""
        self.decision_points.clear()
        self.current_decision_point = None
    






    def export_enhanced_analysis_report(self, filepath: str) -> bool:
        """导出增强版详细分析报告（包含更多数据指标）"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 报告头部
                f.write("# AI分析师思考过程详细分析报告\n\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**分析周期**: {self.decision_points[0].date} 至 {self.decision_points[-1].date}\n")
                f.write(f"**总决策点数**: {len(self.decision_points)}\n\n")

                # 生成汇总统计
                stats = self.generate_summary_statistics()
                f.write("## 📊 整体统计概览\n\n")
                f.write(f"- **总分析次数**: {stats.get('总分析次数', 0)}\n")
                f.write(f"- **平均信心度**: {stats.get('平均信心度', 0)}%\n")

                signal_dist = stats.get('信号分布', {})
                f.write(f"- **信号分布**: 看涨 {signal_dist.get('bullish', 0)} | 看跌 {signal_dist.get('bearish', 0)} | 中性 {signal_dist.get('neutral', 0)}\n")

                # 计算信号比例
                total_signals = sum(signal_dist.values())
                if total_signals > 0:
                    bullish_pct = (signal_dist.get('bullish', 0) / total_signals) * 100
                    bearish_pct = (signal_dist.get('bearish', 0) / total_signals) * 100
                    neutral_pct = (signal_dist.get('neutral', 0) / total_signals) * 100
                    f.write(f"- **信号比例**: 看涨 {bullish_pct:.1f}% | 看跌 {bearish_pct:.1f}% | 中性 {neutral_pct:.1f}%\n")

                f.write("\n")

                # 分析师表现分析
                f.write("## 👥 分析师表现分析\n\n")
                analyst_stats = self._calculate_analyst_performance_stats()

                f.write("| 分析师 | 分析次数 | 平均信心度 | 看涨比例 | 看跌比例 | 中性比例 | 信心度范围 |\n")
                f.write("|--------|----------|------------|----------|----------|----------|------------|\n")

                for analyst_name, data in analyst_stats.items():
                    f.write(f"| {analyst_name} | {data['total']} | {data['avg_confidence']:.1f}% | "
                           f"{data['bullish_pct']:.1f}% | {data['bearish_pct']:.1f}% | {data['neutral_pct']:.1f}% | "
                           f"{data['min_confidence']:.0f}-{data['max_confidence']:.0f}% |\n")

                f.write("\n")

                # 股票分析汇总
                f.write("## 📈 股票分析汇总\n\n")
                ticker_stats = self._calculate_ticker_stats()

                f.write("| 股票代码 | 分析次数 | 看涨次数 | 看跌次数 | 中性次数 | 平均信心度 | 分析师一致性 |\n")
                f.write("|----------|----------|----------|----------|----------|------------|-------------|\n")

                for ticker, data in ticker_stats.items():
                    f.write(f"| {ticker} | {data['total']} | {data['bullish']} | {data['bearish']} | "
                           f"{data['neutral']} | {data['avg_confidence']:.1f}% | {data['consensus_rate']:.1f}% |\n")

                f.write("\n")

                # 时间趋势分析
                f.write("## 📅 时间趋势分析\n\n")
                time_trends = self._calculate_time_trends()

                f.write("| 日期 | 看涨信号 | 看跌信号 | 中性信号 | 平均信心度 | 市场情绪 |\n")
                f.write("|------|----------|----------|----------|------------|----------|\n")

                for date, data in time_trends.items():
                    sentiment = self._determine_market_sentiment(data['bullish'], data['bearish'], data['neutral'])
                    f.write(f"| {date} | {data['bullish']} | {data['bearish']} | {data['neutral']} | "
                           f"{data['avg_confidence']:.1f}% | {sentiment} |\n")

                f.write("\n")

                # 详细决策点分析
                f.write("## 📋 详细决策点分析\n\n")

                for i, dp in enumerate(self.decision_points, 1):
                    f.write(f"### 决策点 {i}: {dp.date}\n\n")
                    f.write(f"**时间**: {dp.timestamp}\n")
                    f.write(f"**股票代码**: {', '.join(dp.tickers)}\n")
                    f.write(f"**投资组合现金**: ${dp.portfolio_state.get('cash', 0):,.2f}\n")

                    # 计算当日投资组合总价值
                    total_value = self._calculate_portfolio_value_for_date(dp)
                    f.write(f"**投资组合总价值**: ${total_value:,.2f}\n")

                    # 当日市场数据
                    if dp.market_data:
                        f.write(f"**市场价格**: ")
                        prices = [f"{ticker}=${price:.2f}" for ticker, price in dp.market_data.items()]
                        f.write(" | ".join(prices))
                        f.write("\n")

                    f.write("\n")

                    # 当日分析师共识
                    daily_consensus = self._calculate_daily_consensus(dp)
                    f.write(f"**当日共识**: {daily_consensus}\n\n")

                    # 按分析师分组显示
                    analyst_groups = {}
                    for reasoning in dp.analyst_reasonings:
                        if reasoning.agent_name not in analyst_groups:
                            analyst_groups[reasoning.agent_name] = []
                        analyst_groups[reasoning.agent_name].append(reasoning)

                    for agent_name, reasonings in analyst_groups.items():
                        f.write(f"#### {agent_name}\n\n")

                        # 分析师当日汇总
                        agent_signals = [r.signal for r in reasonings]
                        agent_avg_conf = sum(r.confidence for r in reasonings) / len(reasonings)
                        f.write(f"**当日信号分布**: 看涨{agent_signals.count('bullish')} | "
                               f"看跌{agent_signals.count('bearish')} | 中性{agent_signals.count('neutral')}\n")
                        f.write(f"**平均信心度**: {agent_avg_conf:.1f}%\n\n")

                        for reasoning in reasonings:
                            f.write(f"**{reasoning.ticker}** - {reasoning.signal.upper()} ({reasoning.confidence}%)\n")
                            f.write(f"> {reasoning.reasoning}\n\n")

                    f.write("---\n\n")

                # 关键洞察和建议
                f.write("## 💡 关键洞察和建议\n\n")
                insights = self._generate_insights()
                for insight in insights:
                    f.write(f"- {insight}\n")

                f.write("\n---\n")
                f.write(f"*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

            return True
        except Exception as e:
            print(f"导出增强版分析报告失败: {e}")
            return False



    def generate_summary_statistics(self) -> Dict[str, Any]:
        """生成汇总统计信息"""
        if not self.decision_points:
            return {"error": "没有决策点数据"}

        total_reasonings = sum(len(dp.analyst_reasonings) for dp in self.decision_points)

        # 统计各分析师的参与次数
        analyst_counts = {}
        signal_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        confidence_scores = []

        for dp in self.decision_points:
            for reasoning in dp.analyst_reasonings:
                analyst_counts[reasoning.agent_name] = analyst_counts.get(reasoning.agent_name, 0) + 1
                signal_counts[reasoning.signal] = signal_counts.get(reasoning.signal, 0) + 1
                confidence_scores.append(reasoning.confidence)

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            "总决策点数": len(self.decision_points),
            "总分析次数": total_reasonings,
            "平均信心度": round(avg_confidence, 2),
            "信号分布": signal_counts,
            "分析师参与统计": analyst_counts,
            "时间范围": {
                "开始": self.decision_points[0].date if self.decision_points else None,
                "结束": self.decision_points[-1].date if self.decision_points else None
            }
        }

    def _calculate_analyst_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """计算各分析师的表现统计"""
        analyst_stats = {}

        for dp in self.decision_points:
            for reasoning in dp.analyst_reasonings:
                agent_name = reasoning.agent_name
                if agent_name not in analyst_stats:
                    analyst_stats[agent_name] = {
                        'total': 0, 'bullish': 0, 'bearish': 0, 'neutral': 0,
                        'confidences': []
                    }

                analyst_stats[agent_name]['total'] += 1
                analyst_stats[agent_name][reasoning.signal] += 1
                analyst_stats[agent_name]['confidences'].append(reasoning.confidence)

        # 计算百分比和统计值
        for agent_name, data in analyst_stats.items():
            total = data['total']
            if total > 0:
                data['bullish_pct'] = (data['bullish'] / total) * 100
                data['bearish_pct'] = (data['bearish'] / total) * 100
                data['neutral_pct'] = (data['neutral'] / total) * 100
                data['avg_confidence'] = sum(data['confidences']) / len(data['confidences'])
                data['min_confidence'] = min(data['confidences'])
                data['max_confidence'] = max(data['confidences'])
            else:
                data.update({'bullish_pct': 0, 'bearish_pct': 0, 'neutral_pct': 0,
                           'avg_confidence': 0, 'min_confidence': 0, 'max_confidence': 0})

        return analyst_stats

    def _calculate_ticker_stats(self) -> Dict[str, Dict[str, Any]]:
        """计算各股票的分析统计"""
        ticker_stats = {}

        for dp in self.decision_points:
            for reasoning in dp.analyst_reasonings:
                ticker = reasoning.ticker
                if ticker not in ticker_stats:
                    ticker_stats[ticker] = {
                        'total': 0, 'bullish': 0, 'bearish': 0, 'neutral': 0,
                        'confidences': [], 'signals_by_date': {}
                    }

                ticker_stats[ticker]['total'] += 1
                ticker_stats[ticker][reasoning.signal] += 1
                ticker_stats[ticker]['confidences'].append(reasoning.confidence)

                # 记录每日信号用于计算一致性
                date = dp.date
                if date not in ticker_stats[ticker]['signals_by_date']:
                    ticker_stats[ticker]['signals_by_date'][date] = []
                ticker_stats[ticker]['signals_by_date'][date].append(reasoning.signal)

        # 计算统计值
        for ticker, data in ticker_stats.items():
            if data['confidences']:
                data['avg_confidence'] = sum(data['confidences']) / len(data['confidences'])
            else:
                data['avg_confidence'] = 0

            # 计算分析师一致性（每日最多信号的比例）
            consensus_count = 0
            total_days = len(data['signals_by_date'])

            for date_signals in data['signals_by_date'].values():
                signal_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
                for signal in date_signals:
                    signal_counts[signal] += 1

                max_count = max(signal_counts.values())
                total_signals = len(date_signals)
                if total_signals > 0:
                    consensus_count += max_count / total_signals

            data['consensus_rate'] = (consensus_count / total_days * 100) if total_days > 0 else 0

        return ticker_stats

    def _calculate_time_trends(self) -> Dict[str, Dict[str, Any]]:
        """计算时间趋势统计"""
        time_trends = {}

        for dp in self.decision_points:
            date = dp.date
            if date not in time_trends:
                time_trends[date] = {
                    'bullish': 0, 'bearish': 0, 'neutral': 0,
                    'confidences': []
                }

            for reasoning in dp.analyst_reasonings:
                time_trends[date][reasoning.signal] += 1
                time_trends[date]['confidences'].append(reasoning.confidence)

        # 计算平均信心度
        for date, data in time_trends.items():
            if data['confidences']:
                data['avg_confidence'] = sum(data['confidences']) / len(data['confidences'])
            else:
                data['avg_confidence'] = 0

        return time_trends

    def _determine_market_sentiment(self, bullish: int, bearish: int, neutral: int) -> str:
        """确定市场情绪"""
        total = bullish + bearish + neutral
        if total == 0:
            return "无数据"

        bullish_pct = bullish / total
        bearish_pct = bearish / total

        if bullish_pct >= 0.6:
            return "强烈看涨"
        elif bullish_pct >= 0.4:
            return "偏向看涨"
        elif bearish_pct >= 0.6:
            return "强烈看跌"
        elif bearish_pct >= 0.4:
            return "偏向看跌"
        else:
            return "中性分歧"

    def _calculate_portfolio_value_for_date(self, dp: DecisionPoint) -> float:
        """计算特定日期的投资组合总价值"""
        cash = dp.portfolio_state.get("cash", 0)
        positions = dp.portfolio_state.get("positions", {})
        market_data = dp.market_data or {}

        total_value = cash
        for ticker, position in positions.items():
            if isinstance(position, dict) and ticker in market_data:
                long_shares = position.get("long", 0)
                short_shares = position.get("short", 0)
                current_price = market_data[ticker]

                # 多头仓位价值
                long_value = long_shares * current_price

                # 空头仓位：减去当前需要归还的股票价值（负债）
                short_liability = 0
                if short_shares > 0:
                    # 当前做空负债 = 需要归还的股票数量 × 当前价格
                    short_liability = short_shares * current_price

                total_value += long_value - short_liability

        return total_value

    def _calculate_daily_consensus(self, dp: DecisionPoint) -> str:
        """计算当日分析师共识"""
        signal_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}

        for reasoning in dp.analyst_reasonings:
            signal_counts[reasoning.signal] += 1

        total = sum(signal_counts.values())
        if total == 0:
            return "无共识数据"

        max_signal = max(signal_counts, key=signal_counts.get)
        max_count = signal_counts[max_signal]
        consensus_pct = (max_count / total) * 100

        signal_names = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}

        if consensus_pct >= 75:
            return f"强烈{signal_names[max_signal]} ({consensus_pct:.0f}%)"
        elif consensus_pct >= 60:
            return f"偏向{signal_names[max_signal]} ({consensus_pct:.0f}%)"
        else:
            return f"分歧较大，略偏{signal_names[max_signal]} ({consensus_pct:.0f}%)"

    def _generate_insights(self) -> List[str]:
        """生成关键洞察和建议"""
        insights = []

        if not self.decision_points:
            return ["无足够数据生成洞察"]

        # 分析师表现洞察
        analyst_stats = self._calculate_analyst_performance_stats()
        if analyst_stats:
            # 找出最活跃的分析师
            most_active = max(analyst_stats.items(), key=lambda x: x[1]['total'])
            insights.append(f"**最活跃分析师**: {most_active[0]}，共进行了{most_active[1]['total']}次分析")

            # 找出信心度最高的分析师
            highest_confidence = max(analyst_stats.items(), key=lambda x: x[1]['avg_confidence'])
            insights.append(f"**最自信分析师**: {highest_confidence[0]}，平均信心度{highest_confidence[1]['avg_confidence']:.1f}%")

            # 找出最看涨/看跌的分析师
            most_bullish = max(analyst_stats.items(), key=lambda x: x[1]['bullish_pct'])
            most_bearish = max(analyst_stats.items(), key=lambda x: x[1]['bearish_pct'])
            insights.append(f"**最乐观分析师**: {most_bullish[0]}，看涨比例{most_bullish[1]['bullish_pct']:.1f}%")
            insights.append(f"**最悲观分析师**: {most_bearish[0]}，看跌比例{most_bearish[1]['bearish_pct']:.1f}%")

        # 股票分析洞察
        ticker_stats = self._calculate_ticker_stats()
        if ticker_stats:
            # 找出分析师最一致的股票
            most_consensus = max(ticker_stats.items(), key=lambda x: x[1]['consensus_rate'])
            insights.append(f"**分析师最一致股票**: {most_consensus[0]}，一致性{most_consensus[1]['consensus_rate']:.1f}%")

            # 找出最受关注的股票
            most_analyzed = max(ticker_stats.items(), key=lambda x: x[1]['total'])
            insights.append(f"**最受关注股票**: {most_analyzed[0]}，共被分析{most_analyzed[1]['total']}次")

        # 时间趋势洞察
        time_trends = self._calculate_time_trends()
        if len(time_trends) >= 2:
            dates = sorted(time_trends.keys())
            first_day = time_trends[dates[0]]
            last_day = time_trends[dates[-1]]

            # 比较首末日的情绪变化
            first_sentiment = self._determine_market_sentiment(
                first_day['bullish'], first_day['bearish'], first_day['neutral']
            )
            last_sentiment = self._determine_market_sentiment(
                last_day['bullish'], last_day['bearish'], last_day['neutral']
            )

            if first_sentiment != last_sentiment:
                insights.append(f"**情绪变化**: 从{first_sentiment}转向{last_sentiment}")

            # 信心度变化
            conf_change = last_day['avg_confidence'] - first_day['avg_confidence']
            if abs(conf_change) > 5:
                direction = "提升" if conf_change > 0 else "下降"
                insights.append(f"**信心度{direction}**: 从{first_day['avg_confidence']:.1f}%变化到{last_day['avg_confidence']:.1f}%")

        # 整体统计洞察
        stats = self.generate_summary_statistics()
        total_signals = stats.get('总分析次数', 0)
        if total_signals > 0:
            signal_dist = stats.get('信号分布', {})
            dominant_signal = max(signal_dist, key=signal_dist.get)
            dominant_pct = (signal_dist[dominant_signal] / total_signals) * 100

            signal_names = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}
            insights.append(f"**主导信号**: {signal_names[dominant_signal]}占{dominant_pct:.1f}%")

            avg_confidence = stats.get('平均信心度', 0)
            if avg_confidence >= 80:
                insights.append("**高信心度**: 分析师整体信心度较高，建议重点关注")
            elif avg_confidence <= 60:
                insights.append("**低信心度**: 分析师整体信心度较低，建议谨慎决策")

        # 风险提示
        if len(set(dp.date for dp in self.decision_points)) < 3:
            insights.append("⚠️ **数据量较少**: 建议增加更多交易日的数据以获得更可靠的分析")

        return insights
