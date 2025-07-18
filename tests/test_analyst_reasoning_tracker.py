"""
测试AI分析师思考过程追踪器功能
"""

import pytest
import tempfile
import os
from datetime import datetime
from src.utils.analyst_reasoning_tracker import AnalystReasoningTracker, AnalystReasoning, DecisionPoint


class TestAnalystReasoningTracker:
    """测试分析师思考过程追踪器"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.tracker = AnalystReasoningTracker()
        self.sample_portfolio = {
            "cash": 100000,
            "positions": {
                "AAPL": {"long": 100, "short": 0},
                "MSFT": {"long": 50, "short": 0}
            }
        }
        self.sample_market_data = {
            "AAPL": 150.0,
            "MSFT": 300.0
        }
    
    def test_start_decision_point(self):
        """测试开始决策点记录"""
        date = "2024-01-15"
        tickers = ["AAPL", "MSFT"]
        
        self.tracker.start_decision_point(
            date=date,
            tickers=tickers,
            portfolio_state=self.sample_portfolio,
            market_data=self.sample_market_data
        )
        
        assert self.tracker.current_decision_point is not None
        assert self.tracker.current_decision_point.date == date
        assert self.tracker.current_decision_point.tickers == tickers
        assert len(self.tracker.current_decision_point.analyst_reasonings) == 0
    
    def test_add_analyst_reasoning(self):
        """测试添加分析师思考记录"""
        # 先开始决策点
        self.tracker.start_decision_point(
            date="2024-01-15",
            tickers=["AAPL"],
            portfolio_state=self.sample_portfolio
        )
        
        # 添加分析师思考
        self.tracker.add_analyst_reasoning(
            agent_id="aswath_damodaran_agent",
            agent_name="阿斯沃斯·达摩达兰",
            ticker="AAPL",
            signal="bullish",
            confidence=85.5,
            reasoning="基于DCF模型分析，AAPL当前估值合理，未来增长前景良好。",
            additional_data={"dcf_value": 160.0, "target_price": 170.0}
        )
        
        assert len(self.tracker.current_decision_point.analyst_reasonings) == 1
        
        reasoning = self.tracker.current_decision_point.analyst_reasonings[0]
        assert reasoning.agent_id == "aswath_damodaran_agent"
        assert reasoning.agent_name == "阿斯沃斯·达摩达兰"
        assert reasoning.ticker == "AAPL"
        assert reasoning.signal == "bullish"
        assert reasoning.confidence == 85.5
        assert "DCF模型" in reasoning.reasoning
        assert reasoning.additional_data["dcf_value"] == 160.0
    
    def test_finish_decision_point(self):
        """测试完成决策点记录"""
        # 开始并添加数据
        self.tracker.start_decision_point(
            date="2024-01-15",
            tickers=["AAPL"],
            portfolio_state=self.sample_portfolio
        )
        
        self.tracker.add_analyst_reasoning(
            agent_id="test_agent",
            agent_name="测试分析师",
            ticker="AAPL",
            signal="neutral",
            confidence=50.0,
            reasoning="测试推理"
        )
        
        # 完成决策点
        self.tracker.finish_decision_point()
        
        assert self.tracker.current_decision_point is None
        assert len(self.tracker.decision_points) == 1
        assert len(self.tracker.decision_points[0].analyst_reasonings) == 1
    
    def test_multiple_decision_points(self):
        """测试多个决策点"""
        dates = ["2024-01-15", "2024-01-16", "2024-01-17"]
        
        for i, date in enumerate(dates):
            self.tracker.start_decision_point(
                date=date,
                tickers=["AAPL"],
                portfolio_state=self.sample_portfolio
            )
            
            self.tracker.add_analyst_reasoning(
                agent_id=f"agent_{i}",
                agent_name=f"分析师{i}",
                ticker="AAPL",
                signal="bullish" if i % 2 == 0 else "bearish",
                confidence=70.0 + i * 5,
                reasoning=f"第{i+1}天的分析"
            )
            
            self.tracker.finish_decision_point()
        
        assert len(self.tracker.decision_points) == 3
        assert self.tracker.get_decision_points_count() == 3
    
    def test_get_analyst_reasoning_by_agent(self):
        """测试按分析师获取思考记录"""
        # 创建测试数据
        self._create_sample_data()
        
        # 获取特定分析师的记录
        reasonings = self.tracker.get_analyst_reasoning_by_agent("aswath_damodaran_agent")
        
        assert len(reasonings) == 2  # 两天的记录
        for reasoning in reasonings:
            assert reasoning.agent_id == "aswath_damodaran_agent"
    
    def test_get_analyst_reasoning_by_ticker(self):
        """测试按股票获取思考记录"""
        self._create_sample_data()
        
        # 获取AAPL的所有分析
        reasonings = self.tracker.get_analyst_reasoning_by_ticker("AAPL")
        
        assert len(reasonings) == 4  # 两个分析师，两天
        for reasoning in reasonings:
            assert reasoning.ticker == "AAPL"
    

    
    def test_export_enhanced_analysis_report(self):
        """测试导出增强版详细分析报告"""
        self._create_sample_data()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name

        try:
            # 导出增强版Markdown报告
            success = self.tracker.export_enhanced_analysis_report(temp_path)
            assert success

            # 验证文件内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert '# AI分析师思考过程详细分析报告' in content
            assert '📊 整体统计概览' in content
            assert '👥 分析师表现分析' in content
            assert '阿斯沃斯·达摩达兰' in content
            assert '查理·芒格' in content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_generate_summary_statistics(self):
        """测试生成汇总统计"""
        self._create_sample_data()
        
        stats = self.tracker.generate_summary_statistics()
        
        assert '总决策点数' in stats
        assert '总分析次数' in stats
        assert '平均信心度' in stats
        assert '信号分布' in stats
        assert '分析师参与统计' in stats
        
        assert stats['总决策点数'] == 2
        assert stats['总分析次数'] == 4
        assert 'bullish' in stats['信号分布']
        assert 'bearish' in stats['信号分布']
    
    def test_clear(self):
        """测试清空功能"""
        self._create_sample_data()
        
        assert len(self.tracker.decision_points) > 0
        
        self.tracker.clear()
        
        assert len(self.tracker.decision_points) == 0
        assert self.tracker.current_decision_point is None
    

    
    def _create_sample_data(self):
        """创建示例数据"""
        dates = ["2024-01-15", "2024-01-16"]
        agents = [
            ("aswath_damodaran_agent", "阿斯沃斯·达摩达兰"),
            ("charlie_munger_agent", "查理·芒格")
        ]
        
        for date in dates:
            self.tracker.start_decision_point(
                date=date,
                tickers=["AAPL"],
                portfolio_state=self.sample_portfolio,
                market_data=self.sample_market_data
            )
            
            for agent_id, agent_name in agents:
                self.tracker.add_analyst_reasoning(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    ticker="AAPL",
                    signal="bullish" if "damodaran" in agent_id else "bearish",
                    confidence=80.0 if "damodaran" in agent_id else 75.0,
                    reasoning=f"{agent_name}的详细分析 - {date}"
                )
            
            self.tracker.finish_decision_point()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
