"""
测试修复后的代码功能
验证技术指标计算、财务指标计算和数据验证的正确性
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.technicals import calculate_rsi, calculate_hurst_exponent, safe_float
from agents.peter_lynch import analyze_lynch_valuation
from utils.data_validation import DataValidator
from utils.scoring_utils import ScoreNormalizer, normalize_and_aggregate_scores
from data.models import FinancialMetrics, Price


class TestTechnicalIndicatorFixes:
    """测试技术指标修复"""
    
    def test_rsi_calculation_fix(self):
        """测试RSI计算修复 - 应该使用指数移动平均"""
        # 创建测试数据
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        df = pd.DataFrame({'close': prices})
        
        # 计算RSI
        rsi = calculate_rsi(df, period=14)
        
        # 验证RSI值在合理范围内
        assert not rsi.isna().all(), "RSI should not be all NaN"
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi), "RSI values should be between 0 and 100"
        
        # 验证RSI对价格上涨的响应
        assert valid_rsi.iloc[-1] > 50, "RSI should be above 50 for upward trending prices"
    
    def test_hurst_exponent_fix(self):
        """测试Hurst指数计算修复"""
        # 创建趋势性数据
        trending_prices = pd.Series([100 + i * 0.5 + np.random.normal(0, 0.1) for i in range(100)])
        
        # 计算Hurst指数
        hurst = calculate_hurst_exponent(trending_prices)
        
        # 验证Hurst指数在合理范围内
        assert 0 <= hurst <= 1, f"Hurst exponent should be between 0 and 1, got {hurst}"
        
        # 对于趋势性数据，Hurst指数应该大于0.5
        assert hurst > 0.4, f"Hurst exponent for trending data should be > 0.4, got {hurst}"
    
    def test_safe_float_improvements(self):
        """测试safe_float函数的改进"""
        # 测试各种输入
        assert safe_float(None) == 0.0
        assert safe_float(np.nan) == 0.0
        assert safe_float(np.inf) == 0.0
        assert safe_float(-np.inf) == 0.0
        assert safe_float("invalid") == 0.0
        assert safe_float(42.5) == 42.5
        
        # 测试pandas Series
        series = pd.Series([1, 2, 3])
        assert safe_float(series) == 3.0  # Should get the last value
        
        empty_series = pd.Series([])
        assert safe_float(empty_series) == 0.0


class TestFinancialIndicatorFixes:
    """测试财务指标修复"""
    
    def test_peg_ratio_calculation(self):
        """测试PEG比率计算修复"""
        # 模拟财务数据
        financial_line_items = [
            type('MockLineItem', (), {
                'net_income': 1000000,
                'earnings_per_share': 2.0
            })(),
            type('MockLineItem', (), {
                'net_income': 800000,
                'earnings_per_share': 1.6
            })(),
            type('MockLineItem', (), {
                'net_income': 600000,
                'earnings_per_share': 1.2
            })()
        ]
        
        market_cap = 50000000  # 50M market cap
        
        # 测试PEG计算
        result = analyze_lynch_valuation(financial_line_items, market_cap)
        
        assert 'score' in result
        assert 'details' in result
        assert isinstance(result['score'], (int, float))
        assert 0 <= result['score'] <= 10


class TestDataValidation:
    """测试数据验证功能"""
    
    def test_financial_metrics_validation(self):
        """测试财务指标验证"""
        validator = DataValidator()
        
        # 创建有效的财务指标
        valid_metrics = [
            FinancialMetrics(
                ticker="AAPL",
                report_period="2023-12-31",
                period="annual",
                currency="USD",
                market_cap=3000000000000,  # 3T
                price_to_earnings_ratio=25.0,
                return_on_equity=0.25
            )
        ]
        
        result = validator.validate_financial_metrics(valid_metrics)
        assert result['is_valid'] == True
        assert len(result['issues']) == 0
        
        # 创建无效的财务指标
        invalid_metrics = [
            FinancialMetrics(
                ticker="",  # 空ticker
                report_period="2023-12-31",
                period="annual",
                currency="USD",
                market_cap=-1000000,  # 负市值
                price_to_earnings_ratio=2000.0,  # 极高P/E
                return_on_equity=10.0  # 1000% ROE
            )
        ]
        
        result = validator.validate_financial_metrics(invalid_metrics)
        assert result['is_valid'] == False
        assert len(result['issues']) > 0
    
    def test_price_data_validation(self):
        """测试价格数据验证"""
        validator = DataValidator()
        
        # 创建有效的价格数据
        valid_prices = [
            Price(
                ticker="AAPL",
                time="2023-12-01",
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000
            )
        ]
        
        result = validator.validate_price_data(valid_prices)
        assert result['is_valid'] == True
        
        # 创建无效的价格数据
        invalid_prices = [
            Price(
                ticker="AAPL",
                time="2023-12-01",
                open=150.0,
                high=140.0,  # High < Open (invalid)
                low=160.0,   # Low > Open (invalid)
                close=-10.0, # Negative close price
                volume=-1000 # Negative volume
            )
        ]
        
        result = validator.validate_price_data(invalid_prices)
        assert result['is_valid'] == False
        assert len(result['issues']) > 0


class TestScoreNormalization:
    """测试评分标准化功能"""
    
    def test_score_normalization(self):
        """测试评分标准化"""
        normalizer = ScoreNormalizer()
        
        # 测试不同范围的评分标准化
        assert normalizer.normalize_score(5, (0, 10)) == 5.0  # 已经在目标范围内
        assert normalizer.normalize_score(50, (0, 100)) == 5.0  # 从0-100标准化到0-10
        assert normalizer.normalize_score(2.5, (0, 5)) == 5.0  # 从0-5标准化到0-10
    
    def test_agent_score_aggregation(self):
        """测试智能体评分聚合"""
        # 模拟多个智能体的评分
        agent_scores = {
            'stanley_druckenmiller_agent': {
                'AAPL': {'score': 8.0, 'confidence': 85, 'signal': 'bullish'}
            },
            'charlie_munger_agent': {
                'AAPL': {'score': 7.5, 'confidence': 90, 'signal': 'bullish'}
            },
            'cathie_wood_agent': {
                'AAPL': {'score': 4.0, 'confidence': 75, 'signal': 'bullish'}  # 0-5范围
            }
        }
        
        # 聚合评分
        aggregated = normalize_and_aggregate_scores(agent_scores)
        
        assert 'AAPL' in aggregated
        assert 'score' in aggregated['AAPL']
        assert 'confidence' in aggregated['AAPL']
        assert 'signal' in aggregated['AAPL']
        
        # 验证聚合后的评分在合理范围内
        assert 0 <= aggregated['AAPL']['score'] <= 10
        assert 0 <= aggregated['AAPL']['confidence'] <= 100


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
