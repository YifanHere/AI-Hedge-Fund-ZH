"""
数据验证工具模块
用于验证从API获取的财务数据的质量和合理性
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.data.models import FinancialMetrics, Price, LineItem
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器类，用于检查财务数据的质量和合理性"""
    
    def __init__(self):
        # 定义合理的数值范围
        self.reasonable_ranges = {
            'market_cap': (1e6, 1e13),  # 100万到10万亿
            'price_to_earnings_ratio': (0, 1000),  # P/E比率
            'price_to_book_ratio': (0, 100),  # P/B比率
            'price_to_sales_ratio': (0, 100),  # P/S比率
            'return_on_equity': (-1, 5),  # ROE: -100% to 500%
            'return_on_assets': (-1, 1),  # ROA: -100% to 100%
            'debt_to_equity': (0, 50),  # 债务权益比
            'current_ratio': (0, 100),  # 流动比率
            'gross_margin': (-1, 1),  # 毛利率
            'operating_margin': (-1, 1),  # 营业利润率
            'net_margin': (-1, 1),  # 净利率
        }
    
    def validate_financial_metrics(self, metrics: List[FinancialMetrics]) -> Dict[str, Any]:
        """
        验证财务指标数据
        
        Args:
            metrics: 财务指标列表
            
        Returns:
            验证结果字典，包含是否通过验证和具体问题
        """
        if not metrics:
            return {
                'is_valid': False,
                'issues': ['No financial metrics data provided'],
                'warnings': []
            }
        
        issues = []
        warnings = []
        
        for i, metric in enumerate(metrics):
            # 检查基本数据完整性
            if not metric.ticker:
                issues.append(f"Metric {i}: Missing ticker")
            
            if not metric.report_period:
                issues.append(f"Metric {i}: Missing report period")
            
            # 检查数值合理性
            for field_name, (min_val, max_val) in self.reasonable_ranges.items():
                value = getattr(metric, field_name, None)
                if value is not None:
                    if not (min_val <= value <= max_val):
                        warnings.append(
                            f"Metric {i}: {field_name} = {value} is outside reasonable range "
                            f"[{min_val}, {max_val}]"
                        )
            
            # 检查特殊的逻辑关系
            if metric.market_cap is not None and metric.market_cap <= 0:
                issues.append(f"Metric {i}: Market cap must be positive, got {metric.market_cap}")
            
            # 检查比率的一致性
            if (metric.price_to_earnings_ratio is not None and 
                metric.price_to_earnings_ratio < 0):
                warnings.append(f"Metric {i}: Negative P/E ratio: {metric.price_to_earnings_ratio}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def validate_price_data(self, prices: List[Price]) -> Dict[str, Any]:
        """
        验证价格数据
        
        Args:
            prices: 价格数据列表
            
        Returns:
            验证结果字典
        """
        if not prices:
            return {
                'is_valid': False,
                'issues': ['No price data provided'],
                'warnings': []
            }
        
        issues = []
        warnings = []
        
        # 按时间排序检查
        sorted_prices = sorted(prices, key=lambda p: p.time)
        
        for i, price in enumerate(sorted_prices):
            # 检查基本字段
            if price.close is None or price.close <= 0:
                issues.append(f"Price {i}: Invalid close price: {price.close}")
            
            if price.volume is not None and price.volume < 0:
                warnings.append(f"Price {i}: Negative volume: {price.volume}")
            
            # 检查OHLC关系
            if all(x is not None for x in [price.open, price.high, price.low, price.close]):
                if not (price.low <= price.open <= price.high and 
                       price.low <= price.close <= price.high):
                    issues.append(f"Price {i}: Invalid OHLC relationship")
            
            # 检查异常的价格变动
            if i > 0:
                prev_price = sorted_prices[i-1]
                if prev_price.close and price.close:
                    change = abs(price.close - prev_price.close) / prev_price.close
                    if change > 0.5:  # 50%的单日变动
                        warnings.append(
                            f"Price {i}: Large price change: {change:.1%} from previous day"
                        )
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def validate_line_items(self, line_items: List[LineItem]) -> Dict[str, Any]:
        """
        验证财务报表行项目数据
        
        Args:
            line_items: 财务报表行项目列表
            
        Returns:
            验证结果字典
        """
        if not line_items:
            return {
                'is_valid': False,
                'issues': ['No line items data provided'],
                'warnings': []
            }
        
        issues = []
        warnings = []
        
        for i, item in enumerate(line_items):
            # 检查基本字段
            if not item.ticker:
                issues.append(f"Line item {i}: Missing ticker")
            
            if not item.report_period:
                issues.append(f"Line item {i}: Missing report period")
            
            # 检查财务数据的逻辑关系
            if item.revenue is not None and item.revenue < 0:
                warnings.append(f"Line item {i}: Negative revenue: {item.revenue}")
            
            if (item.total_assets is not None and item.total_liabilities is not None and
                item.shareholders_equity is not None):
                # 资产 = 负债 + 股东权益 (会计恒等式)
                calculated_equity = item.total_assets - item.total_liabilities
                if abs(calculated_equity - item.shareholders_equity) / max(abs(item.shareholders_equity), 1) > 0.1:
                    warnings.append(
                        f"Line item {i}: Accounting equation imbalance. "
                        f"Assets - Liabilities = {calculated_equity}, "
                        f"but Shareholders Equity = {item.shareholders_equity}"
                    )
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def log_validation_results(self, validation_result: Dict[str, Any], data_type: str):
        """记录验证结果到日志"""
        if not validation_result['is_valid']:
            logger.error(f"{data_type} validation failed:")
            for issue in validation_result['issues']:
                logger.error(f"  - {issue}")
        
        if validation_result['warnings']:
            logger.warning(f"{data_type} validation warnings:")
            for warning in validation_result['warnings']:
                logger.warning(f"  - {warning}")


# 全局验证器实例
_validator = DataValidator()


def get_validator() -> DataValidator:
    """获取全局数据验证器实例"""
    return _validator


def validate_and_log(data, data_type: str) -> bool:
    """
    验证数据并记录结果的便捷函数
    
    Args:
        data: 要验证的数据
        data_type: 数据类型 ('financial_metrics', 'prices', 'line_items')
        
    Returns:
        是否通过验证
    """
    validator = get_validator()
    
    if data_type == 'financial_metrics':
        result = validator.validate_financial_metrics(data)
    elif data_type == 'prices':
        result = validator.validate_price_data(data)
    elif data_type == 'line_items':
        result = validator.validate_line_items(data)
    else:
        logger.error(f"Unknown data type for validation: {data_type}")
        return False
    
    validator.log_validation_results(result, data_type)
    return result['is_valid']
