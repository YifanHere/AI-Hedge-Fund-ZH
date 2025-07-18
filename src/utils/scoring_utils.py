"""
评分标准化工具模块
用于统一不同智能体的评分标准和权重计算
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ScoreNormalizer:
    """评分标准化器，确保所有智能体的评分在统一的范围内"""
    
    def __init__(self, target_range: tuple = (0, 10)):
        """
        初始化评分标准化器
        
        Args:
            target_range: 目标评分范围，默认为(0, 10)
        """
        self.target_min, self.target_max = target_range
    
    def normalize_score(self, score: float, source_range: tuple, 
                       clamp: bool = True) -> float:
        """
        将评分从源范围标准化到目标范围
        
        Args:
            score: 原始评分
            source_range: 源评分范围 (min, max)
            clamp: 是否将结果限制在目标范围内
            
        Returns:
            标准化后的评分
        """
        if score is None:
            return 0.0
        
        source_min, source_max = source_range
        
        # 避免除零错误
        if source_max == source_min:
            return self.target_min
        
        # 线性标准化
        normalized = (score - source_min) / (source_max - source_min)
        normalized = normalized * (self.target_max - self.target_min) + self.target_min
        
        # 可选的范围限制
        if clamp:
            normalized = max(self.target_min, min(self.target_max, normalized))
        
        return normalized
    
    def normalize_agent_scores(self, agent_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        标准化多个智能体的评分
        
        Args:
            agent_scores: 智能体评分字典，格式为 {agent_id: {ticker: {score, confidence, ...}}}
            
        Returns:
            标准化后的评分字典
        """
        # 定义各个智能体的评分范围
        agent_score_ranges = {
            'stanley_druckenmiller_agent': (0, 10),
            'charlie_munger_agent': (0, 10),
            'peter_lynch_agent': (0, 10),
            'warren_buffett_agent': (0, 10),
            'technical_analyst_agent': (0, 100),  # 技术分析使用百分比
            'fundamentals_analyst_agent': (0, 100),  # 基本面分析使用百分比
            'sentiment_analyst_agent': (0, 10),
            'cathie_wood_agent': (0, 5),  # Cathie Wood使用0-5范围
            'bill_ackman_agent': (0, 20),  # Bill Ackman使用0-20范围
            'phil_fisher_agent': (0, 10),
            'michael_burry_agent': (0, 10),
            'aswath_damodaran_agent': (0, 10),
            'ben_graham_agent': (0, 10),
            'rakesh_jhunjhunwala_agent': (0, 24),  # 特殊的评分范围
            'valuation_agent': (0, 10),
        }
        
        normalized_scores = {}
        
        for agent_id, ticker_scores in agent_scores.items():
            if agent_id not in agent_score_ranges:
                logger.warning(f"Unknown agent {agent_id}, using default range (0, 10)")
                source_range = (0, 10)
            else:
                source_range = agent_score_ranges[agent_id]
            
            normalized_scores[agent_id] = {}
            
            for ticker, score_data in ticker_scores.items():
                if isinstance(score_data, dict):
                    normalized_data = score_data.copy()
                    
                    # 标准化主要评分
                    if 'score' in score_data:
                        original_score = score_data['score']
                        normalized_score = self.normalize_score(original_score, source_range)
                        normalized_data['score'] = normalized_score
                        normalized_data['original_score'] = original_score
                        normalized_data['source_range'] = source_range
                    
                    # 确保置信度在0-100范围内
                    if 'confidence' in score_data:
                        confidence = score_data['confidence']
                        if confidence > 1:  # 假设已经是百分比
                            normalized_data['confidence'] = max(0, min(100, confidence))
                        else:  # 假设是0-1范围
                            normalized_data['confidence'] = max(0, min(100, confidence * 100))
                    
                    normalized_scores[agent_id][ticker] = normalized_data
                else:
                    # 如果不是字典格式，直接复制
                    normalized_scores[agent_id][ticker] = score_data
        
        return normalized_scores


class WeightedScoreAggregator:
    """加权评分聚合器，用于组合多个智能体的评分"""
    
    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        """
        初始化加权评分聚合器
        
        Args:
            default_weights: 默认权重字典
        """
        self.default_weights = default_weights or {
            'stanley_druckenmiller_agent': 0.15,
            'charlie_munger_agent': 0.15,
            'peter_lynch_agent': 0.12,
            'warren_buffett_agent': 0.15,
            'technical_analyst_agent': 0.10,
            'fundamentals_analyst_agent': 0.10,
            'sentiment_analyst_agent': 0.05,
            'cathie_wood_agent': 0.08,
            'bill_ackman_agent': 0.05,
            'phil_fisher_agent': 0.05,
        }
    
    def aggregate_scores(self, normalized_scores: Dict[str, Dict[str, Any]], 
                        custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """
        聚合多个智能体的标准化评分
        
        Args:
            normalized_scores: 标准化后的评分字典
            custom_weights: 自定义权重字典
            
        Returns:
            聚合后的评分字典，格式为 {ticker: {aggregated_score, confidence, details}}
        """
        weights = custom_weights or self.default_weights
        
        # 获取所有股票代码
        all_tickers = set()
        for agent_scores in normalized_scores.values():
            all_tickers.update(agent_scores.keys())
        
        aggregated_results = {}
        
        for ticker in all_tickers:
            weighted_score_sum = 0.0
            total_weight = 0.0
            confidence_scores = []
            contributing_agents = []
            
            for agent_id, agent_scores in normalized_scores.items():
                if ticker in agent_scores and agent_id in weights:
                    score_data = agent_scores[ticker]
                    
                    if isinstance(score_data, dict) and 'score' in score_data:
                        score = score_data['score']
                        confidence = score_data.get('confidence', 50)  # 默认50%置信度
                        weight = weights[agent_id]
                        
                        # 使用置信度调整权重
                        adjusted_weight = weight * (confidence / 100)
                        
                        weighted_score_sum += score * adjusted_weight
                        total_weight += adjusted_weight
                        confidence_scores.append(confidence)
                        contributing_agents.append(agent_id)
            
            if total_weight > 0:
                aggregated_score = weighted_score_sum / total_weight
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                
                # 生成信号
                if aggregated_score >= 7.0:
                    signal = "bullish"
                elif aggregated_score <= 3.0:
                    signal = "bearish"
                else:
                    signal = "neutral"
                
                aggregated_results[ticker] = {
                    'score': aggregated_score,
                    'confidence': avg_confidence,
                    'signal': signal,
                    'contributing_agents': contributing_agents,
                    'total_weight': total_weight,
                    'details': f"Aggregated from {len(contributing_agents)} agents with total weight {total_weight:.2f}"
                }
            else:
                # 没有有效评分的情况
                aggregated_results[ticker] = {
                    'score': 5.0,  # 中性评分
                    'confidence': 0,
                    'signal': 'neutral',
                    'contributing_agents': [],
                    'total_weight': 0,
                    'details': "No valid scores from any agent"
                }
        
        return aggregated_results


# 全局实例
_score_normalizer = ScoreNormalizer()
_score_aggregator = WeightedScoreAggregator()


def get_score_normalizer() -> ScoreNormalizer:
    """获取全局评分标准化器实例"""
    return _score_normalizer


def get_score_aggregator() -> WeightedScoreAggregator:
    """获取全局评分聚合器实例"""
    return _score_aggregator


def normalize_and_aggregate_scores(agent_scores: Dict[str, Dict[str, Any]], 
                                 custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
    """
    便捷函数：标准化并聚合智能体评分
    
    Args:
        agent_scores: 原始智能体评分
        custom_weights: 自定义权重
        
    Returns:
        聚合后的评分结果
    """
    normalizer = get_score_normalizer()
    aggregator = get_score_aggregator()
    
    # 标准化评分
    normalized_scores = normalizer.normalize_agent_scores(agent_scores)
    
    # 聚合评分
    aggregated_scores = aggregator.aggregate_scores(normalized_scores, custom_weights)
    
    return aggregated_scores
