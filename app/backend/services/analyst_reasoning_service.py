"""
AI分析师思考过程数据库服务
用于存储、查询和分析AI分析师的思考过程数据
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from ..database.models import AnalystReasoning, AnalystPerformanceMetrics, HedgeFundFlowRunCycle
from ..database.connection import get_db


class AnalystReasoningService:
    """AI分析师思考过程服务"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_reasoning_batch(self, cycle_id: int, reasoning_data: List[Dict[str, Any]]) -> bool:
        """批量保存分析师思考过程"""
        try:
            reasoning_records = []
            for data in reasoning_data:
                record = AnalystReasoning(
                    flow_run_cycle_id=cycle_id,
                    decision_date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
                    agent_id=data['agent_id'],
                    agent_name=data['agent_name'],
                    ticker=data['ticker'],
                    signal=data['signal'],
                    confidence=data['confidence'],
                    reasoning=data['reasoning'],
                    stock_price=data.get('stock_price'),
                    market_data=data.get('market_data'),
                    additional_data=data.get('additional_data'),
                    portfolio_cash=data.get('portfolio_cash'),
                    portfolio_positions=data.get('portfolio_positions')
                )
                reasoning_records.append(record)
            
            self.db.add_all(reasoning_records)
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            print(f"保存分析师思考过程失败: {e}")
            return False
    
    def get_reasoning_by_date_range(self, start_date: date, end_date: date, 
                                  agent_id: Optional[str] = None, 
                                  ticker: Optional[str] = None) -> List[AnalystReasoning]:
        """根据日期范围查询思考过程"""
        query = self.db.query(AnalystReasoning).filter(
            and_(
                AnalystReasoning.decision_date >= start_date,
                AnalystReasoning.decision_date <= end_date
            )
        )
        
        if agent_id:
            query = query.filter(AnalystReasoning.agent_id == agent_id)
        
        if ticker:
            query = query.filter(AnalystReasoning.ticker == ticker)
        
        return query.order_by(desc(AnalystReasoning.decision_date)).all()
    
    def get_reasoning_by_agent(self, agent_id: str, limit: int = 100) -> List[AnalystReasoning]:
        """获取特定分析师的思考过程"""
        return self.db.query(AnalystReasoning).filter(
            AnalystReasoning.agent_id == agent_id
        ).order_by(desc(AnalystReasoning.decision_date)).limit(limit).all()
    
    def get_reasoning_by_ticker(self, ticker: str, limit: int = 100) -> List[AnalystReasoning]:
        """获取特定股票的所有分析师思考过程"""
        return self.db.query(AnalystReasoning).filter(
            AnalystReasoning.ticker == ticker
        ).order_by(desc(AnalystReasoning.decision_date)).limit(limit).all()
    
    def get_signal_distribution(self, agent_id: Optional[str] = None, 
                              start_date: Optional[date] = None,
                              end_date: Optional[date] = None) -> Dict[str, int]:
        """获取信号分布统计"""
        query = self.db.query(
            AnalystReasoning.signal,
            func.count(AnalystReasoning.signal).label('count')
        )
        
        if agent_id:
            query = query.filter(AnalystReasoning.agent_id == agent_id)
        
        if start_date and end_date:
            query = query.filter(
                and_(
                    AnalystReasoning.decision_date >= start_date,
                    AnalystReasoning.decision_date <= end_date
                )
            )
        
        results = query.group_by(AnalystReasoning.signal).all()
        return {signal: count for signal, count in results}
    
    def get_confidence_statistics(self, agent_id: Optional[str] = None) -> Dict[str, float]:
        """获取信心度统计"""
        query = self.db.query(AnalystReasoning.confidence)
        
        if agent_id:
            query = query.filter(AnalystReasoning.agent_id == agent_id)
        
        confidences = [r.confidence for r in query.all()]
        
        if not confidences:
            return {}
        
        return {
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'total_predictions': len(confidences)
        }
    
    def get_agent_activity_summary(self, start_date: Optional[date] = None,
                                 end_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """获取分析师活动汇总"""
        query = self.db.query(
            AnalystReasoning.agent_id,
            AnalystReasoning.agent_name,
            func.count(AnalystReasoning.id).label('total_predictions'),
            func.avg(AnalystReasoning.confidence).label('avg_confidence'),
            func.count(func.distinct(AnalystReasoning.ticker)).label('unique_tickers'),
            func.count(func.distinct(AnalystReasoning.decision_date)).label('active_days')
        )
        
        if start_date and end_date:
            query = query.filter(
                and_(
                    AnalystReasoning.decision_date >= start_date,
                    AnalystReasoning.decision_date <= end_date
                )
            )
        
        results = query.group_by(
            AnalystReasoning.agent_id, 
            AnalystReasoning.agent_name
        ).all()
        
        return [
            {
                'agent_id': r.agent_id,
                'agent_name': r.agent_name,
                'total_predictions': r.total_predictions,
                'avg_confidence': round(r.avg_confidence, 2) if r.avg_confidence else 0,
                'unique_tickers': r.unique_tickers,
                'active_days': r.active_days
            }
            for r in results
        ]
    
    def search_reasoning_by_keyword(self, keyword: str, limit: int = 50) -> List[AnalystReasoning]:
        """根据关键词搜索思考过程"""
        return self.db.query(AnalystReasoning).filter(
            AnalystReasoning.reasoning.contains(keyword)
        ).order_by(desc(AnalystReasoning.decision_date)).limit(limit).all()
    
    def get_recent_reasoning(self, days: int = 7, limit: int = 100) -> List[AnalystReasoning]:
        """获取最近几天的思考过程"""
        cutoff_date = datetime.now().date() - timedelta(days=days)
        return self.db.query(AnalystReasoning).filter(
            AnalystReasoning.decision_date >= cutoff_date
        ).order_by(desc(AnalystReasoning.decision_date)).limit(limit).all()


class AnalystPerformanceService:
    """分析师表现分析服务"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_performance_metrics(self, agent_id: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """计算分析师表现指标"""
        reasonings = self.db.query(AnalystReasoning).filter(
            and_(
                AnalystReasoning.agent_id == agent_id,
                AnalystReasoning.decision_date >= start_date,
                AnalystReasoning.decision_date <= end_date
            )
        ).all()
        
        if not reasonings:
            return {}
        
        total_predictions = len(reasonings)
        signal_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        confidences = []
        
        for r in reasonings:
            signal_counts[r.signal] = signal_counts.get(r.signal, 0) + 1
            confidences.append(r.confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'agent_id': agent_id,
            'total_predictions': total_predictions,
            'signal_distribution': signal_counts,
            'avg_confidence': round(avg_confidence, 2),
            'confidence_range': {
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            },
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        }
    
    def save_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """保存分析师表现指标"""
        try:
            record = AnalystPerformanceMetrics(
                agent_id=metrics['agent_id'],
                agent_name=metrics.get('agent_name', ''),
                start_date=datetime.strptime(metrics['start_date'], '%Y-%m-%d').date(),
                end_date=datetime.strptime(metrics['end_date'], '%Y-%m-%d').date(),
                total_predictions=metrics.get('total_predictions', 0),
                bullish_predictions=metrics.get('bullish_predictions', 0),
                bearish_predictions=metrics.get('bearish_predictions', 0),
                neutral_predictions=metrics.get('neutral_predictions', 0),
                avg_confidence=metrics.get('avg_confidence'),
                detailed_stats=metrics.get('detailed_stats')
            )
            
            self.db.add(record)
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            print(f"保存分析师表现指标失败: {e}")
            return False


def get_reasoning_service() -> AnalystReasoningService:
    """获取分析师思考过程服务实例"""
    db = next(get_db())
    return AnalystReasoningService(db)


def get_performance_service() -> AnalystPerformanceService:
    """获取分析师表现服务实例"""
    db = next(get_db())
    return AnalystPerformanceService(db)
