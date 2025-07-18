from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, ForeignKey, Float, Date, Index, UniqueConstraint
from sqlalchemy.sql import func
from .connection import Base


class HedgeFundFlow(Base):
    """Table to store React Flow configurations (nodes, edges, viewport)"""
    __tablename__ = "hedge_fund_flows"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Flow metadata
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # React Flow state
    nodes = Column(JSON, nullable=False)  # Store React Flow nodes as JSON
    edges = Column(JSON, nullable=False)  # Store React Flow edges as JSON
    viewport = Column(JSON, nullable=True)  # Store viewport state (zoom, x, y)
    data = Column(JSON, nullable=True)  # Store node internal states (tickers, models, etc.)
    
    # Additional metadata
    is_template = Column(Boolean, default=False)  # Mark as template for reuse
    tags = Column(JSON, nullable=True)  # Store tags for categorization


class HedgeFundFlowRun(Base):
    """Table to track individual execution runs of a hedge fund flow"""
    __tablename__ = "hedge_fund_flow_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    flow_id = Column(Integer, ForeignKey("hedge_fund_flows.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Run execution tracking
    status = Column(String(50), nullable=False, default="IDLE")  # IDLE, IN_PROGRESS, COMPLETE, ERROR
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Run configuration
    trading_mode = Column(String(50), nullable=False, default="one-time")  # one-time, continuous, advisory
    schedule = Column(String(50), nullable=True)  # hourly, daily, weekly (for continuous mode)
    duration = Column(String(50), nullable=True)  # 1day, 1week, 1month (for continuous mode)
    
    # Run data
    request_data = Column(JSON, nullable=True)  # Store the request parameters (tickers, agents, models, etc.)
    initial_portfolio = Column(JSON, nullable=True)  # Store initial portfolio state
    final_portfolio = Column(JSON, nullable=True)  # Store final portfolio state
    results = Column(JSON, nullable=True)  # Store the output/results from the run
    error_message = Column(Text, nullable=True)  # Store error details if run failed
    
    # Metadata
    run_number = Column(Integer, nullable=False, default=1)  # Sequential run number for this flow


class HedgeFundFlowRunCycle(Base):
    """Individual analysis cycles within a trading session"""
    __tablename__ = "hedge_fund_flow_run_cycles"
    
    id = Column(Integer, primary_key=True, index=True)
    flow_run_id = Column(Integer, ForeignKey("hedge_fund_flow_runs.id"), nullable=False, index=True)
    cycle_number = Column(Integer, nullable=False)  # 1, 2, 3, etc. within the run
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Analysis results
    analyst_signals = Column(JSON, nullable=True)  # All agent decisions/signals
    trading_decisions = Column(JSON, nullable=True)  # Portfolio manager decisions
    executed_trades = Column(JSON, nullable=True)  # Actual trades executed (paper trading)
    
    # Portfolio state after this cycle
    portfolio_snapshot = Column(JSON, nullable=True)  # Cash, positions, performance metrics
    
    # Performance metrics for this cycle
    performance_metrics = Column(JSON, nullable=True)  # Returns, sharpe ratio, etc.
    
    # Execution tracking
    status = Column(String(50), nullable=False, default="IN_PROGRESS")  # IN_PROGRESS, COMPLETED, ERROR
    error_message = Column(Text, nullable=True)  # Store error details if cycle failed
    
    # Cost tracking
    llm_calls_count = Column(Integer, nullable=True, default=0)  # Number of LLM calls made
    api_calls_count = Column(Integer, nullable=True, default=0)  # Number of financial API calls made
    estimated_cost = Column(String(20), nullable=True)  # Estimated cost in USD
    
    # Metadata
    trigger_reason = Column(String(100), nullable=True)  # scheduled, manual, market_event, etc.
    market_conditions = Column(JSON, nullable=True)  # Market data snapshot at cycle start


class AnalystReasoning(Base):
    """AI分析师思考过程详细记录表"""
    __tablename__ = "analyst_reasoning"

    id = Column(Integer, primary_key=True, index=True)

    # 关联到决策周期
    flow_run_cycle_id = Column(Integer, ForeignKey("hedge_fund_flow_run_cycles.id"), nullable=False, index=True)

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    decision_date = Column(Date, nullable=False, index=True)  # 决策日期

    # 分析师信息
    agent_id = Column(String(100), nullable=False, index=True)  # 分析师ID (如 "aswath_damodaran_agent")
    agent_name = Column(String(200), nullable=False)  # 分析师中文名称

    # 股票信息
    ticker = Column(String(20), nullable=False, index=True)  # 股票代码

    # 分析结果
    signal = Column(String(20), nullable=False)  # bullish, bearish, neutral
    confidence = Column(Float, nullable=False)  # 信心度 0-100
    reasoning = Column(Text, nullable=False)  # 详细思考过程

    # 市场数据快照
    stock_price = Column(Float, nullable=True)  # 当时的股票价格
    market_data = Column(JSON, nullable=True)  # 其他市场数据

    # 额外分析数据
    additional_data = Column(JSON, nullable=True)  # 分析师特定的额外数据

    # 投资组合状态快照
    portfolio_cash = Column(Float, nullable=True)  # 当时的现金余额
    portfolio_positions = Column(JSON, nullable=True)  # 当时的持仓情况

    # 索引
    __table_args__ = (
        Index('idx_analyst_reasoning_date_agent', 'decision_date', 'agent_id'),
        Index('idx_analyst_reasoning_ticker_date', 'ticker', 'decision_date'),
        Index('idx_analyst_reasoning_signal', 'signal'),
    )


class AnalystPerformanceMetrics(Base):
    """分析师表现指标表"""
    __tablename__ = "analyst_performance_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # 分析师信息
    agent_id = Column(String(100), nullable=False, index=True)
    agent_name = Column(String(200), nullable=False)

    # 时间范围
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 基础统计
    total_predictions = Column(Integer, nullable=False, default=0)
    bullish_predictions = Column(Integer, nullable=False, default=0)
    bearish_predictions = Column(Integer, nullable=False, default=0)
    neutral_predictions = Column(Integer, nullable=False, default=0)

    # 准确性指标
    correct_predictions = Column(Integer, nullable=False, default=0)
    accuracy_rate = Column(Float, nullable=True)  # 准确率

    # 信心度统计
    avg_confidence = Column(Float, nullable=True)
    confidence_accuracy_correlation = Column(Float, nullable=True)  # 信心度与准确性的相关性

    # 收益指标
    total_return = Column(Float, nullable=True)  # 如果按照该分析师建议的总收益
    sharpe_ratio = Column(Float, nullable=True)  # 夏普比率

    # 详细统计数据
    detailed_stats = Column(JSON, nullable=True)  # 更详细的统计信息

    # 索引
    __table_args__ = (
        Index('idx_analyst_performance_agent_date', 'agent_id', 'start_date', 'end_date'),
        UniqueConstraint('agent_id', 'start_date', 'end_date', name='uq_analyst_performance_period'),
    )


