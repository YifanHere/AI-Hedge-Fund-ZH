#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文状态消息映射
"""

def get_chinese_status_message(english_message: str) -> str:
    """
    将英文状态消息转换为中文
    
    Args:
        english_message: 英文状态消息
        
    Returns:
        对应的中文状态消息，如果没有映射则返回原消息
    """
    
    # 状态消息映射表
    status_mapping = {
        # 通用状态
        "Starting": "开始分析",
        "Done": "完成",
        "Error": "错误",
        "Processing": "处理中",
        "Analyzing": "分析中",
        "Calculating": "计算中",
        "Generating": "生成中",
        
        # 数据获取相关
        "Fetching financial data": "获取财务数据",
        "Fetching financial metrics": "获取财务指标",
        "Fetching financial metric": "获取财务指标",
        "Fetching financial line items": "获取财务项目",
        "Fetching market data": "获取市场数据",
        "Fetching price data": "获取价格数据",
        "Getting financial metrics": "获取财务指标",
        "Getting market cap": "获取市值数据",
        "Fetching market cap": "获取市值数据",
        "Searching line items": "搜索财务项目",
        "Gathering financial line items": "收集财务项目",
        "Gathering line items": "收集财务项目",
        "Fetching line items": "获取财务项目",
        "Fetching insider trades": "获取内幕交易数据",
        "Fetching company news": "获取公司新闻",
        "Fetching recent price data": "获取最新价格数据",
        "Fetching recent price data for momentum": "获取动量分析价格数据",
        "Fetching recent price data for reference": "获取参考价格数据",
        
        # 分析相关
        "Analyzing fundamentals": "分析基本面",
        "Analyzing technicals": "分析技术面",
        "Analyzing valuation": "分析估值",
        "Analyzing growth": "分析成长性",
        "Analyzing profitability": "分析盈利能力",
        "Analyzing balance sheet": "分析资产负债表",
        "Analyzing cash flow": "分析现金流",
        "Analyzing management": "分析管理层",
        "Analyzing sentiment": "分析市场情绪",
        "Analyzing momentum": "分析动量",
        "Analyzing risk": "分析风险",
        
        # 特定分析师状态
        "Generating Warren Buffett analysis": "生成沃伦·巴菲特分析",
        "Generating Ben Graham analysis": "生成本杰明·格雷厄姆分析",
        "Generating Charlie Munger analysis": "生成查理·芒格分析",
        "Generating Michael Burry analysis": "生成迈克尔·伯里分析",
        "Generating Cathie Wood analysis": "生成凯茜·伍德分析",
        "Generating Peter Lynch analysis": "生成彼得·林奇分析",
        "Generating Phil Fisher analysis": "生成菲利普·费雪分析",
        "Generating Bill Ackman analysis": "生成比尔·阿克曼分析",
        "Generating Aswath Damodaran analysis": "生成阿斯沃斯·达摩达兰分析",
        "Generating Rakesh Jhunjhunwala analysis": "生成拉凯什·朱恩朱恩瓦拉分析",
        "Generating Stanley Druckenmiller analysis": "生成斯坦利·德鲁肯米勒分析",
        "Generating Jhunjhunwala analysis": "生成朱恩朱恩瓦拉分析",
        "Generating technical analysis": "生成技术分析",
        "Generating fundamental analysis": "生成基本面分析",
        "Generating sentiment analysis": "生成情绪分析",
        "Generating valuation analysis": "生成估值分析",
        
        # LLM相关
        "Generating LLM output": "生成AI分析",
        "Calling LLM": "调用AI模型",
        "Processing LLM response": "处理AI响应",
        
        # 投资组合相关
        "Generating portfolio decisions": "生成投资组合决策",
        "Calculating portfolio metrics": "计算投资组合指标",
        "Optimizing portfolio": "优化投资组合",
        "Managing risk": "管理风险",
        
        # 计算相关
        "Calculating intrinsic value": "计算内在价值",
        "Calculating margin of safety": "计算安全边际",
        "Calculating growth metrics": "计算成长指标",
        "Calculating profitability metrics": "计算盈利指标",
        "Calculating financial ratios": "计算财务比率",
        "Calculating technical indicators": "计算技术指标",
        "Calculating volatility": "计算波动率",
        "Calculating correlation": "计算相关性",
        
        # 评估相关
        "Evaluating investment opportunity": "评估投资机会",
        "Evaluating risk-reward": "评估风险收益",
        "Evaluating market conditions": "评估市场条件",
        "Evaluating company fundamentals": "评估公司基本面",
        "Evaluating management quality": "评估管理质量",
        "Evaluating competitive position": "评估竞争地位",
        
        # 其他常见状态
        "Initializing": "初始化",
        "Loading data": "加载数据",
        "Validating data": "验证数据",
        "Cleaning data": "清理数据",
        "Preparing analysis": "准备分析",
        "Finalizing results": "完成结果",
        "Saving results": "保存结果",
    }
    
    # 尝试精确匹配
    if english_message in status_mapping:
        return status_mapping[english_message]
    
    # 尝试部分匹配（用于包含变量的消息）
    for english_key, chinese_value in status_mapping.items():
        if english_key.lower() in english_message.lower():
            return chinese_value
    
    # 如果没有找到映射，返回原消息
    return english_message

def get_chinese_analyst_name(agent_id: str) -> str:
    """
    获取分析师的中文名称
    
    Args:
        agent_id: 分析师ID
        
    Returns:
        对应的中文名称
    """
    chinese_names = {
        "aswath_damodaran_agent": "阿斯沃斯·达摩达兰",
        "ben_graham_agent": "本杰明·格雷厄姆",
        "bill_ackman_agent": "比尔·阿克曼",
        "cathie_wood_agent": "凯茜·伍德",
        "charlie_munger_agent": "查理·芒格",
        "michael_burry_agent": "迈克尔·伯里",
        "peter_lynch_agent": "彼得·林奇",
        "phil_fisher_agent": "菲利普·费雪",
        "rakesh_jhunjhunwala_agent": "拉凯什·朱恩朱恩瓦拉",
        "stanley_druckenmiller_agent": "斯坦利·德鲁肯米勒",
        "warren_buffett_agent": "沃伦·巴菲特",
        "technical_analyst_agent": "技术分析师",
        "fundamentals_analyst_agent": "基本面分析师",
        "sentiment_analyst_agent": "情绪分析师",
        "valuation_agent": "估值分析师",
        "valuation_analyst_agent": "估值分析师",
        "risk_management_agent": "风险管理师",
        "portfolio_manager_agent": "投资组合经理",
    }
    
    return chinese_names.get(agent_id, agent_id.replace("_agent", "").replace("_", " ").title())
