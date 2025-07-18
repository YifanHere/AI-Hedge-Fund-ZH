"""与分析师配置相关的常量和工具。"""

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.aswath_damodaran import aswath_damodaran_agent
from src.agents.ben_graham import ben_graham_agent
from src.agents.bill_ackman import bill_ackman_agent
from src.agents.cathie_wood import cathie_wood_agent
from src.agents.charlie_munger import charlie_munger_agent
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.michael_burry import michael_burry_agent
from src.agents.phil_fisher import phil_fisher_agent
from src.agents.peter_lynch import peter_lynch_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.stanley_druckenmiller import stanley_druckenmiller_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.warren_buffett import warren_buffett_agent
from src.agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent

# 定义分析师配置 - 单一数据源
ANALYST_CONFIG = {
    "aswath_damodaran": {
        "display_name": "阿斯沃斯·达摩达兰",
        "description": "估值学院院长",
        "investing_style": "quantitative_analytical",
        "agent_func": aswath_damodaran_agent,
        "type": "analyst",
        "order": 0,
    },
    "ben_graham": {
        "display_name": "本杰明·格雷厄姆",
        "description": "价值投资之父",
        "investing_style": "value_investing",
        "agent_func": ben_graham_agent,
        "type": "analyst",
        "order": 1,
    },
    "bill_ackman": {
        "display_name": "比尔·阿克曼",
        "description": "激进投资者",
        "investing_style": "contrarian_activist",
        "agent_func": bill_ackman_agent,
        "type": "analyst",
        "order": 2,
    },
    "cathie_wood": {
        "display_name": "凯茜·伍德",
        "description": "成长投资女王",
        "investing_style": "growth_investing",
        "agent_func": cathie_wood_agent,
        "type": "analyst",
        "order": 3,
    },
    "charlie_munger": {
        "display_name": "查理·芒格",
        "description": "理性思考者",
        "investing_style": "value_investing",
        "agent_func": charlie_munger_agent,
        "type": "analyst",
        "order": 4,
    },
    "michael_burry": {
        "display_name": "迈克尔·伯里",
        "description": "大空头逆向投资者",
        "investing_style": "contrarian_activist",
        "agent_func": michael_burry_agent,
        "type": "analyst",
        "order": 5,
    },
    "peter_lynch": {
        "display_name": "彼得·林奇",
        "description": "十倍股投资者",
        "investing_style": "growth_investing",
        "agent_func": peter_lynch_agent,
        "type": "analyst",
        "order": 6,
    },
    "phil_fisher": {
        "display_name": "菲利普·费雪",
        "description": "闲聊法投资者",
        "investing_style": "growth_investing",
        "agent_func": phil_fisher_agent,
        "type": "analyst",
        "order": 7,
    },
    "rakesh_jhunjhunwala": {
        "display_name": "拉凯什·朱恩朱恩瓦拉",
        "description": "印度股市大牛",
        "investing_style": "macro_global",
        "agent_func": rakesh_jhunjhunwala_agent,
        "type": "analyst",
        "order": 8,
    },
    "stanley_druckenmiller": {
        "display_name": "斯坦利·德鲁肯米勒",
        "description": "宏观投资大师",
        "investing_style": "macro_global",
        "agent_func": stanley_druckenmiller_agent,
        "type": "analyst",
        "order": 9,
    },
    "warren_buffett": {
        "display_name": "沃伦·巴菲特",
        "description": "奥马哈先知",
        "investing_style": "value_investing",
        "agent_func": warren_buffett_agent,
        "type": "analyst",
        "order": 10,
    },
    "technical_analyst": {
        "display_name": "技术分析师",
        "description": "图表模式专家",
        "investing_style": "technical_analysis",
        "agent_func": technical_analyst_agent,
        "type": "analyst",
        "order": 11,
    },
    "fundamentals_analyst": {
        "display_name": "基本面分析师",
        "description": "财务报表专家",
        "investing_style": "quantitative_analytical",
        "agent_func": fundamentals_analyst_agent,
        "type": "analyst",
        "order": 12,
    },
    "sentiment_analyst": {
        "display_name": "情绪分析师",
        "description": "市场情绪专家",
        "investing_style": "technical_analysis",
        "agent_func": sentiment_analyst_agent,
        "type": "analyst",
        "order": 13,
    },
    "valuation_analyst": {
        "display_name": "估值分析师",
        "description": "公司估值专家",
        "investing_style": "quantitative_analytical",
        "agent_func": valuation_analyst_agent,
        "type": "analyst",
        "order": 14,
    },
}

# 从ANALYST_CONFIG派生ANALYST_ORDER以保持向后兼容性
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """获取分析师键到其(node_name, agent_func)元组的映射。"""
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}


def get_agents_list():
    """获取用于API响应的代理列表。"""
    return [
        {
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "investing_style": config["investing_style"],
            "order": config["order"]
        }
        for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    ]


def get_investing_styles():
    """获取所有独特的投资风格。"""
    return list(set(config["investing_style"] for config in ANALYST_CONFIG.values()))


def get_investing_style_display_names():
    """获取投资风格的显示名称。"""
    return {
        "value_investing": "Value Investing",
        "growth_investing": "Growth Investing", 
        "contrarian_activist": "Contrarian/Activist",
        "macro_global": "Macro/Global",
        "technical_analysis": "Technical Analysis",
        "quantitative_analytical": "Quantitative/Analytical"
    }


def get_agents_by_investing_style():
    """获取按投资风格分组的代理。"""
    groups = {}
    for key, config in ANALYST_CONFIG.items():
        style = config["investing_style"]
        if style not in groups:
            groups[style] = []
        groups[style].append({
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "order": config["order"]
        })
    
    # 在每个组内按顺序排序代理
    for style in groups:
        groups[style].sort(key=lambda x: x["order"])
    
    return groups
