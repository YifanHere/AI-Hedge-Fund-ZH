from datetime import datetime, timezone
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.style import Style
from rich.text import Text
from typing import Dict, Optional, Callable, List
from src.utils.status_messages_zh import get_chinese_status_message
import unicodedata

console = Console()


def get_display_width(text: str) -> int:
    """计算字符串的显示宽度，考虑中文字符占用两个字符位置"""
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('F', 'W'):  # 全角字符
            width += 2
        else:  # 半角字符
            width += 1
    return width


def pad_to_width(text: str, target_width: int) -> str:
    """将字符串填充到指定的显示宽度"""
    current_width = get_display_width(text)
    if current_width >= target_width:
        return text
    padding = target_width - current_width
    return text + ' ' * padding


class AgentProgress:
    """管理多个代理的进度跟踪。"""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, str]] = {}
        self.table = Table(show_header=False, box=None, padding=(0, 1))
        self.live = Live(self.table, console=console, refresh_per_second=4)
        self.started = False
        self.update_handlers: List[Callable[[str, Optional[str], str], None]] = []

    def register_handler(self, handler: Callable[[str, Optional[str], str], None]):
        """注册在代理状态更新时调用的处理器。"""
        self.update_handlers.append(handler)
        return handler  # 返回处理器以支持用作装饰器

    def unregister_handler(self, handler: Callable[[str, Optional[str], str], None]):
        """取消注册先前注册的处理器。"""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)

    def start(self):
        """启动进度显示。"""
        if not self.started:
            self.live.start()
            self.started = True

    def stop(self):
        """停止进度显示。"""
        if self.started:
            self.live.stop()
            self.started = False

    def is_analysis_in_progress(self) -> bool:
        """检查是否有任何代理当前正在分析（未完成）。"""
        for agent_name, info in self.agent_status.items():
            status = info.get("status", "").lower()
            # 如果状态不是"完成"或"done"，则认为分析正在进行中
            # 同时从此检查中排除"system"代理，因为它用于回测协调
            if agent_name != "system" and status and status not in ["完成", "done", "error", "错误", ""]:
                return True
        return False

    def clear_all_status(self):
        """清除所有代理状态 - 用于在回测日期之间重置。"""
        self.agent_status.clear()
        self._refresh_display()

    def update_status(self, agent_name: str, ticker: Optional[str] = None, status: str = "", analysis: Optional[str] = None):
        """更新代理的状态。"""
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = {"status": "", "ticker": None}

        if ticker:
            self.agent_status[agent_name]["ticker"] = ticker
        if status:
            self.agent_status[agent_name]["status"] = status
        if analysis:
            self.agent_status[agent_name]["analysis"] = analysis
        
        # 将时间戳设置为UTC日期时间
        timestamp = datetime.now(timezone.utc).isoformat()
        self.agent_status[agent_name]["timestamp"] = timestamp

        # 通知所有注册的处理器
        for handler in self.update_handlers:
            handler(agent_name, ticker, status, analysis, timestamp)

        self._refresh_display()

    def get_all_status(self):
        """以字典形式获取所有代理的当前状态。"""
        return {agent_name: {"ticker": info["ticker"], "status": info["status"], "display_name": self._get_display_name(agent_name)} for agent_name, info in self.agent_status.items()}

    def _get_display_name(self, agent_name: str) -> str:
        """将agent_name转换为显示友好的格式。"""
        # 中文分析师名称映射
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
            "portfolio_manager": "投资组合经理",
            "portfolio_manager_agent": "投资组合经理",
            "portfolio_management_agent": "投资组合经理",
        }

        # 如果有中文名称映射，使用中文名称
        if agent_name in chinese_names:
            return chinese_names[agent_name]

        # 否则使用原来的逻辑
        return agent_name.replace("_agent", "").replace("_", " ").title()

    def _refresh_display(self):
        """刷新进度显示。"""
        self.table.columns.clear()
        self.table.add_column(width=100)

        # 将风险管理和投资组合管理代理排在底部
        def sort_key(item):
            agent_name = item[0]
            if "risk_management" in agent_name:
                return (2, agent_name)
            elif "portfolio_management" in agent_name:
                return (3, agent_name)
            else:
                return (1, agent_name)

        # 首先计算所有分析师名称的最大显示宽度
        max_name_width = 0
        agent_displays = {}
        for agent_name, info in self.agent_status.items():
            agent_display = self._get_display_name(agent_name)
            agent_displays[agent_name] = agent_display
            max_name_width = max(max_name_width, get_display_width(agent_display))

        # 设置最小宽度为25，确保有足够的对齐空间
        target_width = max(25, max_name_width + 2)

        for agent_name, info in sorted(self.agent_status.items(), key=sort_key):
            status = info["status"]
            ticker = info["ticker"]
            # 创建具有适当样式的状态文本
            if status.lower() == "done" or status == "完成":
                style = Style(color="green", bold=True)
                symbol = "✓"
            elif status.lower() == "error" or status == "错误":
                style = Style(color="red", bold=True)
                symbol = "✗"
            else:
                style = Style(color="yellow")
                symbol = "⋯"

            agent_display = agent_displays[agent_name]
            # 使用动态对齐，确保中文字符正确对齐
            padded_name = pad_to_width(agent_display, target_width)

            status_text = Text()
            status_text.append(f"{symbol} ", style=style)
            status_text.append(padded_name, style=Style(bold=True))

            if ticker:
                status_text.append(f"[{ticker}] ", style=Style(color="cyan"))

            # 使用中文状态消息
            chinese_status = get_chinese_status_message(status)
            status_text.append(chinese_status, style=style)

            self.table.add_row(status_text)


# 创建全局实例
progress = AgentProgress()
