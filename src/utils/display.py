from colorama import Fore, Style
from tabulate import tabulate
from .analysts import ANALYST_ORDER
from .i18n import _
import os
import json
from datetime import datetime

# 统一的视觉样式配置
class DisplayStyles:
    """统一的显示样式配置类"""

    # 颜色配置
    HEADER_COLOR = Fore.CYAN + Style.BRIGHT
    TITLE_COLOR = Fore.WHITE + Style.BRIGHT
    SUCCESS_COLOR = Fore.GREEN + Style.BRIGHT
    WARNING_COLOR = Fore.YELLOW + Style.BRIGHT
    ERROR_COLOR = Fore.RED + Style.BRIGHT
    INFO_COLOR = Fore.BLUE + Style.BRIGHT
    ACCENT_COLOR = Fore.MAGENTA + Style.BRIGHT

    # 表格样式
    TABLE_FORMAT = "fancy_grid"

    # 分隔线配置
    SEPARATOR_CHAR = "="
    MIN_SEPARATOR_LENGTH = 60

    @staticmethod
    def create_separator(text="", length=None):
        """创建动态长度的分隔线"""
        if length is None:
            length = max(DisplayStyles.MIN_SEPARATOR_LENGTH, len(text) + 20)
        return DisplayStyles.SEPARATOR_CHAR * length

    @staticmethod
    def create_title(text, color=None, icon=""):
        """创建统一格式的标题"""
        if color is None:
            color = DisplayStyles.TITLE_COLOR
        return f"{color}{icon} {text}{Style.RESET_ALL}"

    @staticmethod
    def create_section_header(text, icon="", color=None):
        """创建章节标题"""
        if color is None:
            color = DisplayStyles.HEADER_COLOR
        separator = DisplayStyles.create_separator(text)
        title = DisplayStyles.create_title(text, color, icon)
        return f"\n{color}{separator}{Style.RESET_ALL}\n{title}\n{color}{separator}{Style.RESET_ALL}"


def sort_agent_signals(signals):
    """按一致的顺序排序代理信号。"""
    # 从ANALYST_ORDER创建顺序映射
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # 在末尾添加风险管理

    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def print_trading_output(result: dict) -> None:
    """
    打印格式化的交易结果，为多个股票代码显示彩色表格。

    Args:
        result (dict): 包含多个股票代码的决策和分析师信号的字典
    """
    # 添加开始分隔线
    print(f"\n{DisplayStyles.SUCCESS_COLOR}{'🚀 ' + DisplayStyles.create_separator('AI对冲基金分析结果', 80)}{Style.RESET_ALL}")
    print(f"{DisplayStyles.SUCCESS_COLOR}🚀 AI对冲基金分析结果{Style.RESET_ALL}")
    print(f"{DisplayStyles.SUCCESS_COLOR}{DisplayStyles.create_separator('AI对冲基金分析结果', 80)}{Style.RESET_ALL}")

    decisions = result.get("decisions")
    if not decisions:
        print(f"{DisplayStyles.ERROR_COLOR}{_('No trading decisions available')}{Style.RESET_ALL}")
        return

    # 为每个股票代码打印决策，增强视觉层次
    for ticker, decision in decisions.items():
        # 使用统一的样式创建标题
        print(DisplayStyles.create_section_header(f"股票分析: {ticker}", "🔍", DisplayStyles.TITLE_COLOR))

        # 为此股票代码准备分析师信号表
        table_data = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals:
                continue
                
            # 在信号部分跳过风险管理代理
            if agent == "risk_management_agent":
                continue

            signal = signals[ticker]

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
                "portfolio_manager": "投资组合经理",
                "portfolio_manager_agent": "投资组合经理",
                "portfolio_management_agent": "投资组合经理",
                "risk_management_agent": "风险管理师",
            }

            # 使用中文名称或默认格式
            agent_name = chinese_names.get(agent, agent.replace("_agent", "").replace("_", " ").title())
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)

            # 翻译信号类型
            signal_type_cn = {
                "BULLISH": _("bullish"),
                "BEARISH": _("bearish"),
                "NEUTRAL": _("neutral"),
            }.get(signal_type, signal_type)

            signal_color = {
                "BULLISH": Fore.GREEN,
                "BEARISH": Fore.RED,
                "NEUTRAL": Fore.YELLOW,
            }.get(signal_type, Fore.WHITE)
            
            # 如果可用，获取推理
            reasoning_str = ""
            if "reasoning" in signal and signal["reasoning"]:
                reasoning = signal["reasoning"]
                
                # 处理不同类型的推理（字符串、字典等）
                if isinstance(reasoning, str):
                    reasoning_str = reasoning
                elif isinstance(reasoning, dict):
                    # Convert dict to string representation
                    reasoning_str = json.dumps(reasoning, indent=2)
                else:
                    # Convert any other type to string
                    reasoning_str = str(reasoning)
                
                # 包装长推理文本以使中文文本更易读
                wrapped_reasoning = ""
                current_line = ""
                # Use a smaller width for Chinese text in table columns
                max_line_length = 45  # 为表格中更好的中文显示而减少
                for word in reasoning_str.split():
                    # 计算考虑中文字符的显示宽度
                    current_display_width = len(current_line.encode('utf-8')) - len(current_line)
                    word_display_width = len(word.encode('utf-8')) - len(word)

                    if current_display_width + word_display_width + len(current_line) + len(word) + 1 > max_line_length:
                        wrapped_reasoning += current_line + "\n"
                        current_line = word
                    else:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                if current_line:
                    wrapped_reasoning += current_line

                reasoning_str = wrapped_reasoning

            table_data.append(
                [
                    f"{Fore.CYAN}{Style.BRIGHT}{agent_name}{Style.RESET_ALL}",
                    f"{signal_color}{Style.BRIGHT}{signal_type_cn.upper()}{Style.RESET_ALL}",
                    f"{Fore.YELLOW}{Style.BRIGHT}{confidence}%{Style.RESET_ALL}",
                    f"{Fore.WHITE}{reasoning_str}{Style.RESET_ALL}",
                ]
            )

        # 根据预定义顺序排序信号
        table_data = sort_agent_signals(table_data)

        # 使用统一样式创建分析师信号标题
        print(DisplayStyles.create_section_header(f"分析师信号: {ticker}", "🔍", DisplayStyles.ACCENT_COLOR))
        print(
            tabulate(
                table_data,
                headers=[
                    f"{DisplayStyles.HEADER_COLOR}分析师{Style.RESET_ALL}",
                    f"{DisplayStyles.HEADER_COLOR}信号{Style.RESET_ALL}",
                    f"{DisplayStyles.HEADER_COLOR}置信度{Style.RESET_ALL}",
                    f"{DisplayStyles.HEADER_COLOR}分析理由{Style.RESET_ALL}"
                ],
                tablefmt=DisplayStyles.TABLE_FORMAT,
                colalign=("left", "center", "right", "left"),
            )
        )

        # 打印交易决策表
        action = decision.get("action", "").upper()
        # 翻译操作类型
        action_cn = {
            "BUY": _("buy"),
            "SELL": _("sell"),
            "HOLD": _("hold"),
            "COVER": "平仓",
            "SHORT": "做空",
        }.get(action, action)

        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
        }.get(action, Fore.WHITE)

        # 获取推理并格式化
        reasoning = decision.get("reasoning", "")
        # 包装长推理文本以使中文文本更易读
        wrapped_reasoning = ""
        if reasoning:
            current_line = ""
            # 为中文文本使用更宽的宽度（中文字符更宽）
            max_line_length = 50  # 为更好的中文显示而减少
            for word in reasoning.split():
                # 计算考虑中文字符的显示宽度
                current_display_width = len(current_line.encode('utf-8')) - len(current_line)
                word_display_width = len(word.encode('utf-8')) - len(word)

                if current_display_width + word_display_width + len(current_line) + len(word) + 1 > max_line_length:
                    wrapped_reasoning += current_line + "\n"
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            if current_line:
                wrapped_reasoning += current_line

        decision_data = [
            [f"{Fore.CYAN}{Style.BRIGHT}操作类型{Style.RESET_ALL}", f"{action_color}{Style.BRIGHT}{action_cn.upper()}{Style.RESET_ALL}"],
            [f"{Fore.CYAN}{Style.BRIGHT}交易数量{Style.RESET_ALL}", f"{action_color}{Style.BRIGHT}{decision.get('quantity')}{Style.RESET_ALL}"],
            [
                f"{Fore.CYAN}{Style.BRIGHT}置信度{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
            ],
            [f"{Fore.CYAN}{Style.BRIGHT}决策理由{Style.RESET_ALL}", f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"],
        ]

        # 使用统一样式创建交易决策标题
        print(DisplayStyles.create_section_header(f"交易决策: {ticker}", "📊", DisplayStyles.WARNING_COLOR))
        print(tabulate(decision_data, tablefmt=DisplayStyles.TABLE_FORMAT, colalign=("left", "left")))

    # 打印增强格式的投资组合摘要
    print(DisplayStyles.create_section_header("投资组合摘要", "💼", DisplayStyles.SUCCESS_COLOR))

    portfolio_data = []

    # 提取投资组合经理推理（所有股票代码通用）
    portfolio_manager_reasoning = None
    for ticker, decision in decisions.items():
        if decision.get("reasoning"):
            portfolio_manager_reasoning = decision.get("reasoning")
            break

    for ticker, decision in decisions.items():
        action = decision.get("action", "").upper()
        # 翻译操作类型为中文
        action_cn = {
            "BUY": "买入",
            "SELL": "卖出",
            "HOLD": "持有",
            "COVER": "平仓",
            "SHORT": "做空",
        }.get(action, action)

        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
        }.get(action, Fore.WHITE)

        portfolio_data.append(
            [
                f"{Fore.CYAN}{Style.BRIGHT}{ticker}{Style.RESET_ALL}",
                f"{action_color}{Style.BRIGHT}{action_cn}{Style.RESET_ALL}",
                f"{action_color}{Style.BRIGHT}{decision.get('quantity')}{Style.RESET_ALL}",
                f"{Fore.YELLOW}{Style.BRIGHT}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
            ]
        )

    headers = [
        f"{Fore.GREEN}{Style.BRIGHT}股票代码{Style.RESET_ALL}",
        f"{Fore.GREEN}{Style.BRIGHT}操作类型{Style.RESET_ALL}",
        f"{Fore.GREEN}{Style.BRIGHT}交易数量{Style.RESET_ALL}",
        f"{Fore.GREEN}{Style.BRIGHT}置信度{Style.RESET_ALL}"
    ]

    # 打印投资组合摘要表
    print(
        tabulate(
            portfolio_data,
            headers=headers,
            tablefmt=DisplayStyles.TABLE_FORMAT,
            colalign=("left", "center", "right", "right"),
        )
    )
    
    # 如果可用，打印投资组合经理的推理
    if portfolio_manager_reasoning:
        # 处理不同类型的推理（字符串、字典等）
        reasoning_str = ""
        if isinstance(portfolio_manager_reasoning, str):
            reasoning_str = portfolio_manager_reasoning
        elif isinstance(portfolio_manager_reasoning, dict):
            # Convert dict to string representation
            reasoning_str = json.dumps(portfolio_manager_reasoning, indent=2)
        else:
            # Convert any other type to string
            reasoning_str = str(portfolio_manager_reasoning)

        # Wrap long reasoning text to make it more readable for Chinese text
        wrapped_reasoning = ""
        current_line = ""
        # 为中文文本使用更宽的宽度
        max_line_length = 70
        for word in reasoning_str.split():
            # 计算考虑中文字符的显示宽度
            current_display_width = len(current_line.encode('utf-8')) - len(current_line)
            word_display_width = len(word.encode('utf-8')) - len(word)

            if current_display_width + word_display_width + len(current_line) + len(word) + 1 > max_line_length:
                wrapped_reasoning += current_line + "\n"
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        if current_line:
            wrapped_reasoning += current_line

        print(DisplayStyles.create_section_header("投资组合策略", "🎯", DisplayStyles.INFO_COLOR))
        print(f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}")
        print(f"{DisplayStyles.INFO_COLOR}{DisplayStyles.create_separator()}{Style.RESET_ALL}\n")

    # 添加结束分隔线
    print(f"{DisplayStyles.SUCCESS_COLOR}{DisplayStyles.create_separator('分析完成', 80)}{Style.RESET_ALL}")
    print(f"{DisplayStyles.SUCCESS_COLOR}✅ 分析完成{Style.RESET_ALL}")
    print(f"{DisplayStyles.SUCCESS_COLOR}{DisplayStyles.create_separator('分析完成', 80)}{Style.RESET_ALL}\n")


def print_backtest_results(table_rows: list, clear_screen: bool = True, max_rows: int = None) -> None:
    """以格式良好的表格打印回测结果

    Args:
        table_rows: 要显示的表格行列表
        clear_screen: 是否在显示结果前清除屏幕（默认：True）
        max_rows: 要显示的最大行数（None表示所有行）
    """
    # 仅在请求时清除屏幕
    if clear_screen:
        os.system("cls" if os.name == "nt" else "clear")
    else:
        # 如果不清除屏幕，添加一些视觉分隔但保持最小
        # 以避免在AI分析期间使显示混乱
        print(f"\n{Fore.CYAN}📊 {datetime.now().strftime('%H:%M:%S')} - 回测结果更新{Style.RESET_ALL}")
        print("-" * 60)

    # 将行分为股票代码行和摘要行
    ticker_rows = []
    summary_rows = []

    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    
    # 显示增强格式的最新投资组合摘要
    if summary_rows:
        latest_summary = summary_rows[-1]
        print(DisplayStyles.create_section_header("回测投资组合摘要", "📈", DisplayStyles.SUCCESS_COLOR))

        # 提取值并在转换为浮点数前移除逗号
        cash_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        position_str = latest_summary[6].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        # 创建格式化的摘要表
        summary_data = [
            [f"{Fore.CYAN}{Style.BRIGHT}现金余额{Style.RESET_ALL}", f"{Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}"],
            [f"{Fore.YELLOW}{Style.BRIGHT}持仓总价值{Style.RESET_ALL}", f"{Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}"],
            [f"{Fore.WHITE}{Style.BRIGHT}总资产价值{Style.RESET_ALL}", f"{Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}"],
            [f"{Fore.GREEN}{Style.BRIGHT}投资回报{Style.RESET_ALL}", f"{Fore.GREEN}{latest_summary[9]}{Style.RESET_ALL}"],
        ]

        # Add performance metrics if available
        if latest_summary[10]:  # Sharpe ratio
            summary_data.append([f"{Fore.MAGENTA}{Style.BRIGHT}夏普比率{Style.RESET_ALL}", f"{Fore.MAGENTA}{latest_summary[10]}{Style.RESET_ALL}"])
        if latest_summary[11]:  # Sortino ratio
            summary_data.append([f"{Fore.MAGENTA}{Style.BRIGHT}索提诺比率{Style.RESET_ALL}", f"{Fore.MAGENTA}{latest_summary[11]}{Style.RESET_ALL}"])
        if latest_summary[12]:  # Max drawdown
            summary_data.append([f"{Fore.RED}{Style.BRIGHT}最大回撤{Style.RESET_ALL}", f"{Fore.RED}{latest_summary[12]}{Style.RESET_ALL}"])

        print(tabulate(summary_data, tablefmt=DisplayStyles.TABLE_FORMAT, colalign=("left", "right")))

    # 添加垂直间距
    print("\n" * 2)

    # 打印详细交易历史表，可选行数限制
    display_rows = ticker_rows

    # 如果指定且不清除屏幕（在AI分析期间），应用行数限制
    if max_rows is not None and len(ticker_rows) > max_rows:
        if not clear_screen:
            # 在AI分析期间，只显示最近的行
            display_rows = ticker_rows[-max_rows:]
            print(DisplayStyles.create_section_header(f"详细交易历史 (最近 {max_rows} 条记录)", "📊", DisplayStyles.INFO_COLOR))
        else:
            # 清除屏幕时，显示所有行但添加注释
            print(DisplayStyles.create_section_header(f"详细交易历史 (共 {len(ticker_rows)} 条记录)", "📊", DisplayStyles.INFO_COLOR))
    else:
        print(DisplayStyles.create_section_header("详细交易历史", "📊", DisplayStyles.INFO_COLOR))

    print(
        tabulate(
            display_rows,
            headers=[
                f"{DisplayStyles.HEADER_COLOR}日期{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}股票代码{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}操作{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}数量{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}价格{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}持股数{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}持仓价值{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}负债{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}看涨/现金{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}看跌/资产{Style.RESET_ALL}",
                f"{DisplayStyles.HEADER_COLOR}中性/收益率{Style.RESET_ALL}",
            ],
            tablefmt=DisplayStyles.TABLE_FORMAT,
            colalign=(
                "left",  # 日期
                "left",  # 股票代码
                "center",  # 操作
                "right",  # 数量
                "right",  # Price
                "right",  # Shares
                "right",  # Position Value
                "right",  # 负债
                "right",  # Bullish
                "right",  # Bearish
                "right",  # 中性
            ),
        )
    )

    # 如果适用，添加关于截断行的信息
    if max_rows is not None and len(ticker_rows) > max_rows and not clear_screen:
        hidden_count = len(ticker_rows) - max_rows
        print(f"\n{Fore.YELLOW}💡 隐藏了 {hidden_count} 条较早的记录以保持表格可见性{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   完整历史记录将在AI分析完成后显示{Style.RESET_ALL}")

    # 添加垂直间距
    print("\n" * 4)


def format_backtest_row(
    date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    shares_owned: float,
    position_value: float,
    liabilities: float,
    bullish_count: int,
    bearish_count: int,
    neutral_count: int,
    is_summary: bool = False,
    total_value: float = None,
    return_pct: float = None,
    cash_balance: float = None,
    total_position_value: float = None,
    total_liabilities: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    max_drawdown: float = None,
) -> list[any]:
    """为回测结果表格式化一行"""
    # 为操作着色
    action_color = {
        "BUY": Fore.GREEN,
        "COVER": Fore.GREEN,
        "SELL": Fore.RED,
        "SHORT": Fore.RED,
        "HOLD": Fore.WHITE,
    }.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}投资组合摘要{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Shares
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",  # Total Position Value
            f"{Fore.RED}${total_liabilities:,.2f}{Style.RESET_ALL}" if total_liabilities else "",  # Total Liabilities
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",  # Cash Balance
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",  # Total Value
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",  # Return
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",  # Sharpe Ratio
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",  # Sortino Ratio
            f"{Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",  # Max Drawdown
        ]
    else:
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{shares_owned:,.0f}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.RED}{liabilities:,.2f}{Style.RESET_ALL}" if liabilities > 0 else f"{Fore.WHITE}0.00{Style.RESET_ALL}",
            f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
            f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
            f"{Fore.BLUE}{neutral_count}{Style.RESET_ALL}",
        ]
