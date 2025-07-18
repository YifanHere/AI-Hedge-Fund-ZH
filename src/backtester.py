import sys
import threading
import time
import signal
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools

from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.analysts import ANALYST_ORDER
from src.main import run_hedge_fund
from src.tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)
from src.tools.index_scraper import get_market_index_data
from src.utils.display import print_backtest_results, format_backtest_row
from src.utils.progress import progress
from typing_extensions import Callable
from src.utils.ollama import ensure_ollama_and_model
from src.utils.i18n import _, get_text
from src.utils.analyst_reasoning_tracker import AnalystReasoningTracker
from src.utils.status_messages_zh import get_chinese_analyst_name

init(autoreset=True)

class InputHandler:
    """输入处理器基类。"""
    def get_command(self, timeout=1.0):
        """获取用户命令，带超时。"""
        raise NotImplementedError

class WindowsInputHandler(InputHandler):
    """Windows平台的输入处理器。"""
    def __init__(self):
        import queue
        import threading
        self.input_queue = queue.Queue()
        self.input_thread = None
        self._start_input_thread()
    
    def _start_input_thread(self):
        """启动输入线程。"""
        import threading
        
        def input_loop():
            import sys
            while True:
                try:
                    # 使用sys.stdin.readline()而不是input()以获得更好的控制
                    line = sys.stdin.readline()
                    if line:
                        self.input_queue.put(line.strip())
                    else:
                        break
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception:
                    break
        
        self.input_thread = threading.Thread(target=input_loop, daemon=True)
        self.input_thread.start()
    
    def get_command(self, timeout=1.0):
        """获取用户命令。"""
        import queue
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class UnixInputHandler(InputHandler):
    """Unix/Linux/macOS平台的输入处理器。"""
    def __init__(self):
        import queue
        import threading
        self.input_queue = queue.Queue()
        self.input_thread = None
        self._start_input_thread()
    
    def _start_input_thread(self):
        """启动输入线程。"""
        import threading
        
        def input_loop():
            import sys
            import select
            while True:
                try:
                    # 使用select检查stdin是否有数据可读
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        line = sys.stdin.readline()
                        if line:
                            self.input_queue.put(line.strip())
                        else:
                            break
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception:
                    break
        
        self.input_thread = threading.Thread(target=input_loop, daemon=True)
        self.input_thread.start()
    
    def get_command(self, timeout=1.0):
        """获取用户命令。"""
        import queue
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class Backtester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str = "gpt-4.1",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
        initial_margin_requirement: float = 0.0,
        show_reasoning: bool = False,
    ):
        # 美国股市节假日列表（2025年）
        self.us_holidays_2025 = {
            "2025-01-01",  # 新年
            "2025-01-20",  # 马丁·路德·金纪念日
            "2025-02-17",  # 总统节
            "2025-04-18",  # 耶稣受难日
            "2025-05-26",  # 阵亡将士纪念日
            "2025-06-19",  # 六月节
            "2025-07-03",  # 独立日前一天（7月4日是周五，所以7月3日休市）
            "2025-09-01",  # 劳动节
            "2025-11-27",  # 感恩节
            "2025-12-25",  # 圣诞节
        }
        """
        :param agent: 交易代理（可调用对象）。
        :param tickers: 要回测的股票代码列表。
        :param start_date: 开始日期字符串（YYYY-MM-DD）。
        :param end_date: 结束日期字符串（YYYY-MM-DD）。
        :param initial_capital: 初始投资组合现金。
        :param model_name: 要使用的LLM模型名称（gpt-4等）。
        :param model_provider: LLM提供商（OpenAI等）。
        :param selected_analysts: 要纳入的分析师名称或ID列表。
        :param initial_margin_requirement: 保证金比率（例如0.5 = 50%）。
        :param show_reasoning: 是否显示详细的交易决策推理。
        """
        self.agent = agent
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.show_reasoning = show_reasoning

        # 暂停/恢复/终止功能的控制标志
        self.is_paused = False
        self.should_terminate = False
        self.control_thread = None
        self.pause_event = threading.Event()
        self.pause_event.set()  # 初始时未暂停

        # 初始化投资组合
        self.portfolio = {
            "cash": initial_capital,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "margin_requirement": initial_margin_requirement,
                }
                for ticker in tickers
            },
        }

        # 初始化AI分析师思考过程追踪器
        self.reasoning_tracker = AnalystReasoningTracker()

        # 存储投资组合价值历史（用于性能分析）
        self.portfolio_values = []

        # 存储市场基准数据
        self.market_benchmark = None
        self.benchmark_values = []

    def _setup_signal_handlers(self):
        """设置信号处理器以优雅终止。"""
        def signal_handler(signum, frame):
            print(f"\n{Fore.YELLOW}{get_text('Received termination signal. Gracefully stopping backtest...', 'backtester')}{Style.RESET_ALL}")
            self.should_terminate = True
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _start_control_thread(self):
        """启动控制线程以处理用户输入。"""
        def control_loop():
            pause_text = "'p' or 'pause' - Pause the backtest"
            resume_text = "'r' or 'resume' - Resume the backtest"
            quit_text = "'q' or 'quit' - Terminate the backtest"
            status_text = "'s' or 'status' - Show current status"
            help_text = "'h' or 'help' - Show help message"

            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{get_text('🎮 Backtest Control Commands:', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}  {get_text(pause_text, 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}  {get_text(resume_text, 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.RED}  {get_text(quit_text, 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  {get_text(status_text, 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}  {get_text(help_text, 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{get_text('📝 Type command and press Enter:', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}💡 {get_text('Tip: Your input may not be visible while backtest is running, but it will work when you press Enter.', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}💡 {get_text('If you cannot see your typing, just type the command and press Enter - it will work!', 'backtester')}{Style.RESET_ALL}\n")

            # 使用改进的跨平台输入处理
            input_handler = self._create_input_handler()
            
            while not self.should_terminate:
                try:
                    command = input_handler.get_command(timeout=1.0)
                    if command is not None:
                        # 显示用户输入的命令以提供反馈
                        print(f"{Fore.CYAN}> {command}{Style.RESET_ALL}")
                        self._handle_command(command.strip().lower())
                except Exception as e:
                    print(f"{get_text('Error in control thread: ', 'backtester')}{e}")
                    time.sleep(0.5)

        self.control_thread = threading.Thread(target=control_loop, daemon=True)
        self.control_thread.start()

    def _create_input_handler(self):
        """创建跨平台的输入处理器。"""
        import sys
        import platform

        if platform.system() == "Windows":
            return WindowsInputHandler()
        else:
            return UnixInputHandler()

    def _handle_command(self, command):
        """处理用于控制回测的用户命令。"""
        if command in ['p', 'pause']:
            if not self.is_paused:
                self.is_paused = True
                self.pause_event.clear()
                pause_msg = "⏸️  Backtest PAUSED. Type 'r' or 'resume' to continue."
                print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{get_text(pause_msg, 'backtester')}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}{get_text('⚠️  Backtest is already paused.', 'backtester')}{Style.RESET_ALL}")

        elif command in ['r', 'resume']:
            if self.is_paused:
                self.is_paused = False
                self.pause_event.set()
                print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}{get_text('▶️  Backtest RESUMED.', 'backtester')}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.GREEN}{get_text('✅ Backtest is already running.', 'backtester')}{Style.RESET_ALL}")

        elif command in ['q', 'quit']:
            print(f"\n{Fore.RED}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.RED}{get_text('🛑 Terminating backtest...', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
            self.should_terminate = True
            self.pause_event.set()  # Ensure we don't get stuck in pause

        elif command in ['s', 'status']:
            status = get_text("PAUSED", 'backtester') if self.is_paused else get_text("RUNNING", 'backtester')
            color = Fore.YELLOW if self.is_paused else Fore.GREEN
            print(f"\n{color}{'='*60}{Style.RESET_ALL}")
            print(f"{color}{get_text('📊 Backtest Status: ', 'backtester')}{status}{Style.RESET_ALL}")
            if hasattr(self, 'current_date_str'):
                print(f"{color}{get_text('📅 Current Date: ', 'backtester')}{self.current_date_str}{Style.RESET_ALL}")
            if hasattr(self, 'portfolio') and self.portfolio:
                total_value = self.calculate_portfolio_value(getattr(self, 'current_prices', {}))
                print(f"{color}{get_text('💰 Portfolio Value: ', 'backtester')}${total_value:,.2f}{Style.RESET_ALL}")
            print(f"{color}{'='*60}{Style.RESET_ALL}")

        elif command in ['h', 'help']:
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{get_text('📖 Available Commands:', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}  p, pause  - {get_text('Pause the backtest', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}  r, resume - {get_text('Resume the backtest', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.RED}  q, quit   - {get_text('Terminate the backtest', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  s, status - {get_text('Show current status', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}  h, help   - {get_text('Show this help message', 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

        else:
            use_msg = ". Use 'p', 'r', 'q', 's', or 'h' for help."
            print(f"\n{Fore.RED}{get_text('❌ Unknown command: ', 'backtester')}'{command}'{get_text(use_msg, 'backtester')}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{get_text('💡 Type \"h\" or \"help\" to see available commands.', 'backtester')}{Style.RESET_ALL}")

    def _wait_if_paused(self):
        """如果回测暂停则等待。"""
        if self.is_paused:
            wait_msg = "⏸️  Backtest is paused. Waiting for resume command..."
            print(f"\n{Fore.YELLOW}{get_text(wait_msg, 'backtester')}{Style.RESET_ALL}")
        self.pause_event.wait()  # 阻塞直到恢复或终止

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float):
        """
        执行支持多头和空头仓位的交易。
        `quantity` 是代理想要买入/卖出/做空/平仓的股份数量。
        为了简单起见，我们只交易整数股份。
        """
        if quantity <= 0:
            return 0

        quantity = int(quantity)  # 强制整数股份
        position = self.portfolio["positions"][ticker]

        # 改进的仓位管理逻辑 - 针对单股回测优化
        if action in ["buy", "cover"]:
            # 提高单次买入限制到30%，适应单股回测
            max_position_value = self.calculate_portfolio_value({ticker: current_price}) * 0.30
            max_quantity = max_position_value / current_price
            quantity = min(quantity, max_quantity)

        # 改进的动态止损保护 - 更适合高波动股票
        current_portfolio_value = self.calculate_portfolio_value({ticker: current_price})
        max_portfolio_value = max([pv["Portfolio Value"] for pv in self.portfolio_values] + [self.initial_capital])
        current_drawdown = (max_portfolio_value - current_portfolio_value) / max_portfolio_value

        # 调整动态止损策略 - 提高容忍度
        if current_drawdown > 0.50:  # 50%回撤时完全停止新开仓（从30%提高）
            if action in ["buy", "short"]:
                return 0
        elif current_drawdown > 0.35:  # 35%回撤时减少仓位（从20%提高）
            if action in ["buy", "short"]:
                quantity = int(quantity * 0.7)  # 减少到70%（从50%提高）

        if action == "buy":
            # 买入多头仓位
            cost = quantity * current_price
            if self.portfolio["cash"] >= cost:
                # 更新加权平均成本基础
                total_long_shares = position["long"] + quantity
                if total_long_shares > 0:
                    total_cost = (position["long"] * position["long_cost_basis"]) + cost
                    position["long_cost_basis"] = total_cost / total_long_shares

                position["long"] += quantity
                self.portfolio["cash"] -= cost

                # 记录到持仓管理器
                from src.agents.position_manager import position_manager
                entry_reason = "中期技术分析买入"  # 可以根据实际信号调整
                position_manager.add_position(ticker, quantity, current_price,
                                            self.current_date_str, entry_reason)

                return quantity
            else:
                print(f"{get_text('Insufficient funds to buy', 'backtester')} {quantity} {get_text('shares of', 'backtester')} {ticker}")
                return 0

        elif action == "sell":
            # 卖出多头仓位
            if position["long"] >= quantity:
                position["long"] -= quantity
                self.portfolio["cash"] += quantity * current_price

                # 更新持仓管理器
                from src.agents.position_manager import position_manager
                if position["long"] == 0:
                    # 全部卖出，移除持仓记录
                    position_manager.remove_position(ticker)
                    position["long_cost_basis"] = 0.0
                else:
                    # 部分卖出，减少持仓
                    position_manager.remove_position(ticker, quantity)

                return quantity
            else:
                print(f"{get_text('Insufficient long position to sell', 'backtester')} {quantity} {get_text('shares of', 'backtester')} {ticker}")
                return 0

        elif action == "short":
            # 做空
            margin_required = quantity * current_price * position["margin_requirement"]
            short_proceeds = quantity * current_price

            if self.portfolio["cash"] >= margin_required:
                # 更新加权平均成本基础
                total_short_shares = position["short"] + quantity
                if total_short_shares > 0:
                    total_cost = (position["short"] * position["short_cost_basis"]) + (quantity * current_price)
                    position["short_cost_basis"] = total_cost / total_short_shares

                position["short"] += quantity

                # 正确的融券会计处理：
                # 1. 获得做空收入（现金增加）
                # 2. 扣除保证金（现金减少）
                # 3. 记录融券负债（在净资产计算中体现）
                self.portfolio["cash"] += short_proceeds - margin_required

                # 记录做空负债（用于准确计算投资组合价值）
                if "short_proceeds" not in self.portfolio:
                    self.portfolio["short_proceeds"] = {}
                if ticker not in self.portfolio["short_proceeds"]:
                    self.portfolio["short_proceeds"][ticker] = 0
                self.portfolio["short_proceeds"][ticker] += short_proceeds

                return quantity
            else:
                print(f"{get_text('Insufficient margin to short', 'backtester')} {quantity} {get_text('shares of', 'backtester')} {ticker}")
                return 0

        elif action == "cover":
            # 平仓空头仓位
            if position["short"] >= quantity:
                cover_cost = quantity * current_price

                # 计算需要归还的保证金
                margin_to_return = quantity * position.get("short_cost_basis", current_price) * position["margin_requirement"]

                if self.portfolio["cash"] >= cover_cost:
                    position["short"] -= quantity

                    # 正确的平仓会计处理：
                    # 1. 支付买入成本（现金减少）
                    # 2. 归还保证金（现金增加）
                    # 3. 减少融券负债
                    self.portfolio["cash"] -= cover_cost
                    self.portfolio["cash"] += margin_to_return

                    # 减少做空负债
                    if "short_proceeds" in self.portfolio and ticker in self.portfolio["short_proceeds"]:
                        # 按比例减少做空负债
                        original_short_shares = position["short"] + quantity
                        proceeds_to_reduce = (quantity / original_short_shares) * self.portfolio["short_proceeds"][ticker]
                        self.portfolio["short_proceeds"][ticker] -= proceeds_to_reduce

                        # 如果全部平仓，清零负债记录
                        if position["short"] == 0:
                            self.portfolio["short_proceeds"][ticker] = 0

                    # 如果全部平仓，重置成本基础
                    if position["short"] == 0:
                        position["short_cost_basis"] = 0.0

                    return quantity
                else:
                    print(f"{get_text('Insufficient funds to cover', 'backtester')} {quantity} {get_text('shares of', 'backtester')} {ticker}")
                    return 0
            else:
                print(f"{get_text('Insufficient short position to cover', 'backtester')} {quantity} {get_text('shares of', 'backtester')} {ticker}")
                return 0

        return 0

    def calculate_portfolio_value(self, current_prices):
        """
        计算总投资组合价值（净资产价值），包括：
          - 现金（已扣除保证金占用）
          - 多头仓位的市场价值
          - 减去空头仓位的当前负债
        """
        # 基础现金（已扣除保证金）
        total_value = self.portfolio["cash"]

        for ticker, position in self.portfolio["positions"].items():
            price = current_prices.get(ticker, 0)
            if price == 0:
                continue

            # 多头仓位市场价值
            if position["long"] > 0:
                total_value += position["long"] * price

            # 空头仓位：减去当前需要归还的股票价值（负债）
            if position["short"] > 0:
                # 当前做空负债 = 需要归还的股票数量 × 当前价格
                current_short_liability = position["short"] * price
                total_value -= current_short_liability

        return total_value

    def calculate_ticker_liabilities(self, ticker, current_price):
        """
        计算单个股票的负债总额，包括：
          - 做空负债（需要归还的股票市值）
          - 保证金占用
        """
        position = self.portfolio["positions"].get(ticker, {})
        liabilities = 0.0

        if position.get("short", 0) > 0:
            # 做空负债：当前需要归还的股票市值
            short_liability = position["short"] * current_price
            liabilities += short_liability

            # 保证金占用（如果有的话，这部分已经从现金中扣除）
            # 注意：保证金不是负债，而是被占用的资金，这里我们主要关注做空负债

        return liabilities

    def calculate_total_liabilities(self, current_prices):
        """
        计算投资组合总负债
        """
        total_liabilities = 0.0

        for ticker, position in self.portfolio["positions"].items():
            if ticker in current_prices and position.get("short", 0) > 0:
                current_price = current_prices[ticker]
                ticker_liabilities = self.calculate_ticker_liabilities(ticker, current_price)
                total_liabilities += ticker_liabilities

        return total_liabilities

    def prefetch_data(self):
        """预取回测期间所需的所有数据。"""
        print(f"\n{get_text('Pre-fetching data for the entire backtest period...', 'backtester')}")

        # 将结束日期字符串转换为datetime，获取前1年的数据
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        for ticker in self.tickers:
            # 获取整个期间的价格数据，加上1年
            get_prices(ticker, start_date_str, self.end_date)

            # 获取财务指标
            get_financial_metrics(ticker, self.end_date, limit=10)

            # 获取内部交易
            get_insider_trades(ticker, self.end_date, start_date=self.start_date, limit=1000)

            # 获取公司新闻
            get_company_news(ticker, self.end_date, start_date=self.start_date, limit=1000)

        print(get_text("Data pre-fetch complete.", 'backtester'))

        # 获取市场基准数据
        self._fetch_market_benchmark()

    def _get_trading_days(self, start_date: str, end_date: str) -> list:
        """
        生成交易日列表，排除周末和美国联邦假日。

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            list: datetime对象列表，包含所有交易日
        """
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        trading_days = []
        current_date = start_dt

        while current_date <= end_dt:
            # 检查是否为周末 (周六=5, 周日=6)
            if current_date.weekday() < 5:  # 周一到周五
                # 检查是否为美国联邦假日
                date_str = current_date.strftime("%Y-%m-%d")
                if date_str not in self.us_holidays_2025:
                    trading_days.append(current_date)

            current_date += timedelta(days=1)

        return trading_days

    def _fetch_market_benchmark(self):
        """获取市场基准数据（标普500指数）"""
        try:
            print(f"\n{get_text('Fetching market benchmark data (S&P 500)...', 'backtester')}")

            # 使用AKShare获取市场基准数据
            benchmark_data = get_market_index_data(self.start_date, self.end_date)

            if benchmark_data and benchmark_data.get('market_performance'):
                # 提取标普500数据
                spx_data = benchmark_data['market_performance'].get('SPX')
                if spx_data:
                    self.market_benchmark = {
                        'name': spx_data['name'],
                        'return': spx_data['return'],
                        'start_price': spx_data['start_price'],
                        'end_price': spx_data['end_price'],
                        'data_points': spx_data['data_points']
                    }

                    # 如果有原始数据，构建基准价值序列
                    if 'raw_data' in benchmark_data.get('indices', {}).get('SPX', {}):
                        raw_data = benchmark_data['indices']['SPX']['raw_data']
                        start_value = 100000  # 标准化起始值

                        for data_point in raw_data:
                            date_str = data_point['date']
                            close_price = data_point['close']

                            # 计算相对于起始价格的收益率
                            if self.market_benchmark['start_price'] > 0:
                                return_rate = (close_price / self.market_benchmark['start_price']) - 1
                                benchmark_value = start_value * (1 + return_rate)

                                self.benchmark_values.append({
                                    'Date': datetime.strptime(date_str, '%Y-%m-%d'),
                                    'Benchmark_Value': benchmark_value
                                })

                    print(f"{get_text('Market benchmark data loaded successfully.', 'backtester')}")
                    print(f"{get_text('Benchmark period return:', 'backtester')} {self.market_benchmark['return']:.2%}")
                else:
                    print(f"{get_text('Warning: S&P 500 data not found in benchmark data.', 'backtester')}")
            else:
                print(f"{get_text('Warning: Could not fetch market benchmark data.', 'backtester')}")

        except Exception as e:
            print(f"{get_text('Error fetching market benchmark data:', 'backtester')} {e}")
            self.market_benchmark = None

    def run_backtest(self):
        # 设置信号处理器和控制线程
        self._setup_signal_handlers()
        self._start_control_thread()

        # 在开始时预取所有数据
        self.prefetch_data()

        # 生成交易日，排除周末和美国联邦假日
        dates = self._get_trading_days(self.start_date, self.end_date)
        table_rows = []
        performance_metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None, "long_short_ratio": None, "gross_exposure": None, "net_exposure": None}

        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{get_text('🚀 Starting backtest...', 'backtester')}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        control_msg = "You can control the backtest using commands: 'p' (pause), 'r' (resume), 'q' (quit), 's' (status), 'h' (help)"
        print(f"{Fore.CYAN}💡 {get_text(control_msg, 'backtester')}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}💡 {get_text('Remember: Your input may not be visible, but commands will work when you press Enter!', 'backtester')}{Style.RESET_ALL}")

        # 用初始资金初始化投资组合价值列表
        if len(dates) > 0:
            self.portfolio_values = [{"Date": dates[0], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []

        try:
            for current_date in dates:
                # 检查终止请求
                if self.should_terminate:
                    print(f"\n{Fore.RED}{get_text('🛑 Backtest terminated by user request.', 'backtester')}{Style.RESET_ALL}")
                    break

                # 如果暂停则等待
                self._wait_if_paused()

                # 在可能的暂停后再次检查
                if self.should_terminate:
                    print(f"\n{Fore.RED}{get_text('🛑 Backtest terminated by user request.', 'backtester')}{Style.RESET_ALL}")
                    break

                current_date_str = current_date.strftime("%Y-%m-%d")
                self.current_date_str = current_date_str

                try:
                    # 获取所有股票代码的当前价格
                    current_prices = {}
                    missing_data = False
                    for ticker in self.tickers:
                        try:
                            # 获取当前日期的价格数据，使用当前日期作为开始和结束日期
                            price_data = get_price_data(ticker, current_date_str, current_date_str)
                            if not price_data.empty:
                                current_prices[ticker] = price_data.iloc[-1]["close"]
                            else:
                                missing_data = True
                                break
                        except Exception as e:
                            print(f"{get_text('Error fetching price for ', 'backtester')}{ticker}: {e}")
                            missing_data = True
                            break

                    if missing_data:
                        skip_msg = f"{get_text('Skipping trading day ', 'backtester')}{current_date_str}{get_text(' due to missing price data', 'backtester')}"
                        print(skip_msg)
                        continue

                    # 存储当前价格用于状态显示
                    self.current_prices = current_prices

                    # 在执行交易前检查终止/暂停
                    if self.should_terminate:
                        print(f"\n{Fore.RED}{get_text('🛑 Backtest terminated by user request.', 'backtester')}{Style.RESET_ALL}")
                        break

                    self._wait_if_paused()

                    if self.should_terminate:
                        print(f"\n{Fore.RED}{get_text('🛑 Backtest terminated by user request.', 'backtester')}{Style.RESET_ALL}")
                        break
                except Exception as e:
                    # 如果有一般的API错误，记录并跳过这一天
                    error_msg = f"{get_text('Error fetching prices for ', 'backtester')}{current_date_str}{get_text(': ', 'backtester')}{e}"
                    print(error_msg)
                    continue

                # 开始AI分析的进度跟踪
                progress.start()
                progress.update_status("system", None, "分析中...")

                # 调用对冲基金代理
                output = self.agent(
                    tickers=self.tickers,
                    portfolio=self.portfolio,
                    start_date=self.start_date,
                    end_date=current_date_str,
                    show_reasoning=self.show_reasoning,
                    selected_analysts=self.selected_analysts,
                    model_name=self.model_name,
                    model_provider=self.model_provider,
                )

                decisions = output["decisions"]
                analyst_signals = output["analyst_signals"]

                # 记录AI分析师思考过程
                self._record_analyst_reasoning(current_date_str, analyst_signals, current_prices)

                # 标记AI分析已完成
                progress.update_status("system", None, "完成")

                # 为每个股票代码执行交易
                executed_trades = {}
                for ticker in self.tickers:
                    decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                    action, quantity = decision.get("action", "hold"), decision.get("quantity", 0)

                    executed_quantity = self.execute_trade(ticker, action, quantity, current_prices[ticker])
                    executed_trades[ticker] = executed_quantity

                # 交易后计算投资组合价值
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_values.append({"Date": current_date, "Portfolio Value": portfolio_value})

                # 创建用于显示的表格行
                date_rows = []
                for ticker in self.tickers:
                    decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                    action = decision.get("action", "hold")
                    quantity = decision.get("quantity", 0)
                    executed_quantity = executed_trades.get(ticker, 0)
                    current_price = current_prices[ticker]
                    position = self.portfolio["positions"][ticker]

                    # 获取此股票代码的分析师信号 - 修复数据结构访问
                    # analyst_signals结构: {agent_id: {ticker: {signal, confidence, reasoning}}}
                    ticker_signals = {}
                    for agent_id, agent_signals in analyst_signals.items():
                        if isinstance(agent_signals, dict) and ticker in agent_signals:
                            ticker_signals[agent_id] = agent_signals[ticker]

                    # 计算分析师信号 - 修正逻辑
                    bullish_count = sum(1 for signal_data in ticker_signals.values()
                                      if isinstance(signal_data, dict) and signal_data.get("signal", "").lower() == "bullish")
                    bearish_count = sum(1 for signal_data in ticker_signals.values()
                                      if isinstance(signal_data, dict) and signal_data.get("signal", "").lower() == "bearish")
                    neutral_count = sum(1 for signal_data in ticker_signals.values()
                                      if isinstance(signal_data, dict) and signal_data.get("signal", "").lower() == "neutral")

                    # 计算净股份和仓位价值
                    net_shares = position["long"] - position["short"]

                    # 计算仓位价值：多头价值 + 空头未实现损益
                    long_value = position["long"] * current_price
                    short_unrealized_pnl = 0
                    if position["short"] > 0:
                        short_unrealized_pnl = position["short"] * (position["short_cost_basis"] - current_price)
                    position_value = long_value + short_unrealized_pnl

                    # 计算该股票的负债
                    ticker_liabilities = self.calculate_ticker_liabilities(ticker, current_price)

                    row = format_backtest_row(
                        date=current_date_str,
                        ticker=ticker,
                        action=action.upper(),
                        quantity=executed_quantity,  # 使用执行数量而不是计划数量
                        price=current_price,
                        shares_owned=net_shares,
                        position_value=position_value,
                        liabilities=ticker_liabilities,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                    date_rows.append(row)

                table_rows.extend(date_rows)

                # 添加显示现金和总资产的每日汇总行
                total_value = self.calculate_portfolio_value(current_prices)
                portfolio_return = (total_value / self.initial_capital - 1) * 100
                total_liabilities = self.calculate_total_liabilities(current_prices)

                summary_row = format_backtest_row(
                    date=current_date_str,
                    ticker="",
                    action="",
                    quantity=0,
                    price=0,
                    shares_owned=0,
                    position_value=0,
                    liabilities=0,  # 摘要行不显示单个股票负债
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_value - self.portfolio["cash"],
                    total_liabilities=total_liabilities,
                    sharpe_ratio=performance_metrics.get("sharpe_ratio"),
                    sortino_ratio=performance_metrics.get("sortino_ratio"),
                    max_drawdown=performance_metrics.get("max_drawdown"),
                )
                table_rows.append(summary_row)
                # 如果AI分析正在进行，不要清除屏幕以保持表格可见
                clear_screen = not progress.is_analysis_in_progress()

                # 在AI分析期间限制行数以防止表格过长
                max_rows = 10 if not clear_screen else None
                print_backtest_results(table_rows, clear_screen=clear_screen, max_rows=max_rows)

                # 如果有足够的数据，更新性能指标
                if len(self.portfolio_values) > 3:
                    self._update_performance_metrics(performance_metrics)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}{get_text('Backtest interrupted by user (Ctrl+C).', 'backtester')}{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}{get_text('Error during backtest: ', 'backtester')}{e}{Style.RESET_ALL}")
            raise
        finally:
            # 清理控制线程
            self.should_terminate = True
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=2)

            # 最终状态消息
            if not self.should_terminate:
                print(f"\n{Fore.GREEN}{get_text('✅ Backtest completed successfully!', 'backtester')}{Style.RESET_ALL}")

            # 存储最终性能指标以供analyze_performance参考
            self.performance_metrics = performance_metrics

        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """基于当前投资组合价值更新性能指标。"""
        if len(self.portfolio_values) < 2:
            return

        # 转换为DataFrame以便于计算
        df = pd.DataFrame(self.portfolio_values)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # 计算每日收益
        df['Daily_Return'] = df['Portfolio Value'].pct_change()

        # 移除NaN值
        returns = df['Daily_Return'].dropna()

        if len(returns) < 2:
            return

        # 计算夏普比率（假设每年252个交易日）
        # Note: Using 0% risk-free rate as default, but this should be configurable
        risk_free_rate = 0.0  # Annual risk-free rate (should be configurable)
        daily_risk_free_rate = risk_free_rate / 252

        if returns.std() != 0:
            excess_returns = returns - daily_risk_free_rate
            performance_metrics["sharpe_ratio"] = (excess_returns.mean() / returns.std()) * np.sqrt(252)

        # 计算索提诺比率（下行偏差）
        # Use the same risk-free rate for consistency
        downside_returns = returns[returns < daily_risk_free_rate]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            excess_returns_mean = excess_returns.mean()
            performance_metrics["sortino_ratio"] = (excess_returns_mean / downside_returns.std()) * np.sqrt(252)

        # 计算最大回撤
        df['Cumulative_Max'] = df['Portfolio Value'].cummax()
        df['Drawdown'] = (df['Portfolio Value'] - df['Cumulative_Max']) / df['Cumulative_Max']
        performance_metrics["max_drawdown"] = df['Drawdown'].min()

    def _record_analyst_reasoning(self, date: str, analyst_signals: dict, current_prices: dict):
        """记录AI分析师的思考过程"""
        try:
            # 开始新的决策点记录
            self.reasoning_tracker.start_decision_point(
                date=date,
                tickers=self.tickers,
                portfolio_state=self.portfolio,
                market_data=current_prices
            )

            # 遍历所有分析师的信号
            for agent_id, agent_signals in analyst_signals.items():
                # 获取分析师的中文名称
                agent_name = get_chinese_analyst_name(agent_id) if "_agent" in agent_id else agent_id

                # 遍历每个股票的分析
                if isinstance(agent_signals, dict):
                    for ticker, signal_data in agent_signals.items():
                        if isinstance(signal_data, dict) and ticker in self.tickers:
                            # 提取思考过程数据
                            signal = signal_data.get("signal", "neutral")
                            confidence = signal_data.get("confidence", 0.0)
                            reasoning = signal_data.get("reasoning", "无推理信息")

                            # 收集额外数据（如果有的话）
                            additional_data = {}
                            for key, value in signal_data.items():
                                if key not in ["signal", "confidence", "reasoning"]:
                                    additional_data[key] = value

                            # 添加到追踪器
                            self.reasoning_tracker.add_analyst_reasoning(
                                agent_id=agent_id,
                                agent_name=agent_name,
                                ticker=ticker,
                                signal=signal,
                                confidence=confidence,
                                reasoning=reasoning,
                                additional_data=additional_data if additional_data else None
                            )

            # 完成决策点记录
            self.reasoning_tracker.finish_decision_point()

        except Exception as e:
            print(f"{Fore.YELLOW}警告: 记录分析师思考过程时出错: {e}{Style.RESET_ALL}")

    def analyze_performance(self):
        """分析并显示性能指标。"""
        if not hasattr(self, 'portfolio_values') or len(self.portfolio_values) < 2:
            print(get_text("Insufficient data for performance analysis.", "backtester"))
            return None

        # 转换为DataFrame
        df = pd.DataFrame(self.portfolio_values)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # 计算指标
        initial_value = df['Portfolio Value'].iloc[0]
        final_value = df['Portfolio Value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100

        # 计算每日收益
        df['Daily_Return'] = df['Portfolio Value'].pct_change()
        returns = df['Daily_Return'].dropna()

        # 性能指标
        metrics = {}
        if len(returns) > 0:
            metrics[get_text('Total Return (%)', 'backtester')] = total_return

            # 修正年化收益率计算 - 使用交易日数而不是数据点数
            trading_days = len(returns)
            if trading_days > 0:
                annualized_return = ((final_value / initial_value) ** (252 / trading_days) - 1) * 100
                metrics[get_text('Annualized Return (%)', 'backtester')] = annualized_return

            metrics[get_text('Volatility (%)', 'backtester')] = returns.std() * np.sqrt(252) * 100

            # 修正夏普比率计算 - 使用无风险利率
            risk_free_rate = 0.0  # 年化无风险利率，应该可配置
            daily_risk_free_rate = risk_free_rate / 252

            if returns.std() != 0:
                excess_returns = returns - daily_risk_free_rate
                sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)
                metrics[get_text('Sharpe Ratio', 'backtester')] = sharpe_ratio

            # 修正索提诺比率计算
            downside_returns = returns[returns < daily_risk_free_rate]
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                excess_returns_mean = (returns - daily_risk_free_rate).mean()
                sortino_ratio = (excess_returns_mean / downside_returns.std()) * np.sqrt(252)
                metrics[get_text('Sortino Ratio', 'backtester')] = sortino_ratio

            # 最大回撤
            df['Cumulative_Max'] = df['Portfolio Value'].cummax()
            df['Drawdown'] = (df['Portfolio Value'] - df['Cumulative_Max']) / df['Cumulative_Max']
            metrics[get_text('Max Drawdown (%)', 'backtester')] = df['Drawdown'].min() * 100

            # 胜率
            positive_returns = returns[returns > 0]
            metrics[get_text('Win Rate (%)', 'backtester')] = (len(positive_returns) / len(returns)) * 100

            # 添加与市场基准的比较
            if self.market_benchmark:
                benchmark_return = self.market_benchmark['return'] * 100
                metrics[get_text('Benchmark Return (S&P 500) (%)', 'backtester')] = benchmark_return

                # 计算Alpha（超额收益）
                alpha = total_return - benchmark_return
                metrics[get_text('Alpha vs S&P 500 (%)', 'backtester')] = alpha

                # 计算Beta（如果有足够的基准数据）
                if len(self.benchmark_values) > 1:
                    beta = self._calculate_beta(df, returns)
                    if beta is not None:
                        metrics[get_text('Beta vs S&P 500', 'backtester')] = beta

        # 显示结果
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{get_text('📊 PERFORMANCE ANALYSIS', 'backtester')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

        # 分组显示指标
        basic_metrics = {}
        risk_metrics = {}
        benchmark_metrics = {}

        for metric, value in metrics.items():
            if 'Benchmark' in metric or 'Alpha' in metric or 'Beta' in metric:
                benchmark_metrics[metric] = value
            elif 'Sharpe' in metric or 'Sortino' in metric or 'Drawdown' in metric or 'Volatility' in metric:
                risk_metrics[metric] = value
            else:
                basic_metrics[metric] = value

        # 显示基本收益指标
        print(f"{Fore.YELLOW}📈 {get_text('Return Metrics:', 'backtester')}{Style.RESET_ALL}")
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                color = Fore.GREEN if value > 0 else Fore.RED
                print(f"  {Fore.WHITE}{metric}: {color}{value:.2f}{Style.RESET_ALL}")
            else:
                print(f"  {Fore.WHITE}{metric}: {Fore.GREEN}{value}{Style.RESET_ALL}")

        # 显示风险指标
        if risk_metrics:
            print(f"\n{Fore.YELLOW}⚠️  {get_text('Risk Metrics:', 'backtester')}{Style.RESET_ALL}")
            for metric, value in risk_metrics.items():
                if isinstance(value, float):
                    print(f"  {Fore.WHITE}{metric}: {Fore.GREEN}{value:.2f}{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.WHITE}{metric}: {Fore.GREEN}{value}{Style.RESET_ALL}")

        # 显示基准比较指标
        if benchmark_metrics:
            print(f"\n{Fore.YELLOW}📊 {get_text('Benchmark Comparison (S&P 500):', 'backtester')}{Style.RESET_ALL}")
            for metric, value in benchmark_metrics.items():
                if isinstance(value, float):
                    if 'Alpha' in metric:
                        color = Fore.GREEN if value > 0 else Fore.RED
                        print(f"  {Fore.WHITE}{metric}: {color}{value:.2f}{Style.RESET_ALL}")
                    else:
                        print(f"  {Fore.WHITE}{metric}: {Fore.GREEN}{value:.2f}{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.WHITE}{metric}: {Fore.GREEN}{value}{Style.RESET_ALL}")

        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

        # 导出AI分析师思考过程
        self._export_analyst_reasoning()

        return df

    def _calculate_beta(self, portfolio_df, portfolio_returns):
        """计算投资组合相对于市场基准的Beta值"""
        try:
            if not self.benchmark_values or len(self.benchmark_values) < 2:
                return None

            # 创建基准数据DataFrame
            benchmark_df = pd.DataFrame(self.benchmark_values)
            benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
            benchmark_df = benchmark_df.sort_values('Date')

            # 计算基准收益率
            benchmark_df['Benchmark_Return'] = benchmark_df['Benchmark_Value'].pct_change()
            benchmark_returns = benchmark_df['Benchmark_Return'].dropna()

            # 确保日期对齐
            portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
            merged_df = pd.merge(
                portfolio_df[['Date', 'Daily_Return']].dropna(),
                benchmark_df[['Date', 'Benchmark_Return']].dropna(),
                on='Date',
                how='inner'
            )

            if len(merged_df) < 2:
                return None

            # 计算Beta = Cov(Portfolio, Market) / Var(Market)
            portfolio_aligned = merged_df['Daily_Return']
            benchmark_aligned = merged_df['Benchmark_Return']

            if benchmark_aligned.var() == 0:
                return None

            beta = portfolio_aligned.cov(benchmark_aligned) / benchmark_aligned.var()
            return beta

        except Exception as e:
            print(f"{get_text('Error calculating beta:', 'backtester')} {e}")
            return None

    def _export_analyst_reasoning(self):
        """导出AI分析师思考过程到文件"""
        try:
            if self.reasoning_tracker.get_decision_points_count() == 0:
                print(f"{Fore.YELLOW}没有AI分析师思考过程数据可导出{Style.RESET_ALL}")
                return

            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 导出增强版详细分析报告
            md_file = f"analyst_reasoning_report_{timestamp}.md"
            if self.reasoning_tracker.export_enhanced_analysis_report(md_file):
                print(f"{Fore.GREEN}✅ AI分析师详细分析报告已导出到: {md_file}{Style.RESET_ALL}")

            # 显示汇总统计
            stats = self.reasoning_tracker.generate_summary_statistics()
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}🧠 AI分析师思考过程统计{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{Fore.WHITE}{key}:{Style.RESET_ALL}")
                    for sub_key, sub_value in value.items():
                        print(f"  {Fore.YELLOW}{sub_key}: {Fore.GREEN}{sub_value}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.WHITE}{key}: {Fore.GREEN}{value}{Style.RESET_ALL}")

            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}导出AI分析师思考过程时出错: {e}{Style.RESET_ALL}")

    def get_reasoning_data_for_database(self) -> List[Dict[str, Any]]:
        """获取用于数据库存储的思考过程数据"""
        reasoning_data = []

        for decision_point in self.reasoning_tracker.decision_points:
            for reasoning in decision_point.analyst_reasonings:
                data = {
                    'date': decision_point.date,
                    'agent_id': reasoning.agent_id,
                    'agent_name': reasoning.agent_name,
                    'ticker': reasoning.ticker,
                    'signal': reasoning.signal,
                    'confidence': reasoning.confidence,
                    'reasoning': reasoning.reasoning,
                    'stock_price': decision_point.market_data.get(reasoning.ticker) if decision_point.market_data else None,
                    'market_data': decision_point.market_data,
                    'additional_data': reasoning.additional_data,
                    'portfolio_cash': decision_point.portfolio_state.get('cash'),
                    'portfolio_positions': decision_point.portfolio_state.get('positions')
                }
                reasoning_data.append(data)

        return reasoning_data


### 4. 运行回测 #####
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=_("Run backtesting simulation"))
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        help=_("Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)"),
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help=_("End date in YYYY-MM-DD format"),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help=_("Start date in YYYY-MM-DD format"),
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help=_("Initial capital amount (default: 100000)"),
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help=_("Margin requirement for short positions (0.0 to 1.0, default: 0.0)"),
    )
    parser.add_argument(
        "--analysts",
        type=str,
        help=_("Comma-separated list of analyst names to include"),
    )
    parser.add_argument(
        "--analysts-all",
        action="store_true",
        help=_("Include all available analysts"),
    )
    parser.add_argument("--ollama", action="store_true", help=_("Use Ollama for local LLM inference"))
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help=_("Show detailed reasoning for trading decisions"),
    )

    args = parser.parse_args()

    # 从逗号分隔的字符串解析股票代码
    tickers = [ticker.strip() for ticker in args.tickers.split(",")] if args.tickers else []

    # 从命令行标志解析分析师
    selected_analysts = None
    model_name = ""
    model_provider = None

    # 检查是否使用命令行参数指定分析师
    if args.analysts_all:
        selected_analysts = [a[1] for a in ANALYST_ORDER]
    elif args.analysts:
        selected_analysts = [a.strip() for a in args.analysts.split(",") if a.strip()]

    # 如果没有通过命令行指定分析师，使用交互式选择流程
    if selected_analysts is None:
        # 使用新的选择流程管理器进行交互式选择
        from src.utils.selection_flow import SelectionFlowManager

        flow_manager = SelectionFlowManager(
            use_ollama=args.ollama,
            title_prefix="AI对冲基金回测系统"
        )

        # 运行选择流程
        selected_analysts, model_name, model_provider = flow_manager.run_flow()

        # 检查用户是否取消了操作
        if selected_analysts is None or model_name is None or model_provider is None:
            print(f"\n\n{_('Operation cancelled. Exiting...')}")
            sys.exit(0)
    else:
        # 如果通过命令行指定了分析师，仍需要选择模型
        if args.ollama:
            # 确保Ollama和模型可用
            ensure_ollama_and_model()

            # 获取第一个可用的Ollama模型
            available_models = [model for model in OLLAMA_LLM_ORDER if get_model_info(model[1], model[2])]
            if available_models:
                model_name = available_models[0][1]  # 获取model_name
                model_provider = ModelProvider.OLLAMA.value  # 使用字符串值
            else:
                print(f"{Fore.RED}{get_text('No available Ollama models. Please install models first.', 'backtester')}{Style.RESET_ALL}")
                sys.exit(1)
        else:
            # 使用默认的OpenAI模型
            model_name = "gpt-4o-mini"
            model_provider = ModelProvider.OPENAI.value  # 使用字符串值

    # 验证必需的参数
    if not tickers:
        print(f"{Fore.RED}{get_text('Error: --tickers parameter is required', 'backtester')}{Style.RESET_ALL}")
        parser.print_help()
        sys.exit(1)

    # 验证日期格式
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print(f"{Fore.RED}{get_text('Error: Dates must be in YYYY-MM-DD format', 'backtester')}{Style.RESET_ALL}")
        sys.exit(1)

    # 验证日期逻辑
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    if start_dt >= end_dt:
        print(f"{Fore.RED}{get_text('Error: Start date must be earlier than end date', 'backtester')}{Style.RESET_ALL}")
        sys.exit(1)

    # 显示配置信息
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{get_text('🚀 AI Hedge Fund Backtesting System', 'backtester')}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{get_text('Tickers:', 'backtester')} {Fore.GREEN}{', '.join(tickers)}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{get_text('Start Date:', 'backtester')} {Fore.GREEN}{args.start_date}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{get_text('End Date:', 'backtester')} {Fore.GREEN}{args.end_date}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{get_text('Initial Capital:', 'backtester')} {Fore.GREEN}${args.initial_capital:,.2f}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{get_text('Margin Requirement:', 'backtester')} {Fore.GREEN}{args.margin_requirement:.1%}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{get_text('Model:', 'backtester')} {Fore.GREEN}{model_name} ({model_provider}){Style.RESET_ALL}")

    # 将分析师键名转换为中文显示名称
    from src.utils.analysts import ANALYST_CONFIG
    chinese_analyst_names = []
    for analyst_key in selected_analysts:
        config = ANALYST_CONFIG.get(analyst_key, {})
        display_name = config.get("display_name", analyst_key.title().replace('_', ' '))
        chinese_analyst_names.append(display_name)

    print(f"{Fore.WHITE}{get_text('Analysts:', 'backtester')} {Fore.GREEN}{', '.join(chinese_analyst_names)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    # 创建并运行回测器
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
        show_reasoning=args.show_reasoning,
    )

    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()

    # 可选：保存结果到文件
    if performance_df is not None:
        output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        performance_df.to_csv(output_file, index=False)
        print(f"\n{Fore.GREEN}{get_text('Results saved to: ', 'backtester')}{output_file}{Style.RESET_ALL}")







