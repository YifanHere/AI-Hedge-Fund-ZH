# -*- coding: utf-8 -*-
"""
中文本地化配置文件
包含所有需要翻译的英文文本及其中文对应版本
"""

# 主程序界面文本
MAIN_TEXTS = {
    # 分析师选择
    "Select your AI analysts.": "选择您的AI分析师。",
    "Instructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n": 
        "操作说明：\n1. 按空格键选择/取消选择分析师。\n2. 按 'a' 键全选/全不选。\n3. 完成后按回车键运行对冲基金。\n",
    "You must select at least one analyst.": "您必须至少选择一个分析师。",
    "Selected analysts: ": "已选择的分析师：",
    
    # 模型选择
    "Using Ollama for local LLM inference.": "使用Ollama进行本地LLM推理。",
    "Select your Ollama model:": "选择您的Ollama模型：",
    "Select your LLM model:": "选择您的LLM模型：",
    "Enter the custom model name:": "输入自定义模型名称：",
    "Selected": "已选择",
    "model:": "模型：",
    "Selected model:": "已选择的模型：",

    # 确认流程
    "确认并继续": "确认并继续",
    "修改模型选择": "修改模型选择",
    "修改分析师选择": "修改分析师选择",
    "取消": "取消",
    "您希望执行什么操作？": "您希望执行什么操作？",
    "您必须至少选择一个分析师或返回上一步。": "您必须至少选择一个分析师或返回上一步。",
    "操作说明: \n1. 按空格键选择/取消选择分析师。\n2. 按 'a' 键全选/全不选。\n3. 按回车键完成选择。\n4. 选择 '← 返回上一步' 可返回上一个操作。\n": "操作说明: \n1. 按空格键选择/取消选择分析师。\n2. 按 'a' 键全选/全不选。\n3. 按回车键完成选择。\n4. 选择 '← 返回上一步' 可返回上一个操作。\n",
    
    # 错误和退出消息
    "Interrupt received. Exiting...": "接收到中断信号。正在退出...",
    "Operation cancelled. Exiting...": "操作已取消。正在退出...",
    "Cannot proceed without Ollama and the selected model.": "无法在没有Ollama和所选模型的情况下继续。",
    
    # JSON解析错误
    "JSON decoding error:": "JSON解码错误：",
    "Response:": "响应：",
    "Invalid response type (expected string, got": "无效的响应类型（期望字符串，得到",
    "Unexpected error while parsing response:": "解析响应时发生意外错误：",
    
    # 参数帮助文本
    "Run the hedge fund trading system": "运行对冲基金交易系统",
    "Initial cash position. Defaults to 100000.0)": "初始现金头寸。默认为100000.0",
    "Initial margin requirement. Defaults to 0.0": "初始保证金要求。默认为0.0",
    "Comma-separated list of stock ticker symbols": "以逗号分隔的股票代码列表",
    "Start date (YYYY-MM-DD). Defaults to 3 months before end date": "开始日期(YYYY-MM-DD)。默认为结束日期前3个月",
    "End date (YYYY-MM-DD). Defaults to today": "结束日期(YYYY-MM-DD)。默认为今天",
    "Show reasoning from each agent": "显示每个代理的推理过程",
    "Show the agent graph": "显示代理图",
    "Use Ollama for local LLM inference": "使用Ollama进行本地LLM推理",
    "Start date must be in YYYY-MM-DD format": "开始日期必须为YYYY-MM-DD格式",
    "End date must be in YYYY-MM-DD format": "结束日期必须为YYYY-MM-DD格式",
}

# 回测程序界面文本
BACKTESTER_TEXTS = {
    "Run backtesting simulation": "运行回测模拟",
    "Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)": "以逗号分隔的股票代码列表(例如: AAPL,MSFT,GOOGL)",
    "End date in YYYY-MM-DD format": "结束日期，格式为YYYY-MM-DD",
    "Start date in YYYY-MM-DD format": "开始日期，格式为YYYY-MM-DD",
    "Initial capital amount (default: 100000)": "初始资金金额(默认: 100000)",
    "Use the Space bar to select/unselect analysts.": "使用空格键选择/取消选择分析师。",
    "Press 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.": "按 'a' 键切换全选。\n\n完成后按回车键运行对冲基金。",
    "Use all available analysts (overrides --analysts)": "使用所有可用的分析师(覆盖--analysts参数)",

    # 回测控制命令
    "🎮 Backtest Control Commands:": "🎮 回测控制命令：",
    "Backtest Control Commands:": "回测控制命令：",
    "'p' or 'pause' - Pause the backtest": "'p' 或 'pause' - 暂停回测",
    "'r' or 'resume' - Resume the backtest": "'r' 或 'resume' - 恢复回测",
    "'q' or 'quit' - Terminate the backtest": "'q' 或 'quit' - 终止回测",
    "'s' or 'status' - Show current status": "'s' 或 'status' - 显示当前状态",
    "'h' or 'help' - Show help message": "'h' 或 'help' - 显示帮助信息",
    "📝 Type command and press Enter:": "📝 输入命令并按回车键：",
    "Type command and press Enter:": "输入命令并按回车键：",
    "Tip: Your input may not be visible while backtest is running, but it will work when you press Enter.": "提示：回测运行时您的输入可能不可见，但按回车键后仍会生效。",
    "If you cannot see your typing, just type the command and press Enter - it will work!": "如果您看不到输入的内容，只需输入命令并按回车键即可！",

    # 回测状态消息
    "Pre-fetching data for the entire backtest period...": "正在预取整个回测期间的数据...",
    "Data pre-fetch complete.": "数据预取完成。",
    "Starting backtest...": "开始回测...",
    "🚀 Starting backtest...": "🚀 开始回测...",
    "You can control the backtest using commands: 'p' (pause), 'r' (resume), 'q' (quit), 's' (status)": "您可以使用以下命令控制回测：'p'（暂停）、'r'（恢复）、'q'（退出）、's'（状态）",
    "You can control the backtest using commands: 'p' (pause), 'r' (resume), 'q' (quit), 's' (status), 'h' (help)": "您可以使用以下命令控制回测：'p'（暂停）、'r'（恢复）、'q'（退出）、's'（状态）、'h'（帮助）",
    "Remember: Your input may not be visible, but commands will work when you press Enter!": "请记住：您的输入可能不可见，但按回车键后命令仍会生效！",

    # 回测控制命令帮助
    "📖 Available Commands:": "📖 可用命令：",
    "Pause the backtest": "暂停回测",
    "Resume the backtest": "恢复回测",
    "Terminate the backtest": "终止回测",
    "Show current status": "显示当前状态",
    "Show this help message": "显示此帮助信息",

    # 暂停/恢复消息
    "⏸️  Backtest PAUSED. Type 'r' or 'resume' to continue.": "⏸️  回测已暂停。输入 'r' 或 'resume' 继续。",
    "⚠️  Backtest is already paused.": "⚠️  回测已经暂停。",
    "Backtest is already paused.": "回测已经暂停。",
    "▶️  Backtest RESUMED.": "▶️  回测已恢复。",
    "✅ Backtest is already running.": "✅ 回测已经在运行。",
    "Backtest is already running.": "回测已经在运行。",
    "⏸️  Backtest is paused. Waiting for resume command...": "⏸️  回测已暂停。等待恢复命令...",

    # 终止消息
    "🛑 Terminating backtest...": "🛑 正在终止回测...",
    "🛑 Backtest terminated by user request.": "🛑 回测已根据用户请求终止。",
    "Received termination signal. Gracefully stopping backtest...": "收到终止信号。正在优雅地停止回测...",

    # 状态显示
    "📊 Backtest Status: ": "📊 回测状态：",
    "📅 Current Date: ": "📅 当前日期：",
    "💰 Portfolio Value: ": "💰 投资组合价值：",
    "PAUSED": "已暂停",
    "RUNNING": "运行中",
    "Current Date: ": "当前日期：",
    "Portfolio Value: ": "投资组合价值：",

    # 错误消息
    "❌ Unknown command: ": "❌ 未知命令：",
    "Unknown command: ": "未知命令：",
    ". Use 'p', 'r', 'q', 's', or 'h' for help.": "。请使用 'p'、'r'、'q'、's' 或 'h' 获取帮助。",
    ". Use 'p', 'r', 'q', or 's'.": "。请使用 'p'、'r'、'q' 或 's'。",
    "💡 Type \"h\" or \"help\" to see available commands.": "💡 输入 \"h\" 或 \"help\" 查看可用命令。",
    "Error in control thread: ": "控制线程错误：",
    "Warning: No price data for ": "警告：没有价格数据 ",
    " on ": " 在 ",
    "Error fetching price for ": "获取价格数据错误 ",
    " between ": " 在 ",
    " and ": " 和 ",
    ": ": "：",
    "Skipping trading day ": "跳过交易日 ",
    " due to missing price data": " 由于缺少价格数据",
    "Error fetching prices for ": "获取价格数据错误 ",
    "Backtest interrupted by user (Ctrl+C).": "回测被用户中断 (Ctrl+C)。",
    "Error during backtest: ": "回测过程中出错：",

    # 性能分析相关
    "📊 PERFORMANCE ANALYSIS": "📊 性能分析",
    "PERFORMANCE ANALYSIS": "性能分析",
    "Total Return (%)": "总收益率 (%)",
    "Annualized Return (%)": "年化收益率 (%)",
    "Volatility (%)": "波动率 (%)",
    "Sharpe Ratio": "夏普比率",
    "Sortino Ratio": "索提诺比率",
    "Max Drawdown (%)": "最大回撤 (%)",
    "Win Rate (%)": "胜率 (%)",
    "Insufficient data for performance analysis.": "数据不足，无法进行性能分析。",
    "Results saved to: ": "结果已保存至：",
    "✅ Backtest completed successfully!": "✅ 回测成功完成！",

    # 交易错误消息
    "Insufficient funds to buy": "资金不足，无法买入",
    "Insufficient long position to sell": "多头仓位不足，无法卖出",
    "Insufficient margin to short": "保证金不足，无法做空",
    "Insufficient funds to cover": "资金不足，无法平仓",
    "Insufficient short position to cover": "空头仓位不足，无法平仓",
    "shares of": "股",

    # 系统错误消息
    "No available Ollama models. Please install models first.": "没有可用的Ollama模型。请先安装模型。",
    "Error: --tickers parameter is required": "错误：需要 --tickers 参数",
    "Error: Dates must be in YYYY-MM-DD format": "错误：日期必须为 YYYY-MM-DD 格式",
    "Error: Start date must be earlier than end date": "错误：开始日期必须早于结束日期",

    # 配置信息显示
    "🚀 AI Hedge Fund Backtesting System": "🚀 AI对冲基金回测系统",
    "Tickers:": "股票代码：",
    "Start Date:": "开始日期：",
    "End Date:": "结束日期：",
    "Initial Capital:": "初始资金：",
    "Margin Requirement:": "保证金要求：",
    "Model:": "模型：",
    "Analysts:": "分析师：",
}

# Ollama工具界面文本
OLLAMA_TEXTS = {
    "Please visit https://ollama.com/download to install Ollama manually.": "请访问 https://ollama.com/download 手动安装Ollama。",
    "Ollama for Mac is available as an application download.": "Mac版Ollama可作为应用程序下载。",
    "Would you like to download the Ollama application?": "您想要下载Ollama应用程序吗？",
    "Please download and install the application, then restart this program.": "请下载并安装应用程序，然后重新启动此程序。",
    "After installation, you may need to open the Ollama app once before continuing.": "安装后，您可能需要先打开一次Ollama应用程序才能继续。",
    "Have you installed the Ollama app and opened it at least once?": "您是否已安装Ollama应用程序并至少打开过一次？",
    "Ollama is now properly installed and running!": "Ollama现在已正确安装并运行！",
    "Ollama installation not detected. Please restart this application after installing Ollama.": "未检测到Ollama安装。请在安装Ollama后重新启动此应用程序。",
    "Failed to open browser:": "无法打开浏览器：",
    "Would you like to try the command-line installation instead? (For advanced users)": "您想尝试命令行安装吗？（适用于高级用户）",
    "Attempting command-line installation...": "正在尝试命令行安装...",
    "Ollama installed successfully via command line.": "通过命令行成功安装Ollama。",
    "Command-line installation failed. Please use the app download method instead.": "命令行安装失败。请改用应用程序下载方法。",
    "Error during command-line installation:": "命令行安装过程中出错：",
    "Installing Ollama...": "正在安装Ollama...",
    "Ollama installed successfully.": "Ollama安装成功。",
    "Failed to install Ollama. Error:": "安装Ollama失败。错误：",
    "Error during Ollama installation:": "Ollama安装过程中出错：",
    "Automatic installation on Windows is not supported.": "Windows上不支持自动安装。",
    "Please download and install Ollama from:": "请从以下地址下载并安装Ollama：",
    "Do you want to open the Ollama download page in your browser?": "您想在浏览器中打开Ollama下载页面吗？",
    "After installation, please restart this application.": "安装后，请重新启动此应用程序。",
    "Have you installed Ollama?": "您是否已安装Ollama？",
    "Downloading model": "正在下载模型",
    "This may take a while depending on your internet speed and the model size.": "这可能需要一些时间，具体取决于您的网络速度和模型大小。",
    "The download is happening in the background. Please be patient...": "下载正在后台进行。请耐心等待...",
}

# 显示工具界面文本
DISPLAY_TEXTS = {
    "No trading decisions available": "没有可用的交易决策",
    "Analysis for": "分析",
    "TRADING DECISION:": "交易决策：",
    "Action": "操作",
    "Quantity": "数量",
    "Confidence": "置信度",
    "Reasoning": "推理",
    "Portfolio Strategy:": "投资组合策略：",
    "ANALYST SIGNALS:": "分析师信号：",
    "Signal": "信号",
    "bullish": "看涨",
    "bearish": "看跌", 
    "neutral": "中性",
    "buy": "买入",
    "sell": "卖出",
    "hold": "持有",
}

# 进度显示文本
PROGRESS_TEXTS = {
    "Error - retry": "错误 - 重试",
    "Error in LLM call after": "LLM调用在",
    "attempts:": "次尝试后出错：",
}

# 通用文本
COMMON_TEXTS = {
    "Unknown": "未知",
    "Error": "错误",
    "Success": "成功",
    "Warning": "警告",
    "Info": "信息",
}

def get_text(key: str, category: str = "main") -> str:
    """
    获取本地化文本
    
    Args:
        key: 英文文本键
        category: 文本类别 ("main", "backtester", "ollama", "display", "progress", "common")
    
    Returns:
        对应的中文文本，如果未找到则返回原始键值
    """
    text_maps = {
        "main": MAIN_TEXTS,
        "backtester": BACKTESTER_TEXTS,
        "ollama": OLLAMA_TEXTS,
        "display": DISPLAY_TEXTS,
        "progress": PROGRESS_TEXTS,
        "common": COMMON_TEXTS,
    }
    
    text_map = text_maps.get(category, MAIN_TEXTS)
    return text_map.get(key, key)

def _(key: str) -> str:
    """
    简化的本地化函数，默认从主文本中获取
    """
    # 尝试从所有类别中查找
    for category in ["main", "backtester", "ollama", "display", "progress", "common"]:
        result = get_text(key, category)
        if result != key:  # 找到了翻译
            return result
    return key  # 未找到翻译，返回原文
