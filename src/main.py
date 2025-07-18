import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.ollama import ensure_ollama_and_model
from src.utils.i18n import _

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.utils.visualize import save_graph_as_png
import json

# 从 .env 文件加载环境变量
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    """解析JSON字符串并返回字典。"""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"{_('JSON decoding error:')}: {e}\n{_('Response:')}: {repr(response)}")
        return None
    except TypeError as e:
        print(f"{_('Invalid response type (expected string, got')} {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"{_('Unexpected error while parsing response:')}: {e}\n{_('Response:')}: {repr(response)}")
        return None


##### 运行对冲基金 #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
):
    # 开始进度跟踪
    progress.start()

    try:
        # 如果分析师被自定义，创建新的工作流
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="根据提供的数据做出交易决策。",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "current_date": end_date,  # 添加当前日期用于持仓管理
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # 停止进度跟踪
        progress.stop()


def start(state: AgentState):
    """使用输入消息初始化工作流。"""
    return state


def create_workflow(selected_analysts=None):
    """使用选定的分析师创建工作流。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # 从配置中获取分析师节点
    analyst_nodes = get_analyst_nodes()

    # 如果没有选择分析师，默认使用所有分析师
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # 添加选定的分析师节点
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # 始终添加风险和投资组合管理
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # 将选定的分析师连接到风险管理
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_("Run the hedge fund trading system"))
    parser.add_argument("--initial-cash", type=float, default=100000.0, help=_("Initial cash position. Defaults to 100000.0)"))
    parser.add_argument("--margin-requirement", type=float, default=0.0, help=_("Initial margin requirement. Defaults to 0.0"))
    parser.add_argument("--tickers", type=str, required=True, help=_("Comma-separated list of stock ticker symbols"))
    parser.add_argument(
        "--start-date",
        type=str,
        help=_("Start date (YYYY-MM-DD). Defaults to 3 months before end date"),
    )
    parser.add_argument("--end-date", type=str, help=_("End date (YYYY-MM-DD). Defaults to today"))
    parser.add_argument("--show-reasoning", action="store_true", help=_("Show reasoning from each agent"))
    parser.add_argument("--show-agent-graph", action="store_true", help=_("Show the agent graph"))
    parser.add_argument("--ollama", action="store_true", help=_("Use Ollama for local LLM inference"))

    args = parser.parse_args()

    # 从逗号分隔的字符串解析股票代码
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # 使用新的选择流程管理器
    from src.utils.selection_flow import SelectionFlowManager

    flow_manager = SelectionFlowManager(
        use_ollama=args.ollama,
        title_prefix="AI对冲基金系统"
    )

    # 运行选择流程
    selected_analysts, model_name, model_provider = flow_manager.run_flow()

    # 检查用户是否取消了操作
    if selected_analysts is None or model_name is None or model_provider is None:
        print(f"\n\n{_('Operation cancelled. Exiting...')}")
        sys.exit(0)

    # 使用选定的分析师创建工作流
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = ""
        if selected_analysts is not None:
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            file_path += "graph.png"
        save_graph_as_png(app, file_path)

    # 如果提供了日期，验证日期格式
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(_("Start date must be in YYYY-MM-DD format"))

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(_("End date must be in YYYY-MM-DD format"))

    # 设置开始和结束日期
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # 计算结束日期前3个月
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # 使用现金金额和股票仓位初始化投资组合
    portfolio = {
        "cash": args.initial_cash,  # 初始现金金额
        "margin_requirement": args.margin_requirement,  # 初始保证金要求
        "margin_used": 0.0,  # 所有空头仓位的总保证金使用量
        "positions": {
            ticker: {
                "long": 0,  # 持有的多头股份数量
                "short": 0,  # 持有的空头股份数量
                "long_cost_basis": 0.0,  # 多头仓位的平均成本基础
                "short_cost_basis": 0.0,  # 卖空股票的平均价格
                "short_margin_used": 0.0,  # 该股票空头使用的保证金金额
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # 多头仓位的已实现收益
                "short": 0.0,  # 空头仓位的已实现收益
            }
            for ticker in tickers
        },
    }

    # 运行对冲基金
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )
    print_trading_output(result)
