from typing_extensions import Annotated, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage


import json
from src.utils.status_messages_zh import get_chinese_analyst_name


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    return {**a, **b}


# 定义代理状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]


def show_agent_reasoning(output, agent_name):
    # 获取中文分析师名称
    chinese_name = get_chinese_analyst_name(agent_name) if "_agent" in agent_name else agent_name
    print(f"\n{'=' * 10} {chinese_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):  # 处理Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # 处理自定义对象
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # 回退到字符串表示

    if isinstance(output, (dict, list)):
        # 将输出转换为JSON可序列化格式
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            # 将字符串解析为JSON并美化打印
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # 如果不是有效的JSON，回退到原始字符串
            print(output)

    print("=" * 48)
