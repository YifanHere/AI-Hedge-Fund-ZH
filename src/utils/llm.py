"""LLM的辅助函数"""

import json
from typing import Callable
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info, ModelProvider
from src.utils.progress import progress
from src.graph.state import AgentState
from src.utils.i18n import _


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory: Callable | None = None,
) -> BaseModel:
    """
    使用重试逻辑进行LLM调用，处理支持JSON和不支持JSON的模型。

    Args:
        prompt: 发送给LLM的提示
        pydantic_model: 用于结构化输出的Pydantic模型类
        agent_name: 代理的可选名称，用于进度更新和模型配置提取
        state: 可选的状态对象，用于提取代理特定的模型配置
        max_retries: 最大重试次数（默认：3）
        default_factory: 失败时创建默认响应的可选工厂函数

    Returns:
        指定Pydantic模型的实例
    """

    # 如果提供了状态和代理名称，提取模型配置
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        # 当没有提供状态或代理名称时使用系统默认值
        model_name = "gpt-4.1"
        model_provider = "OPENAI"

    model_info = get_model_info(model_name, model_provider)
    # 将字符串转换为ModelProvider枚举
    if isinstance(model_provider, str):
        model_provider_enum = ModelProvider(model_provider)
    else:
        model_provider_enum = model_provider
    llm = get_model(model_name, model_provider_enum)

    # 检查模型初始化是否失败
    if llm is None:
        print(f"错误：无法使用提供商'{model_provider}'初始化模型'{model_name}'")
        if default_factory:
            return default_factory()
        return create_default_response(pydantic_model)

    # 对于不支持JSON的模型，我们可以使用结构化输出
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # 使用重试调用LLM
    for attempt in range(max_retries):
        try:
            # 调用LLM
            result = llm.invoke(prompt)

            # 对于不支持JSON的模型，我们需要手动提取和解析JSON
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
                else:
                    # JSON提取失败，尝试下一次
                    print(f"尝试 {attempt + 1} 的JSON提取失败。原始响应：{result.content[:200]}...")
                    continue
            else:
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"{_('Error - retry')} {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"{_('Error in LLM call after')} {max_retries} {_('attempts:')}: {e}")
                # 如果提供了default_factory则使用，否则创建基本默认值
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # 由于上面的重试逻辑，这应该永远不会到达
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """基于模型字段创建安全的默认响应。"""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "分析出错，使用默认值"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # 对于其他类型（如Literal），尝试使用第一个允许的值
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """从markdown格式的响应或直接JSON中提取JSON。"""
    import re

    def fix_json_string_values(json_str: str) -> str:
        """通过仅在字符串值内正确转义控制字符来修复JSON。"""

        def escape_string_content(match):
            """在JSON字符串值内转义控制字符。"""
            quote = match.group(1)  # 开始引号
            content = match.group(2)  # 字符串内容
            closing_quote = match.group(3)  # 结束引号

            # 转义控制字符
            content = content.replace('\\', '\\\\')  # 首先转义反斜杠
            content = content.replace('\n', '\\n')
            content = content.replace('\r', '\\r')
            content = content.replace('\t', '\\t')
            content = content.replace('\b', '\\b')
            content = content.replace('\f', '\\f')
            content = content.replace('"', '\\"')  # 转义引号

            # 处理其他控制字符（ASCII 0-31）
            for i in range(32):
                char = chr(i)
                if char not in ['\n', '\r', '\t', '\b', '\f']:
                    content = content.replace(char, f'\\u{i:04x}')

            return f'{quote}{content}{closing_quote}'

        # 匹配JSON字符串值的模式
        # 此模式匹配："key": "value with potential control chars"
        pattern = r'(")((?:[^"\\]|\\.)*)(")'

        return re.sub(pattern, escape_string_content, json_str)

    try:
        # 首先尝试在markdown代码块中查找JSON
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # 跳过```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                json_text = fix_json_string_values(json_text)
                return json.loads(json_text)

        # 如果没有找到markdown代码块，尝试直接提取JSON
        # 查找以{开始并以}结束的内容
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_text = json_match.group(0).strip()
            json_text = fix_json_string_values(json_text)
            return json.loads(json_text)

    except Exception as e:
        print(f"从响应中提取JSON时出错：{e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    从状态中获取特定代理的模型配置。
    如果代理特定配置不可用，则回退到全局模型配置。
    始终返回有效的model_name和model_provider值。
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # 获取代理特定的模型配置
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # 确保我们有有效的值
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # 回退到全局配置（系统默认值）
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4.1"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENAI"
    
    # 如果需要，将枚举转换为字符串
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider
