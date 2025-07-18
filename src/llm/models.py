import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from enum import Enum
from pydantic import BaseModel
from typing import Tuple, List
from pathlib import Path


class ModelProvider(str, Enum):
    """支持的LLM提供商的枚举"""

    ALIBABA = "Alibaba"
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    GROQ = "Groq"
    META = "Meta"
    MISTRAL = "Mistral"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"


class LLMModel(BaseModel):
    """表示LLM模型配置"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """转换为questionary选择所需的格式"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """检查模型是否为Gemini模型"""
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        """检查模型是否支持JSON模式"""
        if self.is_deepseek() or self.is_gemini():
            return False
        # 只有某些Ollama模型支持JSON模式
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        """检查模型是否为DeepSeek模型"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """检查模型是否为Gemini模型"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """检查模型是否为Ollama模型"""
        return self.provider == ModelProvider.OLLAMA


# 从JSON文件加载模型
def load_models_from_json(json_path: str) -> List[LLMModel]:
    """从JSON文件加载模型"""
    with open(json_path, 'r') as f:
        models_data = json.load(f)

    models = []
    for model_data in models_data:
        # 将字符串提供商转换为ModelProvider枚举
        provider_enum = ModelProvider(model_data["provider"])
        models.append(
            LLMModel(
                display_name=model_data["display_name"],
                model_name=model_data["model_name"],
                provider=provider_enum
            )
        )
    return models


# 获取JSON文件的路径
current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"
ollama_models_json_path = current_dir / "ollama_models.json"

# 从JSON加载可用模型
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# 从JSON加载Ollama模型
OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

# 创建UI期望格式的LLM_ORDER
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# 单独创建Ollama LLM_ORDER
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """根据model_name获取模型信息"""
    all_models = AVAILABLE_MODELS + OLLAMA_MODELS
    return next((model for model in all_models if model.model_name == model_name and model.provider == model_provider), None)


def get_models_list():
    """获取用于API响应的模型列表。"""
    return [
        {
            "display_name": model.display_name,
            "model_name": model.model_name,
            "provider": model.provider.value
        }
        for model in AVAILABLE_MODELS
    ]


def get_model(model_name: str, model_provider: ModelProvider) -> ChatOpenAI | ChatGroq | ChatOllama | None:
    if model_provider == ModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # 向控制台打印错误
            print(f"API密钥错误：请确保在.env文件中设置了GROQ_API_KEY。")
            raise ValueError("未找到Groq API密钥。请确保在.env文件中设置了GROQ_API_KEY。")
        return ChatGroq(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OPENAI:
        # 获取并验证API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if not api_key:
            # 向控制台打印错误
            print(f"API密钥错误：请确保在.env文件中设置了OPENAI_API_KEY。")
            raise ValueError("未找到OpenAI API密钥。请确保在.env文件中设置了OPENAI_API_KEY。")
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"API密钥错误：请确保在.env文件中设置了ANTHROPIC_API_KEY。")
            raise ValueError("未找到Anthropic API密钥。请确保在.env文件中设置了ANTHROPIC_API_KEY。")
        return ChatAnthropic(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"API密钥错误：请确保在.env文件中设置了DEEPSEEK_API_KEY。")
            raise ValueError("未找到DeepSeek API密钥。请确保在.env文件中设置了DEEPSEEK_API_KEY。")
        return ChatDeepSeek(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.GOOGLE:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"API密钥错误：请确保在.env文件中设置了GOOGLE_API_KEY。")
            raise ValueError("未找到Google API密钥。请确保在.env文件中设置了GOOGLE_API_KEY。")
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OLLAMA:
        # 对于Ollama，我们使用基础URL而不是API密钥
        # 检查是否设置了OLLAMA_HOST（用于macOS上的Docker）
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
        )
    else:
        # 处理未知的模型提供商
        print(f"错误：未知的模型提供商'{model_provider}'。支持的提供商：{[p.value for p in ModelProvider]}")
        return None
