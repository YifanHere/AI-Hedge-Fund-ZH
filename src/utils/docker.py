"""在Docker环境中使用Ollama模型的工具"""

import requests
import time
from colorama import Fore, Style
import questionary
from .i18n import _

def ensure_ollama_and_model(model_name: str, ollama_url: str) -> bool:
    """确保Ollama模型在Docker环境中可用。"""
    print(f"{Fore.CYAN}检测到Docker环境。{Style.RESET_ALL}")

    # 步骤1：检查Ollama服务是否可用
    if not is_ollama_available(ollama_url):
        return False

    # 步骤2：检查模型是否已经可用
    available_models = get_available_models(ollama_url)
    if model_name in available_models:
        print(f"{Fore.GREEN}模型 {model_name} 在Docker Ollama容器中可用。{Style.RESET_ALL}")
        return True

    # 步骤3：模型不可用 - 询问用户是否要下载
    print(f"{Fore.YELLOW}模型 {model_name} 在Docker Ollama容器中不可用。{Style.RESET_ALL}")

    if not questionary.confirm(f"您想要下载 {model_name} 吗？").ask():
        print(f"{Fore.RED}没有模型无法继续。{Style.RESET_ALL}")
        return False

    # 步骤4：下载模型
    return download_model(model_name, ollama_url)


def is_ollama_available(ollama_url: str) -> bool:
    """检查Ollama服务在Docker环境中是否可用。"""
    try:
        response = requests.get(f"{ollama_url}/api/version", timeout=5)
        if response.status_code == 200:
            return True
            
        print(f"{Fore.RED}无法连接到 {ollama_url} 的Ollama服务。{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}确保Ollama服务在您的Docker环境中运行。{Style.RESET_ALL}")
        return False
    except requests.RequestException as e:
        print(f"{Fore.RED}连接Ollama服务时出错：{e}{Style.RESET_ALL}")
        return False


def get_available_models(ollama_url: str) -> list:
    """获取Docker环境中可用模型的列表。"""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
            
        print(f"{Fore.RED}从Ollama服务获取可用模型失败。状态码：{response.status_code}{Style.RESET_ALL}")
        return []
    except requests.RequestException as e:
        print(f"{Fore.RED}获取可用模型时出错：{e}{Style.RESET_ALL}")
        return []


def download_model(model_name: str, ollama_url: str) -> bool:
    """在Docker环境中下载模型。"""
    print(f"{Fore.YELLOW}正在将模型 {model_name} 下载到Docker Ollama容器...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}这可能需要一些时间。请耐心等待。{Style.RESET_ALL}")
    
    # 步骤1：启动下载
    try:
        response = requests.post(f"{ollama_url}/api/pull", json={"name": model_name}, timeout=10)
        if response.status_code != 200:
            print(f"{Fore.RED}启动模型下载失败。状态码：{response.status_code}{Style.RESET_ALL}")
            if response.text:
                print(f"{Fore.RED}错误：{response.text}{Style.RESET_ALL}")
            return False
    except requests.RequestException as e:
        print(f"{Fore.RED}启动下载请求时出错：{e}{Style.RESET_ALL}")
        return False
    
    # 步骤2：监控下载进度
    print(f"{Fore.CYAN}下载已启动。定期检查完成情况...{Style.RESET_ALL}")

    total_wait_time = 0
    max_wait_time = 1800  # 最多等待30分钟
    check_interval = 10  # 每10秒检查一次
    
    while total_wait_time < max_wait_time:
        # 检查模型是否已下载
        available_models = get_available_models(ollama_url)
        if model_name in available_models:
            print(f"{Fore.GREEN}模型 {model_name} 下载成功。{Style.RESET_ALL}")
            return True
            
        # 再次检查前等待
        time.sleep(check_interval)
        total_wait_time += check_interval
        
        # 每分钟打印一次状态消息
        if total_wait_time % 60 == 0:
            minutes = total_wait_time // 60
            print(f"{Fore.CYAN}下载进行中... (已过去 {minutes} 分钟){Style.RESET_ALL}")
    
    # 如果到达这里，说明超时了
    print(f"{Fore.RED}等待模型下载完成超时，已等待 {max_wait_time // 60} 分钟。{Style.RESET_ALL}")
    return False


def delete_model(model_name: str, ollama_url: str) -> bool:
    """在Docker环境中删除模型。"""
    print(f"{Fore.YELLOW}正在从Docker容器中删除模型 {model_name}...{Style.RESET_ALL}")
    
    try:
        response = requests.delete(f"{ollama_url}/api/delete", json={"name": model_name}, timeout=10)
        if response.status_code == 200:
            print(f"{Fore.GREEN}模型 {model_name} 删除成功。{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}删除模型失败。状态码：{response.status_code}{Style.RESET_ALL}")
            if response.text:
                print(f"{Fore.RED}错误：{response.text}{Style.RESET_ALL}")
            return False
    except requests.RequestException as e:
        print(f"{Fore.RED}删除模型时出错：{e}{Style.RESET_ALL}")
        return False 