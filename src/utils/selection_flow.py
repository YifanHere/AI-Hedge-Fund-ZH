"""
选择流程管理器 - 支持在分析师选择和LLM模型选择之间进行前进和后退导航
"""

import sys
from typing import List, Optional, Tuple, Any, Dict, Callable
from enum import Enum
from colorama import Fore, Style
import questionary
from questionary import Choice

from src.llm.models import ModelProvider, get_model_info, LLM_ORDER, OLLAMA_LLM_ORDER
from src.utils.analysts import ANALYST_ORDER, ANALYST_CONFIG
from src.utils.ollama import ensure_ollama_and_model
from src.utils.i18n import _


class FlowStep(Enum):
    """流程步骤枚举"""
    ANALYST_SELECTION = "analyst_selection"
    MODEL_SELECTION = "model_selection"
    CONFIRMATION = "confirmation"


class SelectionFlowManager:
    """选择流程管理器，支持前进和后退导航"""
    
    def __init__(self, use_ollama: bool = False, title_prefix: str = "AI对冲基金系统"):
        self.use_ollama = use_ollama
        self.title_prefix = title_prefix
        self.current_step = FlowStep.ANALYST_SELECTION
        self.selected_analysts: Optional[List[str]] = None
        self.selected_model_name: Optional[str] = None
        self.selected_model_provider: Optional[str] = None
        self.flow_history: List[FlowStep] = []
        
    def _show_title(self, title: str):
        """显示标题"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{title:^60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 60}{Style.RESET_ALL}\n")
    
    def _create_back_choice(self) -> Choice:
        """创建返回选项"""
        return Choice(f"← {_('返回上一步')}", value="__BACK__")
    
    def _handle_back_navigation(self) -> bool:
        """处理返回导航，返回True表示需要返回上一步"""
        if self.flow_history:
            self.current_step = self.flow_history.pop()
            return True
        return False
    
    def _analyst_selection_step(self) -> bool:
        """分析师选择步骤，返回True表示继续，False表示退出"""
        self._show_title(f"{self.title_prefix} - 分析师选择")

        # 构建选择项，如果有之前的选择，则预选中相应的项目
        choices = []
        for display, value in ANALYST_ORDER:
            # 如果这个分析师之前被选中，则设置为checked
            checked = self.selected_analysts and value in self.selected_analysts
            choices.append(Choice(display, value=value, checked=checked))

        # 如果有历史记录，添加返回选项
        if self.flow_history:
            choices.insert(0, self._create_back_choice())

        # 设置默认选中的分析师（如果之前已经选择过）
        # 注意：questionary.checkbox的default参数只有在有有效值时才设置
        checkbox_kwargs = {
            "message": _("Select your AI analysts."),
            "choices": choices,
            "instruction": _("操作说明: \n1. 按空格键选择/取消选择分析师。\n2. 按 'a' 键全选/全不选。\n3. 按回车键完成选择。\n4. 选择 '← 返回上一步' 可返回上一个操作。\n"),
            "validate": lambda x: (
                len([item for item in x if item != "__BACK__"]) > 0
                or "__BACK__" in x
                or _("您必须至少选择一个分析师或返回上一步。")
            ),
            "style": questionary.Style([
                ("checkbox-selected", "fg:green bold"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "fg:cyan"),
                ("pointer", "fg:cyan bold"),
                ("instruction", "fg:yellow"),
                ("question", "fg:white bold"),
            ])
        }

        # 注意：不再需要设置default参数，因为我们使用Choice对象的checked属性来预选项目

        result = questionary.checkbox(**checkbox_kwargs).ask()
        
        if not result:
            return False  # 用户中断
            
        if "__BACK__" in result:
            return self._handle_back_navigation()
            
        # 过滤掉返回选项
        selected_analysts = [item for item in result if item != "__BACK__"]
        
        if not selected_analysts:
            print(f"\n{Fore.RED}{_('您必须至少选择一个分析师。')}{Style.RESET_ALL}")
            return self._analyst_selection_step()  # 重新选择
            
        self.selected_analysts = selected_analysts
        
        # 显示选择的分析师
        chinese_names = []
        for choice in selected_analysts:
            config = ANALYST_CONFIG.get(choice, {})
            display_name = config.get("display_name", choice.title().replace('_', ' '))
            chinese_names.append(f"{Fore.GREEN}{display_name}{Style.RESET_ALL}")
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}{_('Selected analysts: ')}{Style.RESET_ALL}{', '.join(chinese_names)}\n")
        
        # 记录当前步骤到历史
        self.flow_history.append(self.current_step)
        self.current_step = FlowStep.MODEL_SELECTION
        return True
    
    def _model_selection_step(self) -> bool:
        """模型选择步骤，返回True表示继续，False表示退出"""
        self._show_title("AI模型选择")
        
        if self.use_ollama:
            return self._ollama_model_selection()
        else:
            return self._cloud_model_selection()
    
    def _ollama_model_selection(self) -> bool:
        """Ollama模型选择"""
        print(f"{Fore.YELLOW}{Style.BRIGHT}🔧 {_('Using Ollama for local LLM inference.')}{Style.RESET_ALL}\n")
        
        # 构建选择项
        choices = [Choice(f"📦 {display}", value=value) for display, value, _ in OLLAMA_LLM_ORDER]
        choices.insert(0, self._create_back_choice())
        
        # 设置默认选中的模型
        default_value = self.selected_model_name if self.selected_model_name else None
        
        model_name = questionary.select(
            _("Select your Ollama model:"),
            choices=choices,
            default=default_value,
            instruction="使用方向键选择，回车确认",  # 中文提示替换 "Use arrow keys"
            style=questionary.Style([
                ("selected", "fg:green bold"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan"),
                ("answer", "fg:green bold"),
                ("question", "fg:white bold"),
                ("instruction", "fg:yellow"),
            ]),
        ).ask()
        
        if not model_name:
            return False  # 用户中断
            
        if model_name == "__BACK__":
            return self._handle_back_navigation()
        
        # 处理自定义模型名称
        if model_name == "-":
            model_name = questionary.text(
                _("Enter the custom model name:"),
                default=self.selected_model_name if self.selected_model_name and self.selected_model_name != "-" else "",
                style=questionary.Style([
                    ("question", "fg:white bold"),
                    ("answer", "fg:green bold"),
                ])
            ).ask()
            if not model_name:
                return False
        
        # 确保Ollama和模型可用
        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}{_('Cannot proceed without Ollama and the selected model.')}{Style.RESET_ALL}")
            return self._model_selection_step()  # 重新选择
        
        self.selected_model_name = model_name
        self.selected_model_provider = ModelProvider.OLLAMA.value
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}✅ {_('Selected')} {Fore.CYAN}Ollama{Style.RESET_ALL} {Fore.WHITE}{Style.BRIGHT}{_('model:')}{Style.RESET_ALL} {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        
        # 记录当前步骤到历史
        self.flow_history.append(self.current_step)
        self.current_step = FlowStep.CONFIRMATION
        return True
    
    def _cloud_model_selection(self) -> bool:
        """云端模型选择"""
        print(f"{Fore.YELLOW}{Style.BRIGHT}☁️  使用云端LLM服务{Style.RESET_ALL}\n")
        
        # 构建选择项
        choices = [Choice(f"🤖 {display}", value=(name, provider)) for display, name, provider in LLM_ORDER]
        choices.insert(0, self._create_back_choice())
        
        # 设置默认选中的模型
        default_value = None
        if self.selected_model_name and self.selected_model_provider:
            default_value = (self.selected_model_name, self.selected_model_provider)
        
        model_choice = questionary.select(
            _("Select your LLM model:"),
            choices=choices,
            default=default_value,
            instruction="使用方向键选择，回车确认",  # 中文提示替换 "Use arrow keys"
            style=questionary.Style([
                ("selected", "fg:green bold"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan"),
                ("answer", "fg:green bold"),
                ("question", "fg:white bold"),
                ("instruction", "fg:yellow"),
            ]),
        ).ask()
        
        if not model_choice:
            return False  # 用户中断
            
        if model_choice == "__BACK__":
            return self._handle_back_navigation()
        
        model_name, model_provider = model_choice
        
        # 处理自定义模型
        model_info = get_model_info(model_name, model_provider)
        if model_info and model_info.is_custom():
            model_name = questionary.text(
                _("Enter the custom model name:"),
                default=self.selected_model_name if self.selected_model_name and not self.selected_model_name.startswith("gpt-") else "",
                style=questionary.Style([
                    ("question", "fg:white bold"),
                    ("answer", "fg:green bold"),
                ])
            ).ask()
            if not model_name:
                return False
        
        self.selected_model_name = model_name
        self.selected_model_provider = model_provider
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}✅ {_('Selected model:')}{Style.RESET_ALL} {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL} {Fore.CYAN}({model_provider}){Style.RESET_ALL}\n")
        
        # 记录当前步骤到历史
        self.flow_history.append(self.current_step)
        self.current_step = FlowStep.CONFIRMATION
        return True
    
    def _confirmation_step(self) -> bool:
        """确认步骤"""
        self._show_title("配置确认")
        
        # 显示当前配置
        print(f"{Fore.WHITE}{Style.BRIGHT}当前配置:{Style.RESET_ALL}")
        
        # 显示选择的分析师
        if self.selected_analysts:
            chinese_names = []
            for choice in self.selected_analysts:
                config = ANALYST_CONFIG.get(choice, {})
                display_name = config.get("display_name", choice.title().replace('_', ' '))
                chinese_names.append(display_name)
            print(f"  {Fore.CYAN}分析师:{Style.RESET_ALL} {', '.join(chinese_names)}")
        
        # 显示选择的模型
        if self.selected_model_name and self.selected_model_provider:
            provider_display = "Ollama" if self.selected_model_provider == ModelProvider.OLLAMA.value else self.selected_model_provider
            print(f"  {Fore.CYAN}模型:{Style.RESET_ALL} {self.selected_model_name} ({provider_display})")
        
        print()
        
        # 确认选项
        choices = [
            Choice(f"✅ {_('确认并继续')}", value="confirm"),
            Choice(f"🔄 {_('修改模型选择')}", value="modify_model"),
            Choice(f"👥 {_('修改分析师选择')}", value="modify_analysts"),
            Choice(f"❌ {_('取消')}", value="cancel"),
        ]

        action = questionary.select(
            _("您希望执行什么操作？"),
            choices=choices,
            instruction="使用方向键选择，回车确认",  # 中文提示替换 "Use arrow keys"
            style=questionary.Style([
                ("selected", "fg:green bold"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan"),
                ("answer", "fg:green bold"),
                ("question", "fg:white bold"),
                ("instruction", "fg:yellow"),
            ]),
        ).ask()
        
        if not action or action == "cancel":
            return False
        elif action == "confirm":
            return True
        elif action == "modify_model":
            self.current_step = FlowStep.MODEL_SELECTION
            return True
        elif action == "modify_analysts":
            self.current_step = FlowStep.ANALYST_SELECTION
            return True
        
        return False
    
    def run_flow(self) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
        """
        运行选择流程
        
        Returns:
            Tuple[Optional[List[str]], Optional[str], Optional[str]]: 
            (selected_analysts, model_name, model_provider)
            如果用户取消，返回 (None, None, None)
        """
        while True:
            try:
                if self.current_step == FlowStep.ANALYST_SELECTION:
                    if not self._analyst_selection_step():
                        print(f"\n\n{_('操作已取消。')}")
                        return None, None, None

                elif self.current_step == FlowStep.MODEL_SELECTION:
                    if not self._model_selection_step():
                        print(f"\n\n{_('操作已取消。')}")
                        return None, None, None
                        
                elif self.current_step == FlowStep.CONFIRMATION:
                    if self._confirmation_step():
                        return self.selected_analysts, self.selected_model_name, self.selected_model_provider
                    # 如果确认步骤返回False但没有退出，说明用户选择了修改，继续循环
                    
            except KeyboardInterrupt:
                print(f"\n\n{_('Interrupt received. Exiting...')}")
                return None, None, None
            except Exception as e:
                print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
                return None, None, None
