# AI对冲基金 - 增强版

本项目基于[ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)进行了部分重构和增强，实现了较完整的中文本地化

⚠️ **重要声明**：本项目还处于初始阶段，仅用于**教育和研究目的**，不适用于真实交易或投资。AI投资决策系统的构建是一项复杂的工程，本项目仍处于探索和完善阶段。

## 🚀 核心特性

### 🧠 **智能体生态系统**
- **16位投资大师智能体**：每位都有独特的投资哲学和完整的中文提示词
- **集成风险引擎**：多维度风险分析，包括系统性风险、宏观基本面、市场广度等
- **增强投资组合管理器**：智能仓位管理、动态风险调整、融券决策
- **实时分析师思考追踪**：完整记录每个智能体的决策过程

### 🌐 **现代化Web应用程序**
- **专业VSCode风格界面**：熟悉的开发工具体验
- **实时流程可视化**：React Flow驱动的智能体工作流展示
- **FastAPI后端**：企业级API架构，支持实时数据流
- **SQLite数据库**：完整的数据持久化和历史记录

### 🔍 **多维度风险分析**
- **系统性风险分析器**：识别市场崩盘和系统性风险
- **宏观基本面分析器**：经济指标和政策影响评估
- **市场广度分析器**：板块轮动和市场健康度分析
- **消息面影响分析器**：新闻事件对市场的影响评估
- **情绪与资金流分析器**：市场情绪和资金流向分析

### 🇨🇳 **完整中文本地化**
- **280+条翻译条目**：覆盖所有用户界面文本
- **专业投资术语**：准确的金融专业术语翻译
- **中文智能体提示词**：每位投资大师都有详细的中文投资哲学
- **智能字符处理**：针对中文字符优化的显示和换行

## 🏗️ 系统架构

### 智能体协同工作流程：

**投资大师智能体（11位）：**
1. **Warren Buffett** - 奥马哈先知，价值投资理念
2. **Charlie Munger** - 多学科思维，心理学偏见检查
3. **Ben Graham** - 价值投资之父，安全边际理念
4. **Aswath Damodaran** - 估值专家，DCF模型大师
5. **Michael Burry** - 逆向投资者，深度价值挖掘
6. **Peter Lynch** - 成长股猎手，"十倍股"理念
7. **Phil Fisher** - 成长投资先驱，"小道消息"研究
8. **Cathie Wood** - 创新投资女王，颠覆性技术
9. **Bill Ackman** - 激进投资者，企业变革推动者
10. **Stanley Druckenmiller** - 宏观交易传奇，不对称机会
11. **Rakesh Jhunjhunwala** - 印度股神，新兴市场专家

**专业分析智能体（6个）：**
12. **估值分析师** - DCF模型、相对估值分析
13. **基本面分析师** - 财务指标、行业分析
14. **技术分析师** - 技术指标、图表形态分析
15. **情绪分析师** - 市场情绪、投资者行为分析
16. **风险管理器** - 风险评估、仓位控制
17. **投资组合管理器** - 最终决策、订单生成

## 🎯 主要改进和新增功能

### 相比原版的重大改进：

#### 🔧 **架构升级**
- **集成风险引擎**：全新的多维度风险评估系统
- **增强回测器**：更精确的历史数据回测和性能分析
- **模块化设计**：更清晰的代码结构和组件分离

#### 🌐 **Web应用程序**
- **React + TypeScript前端**：现代化的用户界面
- **FastAPI后端**：高性能的API服务
- **实时数据流**：Server-Sent Events支持
- **数据库集成**：SQLAlchemy + Alembic数据管理

#### 🧪 **数据处理增强**
- **市场数据爬虫**：自动化数据获取
- **数据验证框架**：确保数据质量和一致性
- **缓存机制**：提高数据访问效率
- **akshare集成**：支持中国市场数据

#### 🔍 **分析能力提升**
- **波动率分析器**：市场波动性评估
- **相关性分析**：资产间相关性监控
- **情绪流分析**：资金流向和市场情绪
- **新闻影响分析**：消息面对市场的影响

## 🚀 运行方式

截至2025年7月，有**两种方式**运行AI对冲基金：

1. **⌨️ 命令行界面** - 专业用户的终端方法
2. **🖥️ Web应用程序** - 推荐的可视化界面

**注意**：该系统模拟交易决策，并不实际进行交易。

## 免责声明

该项目仅用于**教育和研究目的**。

- 不适用于真实交易或投资
- 不提供投资建议或保证
- 创建者对财务损失不承担任何责任
- 请咨询财务顾问进行投资决策
- 过往表现不代表未来结果

使用此软件即表示您同意仅将其用于学习目的。

## 目录

- [如何安装](#如何安装)
- [如何运行](#如何运行)
  - [⌨️ 命令行界面](#️-命令行界面)
  - [🖥️ Web应用程序（新功能！）](#️-web应用程序新功能)
- [贡献](#贡献)
- [功能请求](#功能请求)
- [许可证](#许可证)

## 如何安装

在运行AI对冲基金之前，您需要安装它并设置API密钥。这些步骤对于全栈Web应用程序和命令行界面都是通用的。

### 1. 克隆仓库

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. 设置API密钥

为您的API密钥创建`.env`文件：

```bash
# 创建API密钥的.env文件（在根目录中）
cp .env.example .env
```

打开并编辑`.env`文件以添加您的API密钥：

```bash
# 用于运行由openai托管的LLM（gpt-4o、gpt-4o-mini等）
OPENAI_API_KEY=your-openai-api-key

# 用于运行由groq托管的LLM（deepseek、llama3等）
GROQ_API_KEY=your-groq-api-key

# 用于获取为对冲基金提供动力的金融数据
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

**重要**：您必须设置至少一个LLM API密钥（`OPENAI_API_KEY`、`GROQ_API_KEY`、`ANTHROPIC_API_KEY`或`DEEPSEEK_API_KEY`）才能使对冲基金正常工作。

**金融数据**：AAPL、GOOGL、MSFT、NVDA和TSLA的数据是免费的，不需要API密钥。对于任何其他股票代码，您需要在.env文件中设置`FINANCIAL_DATASETS_API_KEY`。

## 如何运行

### ⌨️ 命令行界面

对于喜欢使用命令行工具的用户，您可以直接通过终端运行AI对冲基金。这种方法提供更精细的控制，对于自动化、脚本编写和集成目的很有用。

![Screenshot 2025-01-06 at 5 50 17 PM](https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b)

选择以下安装方法之一：

#### 使用Poetry

1. 安装Poetry（如果尚未安装）：

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 安装依赖项：

```bash
poetry install
```

#### 使用Docker

1. 确保您的系统上安装了Docker。如果没有，您可以从[Docker官方网站](https://www.docker.com/get-started)下载。

2. 导航到docker目录：

```bash
cd docker
```

3. 构建Docker镜像：

```bash
# 在Linux/Mac上：
./run.sh build

# 在Windows上：
run.bat build
```

#### 运行AI对冲基金（使用Poetry）

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

#### 运行AI对冲基金（使用Docker）

```bash
# 首先导航到docker目录
cd docker

# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA main

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA main
```

您还可以指定`--ollama`标志来使用本地LLM运行AI对冲基金。

```bash
# 使用Poetry：
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama

# 使用Docker（从docker/目录）：
# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA --ollama main

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA --ollama main
```

您还可以指定`--show-reasoning`标志将每个智能体的推理过程打印到控制台。

```bash
# 使用Poetry：
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning

# 使用Docker（从docker/目录）：
# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA --show-reasoning main

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA --show-reasoning main
```

您可以选择指定开始和结束日期，以便为特定时间段做出决策。

```bash
# 使用Poetry：
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# 使用Docker（从docker/目录）：
# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main
```

#### 运行回测器（使用Poetry）

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

#### 运行回测器（使用Docker）

```bash
# 首先导航到docker目录
cd docker

# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA backtest

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA backtest
```

**示例输出：**
![Screenshot 2025-01-06 at 5 47 52 PM](https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47)

您可以选择指定开始和结束日期，以便在特定时间段内进行回测。

```bash
# 使用Poetry：
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# 使用Docker（从docker/目录）：
# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest
```

您还可以指定`--ollama`标志来使用本地LLM运行回测器。

```bash
# 使用Poetry：
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --ollama

# 使用Docker（从docker/目录）：
# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA --ollama backtest

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA --ollama backtest
```

### 🖥️ Web应用程序（新功能！）

运行AI对冲基金的新方式是通过我们的Web应用程序，它提供用户友好的界面。**这是大多数用户的推荐方式，特别是那些喜欢可视化界面而不是命令行工具的用户。**

![Screenshot 2025-06-28 at 6 41 03 PM](https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b)

#### 对于Mac/Linux

```bash
cd app && ./run.sh
```

如果您遇到"权限被拒绝"错误，请先运行：

```bash
cd app && chmod +x run.sh && ./run.sh
```

#### 对于Windows

```bash
# 进入/app目录
cd app

# 运行应用程序
\.run.bat
```

**就是这样！** 这些脚本将：

1. 检查所需的依赖项（Node.js、Python、Poetry）
2. 自动安装所有依赖项
3. 启动前端和后端服务
4. **自动打开您的Web浏览器**到应用程序

#### 详细设置说明

有关详细的设置说明、故障排除和高级配置选项，请参阅：

- [全栈应用程序文档](./app/README.md)
- [前端文档](./app/frontend/README.md)
- [后端文档](./app/backend/README.md)

## 贡献

1. Fork仓库
2. 创建功能分支
3. 提交您的更改
4. 推送到分支
5. 创建Pull Request

**重要**：请保持您的pull request小而专注。这将使审查和合并更容易。

## 功能请求

如果您有功能请求，请打开一个[issue](https://github.com/virattt/ai-hedge-fund/issues)并确保它被标记为`enhancement`。

## 许可证

该项目根据MIT许可证授权 - 有关详细信息，请参阅LICENSE文件。
