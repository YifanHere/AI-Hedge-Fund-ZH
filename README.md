# AI对冲基金

本项目基于[ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)汉化，并添加了一些新的功能，同时对原有的AI智能体决策策略进行了大幅度重写，需要注意的是，本项目仅适用于学习研究，为AI规制科学合理的工作流程是一项繁杂的工作，如今只是初步探索阶段

这是一个AI驱动对冲基金的概念验证项目。该项目的目标是探索使用AI进行交易决策。该项目仅用于**教育**目的，不适用于真实交易或投资

该系统采用多个智能体协同工作：

1. Aswath Damodaran智能体 - 估值学院院长，专注于故事、数字和严格的估值
2. Ben Graham智能体 - 价值投资之父，只购买具有安全边际的隐藏宝石
3. Bill Ackman智能体 - 激进投资者，采取大胆立场并推动变革
4. Cathie Wood智能体 - 成长投资女王，相信创新和颠覆的力量
5. Charlie Munger智能体 - 沃伦·巴菲特的合伙人，只以合理价格购买优秀企业
6. Michael Burry智能体 - 《大空头》逆向投资者，寻找深度价值
7. Peter Lynch智能体 - 实用投资者，在日常企业中寻找"十倍股"
8. Phil Fisher智能体 - 细致的成长投资者，使用深度"小道消息"研究
9. Rakesh Jhunjhunwala智能体 - 印度的大牛
10. Stanley Druckenmiller智能体 - 宏观传奇，寻找具有增长潜力的不对称机会
11. Warren Buffett智能体 - 奥马哈先知，寻找价格合理的优秀公司
12. 估值智能体 - 计算股票内在价值并生成交易信号
13. 情绪智能体 - 分析市场情绪并生成交易信号
14. 基本面智能体 - 分析基本面数据并生成交易信号
15. 技术面智能体 - 分析技术指标并生成交易信号
16. 风险管理器 - 计算风险指标并设置仓位限制
17. 投资组合管理器 - 做出最终交易决策并生成订单

![Screenshot 2025-03-22 at 6 19 07 PM](https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4)

截至2025年6月，有**两种方式**运行AI对冲基金：

1. **⌨️ 命令行界面** - 基于终端的方法
2. **🖥️ Web应用程序（新功能！）** - 用户友好的Web界面

**注意**：该系统模拟交易决策，并不实际进行交易。

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

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
