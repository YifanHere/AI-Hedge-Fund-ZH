import datetime
import os
import pandas as pd
import requests
import time

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)
from src.utils.data_validation import validate_and_log

# 全局缓存实例
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict | None = None, max_retries: int = 3) -> requests.Response:
    """
    发起API请求，处理速率限制和适度退避。

    Args:
        url: 请求的URL
        headers: 请求中包含的头部
        method: HTTP方法（GET或POST）
        json_data: POST请求的JSON数据
        max_retries: 最大重试次数（默认：3）

    Returns:
        requests.Response: 响应对象

    Raises:
        Exception: 如果所有重试后请求仍然失败
    """
    last_exception = None

    for attempt in range(max_retries + 1):  # +1用于初始尝试
        try:
            if method.upper() == "POST":
                response = requests.post(url, headers=headers, json=json_data, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 429 and attempt < max_retries:
                # 线性退避：60秒、90秒、120秒、150秒...
                delay = 60 + (30 * attempt)
                print(f"速率限制 (429)。尝试 {attempt + 1}/{max_retries + 1}。等待 {delay}秒后重试...")
                time.sleep(delay)
                continue

            # 返回响应（无论成功、其他错误或最终429）
            return response

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                ConnectionResetError) as e:
            last_exception = e
            if attempt < max_retries:
                # 连接错误的指数退避：2秒、4秒、8秒...
                delay = 2 ** attempt
                print(f"尝试 {attempt + 1}/{max_retries + 1} 时连接错误：{str(e)[:100]}... {delay}秒后重试...")
                time.sleep(delay)
                continue
            else:
                # 最终尝试失败，抛出异常
                print(f"所有 {max_retries + 1} 次尝试都失败了。最后错误：{str(e)}")
                raise e
        except Exception as e:
            # 对于其他异常，不重试
            print(f"不可重试的错误：{str(e)}")
            raise e

    # 这应该永远不会到达，但以防万一
    if last_exception:
        raise last_exception
    else:
        raise Exception("API请求中的意外错误")


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """从缓存或API获取价格数据。"""
    # 创建包含所有参数的缓存键以确保精确匹配
    # Include API endpoint info to avoid confusion with other data types
    cache_key = f"prices_{ticker}_{start_date}_{end_date}_day_1"

    # 首先检查缓存 - 简单的精确匹配
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # 检查是否有有效的API密钥
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if not api_key or api_key == "your-financial-datasets-api-key":
        print(f"No valid API key found for {ticker}")
        return []

    # 如果有有效API密钥，从API获取
    headers = {"X-API-KEY": api_key}
    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"

    try:
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f"API request failed for {ticker} (status: {response.status_code})")
            return []

        # 使用Pydantic模型解析响应
        response_data = response.json()
        price_response = PriceResponse(**response_data)
        prices = price_response.prices

        if not prices:
            print(f"No price data returned from API for {ticker}")
            return []

        # 验证价格数据质量
        if not validate_and_log(prices, 'prices'):
            print(f"Warning: Price data validation failed for {ticker}")

        # 使用综合缓存键缓存结果
        try:
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        except Exception as e:
            print(f"Warning: Failed to cache price data for {ticker}: {e}")

        return prices

    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """从缓存或API获取财务指标。"""
    # 创建包含所有参数的缓存键以确保精确匹配
    # Include data type prefix to avoid confusion
    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"

    # 首先检查缓存 - 简单的精确匹配
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # 检查是否有有效的API密钥
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if not api_key or api_key == "your-financial-datasets-api-key":
        print(f"No valid API key found for {ticker}")
        return []

    # 如果有有效API密钥，从API获取
    headers = {"X-API-KEY": api_key}
    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"

    try:
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f"API request failed for financial metrics {ticker} (status: {response.status_code})")
            return []

        # 使用Pydantic模型解析响应
        response_data = response.json()
        metrics_response = FinancialMetricsResponse(**response_data)
        financial_metrics = metrics_response.financial_metrics

        if not financial_metrics:
            print(f"No financial metrics returned from API for {ticker}")
            return []

        # 验证财务指标数据质量
        if not validate_and_log(financial_metrics, 'financial_metrics'):
            print(f"Warning: Financial metrics validation failed for {ticker}")

        # 使用综合缓存键将结果作为字典缓存
        try:
            _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
        except Exception as e:
            print(f"Warning: Failed to cache financial metrics for {ticker}: {e}")

        return financial_metrics

    except Exception as e:
        print(f"Error fetching financial metrics for {ticker}: {e}")
        return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """从API获取行项目。"""
    # 如果不在缓存中或数据不足，从API获取
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # 缓存结果
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """从缓存或API获取内部交易。"""
    # 创建包含所有参数的缓存键以确保精确匹配
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    # 首先检查缓存 - 简单的精确匹配
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # 检查是否有有效的API密钥
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if not api_key or api_key == "your-financial-datasets-api-key":
        print(f"No valid API key found for {ticker}")
        return []

    # 如果有有效API密钥，从API获取
    headers = {"X-API-KEY": api_key}

    try:
        all_trades = []
        current_end_date = end_date

        while True:
            url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
            if start_date:
                url += f"&filing_date_gte={start_date}"
            url += f"&limit={limit}"

            response = _make_api_request(url, headers)
            if response.status_code != 200:
                print(f"API request failed for insider trades {ticker} (status: {response.status_code})")
                return []

            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades

            if not insider_trades:
                break

            all_trades.extend(insider_trades)

            # 只有在有start_date且获得完整页面时才继续分页
            if not start_date or len(insider_trades) < limit:
                break

            # 将end_date更新为当前批次中最旧的申报日期，用于下一次迭代
            current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

            # 如果我们已经到达或超过了start_date，可以停止
            if current_end_date <= start_date:
                break

        if not all_trades:
            print(f"No insider trades returned from API for {ticker}")
            return []

        # 使用综合缓存键缓存结果
        _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
        return all_trades

    except Exception as e:
        print(f"Error fetching insider trades for {ticker}: {e}")
        return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """从缓存或API获取公司新闻。"""
    # 创建包含所有参数的缓存键以确保精确匹配
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    # 首先检查缓存 - 简单的精确匹配
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # 如果不在缓存中，从API获取
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news

        if not company_news:
            break

        all_news.extend(company_news)

        # 只有在有start_date且获得完整页面时才继续分页
        if not start_date or len(company_news) < limit:
            break

        # 将end_date更新为当前批次中最旧的日期，用于下一次迭代
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # 如果我们已经到达或超过了start_date，可以停止
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # 使用综合缓存键缓存结果
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """从API获取市值。"""
    # 检查end_date是否为今天
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # 从公司事实API获取市值
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """将价格转换为DataFrame。"""
    df = pd.DataFrame([p.model_dump() for p in prices])
    # 确保日期时间索引不带时区，避免与不带时区的datetime对象比较时出错
    df["Date"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# 更新get_price_data函数以使用新函数
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)



