"""
市场数据爬虫模块
用于获取SPY、QQQ等指数的真实价格数据，作为API失败时的备用数据源
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf
from bs4 import BeautifulSoup
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataScraper:
    """市场数据爬虫类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_yahoo_finance_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        使用yfinance库获取真实数据（最可靠的方法）
        """
        try:
            logger.info(f"Fetching {symbol} data from Yahoo Finance using yfinance")

            # 使用yfinance获取数据，添加重试机制
            import time
            for attempt in range(3):
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)

                    if not hist.empty:
                        break
                    else:
                        logger.warning(f"Attempt {attempt + 1}: No data returned for {symbol}")
                        if attempt < 2:
                            time.sleep(2)  # 等待2秒后重试
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        raise e

            if hist.empty:
                logger.warning(f"No data returned for {symbol} after all attempts")
                return None

            # 转换为我们需要的格式
            data = []
            for date, row in hist.iterrows():
                # 确保所有数据都是有效的
                if pd.isna(row['Close']) or pd.isna(row['Open']):
                    continue

                data.append({
                    'time': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                })

            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data

        except Exception as e:
            logger.error(f"yfinance failed for {symbol}: {e}")
            return None
    
    def get_investing_com_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        从Investing.com获取数据（公开数据源）
        """
        try:
            logger.info(f"Fetching {symbol} data from Investing.com")

            # Investing.com的历史数据页面
            # 这里使用一个简化的方法，实际中可能需要更复杂的解析
            symbol_map = {
                'SPY': 'spdr-s-p-500',
                'QQQ': 'powershares-qqq-trust-series-1',
                'IWM': 'ishares-russell-2000',
                'VTI': 'vanguard-total-stock-market',
                'DIA': 'spdr-dow-jones-industrial-average'
            }

            if symbol not in symbol_map:
                logger.warning(f"Symbol {symbol} not supported by Investing.com scraper")
                return None

            investing_symbol = symbol_map[symbol]
            url = f"https://www.investing.com/etf/{investing_symbol}-historical-data"

            # 添加更多headers来模拟真实浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # 这里应该解析HTML，但为了简化，我们返回None
            # 实际实现需要使用BeautifulSoup解析表格数据
            logger.warning(f"Investing.com scraping not fully implemented for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Investing.com failed for {symbol}: {e}")
            return None

    def get_finnhub_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        使用Finnhub免费API获取真实数据
        """
        try:
            logger.info(f"Fetching {symbol} data from Finnhub")

            # 转换日期为时间戳
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            start_timestamp = int(start_dt.timestamp())
            end_timestamp = int(end_dt.timestamp())

            # Finnhub免费API（无需注册）
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',
                'from': start_timestamp,
                'to': end_timestamp,
                'token': 'demo'  # 使用演示token
            }

            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()

            data_json = response.json()

            if data_json.get('s') != 'ok' or not data_json.get('c'):
                logger.warning(f"No data returned from Finnhub for {symbol}")
                return None

            # 转换数据格式
            data = []
            timestamps = data_json['t']
            opens = data_json['o']
            highs = data_json['h']
            lows = data_json['l']
            closes = data_json['c']
            volumes = data_json['v']

            for i in range(len(timestamps)):
                date_str = datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d')
                data.append({
                    'time': date_str,
                    'open': float(opens[i]),
                    'high': float(highs[i]),
                    'low': float(lows[i]),
                    'close': float(closes[i]),
                    'volume': int(volumes[i])
                })

            # 按日期排序
            data.sort(key=lambda x: x['time'])

            logger.info(f"Successfully fetched {len(data)} data points for {symbol} from Finnhub")
            return data

        except Exception as e:
            logger.error(f"Finnhub failed for {symbol}: {e}")
            return None

    def get_polygon_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        使用Polygon.io免费API获取真实数据
        """
        try:
            logger.info(f"Fetching {symbol} data from Polygon.io")

            # Polygon.io免费API
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {
                'apikey': 'demo'  # 使用演示key
            }

            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()

            data_json = response.json()

            if data_json.get('status') != 'OK' or not data_json.get('results'):
                logger.warning(f"No data returned from Polygon.io for {symbol}")
                return None

            # 转换数据格式
            data = []
            for result in data_json['results']:
                timestamp = result['t'] / 1000  # 转换为秒
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

                data.append({
                    'time': date_str,
                    'open': float(result['o']),
                    'high': float(result['h']),
                    'low': float(result['l']),
                    'close': float(result['c']),
                    'volume': int(result['v'])
                })

            # 按日期排序
            data.sort(key=lambda x: x['time'])

            logger.info(f"Successfully fetched {len(data)} data points for {symbol} from Polygon.io")
            return data

        except Exception as e:
            logger.error(f"Polygon.io failed for {symbol}: {e}")
            return None
    
    def get_marketwatch_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        从MarketWatch网站爬取数据
        """
        try:
            logger.info(f"Scraping {symbol} data from MarketWatch")
            
            # MarketWatch历史数据URL
            url = f"https://www.marketwatch.com/investing/fund/{symbol.lower()}/downloaddatapartial"
            
            # 计算日期范围
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            params = {
                'startdate': start_dt.strftime('%m/%d/%Y'),
                'enddate': end_dt.strftime('%m/%d/%Y'),
                'daterange': 'd30',
                'frequency': 'p1d',
                'csvdownload': 'true',
                'downloadpartial': 'false'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # 解析CSV数据
            lines = response.text.strip().split('\n')
            if len(lines) < 2:
                logger.warning(f"No data returned from MarketWatch for {symbol}")
                return None
            
            # 跳过标题行
            data = []
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        date_str = parts[0].strip()
                        # 转换日期格式
                        date_dt = datetime.strptime(date_str, '%m/%d/%Y')
                        
                        data.append({
                            'time': date_dt.strftime('%Y-%m-%d'),
                            'open': float(parts[1].strip()),
                            'high': float(parts[2].strip()),
                            'low': float(parts[3].strip()),
                            'close': float(parts[4].strip()),
                            'volume': int(parts[5].strip()) if parts[5].strip().isdigit() else 0
                        })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse line: {line}, error: {e}")
                        continue
            
            # 按日期排序
            data.sort(key=lambda x: x['time'])
            
            logger.info(f"Successfully scraped {len(data)} data points for {symbol} from MarketWatch")
            return data
            
        except Exception as e:
            logger.error(f"MarketWatch scraping failed for {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        获取市场数据的主方法，按优先级尝试多个数据源
        """
        logger.info(f"Attempting to fetch market data for {symbol} from {start_date} to {end_date}")
        
        # 数据源优先级：只使用真实数据源
        data_sources = [
            ("Yahoo Finance Web", self.get_yahoo_finance_data),
            ("Finnhub API", self.get_finnhub_data),
            ("Polygon.io API", self.get_polygon_data),
            ("Investing.com", self.get_investing_com_data)
        ]
        
        for source_name, fetch_func in data_sources:
            try:
                logger.info(f"Trying {source_name} for {symbol}")
                data = fetch_func(symbol, start_date, end_date)
                
                if data and len(data) > 0:
                    logger.info(f"Successfully obtained {len(data)} data points from {source_name}")
                    return data
                else:
                    logger.warning(f"{source_name} returned no data for {symbol}")
                    
            except Exception as e:
                logger.error(f"{source_name} failed for {symbol}: {e}")
                continue
            
            # 在尝试之间添加短暂延迟
            time.sleep(1)
        
        logger.error(f"All data sources failed for {symbol}")
        return None


def get_market_data_with_fallback(symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
    """
    获取市场数据的便捷函数
    """
    scraper = MarketDataScraper()
    return scraper.get_market_data(symbol, start_date, end_date)


# 测试函数
if __name__ == "__main__":
    # 测试爬虫功能
    scraper = MarketDataScraper()
    
    # 测试获取SPY数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    print(f"Testing data scraping for SPY from {start_date} to {end_date}")
    data = scraper.get_market_data("SPY", start_date, end_date)
    
    if data:
        print(f"Successfully obtained {len(data)} data points")
        print("Sample data:")
        for i, point in enumerate(data[:3]):
            print(f"  {i+1}: {point}")
    else:
        print("Failed to obtain data")
