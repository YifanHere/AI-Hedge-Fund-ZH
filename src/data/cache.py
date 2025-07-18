class Cache:
    """API响应的内存缓存。"""

    def __init__(self):
        self._prices_cache: dict[str, list[dict[str, any]]] = {}
        self._financial_metrics_cache: dict[str, list[dict[str, any]]] = {}
        self._line_items_cache: dict[str, list[dict[str, any]]] = {}
        self._insider_trades_cache: dict[str, list[dict[str, any]]] = {}
        self._company_news_cache: dict[str, list[dict[str, any]]] = {}

    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """合并现有数据和新数据，基于键字段避免重复。"""
        if not existing:
            return new_data.copy() if new_data else []

        if not new_data:
            return existing.copy()

        # 创建现有键的集合以进行O(1)查找，处理None值
        existing_keys = set()
        for item in existing:
            if isinstance(item, dict) and key_field in item and item[key_field] is not None:
                existing_keys.add(item[key_field])

        # 只添加尚不存在的项目，同时处理None值
        merged = existing.copy()
        for item in new_data:
            if (isinstance(item, dict) and
                key_field in item and
                item[key_field] is not None and
                item[key_field] not in existing_keys):
                merged.append(item)
                existing_keys.add(item[key_field])  # 避免在同一批次中重复添加

        return merged

    def get_prices(self, ticker: str) -> list[dict[str, any]] | None:
        """如果可用，获取缓存的价格数据。"""
        return self._prices_cache.get(ticker)

    def set_prices(self, ticker: str, data: list[dict[str, any]]):
        """将新价格数据追加到缓存。"""
        self._prices_cache[ticker] = self._merge_data(self._prices_cache.get(ticker), data, key_field="time")

    def get_financial_metrics(self, ticker: str) -> list[dict[str, any]]:
        """如果可用，获取缓存的财务指标。"""
        return self._financial_metrics_cache.get(ticker)

    def set_financial_metrics(self, ticker: str, data: list[dict[str, any]]):
        """将新财务指标追加到缓存。"""
        self._financial_metrics_cache[ticker] = self._merge_data(self._financial_metrics_cache.get(ticker), data, key_field="report_period")

    def get_line_items(self, ticker: str) -> list[dict[str, any]] | None:
        """如果可用，获取缓存的行项目。"""
        return self._line_items_cache.get(ticker)

    def set_line_items(self, ticker: str, data: list[dict[str, any]]):
        """将新行项目追加到缓存。"""
        self._line_items_cache[ticker] = self._merge_data(self._line_items_cache.get(ticker), data, key_field="report_period")

    def get_insider_trades(self, ticker: str) -> list[dict[str, any]] | None:
        """如果可用，获取缓存的内部交易。"""
        return self._insider_trades_cache.get(ticker)

    def set_insider_trades(self, ticker: str, data: list[dict[str, any]]):
        """将新内部交易追加到缓存。"""
        self._insider_trades_cache[ticker] = self._merge_data(self._insider_trades_cache.get(ticker), data, key_field="filing_date")  # 如果需要，也可以使用transaction_date

    def get_company_news(self, ticker: str) -> list[dict[str, any]] | None:
        """如果可用，获取缓存的公司新闻。"""
        return self._company_news_cache.get(ticker)

    def set_company_news(self, ticker: str, data: list[dict[str, any]]):
        """将新公司新闻追加到缓存。"""
        self._company_news_cache[ticker] = self._merge_data(self._company_news_cache.get(ticker), data, key_field="date")


# 全局缓存实例
_cache = Cache()


def get_cache() -> Cache:
    """获取全局缓存实例。"""
    return _cache
