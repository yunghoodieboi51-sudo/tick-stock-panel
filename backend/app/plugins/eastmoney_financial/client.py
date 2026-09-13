"""东方财富 F10 主要财务指标 HTTP 客户端。"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import httpx

from app.data_providers.base import DataProviderRequestError

logger = logging.getLogger(__name__)

_ENDPOINT = "https://datacenter.eastmoney.com/securities/api/data/get"
_PAGE_SIZE = 500
_MAX_PAGES = 100
_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 0.5


class EastmoneyFinancialError(DataProviderRequestError):
    """上游请求或响应契约异常。"""


class EastmoneyFinancialClient:
    """可分页查询多个 A 股代码的主要财务指标。"""

    def __init__(self, *, timeout: float = 20.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Accept": "application/json, text/plain, */*",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/138.0.0.0 Safari/537.36"
                ),
            },
        )

    def close(self) -> None:
        self._client.close()

    def fetch_metrics(
        self,
        symbols: Iterable[str],
        *,
        report_start: str,
    ) -> list[dict]:
        """获取指定代码集合从 report_start 起的全部报告期。

        单批任何一页失败时整批抛错, 调用方负责隔离该批; 不会把残缺分页结果
        当成完整响应写入。
        """
        symbol_list = list(symbols)
        if not symbol_list:
            return []
        quoted = ",".join(f'"{symbol}"' for symbol in symbol_list)
        filter_value = f"(SECUCODE in ({quoted}))(REPORT_DATE>='{report_start}')"
        rows: list[dict] = []
        page = 1
        pages = 1
        while page <= pages:
            payload = self._request_page(filter_value, page)
            result = payload.get("result")
            if not isinstance(result, dict):
                raise EastmoneyFinancialError("响应缺少 result")
            page_rows = result.get("data")
            if page_rows is None:
                page_rows = []
            if not isinstance(page_rows, list):
                raise EastmoneyFinancialError("响应 result.data 不是列表")
            if any(not isinstance(row, dict) for row in page_rows):
                raise EastmoneyFinancialError("响应包含非对象财务行")
            rows.extend(page_rows)
            try:
                pages = max(1, int(result.get("pages") or 1))
            except (TypeError, ValueError) as exc:
                raise EastmoneyFinancialError("响应 pages 非法") from exc
            if pages > _MAX_PAGES:
                raise EastmoneyFinancialError(f"响应分页数 {pages} 超过安全上限 {_MAX_PAGES}")
            if not page_rows and pages > 1:
                raise EastmoneyFinancialError(f"第 {page} 页为空, 但响应声明共 {pages} 页")
            if not page_rows:
                break
            page += 1
        return rows

    def _request_page(self, filter_value: str, page: int) -> dict:
        params = {
            "type": "RPT_F10_FINANCE_MAINFINADATA",
            "sty": "APP_F10_MAINFINADATA",
            "quoteColumns": "",
            "filter": filter_value,
            "p": str(page),
            "ps": str(_PAGE_SIZE),
            "sr": "1,-1",
            "st": "SECUCODE,REPORT_DATE",
            "source": "HSF10",
            "client": "PC",
        }
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.get(_ENDPOINT, params=params)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise EastmoneyFinancialError("响应不是 JSON 对象")
                if payload.get("success") is not True:
                    raise EastmoneyFinancialError(str(payload.get("message") or "上游返回失败"))
                return payload
            except (httpx.HTTPError, ValueError, EastmoneyFinancialError) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
        raise EastmoneyFinancialError(f"请求失败: {last_error}") from last_error
