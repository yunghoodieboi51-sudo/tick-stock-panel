"""Optional Tushare adapter for historical research master data.

It uses the existing ``httpx`` dependency and reads ``TUSHARE_TOKEN`` only at
construction time.  It is intentionally not registered as a live data source.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime

import httpx

from app.data_providers.historical_master import (
    HistoricalCapability,
    HistoricalCredentialUnavailable,
    HistoricalMasterCapabilities,
    HistoricalSourcePayload,
)

_API_URL = "https://api.tushare.pro"
_STATUSES = ("L", "D", "P", "G", "UN")


class TushareHistoricalMasterProvider:
    name = "tushare"
    capabilities = HistoricalMasterCapabilities(
        historical_lifecycle=HistoricalCapability.PARTIAL,
        delisted_instruments=HistoricalCapability.SUPPORTED,
        historical_st=HistoricalCapability.PARTIAL,
        suspension_status=HistoricalCapability.PARTIAL,
    )

    def __init__(self, token: str | None = None, client: httpx.Client | None = None) -> None:
        self._token = token if token is not None else os.getenv("TUSHARE_TOKEN")
        self._client = client or httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def fetch_instrument_lifecycle(self) -> Sequence[HistoricalSourcePayload]:
        return tuple(self._request("stock_basic", {"list_status": status}) for status in _STATUSES)

    def fetch_historical_st(self, start: date, end: date) -> HistoricalSourcePayload:
        return self._request("stock_st", _date_params(start, end))

    def fetch_suspensions(self, start: date, end: date) -> HistoricalSourcePayload:
        return self._request("suspend_d", _date_params(start, end))

    def _request(self, endpoint: str, params: Mapping[str, object]) -> HistoricalSourcePayload:
        if not self._token:
            raise HistoricalCredentialUnavailable("TUSHARE_TOKEN is not configured")
        response = self._client.post(
            _API_URL,
            json={"api_name": endpoint, "token": self._token, "params": dict(params)},
        )
        response.raise_for_status()
        body = response.json()
        if body.get("code", 0) != 0:
            raise RuntimeError(f"tushare {endpoint} request failed: code={body.get('code')}")
        data = body.get("data") or {}
        fields = data.get("fields") or []
        items = data.get("items") or []
        if not isinstance(fields, list) or not isinstance(items, list):
            raise RuntimeError(f"tushare {endpoint} response has invalid data schema")
        rows = tuple(
            {str(field): value for field, value in zip(fields, item, strict=False)}
            for item in items
            if isinstance(item, list)
        )
        return HistoricalSourcePayload(
            provider=self.name,
            provider_version=None,
            endpoint=endpoint,
            retrieved_at=datetime.now(UTC),
            request_params=dict(params),
            rows=rows,
            source_schema=",".join(str(field) for field in fields),
        )


def _date_params(start: date, end: date) -> dict[str, str]:
    if end < start:
        raise ValueError("end must not precede start")
    return {"start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d")}
