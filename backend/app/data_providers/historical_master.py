"""Provider contract for offline historical master and trading-status research.

This contract deliberately sits beside, rather than inside, the live daily-data
provider contract.  It is consumed only by the historical research generation
tooling; it must not change the current instruments snapshot or live routes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol


class HistoricalCapability(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class HistoricalMasterCapabilities:
    historical_lifecycle: HistoricalCapability
    delisted_instruments: HistoricalCapability
    historical_st: HistoricalCapability
    suspension_status: HistoricalCapability


@dataclass(frozen=True)
class HistoricalSourcePayload:
    """Unmodified source rows plus non-secret retrieval metadata.

    ``request_params`` must never contain credentials.  A snapshot writer owns
    persistence; adapters only return this value object.
    """

    provider: str
    provider_version: str | None
    endpoint: str
    retrieved_at: datetime
    request_params: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]
    source_schema: str | None = None


class HistoricalMasterProvider(Protocol):
    name: str
    capabilities: HistoricalMasterCapabilities

    def fetch_instrument_lifecycle(self) -> Sequence[HistoricalSourcePayload]: ...

    def fetch_historical_st(self, start: date, end: date) -> HistoricalSourcePayload: ...

    def fetch_suspensions(self, start: date, end: date) -> HistoricalSourcePayload: ...


class HistoricalCredentialUnavailable(RuntimeError):  # noqa: N818
    """Raised before a network request when the optional provider has no token."""
