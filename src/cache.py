"""Local CSV caching engine for the multi-asset backtesting pipeline.

Every fetched dataset (execution OHLCV+OI, daily OHLCV, 8h funding rates) is
snapshotted into ``data/`` as per-asset CSVs plus a small JSON sidecar holding
fetch metadata (loaded lookback days, listing date, ``fetched_at``). Re-runs
inside the freshness window (default 24h, override with ``CACHE_MAX_AGE_HOURS``)
reload straight from disk instead of re-hitting the Binance API.

Layout per asset (lookback 3650 days, 4h timeframe):

    data/btc_4h_10y.csv           <- execution frame (OI merged)
    data/btc_4h_10y_daily.csv     <- 1d macro-trend frame
    data/btc_4h_10y_funding.csv   <- 8h funding-rate frame
    data/btc_4h_10y_meta.json     <- fetch metadata for cache validation
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_CACHE_MAX_AGE_HOURS = 24.0


def asset_slug(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'btc'; 'HYPE/USDT' -> 'hype'."""
    base = symbol.split(":")[0].split("/")[0]
    return base.lower().replace("-", "").replace("_", "")


def _duration_label(lookback_days: int) -> str:
    """3650 -> '10y'; 451 -> '451d'; 365 -> '1y'."""
    days = int(lookback_days)
    if days > 0 and days % 365 == 0:
        return f"{days // 365}y"
    return f"{days}d"


def cache_paths(symbol: str, execution_tf: str, lookback_days: int) -> Dict[str, str]:
    """Return the per-asset cache file paths for the requested key."""
    slug = asset_slug(symbol)
    dur = _duration_label(lookback_days)
    stem = os.path.join(DATA_DIR, f"{slug}_{execution_tf}_{dur}")
    return {
        "execution": f"{stem}.csv",
        "daily": f"{stem}_daily.csv",
        "funding": f"{stem}_funding.csv",
        "meta": f"{stem}_meta.json",
    }


def _read_frame(path: str) -> Optional[pd.DataFrame]:
    """Read a cached CSV back into a tz-aware, timestamp-keyed DataFrame."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    if df.empty or "timestamp" not in df.columns:
        return None
    if "datetime" in df.columns:
        df = df.drop(columns=["datetime"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("datetime").sort_index()


def _read_meta(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("Cache meta unreadable (%s): %s", path, exc)
        return None


def is_fresh(meta: Optional[Dict]) -> bool:
    """True when the cached snapshot is within ``CACHE_MAX_AGE_HOURS`` (default 24h)."""
    if not meta or "fetched_at" not in meta:
        return True
    try:
        fetched = _dt.datetime.fromisoformat(str(meta["fetched_at"]))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=_dt.timezone.utc)
        max_age = float(os.getenv("CACHE_MAX_AGE_HOURS",
                                  str(DEFAULT_CACHE_MAX_AGE_HOURS)))
        age_hours = (_dt.datetime.now(_dt.timezone.utc) - fetched).total_seconds() / 3600.0
        return age_hours <= max_age
    except (TypeError, ValueError):
        return True


def load_cached_dataset(symbol: str, execution_tf: str,
                        lookback_days: int) -> Optional[Dict]:
    """Return the cached dataset dict (or None on any miss/staleness)."""
    paths = cache_paths(symbol, execution_tf, lookback_days)
    meta = _read_meta(paths["meta"])
    if not is_fresh(meta):
        logger.info("Cache STALE for %s; refetching from the exchange.", symbol)
        return None

    required = ["execution", "daily"]
    if not all(os.path.isfile(paths[k]) and os.path.getsize(paths[k]) > 0
               for k in required):
        return None

    try:
        data = {
            "execution": _read_frame(paths["execution"]),
            "daily": _read_frame(paths["daily"]),
            "funding": _read_frame(paths["funding"]),
        }
        if (data["execution"] is None or data["execution"].empty
                or data["daily"] is None or data["daily"].empty):
            return None
        data["meta"] = meta or {}
        logger.info("Cache HIT for %s <- %s", symbol, paths["execution"])
        return data
    except Exception as exc:  # noqa: BLE001 - fall back to a live fetch
        logger.warning("Cache read failed for %s (%s); refetching live.",
                       symbol, exc)
        return None


def store_cached_dataset(data: Dict, symbol: str, execution_tf: str,
                         lookback_days: int) -> None:
    """Persist a fetched dataset (frames + meta) to the ``data/`` cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = cache_paths(symbol, execution_tf, lookback_days)

    for key, path in (("execution", paths["execution"]),
                      ("daily", paths["daily"]),
                      ("funding", paths["funding"])):
        frame = data.get(key)
        if frame is None or not hasattr(frame, "to_csv"):
            continue
        # Round-trip: datetime index becomes a column, rebuilt on read.
        frame.reset_index().to_csv(path, index=False)

    meta = dict(data.get("meta") or {})
    meta.setdefault("fetched_at", _dt.datetime.now(_dt.timezone.utc).isoformat())
    with open(paths["meta"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    logger.info("Cached %s -> %s", symbol, DATA_DIR)