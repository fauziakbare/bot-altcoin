"""Data-loading layer for Binance USD-M (linear) Perpetual Futures.

Fetches and merges the inputs required by the strategy + backtester:

* OHLCV candles  - execution timeframe (1h/4h) and macro-trend timeframe (1d)
* Open Interest  - historical OI, used to confirm "new money" vs liquidation
* Funding Rates  - 8h settlement timestamps (00:00 / 08:00 / 16:00 UTC)

Credentials are loaded securely from a local ``.env`` file via ``python-dotenv``.
Every endpoint used here is public market data, so API keys are optional; the
client still wires them in so private endpoints work if keys are present.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL = "BTC/USDT:USDT"     # Binance USD-M linear perpetual
EXECUTION_TIMEFRAMES = ("1h", "4h")
DEFAULT_DAILY_TF = "1d"
DEFAULT_LOOKBACK_DAYS = 365

# One day in milliseconds (identical to timeframe_to_ms("1d"), kept explicit
# so the adaptive-lookback math below never depends on a tf-string parse).
DAY_MS = 86_400_000

# Exchange pagination caps (Binance USD-M).
BINANCE_OHLCV_LIMIT = 1500
BINANCE_FUNDING_LIMIT = 1000
BINANCE_OI_LIMIT = 500

# Binance USD-M keeps only the trailing ~30 days of Open Interest history;
# an older ``startTime`` triggers API error -1130. Cap OI fetches at 29 days
# to stay safely inside the retention window.
BINANCE_OI_HISTORY_DAYS = 29
BINANCE_OI_HISTORY_MS = BINANCE_OI_HISTORY_DAYS * 24 * 60 * 60 * 1000

# Funding is settled at exactly these UTC hours on Binance USD-M futures.
FUNDING_HOURS_UTC = (0, 8, 16)


def timeframe_to_ms(timeframe: str) -> int:
    """Convert a CCXT timeframe string ('1m', '4h', '1d', ...) to milliseconds."""
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    if unit not in units:
        raise ValueError(f"Unsupported timeframe unit: {timeframe!r}")
    return value * units[unit]


def default_since(exchange: ccxt.Exchange,
                  lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> int:
    """Return a millisecond timestamp `lookback_days` before now."""
    return exchange.milliseconds() - lookback_days * timeframe_to_ms("1d")


def resolve_available_lookback(exchange: ccxt.Exchange, symbol: str,
                               requested_days: int):
    """Return the maximum historical window Binance actually serves for `symbol`.

    Newer contracts (e.g. HYPE/USDT:USDT, listed 2025-05) expose far less
    history than the configured 10-year lookback. Requesting an older
    ``startTime`` would either be silently truncated or (on some endpoints)
    rejected, so we resolve the *real* window up front:

    1. Prefer the ``onboardDate`` advertised in the USD-M market metadata; it
       matches the first OHLCV bar exactly (verified against the exchange).
    2. If absent, probe the earliest 1d bar with a single ``since=1`` request.
    3. If neither works, fall back to the requested window unchanged.

    Returns ``(available_days, listing_timestamp_ms)`` with ``available_days``
    clamped to ``[1, requested_days]``.
    """
    now = exchange.milliseconds()
    requested_days = int(requested_days)

    listing_ms = None
    market = exchange.markets.get(symbol) or {}
    info = market.get("info") or {}
    onboard = info.get("onboardDate") or info.get("onboard_date")
    if onboard:
        try:
            listing_ms = int(onboard)
        except (TypeError, ValueError):
            listing_ms = None

    if listing_ms is None:
        logger.info("No onboardDate metadata for %s; probing the earliest 1d bar.",
                    symbol)
        try:
            earliest = exchange.fetch_ohlcv(symbol, DEFAULT_DAILY_TF,
                                            since=1, limit=1)
            if earliest:
                listing_ms = int(earliest[0][0])
        except Exception as exc:  # noqa: BLE001 - non-fatal; keep requested window
            logger.warning(
                "Unable to determine the listing date for %s (%s); using the "
                "requested %d-day lookback unchanged.",
                symbol, exc, requested_days,
            )
            return requested_days, now - requested_days * DAY_MS

    if listing_ms is None or listing_ms <= 0 or listing_ms >= now:
        return requested_days, now - requested_days * DAY_MS

    available = max(1, min(requested_days, int((now - listing_ms) / DAY_MS)))
    return available, int(listing_ms)


def load_env(env_path: Optional[str] = None) -> bool:
    """Load ``.env`` and return True if valid API credentials are present."""
    load_dotenv(dotenv_path=env_path, override=False)
    return bool(os.getenv("BINANCE_API_KEY")) and bool(os.getenv("BINANCE_API_SECRET"))


def create_exchange(api_key: Optional[str] = None,
                    api_secret: Optional[str] = None) -> ccxt.binanceusdm:
    """Build a configured CCXT client for Binance USD-M futures.

    Credentials are read from the environment (populated by ``.env``) unless
    explicitly supplied. Missing keys trigger a warning and leave the client in
    public-only mode (sufficient for OHLCV / OI / funding-rate history).
    """
    load_dotenv()
    api_key = api_key or os.getenv("BINANCE_API_KEY")
    api_secret = api_secret or os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        logger.warning(
            "Binance API credentials not found in .env - using PUBLIC "
            "market-data endpoints. Authenticated/private endpoints will raise."
        )

    exchange = ccxt.binanceusdm({
        "apiKey": api_key or "",
        "secret": api_secret or "",
        "enableRateLimit": True,           # respect Binance request-weight limits
        "options": {"defaultType": "future"},
    })
    exchange.load_markets()
    logger.info("Connected to %s (%d markets loaded)",
                exchange.name, len(exchange.markets))
    return exchange


def _coerce_timestamp_ms(values: pd.Series) -> pd.Series:
    """Coerce a timestamp column to int64 epoch-milliseconds.

    CCXT/Binance market-data endpoints already return epoch milliseconds, but
    clamping this at every ingest boundary guarantees as-of merges, truncation
    filters and ``pd.to_datetime(..., unit="ms")`` all compare the same unit,
    instead of silently mixing milliseconds with seconds or float/object dtypes.
    """
    return pd.to_numeric(values, errors="coerce").astype("int64")


def _log_df_diagnostics(name: str, df: pd.DataFrame) -> None:
    """Log index/column dtypes and the first rows of a DataFrame (DEBUG aid)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "[%s] index=%s | dtypes=%s",
        name, type(df.index).__name__,
        {col: str(dtype) for col, dtype in df.dtypes.items()},
    )
    logger.debug("[%s] head(2):\n%s", name, df.head(2).to_string())


def _normalize_ohlcv_batch(batch: List[List[float]]) -> pd.DataFrame:
    """Convert raw CCXT OHLCV rows into an ordered, tz-aware DataFrame."""
    df = pd.DataFrame(
        batch,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df["timestamp"] = _coerce_timestamp_ms(df["timestamp"])
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str = DEFAULT_SYMBOL,
                timeframe: str = "1h", since: Optional[int] = None,
                limit: int = BINANCE_OHLCV_LIMIT,
                max_bars: Optional[int] = None,
                lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch OHLCV bars, paginating forward from `since` toward now.

    Returns a tz-aware DataFrame indexed by UTC datetime, with an integer
    ``timestamp`` (ms) column retained for as-of merges.

    If ``since`` is None, a rolling window of ``lookback_days`` is used.
    """
    fetch_limit = min(int(limit), BINANCE_OHLCV_LIMIT)
    step_ms = timeframe_to_ms(timeframe)
    if since is None:
        since = default_since(exchange, lookback_days)

    rows: List[List[float]] = []
    cursor = since
    now = exchange.milliseconds()

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor,
                                     limit=fetch_limit)
        if not batch:
            break
        rows.extend(batch)
        if max_bars is not None and len(rows) >= max_bars:
            break
        next_cursor = int(batch[-1][0]) + step_ms
        # Binance caps some timeframes below ``limit`` (ccxt clamps the request
        # to 1000 bars regardless of the 1500 we pass here), so a short batch
        # does NOT mean "no more data". Keep paginating until the cursor passes
        # ``now``, or the exchange stops advancing the cursor.
        if next_cursor > now or next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        raise RuntimeError(f"No OHLCV data returned for {symbol} @ {timeframe}")

    df = _normalize_ohlcv_batch(rows)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    logger.info("Fetched %d OHLCV bars (%s %s)", len(df), symbol, timeframe)
    return df


def fetch_funding_rates(exchange: ccxt.Exchange, symbol: str = DEFAULT_SYMBOL,
                        since: Optional[int] = None,
                        limit: int = BINANCE_FUNDING_LIMIT,
                        max_items: Optional[int] = None,
                        lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch historical 8-hour funding rates into a tz-aware DataFrame.

    Columns: ``timestamp`` (ms) and ``funding_rate`` (signed decimal).
    The backtester applies the rate whenever a position spans one of the
    funding timestamps (00:00 / 08:00 / 16:00 UTC).
    """
    fetch_limit = min(int(limit), BINANCE_FUNDING_LIMIT)
    if since is None:
        since = default_since(exchange, lookback_days)

    rows: List[Dict[str, float]] = []
    cursor = since
    now = exchange.milliseconds()

    while True:
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor,
                                                    limit=fetch_limit)
        if not batch:
            break
        for r in batch:
            rows.append({
                "timestamp": int(r["timestamp"]),
                "funding_rate": float(r["fundingRate"]),
            })
        if max_items is not None and len(rows) >= max_items:
            break
        next_cursor = int(batch[-1]["timestamp"]) + 8 * 3_600_000
        if len(batch) < fetch_limit or next_cursor > now:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        raise RuntimeError(f"No funding-rate history returned for {symbol}")

    df = pd.DataFrame(rows)
    df["timestamp"] = _coerce_timestamp_ms(df["timestamp"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("datetime").sort_index()
    logger.info("Fetched %d funding-rate records (%s)", len(df), symbol)
    return df


def fetch_open_interest(exchange: ccxt.Exchange, symbol: str = DEFAULT_SYMBOL,
                        timeframe: str = "1h", since: Optional[int] = None,
                        limit: int = BINANCE_OI_LIMIT,
                        max_items: Optional[int] = None,
                        lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch historical Open Interest (coin amount + USDT notional).

    Returns a tz-aware DataFrame with columns ``timestamp`` (ms),
    ``oi_amount`` (base-coin count) and ``oi_value`` (USDT notional).
    """
    fetch_limit = min(int(limit), BINANCE_OI_LIMIT)
    step_ms = timeframe_to_ms(timeframe)
    if since is None:
        since = default_since(exchange, lookback_days)

    # Binance USD-M only retains ~30 days of OI history. Requesting an older
    # startTime raises API error -1130, so clamp the start timestamp to the
    # oldest value the endpoint is guaranteed to serve.
    now = exchange.milliseconds()
    oi_min_since = now - BINANCE_OI_HISTORY_MS
    if since < oi_min_since:
        logger.info(
            "Capping Open Interest fetch to the last 29 days due to Binance API restrictions."
        )
        since = oi_min_since

    rows: List[Dict[str, float]] = []
    cursor = since

    while True:
        batch = exchange.fetch_open_interest_history(symbol, timeframe,
                                                     since=cursor, limit=fetch_limit)
        if not batch:
            break
        for r in batch:
            rows.append({
                "timestamp": int(r["timestamp"]),
                "oi_amount": float(r.get("openInterestAmount") or 0.0),
                "oi_value": float(r.get("openInterestValue") or 0.0),
            })
        if max_items is not None and len(rows) >= max_items:
            break
        next_cursor = int(batch[-1]["timestamp"]) + step_ms
        if len(batch) < fetch_limit or next_cursor > now:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        raise RuntimeError(f"No Open Interest history returned for {symbol}")

    df = pd.DataFrame(rows)
    df["timestamp"] = _coerce_timestamp_ms(df["timestamp"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("datetime").sort_index()
    logger.info("Fetched %d OI records (%s %s)", len(df), symbol, timeframe)
    return df


def merge_open_interest(ohlcv_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """Attach OI to each candle using as-of (backward) alignment.

    OI timestamps rarely coincide exactly with candle open times. ``merge_asof``
    with ``direction="backward"`` attaches the most recent OI known at or before
    each candle timestamp, guaranteeing no look-ahead leakage into the bar.

    Both inputs are normalized onto the same integer-millisecond ``timestamp``
    key (CCXT standard) and a tz-aware UTC ``datetime`` index, so a DatetimeIndex
    vs. timestamp-column or tz-naive vs. tz-aware mismatch cannot silently drop
    rows from the join.
    """
    left = ohlcv_df.reset_index(drop=True).copy()
    right = oi_df.reset_index(drop=True).copy()

    left["timestamp"] = _coerce_timestamp_ms(left["timestamp"])
    right["timestamp"] = _coerce_timestamp_ms(right["timestamp"])

    left = left.sort_values("timestamp")
    right = right.sort_values("timestamp")[["timestamp", "oi_amount", "oi_value"]]

    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True)
    return merged.set_index("datetime").sort_index()


def fetch_market_data(symbol: str = DEFAULT_SYMBOL, execution_tf: str = "1h",
                      since: Optional[int] = None, oi_tf: Optional[str] = None,
                      lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                      include_daily: bool = True,
                      include_funding: bool = True,
                      exchange: Optional[ccxt.Exchange] = None) -> Dict[str, pd.DataFrame]:
    """Fetch the full dataset required by the strategy and backtester.

    Parameters
    ----------
    symbol       : CCXT unified symbol for the linear perpetual.
    execution_tf : execution timeframe ('1h' or '4h').
    since        : optional start timestamp (ms); default is a rolling lookback.
    oi_tf        : OI timeframe; defaults to ``execution_tf``.
    include_daily     : fetch the 1d OHLCV macro-trend series.
    include_funding   : fetch the 8h funding-rate history.

    Returns
    -------
    dict with keys:
        ``execution`` -> execution-tf OHLCV merged with OI
        ``daily``     -> 1d OHLCV (macro-trend filter), if ``include_daily``
        ``funding``   -> funding-rate history, if ``include_funding``
    """
    exchange = exchange or create_exchange()
    oi_tf = oi_tf or execution_tf

    requested_days = int(lookback_days)
    loaded_days = requested_days
    listing_ms: Optional[int] = None
    if since is None:
        # Graceful lookback handling: if the requested window exceeds the
        # asset's actual history on Binance, clamp to the maximum available
        # and log an INFO line with the number of days actually loaded.
        loaded_days, listing_ms = resolve_available_lookback(
            exchange, symbol, requested_days
        )
        if loaded_days < requested_days:
            listing_dt = pd.to_datetime(listing_ms, unit="ms", utc=True)
            logger.info(
                "Symbol %s: requested %d days of history, but only %d days are "
                "available on Binance (listed %s UTC). Loading the maximum "
                "available history: %d days.",
                symbol, requested_days, loaded_days,
                listing_dt.strftime("%Y-%m-%d"), loaded_days,
            )
        if listing_ms:
            since = listing_ms
        else:
            since = exchange.milliseconds() - loaded_days * DAY_MS

    logger.info("Loading %s | execution=%s | oi=%s | daily=%s | lookback=%d days",
                symbol, execution_tf, oi_tf, DEFAULT_DAILY_TF, loaded_days)

    exec_df = fetch_ohlcv(exchange, symbol, execution_tf, since=since,
                          lookback_days=loaded_days)
    oi_df = fetch_open_interest(exchange, symbol, oi_tf, since=since,
                                lookback_days=loaded_days)

    # DEBUG aid: surface index/dtype/unit mismatches before the join and the
    # truncation filter run, since those steps silently drop bars when timestamp
    # units or index types disagree.
    _log_df_diagnostics("execution", exec_df)
    _log_df_diagnostics("open_interest", oi_df)

    exec_df = merge_open_interest(exec_df, oi_df)

    # Binance USD-M retains only ~30 days of Open Interest history. The merged
    # execution series is kept at full lookback length: bars older than the OI
    # window carry NaN in ``oi_amount``/``oi_value``, and the strategy bypasses
    # the OI confirmation filter for those bars (see MultiTimeframeStrategy).
    # The D1 trend + Donchian breakout remain active across the whole period.
    oi_missing = int(exec_df["oi_amount"].isna().sum())
    if oi_missing:
        exec_df.attrs["oi_available_since"] = int(
            _coerce_timestamp_ms(oi_df["timestamp"]).min()
        )
        logger.info(
            "Open Interest unavailable for %d/%d bars older than ~29 days; "
            "NaN-filled (OI confirmation will be bypassed for those bars).",
            oi_missing, len(exec_df),
        )

    result: Dict[str, pd.DataFrame] = {"execution": exec_df}

    daily_bars: Optional[int] = None
    if include_daily:
        daily_df = fetch_ohlcv(exchange, symbol, DEFAULT_DAILY_TF,
                               since=since, lookback_days=loaded_days)
        _log_df_diagnostics("daily", daily_df)
        daily_bars = int(len(daily_df))
        result["daily"] = daily_df

    funding_records: Optional[int] = None
    if include_funding:
        funding_df = fetch_funding_rates(exchange, symbol, since=since,
                                         lookback_days=loaded_days)
        funding_records = int(len(funding_df))
        result["funding"] = funding_df

    # Expose the *actually loaded* window to the rest of the pipeline so the
    # runner and the EBTA validators adapt to each asset's real history length.
    result["meta"] = {
        "symbol": symbol,
        "exchange": exchange.name,
        "execution_tf": execution_tf,
        "oi_tf": oi_tf,
        "requested_lookback_days": int(requested_days),
        "loaded_lookback_days": int(loaded_days),
        "listing_timestamp_ms": int(listing_ms) if listing_ms else None,
        "start_timestamp_ms": int(_coerce_timestamp_ms(exec_df["timestamp"]).min()),
        "end_timestamp_ms": int(_coerce_timestamp_ms(exec_df["timestamp"]).max()),
        "execution_bars": int(len(exec_df)),
        "daily_bars": daily_bars,
        "funding_records": funding_records,
    }
    exec_df.attrs["loaded_lookback_days"] = int(loaded_days)
    return result


def load_dataset(symbol: str = DEFAULT_SYMBOL, execution_tf: str = "4h",
                 lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                 use_cache: bool = True, **kwargs) -> Dict[str, object]:
    """Cache-aware dataset loader for the multi-asset pipeline.

    Checks the local ``data/`` CSV cache first (built by ``src.cache``); on a
    hit the frames reload instantly from disk. On a miss, fetches live from
    Binance, persists the snapshot, and returns it. ``kwargs`` are forwarded to
    :func:`fetch_market_data`.
    """
    from src.cache import load_cached_dataset, store_cached_dataset

    if use_cache:
        cached = load_cached_dataset(symbol, execution_tf, lookback_days)
        if cached is not None:
            return cached

    data = fetch_market_data(
        symbol=symbol, execution_tf=execution_tf,
        lookback_days=lookback_days, **kwargs,
    )

    if use_cache:
        store_cached_dataset(data, symbol, execution_tf, lookback_days)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data = fetch_market_data(execution_tf="4h", lookback_days=30)
    for name, df in data.items():
        print(f"\n=== {name} ===")
        print(df.tail(3))
        print(f"rows: {len(df)}")