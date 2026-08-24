"""High-fidelity USD-M (linear) Perpetual Futures backtesting engine.

Simulates leveraged long/short positions on BTC/USDT:USDT with:

* Market-order execution at the NEXT bar's open (taker fee + slippage)
* 8-hour funding accrual (00:00 / 08:00 / 16:00 UTC), integrated into the
  open trade's P&L and the account's rolling equity (funding-inclusive)
* Maintenance-margin liquidation using conservative High/Low marks

The engine consumes the output of ``src.strategies.MultiTimeframeStrategy``:
its ``signal`` column is already shifted by one bar, so a signal computed at
bar t's close executes at bar t+1's open -> no look-ahead leakage.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "signal")


class PerpBacktester:
    """Event-driven perpetual-futures backtester."""

    def __init__(
        self,
        initial_capital: float = 100.0,
        leverage: float = 10.0,
        taker_fee: float = 0.0005,          # 0.05% market-order exit fee
        maker_fee: float = 0.0002,          # 0.02% limit-order entry fee
        risk_factor: float = 0.035,         # equity fraction risked per ATR stop
        maintenance_margin_rate: float = 0.005,  # 0.5% of notional
        slippage: float = 0.0002,           # 2 bps adverse fill on market orders
        use_atr_sizing: bool = False,       # True -> dynamic ATR sizing w/ cap
        atr_stop_multiplier: float = 2.0,   # ATR trailing stop distance (x ATR)
        use_dynamic_atr: bool = False,      # True -> re-snapshot ATR every bar
        reversal_exit: bool = True,         # True -> exit on Donchian reversal
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.leverage = float(leverage)
        self.taker_fee = float(taker_fee)
        self.maker_fee = float(maker_fee)
        self.risk_factor = float(risk_factor)
        self.maintenance_margin_rate = float(maintenance_margin_rate)
        self.slippage = float(slippage)
        self.use_atr_sizing = bool(use_atr_sizing)
        self.atr_stop_multiplier = float(atr_stop_multiplier)
        self.use_dynamic_atr = bool(use_dynamic_atr)
        self.reversal_exit = bool(reversal_exit)
        self.reset()

    # --- state -----------------------------------------------------------
    def reset(self) -> None:
        self.wallet_balance = self.initial_capital
        self.position_size = 0.0            # signed BTC (>0 long, <0 short)
        self.entry_price = None
        self.trailing_stop = None
        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.stop_atr = None
        self.reversal_upper = None
        self.reversal_lower = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees = 0.0
        self.total_funding_paid = 0.0
        self.total_funding_received = 0.0
        self.trade_funding_pnl = 0.0      # funding accrued by the OPEN trade
        self.liquidation_count = 0
        self.trade_count = 0
        self.win_count = 0
        self.avg_leverage = 0.0
        self._leverage_samples = 0
        self.last_price = None
        self.liquidated_this_bar = False

    # --- fill prices (slippage) ------------------------------------------
    def _buy_fill(self, open_price: float) -> float:
        return open_price * (1.0 + self.slippage)      # pay up

    def _sell_fill(self, open_price: float) -> float:
        return open_price * (1.0 - self.slippage)      # receive less

    # --- order helpers ----------------------------------------------------
    def _open(self, side: int, price: float, atr: Optional[float]) -> None:
        """Open a position: +1 long, -1 short as a MAKER limit order.

        Sizing modes:

        * ``use_atr_sizing == False`` (flat mode): every trade uses flat
          ``self.leverage`` of current equity:
          ``notional = current_equity * leverage``.
        * ``use_atr_sizing == True`` (dynamic mode): ATR-volatility-scaled
          ``notional = current_equity * risk_factor / ATR``, CLAMPED to at
          most ``self.leverage`` of equity. ATR missing/invalid falls back
          to flat leverage.
        """
        stop_atr = float(atr) if atr is not None and np.isfinite(atr) and atr > 0 else None
        available = self.wallet_balance

        if self.use_atr_sizing:
            atr = float(atr) if atr is not None and np.isfinite(atr) and atr > 0 else None
            if atr is not None:
                target_notional = available * self.risk_factor * price / atr
                notional = min(target_notional, available * self.leverage)
                size_btc = notional / price
            else:
                notional = available * self.leverage
                size_btc = notional / price
        else:
            notional = available * self.leverage
            size_btc = notional / price

        if notional <= 0 or size_btc <= 0:
            logger.warning("Insufficient capital to open position; order skipped.")
            return

        entry_fee = notional * self.maker_fee
        avg_lev = notional / available
        self.wallet_balance -= entry_fee
        self.total_fees += entry_fee
        self._leverage_samples += 1
        self.avg_leverage += (avg_lev - self.avg_leverage) / self._leverage_samples
        self.trade_funding_pnl = 0.0
        self.position_size = size_btc * side
        self.entry_price = price
        self.stop_atr = stop_atr
        self.highest_since_entry = price
        self.lowest_since_entry = price
        if stop_atr is not None:
            if side > 0:
                self.trailing_stop = price - self.atr_stop_multiplier * stop_atr
            else:
                self.trailing_stop = price + self.atr_stop_multiplier * stop_atr
        else:
            self.trailing_stop = None
        self.reversal_upper = None
        self.reversal_lower = None
        self.trade_count += 1
        logger.info("OPEN %s @ %.2f | size=%.6f BTC | notional=%.2f | lev=%.2fx | fee=%.4f",
                    "LONG" if side > 0 else "SHORT", price,
                    self.position_size, notional, avg_lev, entry_fee)

    def _close(self, price: float) -> None:
        """Close the current position (market/taker)."""
        if self.position_size == 0:
            return
        notional = abs(self.position_size) * price
        fee = notional * self.taker_fee
        trade_pnl = self.position_size * (price - self.entry_price)
        net = trade_pnl - fee
        # Funding charged while this trade was open was already posted to
        # wallet_balance at each funding timestamp; fold it into the trade's
        # realized P&L and its win/loss flag WITHOUT re-crediting the wallet.
        net_with_funding = net + self.trade_funding_pnl
        self.wallet_balance += net
        self.realized_pnl += net_with_funding
        self.total_fees += fee
        if net_with_funding > 0:
            self.win_count += 1
        logger.info("CLOSE @ %.2f | pnl=%.4f | fee=%.4f | funding=%.4f | wallet=%.4f",
                    price, trade_pnl, fee, self.trade_funding_pnl, self.wallet_balance)
        self.position_size = 0.0
        self.entry_price = None
        self.trailing_stop = None
        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.stop_atr = None
        self.reversal_upper = None
        self.reversal_lower = None
        self.trade_funding_pnl = 0.0

    def _check_trailing_stop(self, h: float, l: float,
                             atr: Optional[float] = None) -> None:
        """ATR trailing stop: trail with this bar's extremes, exit if breached.

        LONG : stop = max(stop, highest_since_entry - mult * ATR); exit if
               low <= stop (taker fill at the stop price, no slippage).
        SHORT: stop = min(stop, lowest_since_entry  + mult * ATR); exit if
               high >= stop (taker fill at the stop price, no slippage).

        When ``use_dynamic_atr`` is True the stop distance is re-snapshotted
        from ``atr`` on every bar instead of the frozen entry-time ATR.
        """
        if self.position_size == 0 or self.trailing_stop is None:
            return

        if self.use_dynamic_atr:
            if (atr is None or not np.isfinite(atr) or atr <= 0
                    or self.stop_atr is None):
                # No fresh ATR yet (or gap): keep previous distance unchanged.
                dist = self.atr_stop_multiplier * self.stop_atr
            else:
                dist = self.atr_stop_multiplier * float(atr)
        else:
            if self.stop_atr is None:
                return
            dist = self.atr_stop_multiplier * self.stop_atr

        if self.position_size > 0:
            self.highest_since_entry = max(self.highest_since_entry, h)
            self.trailing_stop = max(
                self.trailing_stop,
                self.highest_since_entry - dist,
            )
            if l <= self.trailing_stop:
                logger.info("TRAILING STOP (LONG) @ %.2f", self.trailing_stop)
                self._close(self.trailing_stop)
        else:
            self.lowest_since_entry = min(self.lowest_since_entry, l)
            self.trailing_stop = min(
                self.trailing_stop,
                self.lowest_since_entry + dist,
            )
            if h >= self.trailing_stop:
                logger.info("TRAILING STOP (SHORT) @ %.2f", self.trailing_stop)
                self._close(self.trailing_stop)

    def _check_reversal(self, h: float, l: float, row: pd.Series) -> bool:
        """Exit if the opposite Donchian band is touched by this bar.

        Returns True when a reversal exit closed the position.
        """
        if self.position_size == 0:
            return False

        upper = row.get("donchian_upper")
        lower = row.get("donchian_lower")
        if upper is None or lower is None or pd.isna(upper) or pd.isna(lower):
            return False

        closed = False
        if self.position_size > 0 and l <= float(lower):
            logger.info("REVERSAL EXIT (LONG) @ %.2f", float(lower))
            self._close(float(lower))
            closed = True
        elif self.position_size < 0 and h >= float(upper):
            logger.info("REVERSAL EXIT (SHORT) @ %.2f", float(upper))
            self._close(float(upper))
            closed = True
        return closed

    def _liquidate(self, liq_price: float) -> None:
        """Force-close at the liquidation price; the entire margin is forfeited.

        In cross margin the position is closed the moment ``equity == maintenance
        margin``, so the trader loses all of their remaining collateral (equity
        -> 0) and can never go below zero.
        """
        lost = self.wallet_balance
        self.wallet_balance = 0.0
        self.realized_pnl -= lost
        self.liquidation_count += 1
        self.position_size = 0.0
        self.entry_price = None
        self.trailing_stop = None
        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.stop_atr = None
        self.reversal_upper = None
        self.reversal_lower = None
        self.trade_funding_pnl = 0.0
        logger.warning("LIQUIDATION @ %.2f | collateral lost=%.4f", liq_price, lost)

    # --- funding -----------------------------------------------------------
    @staticmethod
    def _prepare_funding(funding_df: Optional[pd.DataFrame]):
        if funding_df is None or funding_df.empty:
            return None, None
        ts = funding_df["timestamp"].astype("int64").tolist()
        rates = funding_df["funding_rate"].astype(float).tolist()
        return sorted(ts), dict(zip(ts, rates))

    def run(self, df: pd.DataFrame,
            funding_df: Optional[pd.DataFrame] = None):
        """Run the backtest and return ``(result_df, summary_dict)``."""
        self.reset()

        if df is None or df.empty:
            logger.warning(
                "signals_df is empty; skipping the backtest loop and returning "
                "zeroed performance metrics."
            )
            empty = pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close", "signal",
                "position", "entry_price", "wallet_balance", "unrealized_pnl",
                "margin_balance", "equity", "realized_pnl", "funding_pnl",
                "liquidated",
            ])
            empty.index = pd.DatetimeIndex([], tz="UTC", name="datetime")
            return empty, self._zero_summary()

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

        funding_ts, rate_map = self._prepare_funding(funding_df)
        ptr = 0
        prev_open = None
        records: list = []

        for idx, row in df.iterrows():
            t_i = int(row["timestamp"])
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            desired = int(row["signal"])
            self.liquidated_this_bar = False

            # (1) Funding accrual over window (prev_open, t_i], on the position
            #     held during the previous bar (before this bar's signal).
            #     Signed Funding_PnL = -1 * Position_Notional * Funding_Rate at
            #     each Binance 8h settlement timestamp (a settled bar every 2nd
            #     4H bar): longs pay when rates are positive, shorts receive.
            #     The cash is posted straight into wallet_balance (rolling
            #     equity) AND attributed to the open trade's P&L via
            #     trade_funding_pnl so the closed trade's realized profit
            #     strictly includes funding erosion.
            bar_funding = 0.0
            if funding_ts:
                while ptr < len(funding_ts) and funding_ts[ptr] <= t_i:
                    f_ts = funding_ts[ptr]
                    if prev_open is not None and f_ts > prev_open and self.position_size != 0:
                        rate = rate_map.get(f_ts, 0.0)
                        price = self.last_price if self.last_price is not None else o
                        cash = -self.position_size * price * rate
                        self.wallet_balance += cash
                        self.trade_funding_pnl += cash
                        bar_funding += cash
                        if cash >= 0:
                            self.total_funding_received += cash
                        else:
                            self.total_funding_paid += -cash
                    ptr += 1

            # (2) Apply this bar's signal at its open (entry/exit/reversal).
            current_side = 0 if self.position_size == 0 else (1 if self.position_size > 0 else -1)
            if desired != current_side:
                if self.position_size != 0:
                    # Conservative stop-loss/exit: taker fill + slippage.
                    exit_fill = self._sell_fill(o) if current_side > 0 else self._buy_fill(o)
                    self._close(exit_fill)
                if desired != 0:
                    # Maker limit entry at the open: no adverse slippage.
                    atr = float(row["atr"]) if "atr" in row and pd.notna(row["atr"]) else np.nan
                    self._open(desired, o, atr)

            # (2b) EXIT orchestration:
            #      * trailing stop, and/or
            #      * Donchian reversal (opposite band touch) -- whichever first.
            #      Reversal runs first: it represents the clean macro-trend flip
            #      and typically fires before a very loose ATR stop. If it does
            #      not fire, the ATR stop still protects the trade.
            if self.position_size != 0:
                exit_by_reversal = False
                if self.reversal_exit:
                    exit_by_reversal = self._check_reversal(h, l, row)
                if self.position_size != 0 and not exit_by_reversal:
                    atr_now = None
                    if self.use_dynamic_atr and "atr" in row and pd.notna(row["atr"]):
                        atr_now = float(row["atr"])
                    self._check_trailing_stop(h, l, atr_now)

            # (3) Mark-to-market + liquidation (liquidation-price model).
            self.unrealized_pnl = 0.0
            if self.position_size != 0:
                self.unrealized_pnl = self.position_size * (c - self.entry_price)
                s = self.position_size
                e = self.entry_price
                w = self.wallet_balance
                mmr = self.maintenance_margin_rate
                if s > 0:
                    liq_price = (e - w / s) / (1.0 - mmr)
                    breached = liq_price > 0.0 and l <= liq_price
                else:
                    a = -s
                    liq_price = (w / a + e) / (1.0 + mmr)
                    breached = h >= liq_price
                if breached:
                    self._liquidate(liq_price)
                    self.unrealized_pnl = 0.0
                    self.liquidated_this_bar = True

            equity = self.wallet_balance + self.unrealized_pnl
            self.last_price = c

            records.append({
                "timestamp": t_i,
                "open": o, "high": h, "low": l, "close": c,
                "signal": desired,
                "position": self.position_size,
                "entry_price": 0.0 if self.entry_price is None else self.entry_price,
                "wallet_balance": self.wallet_balance,
                "unrealized_pnl": self.unrealized_pnl,
                "margin_balance": equity,
                "equity": equity,
                "realized_pnl": self.realized_pnl,
                "funding_pnl": bar_funding,
                "liquidated": int(self.liquidated_this_bar),
            })
            prev_open = t_i

        result = pd.DataFrame.from_records(records)
        result["datetime"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        result = result.set_index("datetime")
        return result, self.summarize(result)

    def _zero_summary(self) -> Dict[str, float]:
        """Return a fully zeroed summary for the empty-input path."""
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.initial_capital,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "total_fees_paid": 0.0,
            "total_funding_paid": 0.0,
            "total_funding_received": 0.0,
            "net_funding": 0.0,
            "liquidations": 0,
            "leverage": self.leverage,
            "leverage_cap": self.leverage,
            "use_atr_sizing": self.use_atr_sizing,
            "risk_factor": self.risk_factor,
            "atr_stop_multiplier": self.atr_stop_multiplier,
        }

    def summarize(self, result_df: pd.DataFrame) -> Dict[str, float]:
        """Compute headline performance/risk metrics."""
        equity = result_df["equity"].astype(float)
        initial = self.initial_capital
        total_return = equity.iloc[-1] / initial - 1.0

        prev = equity.shift(1)
        mask = (equity > 0) & (prev > 0)
        log_ret = np.log(equity[mask] / prev[mask]).dropna()
        delta = result_df["timestamp"].astype("int64").diff().dropna()
        median_sec = float(delta.median()) / 1000.0 if len(delta) else 3600.0
        median_sec = max(median_sec, 1.0)
        periods_per_year = 365.25 * 24.0 * 3600.0 / median_sec

        std = float(log_ret.std(ddof=1))
        sharpe = float(log_ret.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = float(drawdown.min())

        net_funding = self.total_funding_received - self.total_funding_paid
        win_rate = (self.win_count / self.trade_count) if self.trade_count else 0.0

        return {
            "initial_capital": initial,
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(total_return),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "num_trades": self.trade_count,
            "win_rate": win_rate,
            "total_fees_paid": float(self.total_fees),
            "total_funding_paid": float(self.total_funding_paid),
            "total_funding_received": float(self.total_funding_received),
            "net_funding": float(net_funding),
            "liquidations": self.liquidation_count,
            "leverage": self.avg_leverage,
            "leverage_cap": self.leverage,
            "use_atr_sizing": self.use_atr_sizing,
            "risk_factor": self.risk_factor,
            "atr_stop_multiplier": self.atr_stop_multiplier,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Synthetic smoke test (random signals, no funding).
    n = 500
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    price = 40_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n)))
    df = pd.DataFrame({
        "timestamp": ts.astype("int64") // 1_000_000,
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "signal": rng.integers(-1, 2, n),
    })
    bt = PerpBacktester()
    result, summary = bt.run(df)
    print("Final equity:", result["equity"].iloc[-1])
    print(summary)