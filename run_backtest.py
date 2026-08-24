"""Multi-asset perpetual parameter-sweep: RISK_FACTOR x LEVERAGE_CAP grid.

Flow, in order:

    .env -> load/fetch each symbol -> Donchian 120 signals -> grid sweep of
         RISK_FACTOR x LEVERAGE_CAP over the champion strategy
         (4.0x dynamic ATR trailing + reversal exit) -> EBTA validation (MCP
         permutation + bootstrap) -> per-asset grid leaderboards
         -> winners table -> normalized multi-asset dashboard PNG

The champion strategy is fixed for every grid combination:

    * Donchian 120 breakout entry (D1 SMA200 macro-trend filter + OI confirm)
    * 4.0x dynamic ATR trailing stop (re-snapshotted every bar)
    * Donchian reversal exit (opposite-band touch)
    * Maker entry fee 0.02% | Taker exit fee 0.05%
    * Grid: RISK_FACTOR in {3.5%, 5.0%, 7.5%} x LEVERAGE_CAP in {5x, 7.5x, 10x}
    * Requirement: max drawdown strictly under 45% -> 'SAFE' combo

Runs sequentially over BTC, ETH, SOL and HYPE (Binance USD-M perpetuals),
caching every dataset to ``data/`` (e.g. ``data/btc_4h_10y.csv``) so re-runs
reload instantly from disk instead of re-hitting the exchange API.

Run:  python run_backtest.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.backtester import PerpBacktester
from src.statistics import bootstrap_confidence_interval, monte_carlo_permutation
from src.strategies import MultiTimeframeStrategy

# Deliberately NOT imported at module top: data_loader imports ccxt, which may
# be absent; it is imported lazily inside main() with a friendly error message.

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "backtest_results.png"
DAY_MS = 86_400_000

MULTI_ASSET_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "HYPE/USDT:USDT",
]


COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
          "tab:red", "tab:brown", "tab:pink", "tab:gray"]


def resolve_config() -> Dict:
    """Read pipeline configuration from ``.env`` (with sane defaults)."""
    load_dotenv()
    return {
        "symbols": [s.strip() for s in os.getenv(
            "SYMBOLS", ",".join(MULTI_ASSET_SYMBOLS)).split(",") if s.strip()],
        "execution_tf": os.getenv("EXECUTION_TIMEFRAME", "4h"),
        "lookback_days": int(os.getenv("LOOKBACK_DAYS", "3650")),
        "initial_capital": float(os.getenv("INITIAL_CAPITAL", "100")),
        "leverage": float(os.getenv("LEVERAGE", "10")),
        "taker_fee": float(os.getenv("TAKER_FEE", "0.0005")),
        "maker_fee": float(os.getenv("MAKER_FEE", "0.0002")),
        "risk_factor": float(os.getenv("RISK_FACTOR", "0.035")),
        "atr_stop_multiplier": float(os.getenv("ATR_STOP_MULTIPLIER", "4.0")),
        "atr_period": int(os.getenv("ATR_PERIOD", "14")),
        "maintenance_margin_rate": float(os.getenv("MAINTENANCE_MARGIN_RATE", "0.005")),
        "slippage": float(os.getenv("SLIPPAGE", "0.0002")),
        "sma_period": int(os.getenv("SMA_PERIOD", "200")),
        "donchian_period": int(os.getenv("DONCHIAN_PERIOD", "120")),
        "oi_lookback": int(os.getenv("OI_LOOKBACK", "3")),
        "n_permutations": int(os.getenv("N_PERMUTATIONS", "5000")),
        "n_bootstrap": int(os.getenv("N_BOOTSTRAP", "5000")),
        "confidence": float(os.getenv("CONFIDENCE", "0.95")),
        "seed": int(os.getenv("SEED", "42")),
        "output": os.getenv("OUTPUT_PLOT", DEFAULT_OUTPUT),
        "risk_factors": [float(x) for x in os.getenv(
            "GRID_RISK_FACTORS", "0.035,0.050,0.075").split(",") if x.strip()],
        "leverages": [float(x) for x in os.getenv(
            "GRID_LEVERAGES", "5.0,7.5,10.0").split(",") if x.strip()],
        "max_drawdown_cap": float(os.getenv("MAX_DRAWDOWN_CAP", "0.45")),
        "daily_target_usdt": float(os.getenv("DAILY_TARGET_USDT", "10.0")),
    }


def build_dashboard(champions: Dict[str, Dict],
                    permuted: Dict[str, np.ndarray],
                    actual_returns: Dict[str, float],
                    p_values: Dict[str, float],
                    out_path: str) -> None:
    """Render the grid-sweep dashboard PNG.

    Panel 1 : normalized equity curves (start = 100) of each asset's best SAFE
              (max drawdown < cap) grid configuration, overlaid on one axis.
    Panel 2 : per-asset Monte Carlo permutation null distributions with each
              asset's actual rule return drawn as a vertical line.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visual output
        logger.warning("matplotlib unavailable - skipping dashboard: %s", exc)
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [3, 2]},
    )

    # --- Panel 1: best SAFE grid winners' normalized equity (start = 100) ----
    for i, (symbol, champ) in enumerate(champions.items()):
        color = COLORS[i % len(COLORS)]
        equity = champ["equity"]
        normalized = equity / float(equity.iloc[0]) * 100.0
        label = (f"{symbol} | rf={champ['risk_factor'] * 100:.1f}% "
                 f"x{champ['leverage']:g} | ret {champ['net_return'] * 100:+.0f}%")
        ax1.plot(normalized.index, normalized.values,
                 label=label, color=color, linewidth=1.4)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized Equity (start = 100)")
    ax1.set_title(
        "Grid-Sweep Winners - Best SAFE (DD<45%) Net-Return Config per "
        "Asset\nDonchian 120 | 4.0x dynamic ATR trailing | Reversal exit | "
        "funding-inclusive P&L"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # --- Panel 2: per-asset MCP null distributions ---------------------------
    for i, symbol in enumerate(champions):
        color = COLORS[i % len(COLORS)]
        ax2.hist(permuted[symbol], bins=80, color=color, alpha=0.35,
                 edgecolor="none")
        ax2.axvline(actual_returns[symbol], color=color, linewidth=1.8,
                    label=f"{symbol} actual={actual_returns[symbol]:+.4f} "
                          f"(p={p_values[symbol]:.4f})")
    ax2.set_title(
        "Monte Carlo Permutation Null Distributions per Asset "
        "(5000 permutations, detrended)"
    )
    ax2.set_xlabel("Mean rule return (per 4h bar)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved dashboard -> %s", out_path)


def format_grid_leaderboard(grid: List[Dict], cfg: Dict) -> str:
    """Rank all RISK_FACTOR x LEVERAGE_CAP combos of one asset by Net Return."""
    safe_rows = [r for r in grid if r["safe"]]
    if not safe_rows:
        logger.warning("No SAFE combo; flagging least-bad picks from full grid.")
        safe_rows = grid
    best_ret = max(safe_rows, key=lambda r: r["net_return"])
    best_sr = max(safe_rows, key=lambda r: r["sharpe"])
    ranked = sorted(grid, key=lambda r: r["net_return"], reverse=True)

    rule = "=" * 128
    header = (
        f"{'Risk%':>7}{'Lev':>6}{'NetRet':>11}{'Sharpe':>9}{'MaxDD':>10}"
        f"{'FinalEq':>13}{'Trades':>8}{'Liquid':>7}{'NetFund':>11}"
        f"{'USDT/d':>10}  {'Safe':>5}"
    )
    lines = [
        rule,
        " GRID LEADERBOARD | champion strategy: Donchian 120 + 4.0x dynamic "
        "ATR trailing + Reversal exit",
        f" {len(grid)} combos x RISK_FACTOR/LEVERAGE_CAP | "
        f"DD cap {cfg['max_drawdown_cap'] * 100:.0f}% | "
        f"start {cfg['initial_capital']:.0f} USDT | "
        f"target {cfg['daily_target_usdt']:.1f} USDT/day | "
        "funding-inclusive equity",
        rule,
        header,
        rule,
    ]
    for r in ranked:
        flag = ""
        if r is best_ret:
            flag = "  <= best Net Return (safe)"
        elif r is best_sr:
            flag = "  <= best Sharpe (safe)"
        lines.append(
            f"{r['risk_factor'] * 100:>6.1f}%{r['leverage']:>6.1f}"
            f"{r['net_return'] * 100:>10.2f}%{r['sharpe']:>9.2f}"
            f"{r['max_drawdown'] * 100:>9.2f}%{r['final_equity']:>13,.2f}"
            f"{r['num_trades']:>8d}{r['liquidations']:>7d}"
            f"{r['net_funding']:>11,.2f}{r['usdt_per_day']:>10,.2f}"
            f"{'YES' if r['safe'] else 'NO':>5}{flag}"
        )
    lines.append(rule)
    return "\n".join(lines)


def format_winners_table(assets: List[Dict], cfg: Dict) -> str:
    """Render the cross-asset winner leaderboard (best SAFE combo per symbol)."""
    rule = "=" * 128
    header = (
        f"{'Symbol':<14}{'Risk%':>7}{'Lev':>6}{'NetRet':>11}{'Sharpe':>9}"
        f"{'MaxDD':>10}{'FinalEq':>13}{'USDT/day':>11}{'vs target':>11}"
    )
    lines = [
        rule,
        " WINNER LEADERBOARD - best SAFE (DD < "
        f"{cfg['max_drawdown_cap'] * 100:.0f}%) Net-Return config per asset",
        " Fees: Maker 0.02% | Taker 0.05% | Funding: Binance 8h history | "
        f"Initial capital {cfg['initial_capital']:.0f} USDT",
        rule,
        header,
        rule,
    ]
    for a in assets:
        c = a["champion"]
        target_pct = c["usdt_per_day"] / cfg["daily_target_usdt"] * 100.0
        lines.append(
            f"{a['symbol']:<14}{c['risk_factor'] * 100:>6.1f}%"
            f"{c['leverage']:>6.1f}{c['net_return'] * 100:>10.2f}%"
            f"{c['sharpe']:>9.2f}{c['max_drawdown'] * 100:>9.2f}%"
            f"{c['final_equity']:>13,.2f}{c['usdt_per_day']:>11,.2f}"
            f"{target_pct:>10.0f}%"
        )
    lines.append(rule)
    return "\n".join(lines)


def run_asset(cfg: Dict, symbol: str) -> Optional[Dict]:
    """Load data (cached or live), run the backtest and validate one symbol."""
    print(f"\n{'=' * 110}\n  ASSET: {symbol}\n{'=' * 110}")

    try:
        from src.data_loader import load_dataset
    except ImportError as exc:
        print("ERROR: ccxt is required to fetch live market data.")
        print("       Fix:  pip install -r requirements.txt")
        print(f"       Detail: {exc}")
        return None

    print(f"Loading {symbol} market data ({cfg['execution_tf']}, "
          f"requested {cfg['lookback_days']} days) ...")
    t0 = time.time()
    try:
        data = load_dataset(symbol=symbol, execution_tf=cfg["execution_tf"],
                            lookback_days=cfg["lookback_days"])
    except Exception as exc:
        logger.exception("Data fetch failed for %s", symbol)
        print(f"ERROR: data fetch failed for {symbol}: {exc}")
        return None

    exec_df = data["execution"]
    daily_df = data["daily"]
    funding_df = data.get("funding")
    meta = data.get("meta") or {}
    loaded_days = int(meta.get("loaded_lookback_days") or cfg["lookback_days"])

    exec_ts = exec_df["timestamp"].astype("int64")
    days_tested = max(1, int((exec_ts.max() - exec_ts.min()) / DAY_MS))
    print(f"Loaded {len(exec_df)} execution bars / {len(daily_df)} daily bars "
          f"({loaded_days} days loaded, {days_tested} days covered) "
          f"in {time.time() - t0:.1f}s")

    oi_available_since = exec_df.attrs.get("oi_available_since")
    if oi_available_since is not None:
        oi_start = pd.to_datetime(oi_available_since, unit="ms", utc=True)
        print(
            "INFO: Binance Open Interest history is limited to the last ~29 days "
            f"(available since {oi_start:%Y-%m-%d %H:%M} UTC). Older bars are "
            "NaN-filled and will bypass the OI confirmation filter; D1 trend + "
            "Donchian breakout remain fully active across the whole period."
        )

    # Optimized strategy scenario, fixed for every asset.
    scen = {
        "key": "Donchian 120 | 4.0x ATR dynamic + Reversal",
        "donchian_period": int(cfg["donchian_period"]),
        "atr_stop_multiplier": float(cfg["atr_stop_multiplier"]),
    }
    print(f"\nRunning {scen['key']} on {symbol} ...")

    strat = MultiTimeframeStrategy(
        sma_period=cfg["sma_period"],
        donchian_period=scen["donchian_period"],
        oi_lookback=cfg["oi_lookback"],
        atr_period=cfg["atr_period"],
    )
    signals_df = strat.generate_signals(exec_df, daily_df)
    if signals_df.empty or len(signals_df) < cfg["sma_period"] + 10:
        print(f"WARNING: {symbol}: no usable signals after indicator warm-up; "
              "skipping.")
        return None

    close = exec_df["close"].astype(float)
    market_returns = close.pct_change().dropna().to_numpy(dtype=float)
    signals = signals_df["signal"].iloc[1:].to_numpy(dtype=float)
    if len(signals) == 0:
        print(f"WARNING: {symbol}: no traded bars; skipping.")
        return None

    # EBTA validation. Adaptive by construction: it uses exactly the bars
    # actually loaded for this asset (see src/statistics.py).
    mcp = monte_carlo_permutation(
        signals, market_returns,
        n_permutations=cfg["n_permutations"], seed=cfg["seed"],
    )

    # ---- Grid sweep: RISK_FACTOR x LEVERAGE_CAP x champion strategy --------
    grid: List[Dict] = []
    for rf in cfg["risk_factors"]:
        for lev in cfg["leverages"]:
            bt = PerpBacktester(
                initial_capital=cfg["initial_capital"],
                leverage=lev,
                taker_fee=cfg["taker_fee"],
                maker_fee=cfg["maker_fee"],
                risk_factor=rf,
                maintenance_margin_rate=cfg["maintenance_margin_rate"],
                slippage=cfg["slippage"],
                use_atr_sizing=True,
                atr_stop_multiplier=scen["atr_stop_multiplier"],
                use_dynamic_atr=True,
                reversal_exit=True,
            )
            result, summary = bt.run(signals_df, funding_df)
            equity = result["equity"].astype(float)
            run = {
                "symbol": symbol,
                "risk_factor": rf,
                "leverage": lev,
                "final_equity": float(summary["final_equity"]),
                "net_return": float(summary["total_return"]),
                "sharpe": float(summary["sharpe_ratio"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "num_trades": int(summary["num_trades"]),
                "win_rate": float(summary["win_rate"]),
                "net_funding": float(summary["net_funding"]),
                "liquidations": int(summary["liquidations"]),
                "equity": equity,
            }
            run["safe"] = abs(run["max_drawdown"]) < cfg["max_drawdown_cap"]
            run["usdt_per_day"] = (
                run["final_equity"] - cfg["initial_capital"]) / days_tested
            grid.append(run)
            print(f"  [grid] rf={rf * 100:4.1f}%  lev={lev:4.1f}x  ->  "
                  f"finalEq {run['final_equity']:>10,.2f}  "
                  f"ret {run['net_return'] * 100:+8.2f}%  "
                  f"sharpe {run['sharpe']:6.2f}  "
                  f"DD {run['max_drawdown'] * 100:7.2f}%  "
                  f"{'SAFE' if run['safe'] else 'UNSAFE'}")

    safe_runs = [r for r in grid if r["safe"]]
    if not safe_runs:
        print(f"WARNING: {symbol}: NO grid combo stays under the "
              f"{cfg['max_drawdown_cap'] * 100:.0f}% DD cap; "
              "showing least-bad runs.")
        safe_runs = grid
    best_return = max(safe_runs, key=lambda r: r["net_return"])
    best_sharpe = max(safe_runs, key=lambda r: r["sharpe"])
    champion = best_return

    print(f"\n{format_grid_leaderboard(grid, cfg)}")

    # Bootstrap CI over the champion's (best SAFE Net Return) daily bars.
    daily_equity = champion["equity"].resample("1D").last().dropna()
    daily_net = daily_equity.pct_change().dropna()
    if len(daily_net) >= 2:
        boot = bootstrap_confidence_interval(
            daily_net, n_bootstrap=cfg["n_bootstrap"],
            confidence=cfg["confidence"], seed=cfg["seed"],
        )
    else:
        boot = {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    target_pct = champion["usdt_per_day"] / cfg["daily_target_usdt"] * 100.0
    print(f"\n--- {symbol} CHAMPION (best SAFE Net Return) ---")
    print(f"  Config           : risk factor "
          f"{champion['risk_factor'] * 100:.1f}% | leverage cap "
          f"{champion['leverage']:.1f}x")
    print(f"  Days tested      : {days_tested}")
    print(f"  Final equity     : {champion['final_equity']:,.2f} USDT")
    print(f"  Net return       : {champion['net_return'] * 100:+.2f}%")
    print(f"  Annualized Sharpe: {champion['sharpe']:.2f}")
    print(f"  Max drawdown     : {champion['max_drawdown'] * 100:.2f}%")
    print(f"  Total trades     : {champion['num_trades']}")
    print(f"  Win rate         : {champion['win_rate'] * 100:.1f}%")
    print(f"  Net funding      : {champion['net_funding']:,.2f} USDT")
    print(f"  Avg daily        : {champion['usdt_per_day']:+.2f} USDT/day "
          f"= {target_pct:.0f}% of the {cfg['daily_target_usdt']:.1f} "
          "USDT/day target")
    if champion["liquidations"]:
        print(f"  Liquidations     : {champion['liquidations']}")
    print(f"  MCP p-value      : {mcp['p_value']:.4f} "
          f"({mcp['n_permutations']} permutations)")
    print(f"  Bootstrap CI     : mean {boot['mean']:+.6f}  "
          f"[{boot['lower']:+.6f}, {boot['upper']:+.6f}] "
          f"({len(daily_net)} daily bars resampled)")

    return {
        "symbol": symbol,
        "days_tested": days_tested,
        "grid_rows": grid,
        "best_return": best_return,
        "best_sharpe": best_sharpe,
        "champion": champion,
        "mcp": mcp,
        "boot": boot,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = resolve_config()

    symbols = cfg["symbols"]
    print(f"Multi-asset grid sweep: {len(symbols)} symbols -> {symbols}")
    print(f"Champion strategy: Donchian {cfg['donchian_period']} | "
          f"{cfg['atr_stop_multiplier']:.1f}x dynamic ATR trailing | Reversal exit")
    print(f"RISK_FACTOR grid : "
          f"{[f'{x * 100:.1f}%' for x in cfg['risk_factors']]}")
    print(f"LEVERAGE_CAP grid: "
          f"{[f'{x:.1f}x' for x in cfg['leverages']]}")
    print(f"Constraint: max drawdown < {cfg['max_drawdown_cap'] * 100:.0f}% | "
          f"target {cfg['daily_target_usdt']:.1f} USDT/day | "
          f"start {cfg['initial_capital']:.0f} USDT | "
          f"fees Maker {cfg['maker_fee'] * 100:.2f}% / Taker "
          f"{cfg['taker_fee'] * 100:.2f}%")

    assets: List[Dict] = []
    for symbol in symbols:
        asset = run_asset(cfg, symbol)
        if asset is not None:
            assets.append(asset)

    if not assets:
        print("No asset produced a backtest result.")
        return 1

    print()
    print(format_winners_table(assets, cfg))

    print(f"\nRendering multi-asset dashboard -> {cfg['output']} ...")
    build_dashboard(
        champions={a["symbol"]: a["champion"] for a in assets},
        permuted={a["symbol"]: a["mcp"]["permuted_returns"] for a in assets},
        actual_returns={a["symbol"]: a["mcp"]["actual_return"] for a in assets},
        p_values={a["symbol"]: a["mcp"]["p_value"] for a in assets},
        out_path=cfg["output"],
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
