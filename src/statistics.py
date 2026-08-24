"""Statistical validation engine (David Aronson, *Evidence-Based Technical Analysis*).

Implements the EBTA inference toolkit that decides whether a strategy's observed
performance is the product of skill or luck:

1. **Detrending** [p. 3-10] - remove the benchmark drift so the null benchmark
   return is centered exactly at zero, eliminating market-trend / position bias.
2. **Monte Carlo Permutation (MCP) test** [p. 11-15] - test the null hypothesis
   "the strategy signals are random and have no predictive power". The actual
   signal sequence is kept intact (preserving its autocorrelation); only the
   market returns are shuffled (without replacement) and re-paired with the
   signals. The empirical p-value is the fraction of permuted "noise rules"
   whose mean return equals or exceeds the strategy's actual mean return.
3. **Bootstrap resampling** [p. 16-19] - resample the realized net returns with
   replacement to build a confidence interval for the mean return.

All functions are frequency-agnostic: pass bar-frequency returns for the MCP
(one return per signal) and daily net returns for the bootstrap.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PERMUTATIONS = 5000
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_CONFIDENCE = 0.95


def detrend_returns(returns) -> np.ndarray:
    """Center a return series at exactly zero by subtracting its mean.

    ``r_detrended[t] = r[t] - mean(r)``. This removes the market's average
    drift so the benchmark return is zero, isolating the strategy's edge.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {r.shape}")
    return r - r.mean()


def monte_carlo_permutation(
    signals,
    market_returns,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run the EBTA Monte Carlo Permutation test.

    Parameters
    ----------
    signals : 1-D array-like of +1 / -1 / 0, the strategy's position per bar.
        The REAL sequence is preserved (autocorrelation intact).
    market_returns : 1-D array-like of the market's per-bar returns, aligned to
        ``signals`` (market_returns[t] is earned while signal[t] is held).
    n_permutations : number of shuffles used to build the noise-rule null dist.
    seed : optional RNG seed for reproducibility.

    Returns
    -------
    dict with ``actual_return`` (mean signal*return of the true pairing),
    ``permuted_returns`` (array of noise-rule means), and ``p_value``
    (fraction of permuted means >= actual return).

    The test is frequency- and size-adaptive: ``signals``/``market_returns``
    are simply whatever was actually loaded for the asset (e.g. HYPE only has
    ~450 daily-equivalent bars vs BTC's ~2,500), and the null distribution
    automatically reflects that real history length.
    """
    signals = np.asarray(signals, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    if signals.ndim != 1 or market_returns.ndim != 1:
        raise ValueError("signals and market_returns must both be 1-D")

    n = min(len(signals), len(market_returns))
    if n == 0:
        raise ValueError("signals and market_returns are empty")
    signals = signals[-n:]
    market_returns = market_returns[-n:]

    if np.count_nonzero(signals) == 0:
        logger.warning("No non-zero signals supplied; MCP p-value is degenerate.")

    # Detrend benchmark -> null centered at zero (no drift/position bias).
    detrended = market_returns - market_returns.mean()

    actual_return = float(np.mean(signals * detrended))

    rng = np.random.default_rng(seed)
    permuted = np.empty(int(n_permutations), dtype=float)
    for i in range(int(n_permutations)):
        # Shuffle returns WITHOUT replacement, keep signal order intact.
        permuted[i] = np.mean(signals * rng.permutation(detrended))

    p_value = float(np.mean(permuted >= actual_return))

    logger.info("MCP: actual=%.6f p_value=%.4f (%d perms)",
                actual_return, p_value, int(n_permutations))
    return {
        "actual_return": actual_return,
        "permuted_returns": permuted,
        "p_value": p_value,
        "n_permutations": int(n_permutations),
    }


def bootstrap_confidence_interval(
    net_returns,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Resample the strategy's net returns to build a mean-return CI.

    Draws ``n_bootstrap`` samples (with replacement) of the same size as the
    input, records each sample mean, and reports the percentile interval.

    Size-adaptive by construction: pass one entry per actually-loaded daily
    bar for the asset (e.g. the resampled equity curve) and the resampling
    window, sample size and CI automatically match that real history length.
    """
    r = np.asarray(net_returns, dtype=float)
    if r.ndim != 1:
        raise ValueError(f"net_returns must be 1-D, got shape {r.shape}")
    if r.size == 0:
        raise ValueError("net_returns is empty")

    rng = np.random.default_rng(seed)
    means = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        means[i] = rng.choice(r, size=r.size, replace=True).mean()

    alpha = 1.0 - float(confidence)
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))

    logger.info("Bootstrap CI: mean=%.6f [%.6f, %.6f] (%d resamples)",
                float(r.mean()), lower, upper, int(n_bootstrap))
    return {
        "mean": float(r.mean()),
        "confidence": float(confidence),
        "lower": lower,
        "upper": upper,
        "bootstrapped_means": means,
    }


def validate(
    signals,
    market_returns,
    net_returns,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Convenience wrapper running both MCP and bootstrap in one call."""
    mcp = monte_carlo_permutation(
        signals, market_returns,
        n_permutations=n_permutations, seed=seed,
    )
    boot = bootstrap_confidence_interval(
        net_returns,
        n_bootstrap=n_bootstrap, confidence=confidence, seed=seed,
    )
    out = dict(mcp)
    out.update(boot)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    n = 1000
    # Predictive signal (long only on up-bars) -> demonstrates the test
    # detecting a real edge (p -> 0).
    market_returns = rng.normal(0.0002, 0.02, n)
    signals = np.where(market_returns > 0.0, 1.0, 0.0)
    net_returns = signals * market_returns

    mcp = monte_carlo_permutation(signals, market_returns, n_permutations=2000, seed=7)
    boot = bootstrap_confidence_interval(net_returns, n_bootstrap=2000, seed=7)
    print("p_value =", round(mcp["p_value"], 4))
    print("bootstrap mean =", round(boot["mean"], 6),
          "CI = [", round(boot["lower"], 6), ",", round(boot["upper"], 6), "]")