"""
Volatility Tools
=================
- Implied volatility solver (Newton-Raphson method)
- Historical volatility from price data
- Realized vs implied volatility comparison
"""

import numpy as np
from .black_scholes import BlackScholesModel


def implied_volatility(market_price, S, K, T, r, option_type="call", q=0.0,
                       tol=1e-6, max_iter=100):
    """
    Solve for implied volatility using Newton-Raphson method.

    Uses vega as the derivative to iteratively converge on the
    volatility that makes the BSM price equal to the market price.

    Parameters
    ----------
    market_price : float — Observed market price of the option
    S, K, T, r, q : float — BSM parameters
    option_type : str — 'call' or 'put'
    tol : float — Convergence tolerance
    max_iter : int — Maximum iterations

    Returns
    -------
    float — Implied volatility (annualized)
    """
    sigma = 0.30  # Initial guess

    for _ in range(max_iter):
        bsm = BlackScholesModel(S, K, T, r, sigma, q)
        price = bsm.price(option_type)
        vega = bsm.vega() * 100  # Undo the /100 in vega method

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        if abs(vega) < 1e-12:
            break

        sigma -= diff / vega
        sigma = max(sigma, 0.001)  # Prevent negative vol

    return sigma  # Best estimate


def historical_volatility(prices, window=30, annualize=True):
    """
    Calculate rolling historical (realized) volatility from price series.

    Parameters
    ----------
    prices : array-like — Daily closing prices
    window : int — Rolling window in trading days
    annualize : bool — If True, annualize by √252

    Returns
    -------
    np.ndarray — Rolling volatility series
    """
    prices = np.array(prices, dtype=float)
    log_returns = np.diff(np.log(prices))

    if len(log_returns) < window:
        raise ValueError(f"Need at least {window} returns, got {len(log_returns)}")

    vol = np.array([
        np.std(log_returns[i:i + window], ddof=1)
        for i in range(len(log_returns) - window + 1)
    ])

    if annualize:
        vol *= np.sqrt(252)

    return vol


def realized_volatility(prices, annualize=True):
    """
    Calculate total realized volatility over the full period.

    Parameters
    ----------
    prices : array-like — Daily closing prices
    annualize : bool — Annualize the result

    Returns
    -------
    float — Realized volatility
    """
    prices = np.array(prices, dtype=float)
    log_returns = np.diff(np.log(prices))
    vol = np.std(log_returns, ddof=1)
    if annualize:
        vol *= np.sqrt(252)
    return vol
