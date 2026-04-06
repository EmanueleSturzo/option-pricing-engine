"""
Black-Scholes-Merton Option Pricing Model
==========================================
Calculates European option prices and Greeks (1st, 2nd, and 3rd order).

The BSM model assumes:
    - European-style options (exercisable only at expiry)
    - Constant volatility and risk-free rate
    - Log-normal distribution of asset returns
    - No dividends (unless dividend yield q is specified)
    - No transaction costs or taxes
"""

import numpy as np
from scipy.stats import norm


class BlackScholesModel:
    """
    Black-Scholes-Merton pricing model for European options.

    Parameters
    ----------
    S : float — Current spot price of the underlying asset
    K : float — Strike price of the option
    T : float — Time to maturity in years
    r : float — Risk-free interest rate (annualized)
    sigma : float — Volatility of the underlying asset (annualized)
    q : float — Continuous dividend yield (default: 0)
    """

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S = S
        self.K = K
        self.T = max(T, 1e-10)  # Avoid division by zero
        self.r = r
        self.sigma = sigma
        self.q = q
        self._compute_d()

    def _compute_d(self):
        """Compute d1 and d2 used in the BSM formula."""
        self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    # ─── Option Prices ───────────────────────────────────────────

    def call_price(self):
        """European call option price."""
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
                - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))

    def put_price(self):
        """European put option price."""
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
                - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))

    def price(self, option_type="call"):
        """Calculate option price. option_type: 'call' or 'put'."""
        if option_type.lower() == "call":
            return self.call_price()
        return self.put_price()

    # ─── First-Order Greeks ──────────────────────────────────────

    def delta(self, option_type="call"):
        """Sensitivity of option price to $1 change in spot price."""
        if option_type.lower() == "call":
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        return np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)

    def theta(self, option_type="call"):
        """Time decay: sensitivity to passage of 1 day."""
        common = -(self.S * self.sigma * np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (2 * np.sqrt(self.T))
        if option_type.lower() == "call":
            t = common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2) + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
        else:
            t = common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
        return t / 365  # Per-day theta

    def vega(self):
        """Sensitivity to 1% change in volatility."""
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100

    def rho(self, option_type="call"):
        """Sensitivity to 1% change in risk-free rate."""
        if option_type.lower() == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100

    # ─── Second-Order Greeks ─────────────────────────────────────

    def gamma(self):
        """Rate of change of delta per $1 move in spot."""
        return np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vanna(self):
        """Sensitivity of delta to volatility (cross-Greek)."""
        return -np.exp(-self.q * self.T) * norm.pdf(self.d1) * self.d2 / self.sigma

    def charm(self):
        """Rate of change of delta over time (delta decay)."""
        return -np.exp(-self.q * self.T) * (
            norm.pdf(self.d1) * (2 * (self.r - self.q) * self.T - self.d2 * self.sigma * np.sqrt(self.T))
            / (2 * self.T * self.sigma * np.sqrt(self.T))
        )

    # ─── Third-Order Greeks ──────────────────────────────────────

    def speed(self):
        """Rate of change of gamma per $1 move in spot."""
        return -self.gamma() / self.S * (self.d1 / (self.sigma * np.sqrt(self.T)) + 1)

    # ─── Summary ─────────────────────────────────────────────────

    def greeks(self, option_type="call"):
        """Return a dictionary of all Greeks for the given option type."""
        return {
            "price": self.price(option_type),
            "delta": self.delta(option_type),
            "gamma": self.gamma(),
            "theta": self.theta(option_type),
            "vega": self.vega(),
            "rho": self.rho(option_type),
            "vanna": self.vanna(),
            "charm": self.charm(),
        }

    # ─── Put-Call Parity ─────────────────────────────────────────

    def put_call_parity_check(self):
        """
        Verify put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
        Returns the parity error (should be ~0).
        """
        lhs = self.call_price() - self.put_price()
        rhs = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        return abs(lhs - rhs)
