"""
Binomial Tree Option Pricing Model
====================================
Cox-Ross-Rubinstein (CRR) binomial lattice model.

Supports both European and American options. The binomial tree
converges to the Black-Scholes price as the number of steps increases.

At each node:
    - Up move:   u = e^(σ√Δt)
    - Down move:  d = 1/u
    - Risk-neutral probability: p = (e^((r-q)Δt) - d) / (u - d)
"""

import numpy as np


class BinomialTreeModel:
    """
    CRR Binomial Tree for option pricing.

    Parameters
    ----------
    S : float — Spot price
    K : float — Strike price
    T : float — Time to maturity (years)
    r : float — Risk-free rate
    sigma : float — Volatility
    q : float — Dividend yield (default: 0)
    n_steps : int — Number of time steps in the tree
    american : bool — If True, allows early exercise (American option)
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, n_steps=500, american=False):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_steps = n_steps
        self.american = american

        self.dt = T / n_steps
        self.u = np.exp(sigma * np.sqrt(self.dt))       # Up factor
        self.d = 1 / self.u                              # Down factor
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral prob
        self.discount = np.exp(-r * self.dt)

    def price(self, option_type="call"):
        """Calculate option price using backward induction through the tree."""
        n = self.n_steps

        # Terminal asset prices at expiry (all possible nodes)
        ST = self.S * self.u ** np.arange(n, -1, -1) * self.d ** np.arange(0, n + 1)

        # Terminal payoffs
        if option_type.lower() == "call":
            option_values = np.maximum(ST - self.K, 0)
        else:
            option_values = np.maximum(self.K - ST, 0)

        # Backward induction
        for t in range(n - 1, -1, -1):
            # Discounted expected value under risk-neutral measure
            option_values = self.discount * (self.p * option_values[:-1] + (1 - self.p) * option_values[1:])

            # American: check for early exercise
            if self.american:
                St = self.S * self.u ** np.arange(t, -1, -1) * self.d ** np.arange(0, t + 1)
                if option_type.lower() == "call":
                    exercise = np.maximum(St - self.K, 0)
                else:
                    exercise = np.maximum(self.K - St, 0)
                option_values = np.maximum(option_values, exercise)

        return option_values[0]

    def early_exercise_boundary(self, option_type="put"):
        """
        Calculate the early exercise boundary for American options.
        Returns an array of critical stock prices at each time step
        below (put) or above (call) which early exercise is optimal.

        Only meaningful for American options.
        """
        if not self.american:
            return None

        n = self.n_steps
        boundary = np.full(n + 1, np.nan)

        # Build full tree
        tree = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                tree[j, i] = self.S * self.u ** (i - j) * self.d ** j

        # Terminal payoffs
        if option_type.lower() == "call":
            values = np.maximum(tree[:n + 1, n] - self.K, 0)
        else:
            values = np.maximum(self.K - tree[:n + 1, n], 0)

        # Backward induction tracking exercise points
        for t in range(n - 1, -1, -1):
            for j in range(t + 1):
                hold = self.discount * (self.p * values[j] + (1 - self.p) * values[j + 1])
                S_node = tree[j, t]
                if option_type.lower() == "call":
                    exercise = max(S_node - self.K, 0)
                else:
                    exercise = max(self.K - S_node, 0)
                if exercise > hold and exercise > 0:
                    if np.isnan(boundary[t]):
                        boundary[t] = S_node
                values[j] = max(hold, exercise)

        return boundary
