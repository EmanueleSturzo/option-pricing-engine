"""
Monte Carlo Option Pricing
===========================
Simulates price paths using Geometric Brownian Motion (GBM) to price options.

Models:
    - European options: Standard MC with GBM paths
    - American options: Longstaff-Schwartz least-squares regression for early exercise

GBM: dS = S * (r - q) * dt + S * sigma * dW
"""

import numpy as np


class MonteCarloPricing:
    """
    Monte Carlo simulation for European option pricing.

    Parameters
    ----------
    S : float — Spot price
    K : float — Strike price
    T : float — Time to maturity (years)
    r : float — Risk-free rate
    sigma : float — Volatility
    q : float — Dividend yield (default: 0)
    n_simulations : int — Number of simulated paths
    n_steps : int — Time steps per path
    seed : int — Random seed for reproducibility
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, n_simulations=100000, n_steps=252, seed=42):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.seed = seed

    def _generate_paths(self):
        """Generate asset price paths using GBM."""
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        Z = rng.standard_normal((self.n_simulations, self.n_steps))
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)
        log_paths = np.column_stack([np.zeros(self.n_simulations), log_paths])

        paths = self.S * np.exp(log_paths)
        return paths

    def price(self, option_type="call"):
        """Price a European option via Monte Carlo simulation."""
        paths = self._generate_paths()
        ST = paths[:, -1]  # Terminal prices

        if option_type.lower() == "call":
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)

        discounted = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted)
        std_error = np.std(discounted) / np.sqrt(self.n_simulations)

        return {"price": price, "std_error": std_error}

    def get_paths(self, n_paths=50):
        """Return a subset of simulated paths for visualization."""
        paths = self._generate_paths()
        return paths[:n_paths, :]


class AmericanMonteCarlo:
    """
    Longstaff-Schwartz Monte Carlo for American option pricing.

    Uses least-squares regression at each time step to estimate
    the continuation value and determine optimal early exercise.

    Parameters
    ----------
    S, K, T, r, sigma, q : same as MonteCarloPricing
    n_simulations : int — Number of paths (default: 50000)
    n_steps : int — Time steps (default: 100)
    poly_degree : int — Degree of polynomial for regression (default: 3)
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, n_simulations=50000,
                 n_steps=100, poly_degree=3, seed=42):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.poly_degree = poly_degree
        self.seed = seed

    def price(self, option_type="put"):
        """
        Price an American option using the Longstaff-Schwartz algorithm.

        The LS method works backward through time:
        1. At each step, calculate the immediate exercise payoff
        2. For in-the-money paths, regress discounted future cashflows
           on current stock prices to estimate continuation value
        3. Exercise early if immediate payoff > estimated continuation value
        """
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        discount = np.exp(-self.r * dt)

        # Generate paths
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        Z = rng.standard_normal((self.n_simulations, self.n_steps))
        log_returns = drift + diffusion * Z
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(log_returns[:, t - 1])

        # Payoff function
        if option_type.lower() == "call":
            payoff = lambda s: np.maximum(s - self.K, 0)
        else:
            payoff = lambda s: np.maximum(self.K - s, 0)

        # Initialize cashflow matrix with terminal payoffs
        cashflows = payoff(paths[:, -1])

        # Backward induction
        for t in range(self.n_steps - 1, 0, -1):
            exercise = payoff(paths[:, t])
            itm = exercise > 0  # In-the-money paths

            if np.sum(itm) == 0:
                cashflows *= discount
                continue

            # Regression: continuation value ~ polynomial of spot price
            X = paths[itm, t]
            Y = cashflows[itm] * discount

            try:
                coeffs = np.polyfit(X, Y, self.poly_degree)
                continuation = np.polyval(coeffs, X)
            except (np.linalg.LinAlgError, ValueError):
                continuation = Y

            # Exercise where immediate payoff > estimated continuation
            early_exercise = exercise[itm] > continuation
            cashflows[itm] = np.where(early_exercise, exercise[itm], cashflows[itm] * discount)
            cashflows[~itm] *= discount

        # Discount to present
        cashflows *= discount
        price = np.mean(cashflows)
        std_error = np.std(cashflows) / np.sqrt(self.n_simulations)

        return {"price": price, "std_error": std_error}
