"""
Option Pricing Engine
======================
A Python library for pricing European and American options.

Models:
    - BlackScholesModel: Analytical BSM with full Greeks
    - MonteCarloPricing: European options via GBM simulation
    - AmericanMonteCarlo: American options via Longstaff-Schwartz
    - BinomialTreeModel: CRR binomial lattice (European + American)

Tools:
    - implied_volatility: Newton-Raphson IV solver
    - historical_volatility: Rolling realized vol from prices
"""

from .black_scholes import BlackScholesModel
from .monte_carlo import MonteCarloPricing, AmericanMonteCarlo
from .binomial_tree import BinomialTreeModel
from .volatility import implied_volatility, historical_volatility, realized_volatility

__all__ = [
    "BlackScholesModel",
    "MonteCarloPricing",
    "AmericanMonteCarlo",
    "BinomialTreeModel",
    "implied_volatility",
    "historical_volatility",
    "realized_volatility",
]
