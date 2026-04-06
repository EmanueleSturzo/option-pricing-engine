"""
Option Pricing Model Tests
============================
Validates model outputs against known analytical solutions.
"""

import numpy as np
from option_pricing import (
    BlackScholesModel,
    MonteCarloPricing,
    AmericanMonteCarlo,
    BinomialTreeModel,
    implied_volatility,
)


def test_black_scholes():
    """Test BSM against textbook values (Hull, Options Futures and Other Derivatives)."""
    bsm = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    call = bsm.call_price()
    put = bsm.put_price()

    # Known values for S=K=100, T=1, r=5%, σ=20%
    assert abs(call - 10.4506) < 0.01, f"Call price {call:.4f} != 10.4506"
    assert abs(put - 5.5735) < 0.01, f"Put price {put:.4f} != 5.5735"

    # Put-call parity
    assert bsm.put_call_parity_check() < 1e-10, "Put-call parity violated"

    # Greeks sanity checks
    assert 0 < bsm.delta("call") < 1, f"Call delta {bsm.delta('call')} out of range"
    assert -1 < bsm.delta("put") < 0, f"Put delta {bsm.delta('put')} out of range"
    assert bsm.gamma() > 0, "Gamma should be positive"
    assert bsm.vega() > 0, "Vega should be positive"
    assert bsm.theta("call") < 0, "Call theta should be negative"

    print("  ✓ Black-Scholes: all tests passed")


def test_monte_carlo_convergence():
    """MC should converge to BSM for European options."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    bsm = BlackScholesModel(S, K, T, r, sigma)
    bs_call = bsm.call_price()
    bs_put = bsm.put_price()

    mc = MonteCarloPricing(S, K, T, r, sigma, n_simulations=200000)
    mc_call = mc.price("call")["price"]
    mc_put = mc.price("put")["price"]

    assert abs(mc_call - bs_call) < 0.15, f"MC call {mc_call:.4f} too far from BS {bs_call:.4f}"
    assert abs(mc_put - bs_put) < 0.15, f"MC put {mc_put:.4f} too far from BS {bs_put:.4f}"

    print("  ✓ Monte Carlo: convergence to BSM verified")


def test_binomial_convergence():
    """Binomial tree should converge to BSM for European options."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    bsm = BlackScholesModel(S, K, T, r, sigma)
    bs_call = bsm.call_price()

    bt = BinomialTreeModel(S, K, T, r, sigma, n_steps=1000)
    bt_call = bt.price("call")

    assert abs(bt_call - bs_call) < 0.05, f"Tree call {bt_call:.4f} too far from BS {bs_call:.4f}"

    print("  ✓ Binomial Tree: convergence to BSM verified")


def test_american_premium():
    """American put should be worth >= European put."""
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.30

    bt_eu = BinomialTreeModel(S, K, T, r, sigma, american=False)
    bt_am = BinomialTreeModel(S, K, T, r, sigma, american=True)

    eu_put = bt_eu.price("put")
    am_put = bt_am.price("put")

    assert am_put >= eu_put - 0.001, f"American put {am_put:.4f} < European put {eu_put:.4f}"

    # American call on non-dividend stock = European call
    eu_call = bt_eu.price("call")
    am_call = bt_am.price("call")
    assert abs(am_call - eu_call) < 0.05, "American call should equal European call (no dividends)"

    print("  ✓ American options: early exercise premium verified")


def test_implied_volatility():
    """IV solver should recover the input volatility."""
    S, K, T, r, true_sigma = 100, 100, 0.5, 0.05, 0.25

    bsm = BlackScholesModel(S, K, T, r, true_sigma)
    market_price = bsm.call_price()

    iv = implied_volatility(market_price, S, K, T, r, "call")
    assert abs(iv - true_sigma) < 0.001, f"IV {iv:.4f} != true σ {true_sigma:.4f}"

    print("  ✓ Implied Volatility: solver accuracy verified")


def test_put_call_parity():
    """Verify put-call parity across different parameters."""
    test_cases = [
        (50, 55, 0.25, 0.03, 0.30),
        (200, 180, 2.0, 0.08, 0.15),
        (100, 100, 0.5, 0.05, 0.40),
    ]
    for S, K, T, r, sigma in test_cases:
        bsm = BlackScholesModel(S, K, T, r, sigma)
        error = bsm.put_call_parity_check()
        assert error < 1e-10, f"Parity error {error:.2e} for S={S}, K={K}"

    print("  ✓ Put-Call Parity: verified across multiple parameter sets")


if __name__ == "__main__":
    print("\n  Running Option Pricing Tests")
    print("  " + "─" * 45)

    test_black_scholes()
    test_monte_carlo_convergence()
    test_binomial_convergence()
    test_american_premium()
    test_implied_volatility()
    test_put_call_parity()

    print("  " + "─" * 45)
    print("  All tests passed ✓\n")
