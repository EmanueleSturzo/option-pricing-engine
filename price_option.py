"""
Option Pricing Engine — CLI
=============================
Price European and American options using multiple models,
calculate Greeks, and solve for implied volatility.

Usage:
    python price_option.py --spot 100 --strike 105 --maturity 0.5 --rate 0.05 --vol 0.20
    python price_option.py --spot 100 --strike 95 --maturity 1.0 --rate 0.05 --vol 0.30 --type put
    python price_option.py --spot 100 --strike 100 --maturity 0.25 --rate 0.05 --vol 0.25 --american
    python price_option.py --spot 150 --strike 155 --maturity 0.5 --rate 0.05 --market-price 8.50
"""

import numpy as np
import warnings
from argparse import ArgumentParser
from option_pricing import (
    BlackScholesModel,
    MonteCarloPricing,
    AmericanMonteCarlo,
    BinomialTreeModel,
    implied_volatility,
)


def format_price(val, width=10):
    return f"${val:>{width}.4f}"


def format_greek(val, width=10):
    return f"{val:>{width}.6f}"


def main():
    warnings.filterwarnings("ignore")

    parser = ArgumentParser(description="Option Pricing Engine")
    parser.add_argument("--spot", type=float, required=True, help="Current spot price")
    parser.add_argument("--strike", type=float, required=True, help="Strike price")
    parser.add_argument("--maturity", type=float, required=True, help="Time to maturity (years)")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    parser.add_argument("--vol", type=float, default=0.20, help="Volatility (default: 0.20)")
    parser.add_argument("--div", type=float, default=0.0, help="Dividend yield (default: 0)")
    parser.add_argument("--type", default="call", choices=["call", "put"], help="Option type")
    parser.add_argument("--american", action="store_true", help="Price as American option")
    parser.add_argument("--market-price", type=float, default=None, help="Market price (to solve for IV)")
    parser.add_argument("--mc-sims", type=int, default=100000, help="Monte Carlo simulations")
    parser.add_argument("--tree-steps", type=int, default=500, help="Binomial tree steps")
    args = parser.parse_args()

    S, K, T, r, sigma, q = args.spot, args.strike, args.maturity, args.rate, args.vol, args.div
    opt = args.type
    sep = "=" * 65

    print(f"\n{sep}")
    print(f"  OPTION PRICING ENGINE")
    print(f"{sep}")
    print(f"  Underlying Spot:    ${S:.2f}")
    print(f"  Strike Price:       ${K:.2f}")
    print(f"  Time to Maturity:   {T:.4f} years ({T*365:.0f} days)")
    print(f"  Risk-Free Rate:     {r:.2%}")
    print(f"  Volatility (σ):     {sigma:.2%}")
    print(f"  Dividend Yield:     {q:.2%}")
    print(f"  Option Type:        {opt.upper()}")
    print(f"  Style:              {'AMERICAN' if args.american else 'EUROPEAN'}")

    # ─── Implied Volatility ──────────────────────────────────────
    if args.market_price is not None:
        print(f"\n  IMPLIED VOLATILITY SOLVER")
        print(f"  {'─'*45}")
        print(f"  Market Price:       ${args.market_price:.4f}")
        iv = implied_volatility(args.market_price, S, K, T, r, opt, q)
        print(f"  Implied Volatility: {iv:.4%}")
        print(f"  (Using Newton-Raphson with BSM)")
        sigma = iv  # Use IV for subsequent pricing
        print(f"  → Using IV={iv:.4%} for pricing below")

    # ─── Black-Scholes Model ─────────────────────────────────────
    print(f"\n  BLACK-SCHOLES MODEL")
    print(f"  {'─'*45}")
    bsm = BlackScholesModel(S, K, T, r, sigma, q)

    call_price = bsm.call_price()
    put_price = bsm.put_price()

    print(f"  Call Price:         {format_price(call_price)}")
    print(f"  Put Price:          {format_price(put_price)}")

    # Greeks
    greeks = bsm.greeks(opt)
    print(f"\n  GREEKS ({opt.upper()})")
    print(f"  {'─'*45}")
    print(f"  Delta (Δ):          {format_greek(greeks['delta'])}")
    print(f"  Gamma (Γ):          {format_greek(greeks['gamma'])}")
    print(f"  Theta (Θ):          {format_greek(greeks['theta'])}  (per day)")
    print(f"  Vega (ν):           {format_greek(greeks['vega'])}  (per 1% σ)")
    print(f"  Rho (ρ):            {format_greek(greeks['rho'])}  (per 1% r)")
    print(f"  Vanna:              {format_greek(greeks['vanna'])}")
    print(f"  Charm:              {format_greek(greeks['charm'])}")

    # Put-Call Parity
    parity_error = bsm.put_call_parity_check()
    print(f"\n  Put-Call Parity Error: {parity_error:.2e} {'✓' if parity_error < 1e-8 else '✗'}")

    # ─── Monte Carlo Simulation ──────────────────────────────────
    print(f"\n  MONTE CARLO SIMULATION ({args.mc_sims:,} paths)")
    print(f"  {'─'*45}")

    mc = MonteCarloPricing(S, K, T, r, sigma, q, n_simulations=args.mc_sims)
    mc_result = mc.price(opt)
    print(f"  European {opt.upper()} Price: {format_price(mc_result['price'])}")
    print(f"  Std Error:          {format_greek(mc_result['std_error'])}")
    print(f"  95% CI:             [{format_price(mc_result['price'] - 1.96*mc_result['std_error'])}, "
          f"{format_price(mc_result['price'] + 1.96*mc_result['std_error'])}]")

    if args.american:
        print(f"\n  LONGSTAFF-SCHWARTZ AMERICAN MC")
        print(f"  {'─'*45}")
        amc = AmericanMonteCarlo(S, K, T, r, sigma, q, n_simulations=min(args.mc_sims, 50000))
        amc_result = amc.price(opt)
        print(f"  American {opt.upper()} Price: {format_price(amc_result['price'])}")
        print(f"  Std Error:          {format_greek(amc_result['std_error'])}")
        print(f"  Early Exercise Premium: {format_price(amc_result['price'] - mc_result['price'])}")

    # ─── Binomial Tree Model ─────────────────────────────────────
    print(f"\n  BINOMIAL TREE MODEL ({args.tree_steps} steps)")
    print(f"  {'─'*45}")

    bt_eu = BinomialTreeModel(S, K, T, r, sigma, q, n_steps=args.tree_steps, american=False)
    eu_price = bt_eu.price(opt)
    print(f"  European {opt.upper()} Price: {format_price(eu_price)}")

    if args.american:
        bt_am = BinomialTreeModel(S, K, T, r, sigma, q, n_steps=args.tree_steps, american=True)
        am_price = bt_am.price(opt)
        print(f"  American {opt.upper()} Price: {format_price(am_price)}")
        print(f"  Early Exercise Premium: {format_price(am_price - eu_price)}")

    # ─── Model Comparison ────────────────────────────────────────
    print(f"\n  MODEL COMPARISON")
    print(f"  {'─'*45}")
    print(f"  {'Model':<30} {'Price':>12}")
    print(f"  {'─'*42}")
    print(f"  {'Black-Scholes (analytical)':<30} {format_price(bsm.price(opt))}")
    print(f"  {'Monte Carlo (European)':<30} {format_price(mc_result['price'])}")
    print(f"  {'Binomial Tree (European)':<30} {format_price(eu_price)}")
    if args.american:
        print(f"  {'Monte Carlo (American, LS)':<30} {format_price(amc_result['price'])}")
        print(f"  {'Binomial Tree (American)':<30} {format_price(am_price)}")

    bs_price = bsm.price(opt)
    print(f"\n  Convergence vs BSM:")
    print(f"  MC error:    {abs(mc_result['price'] - bs_price):.4f} ({abs(mc_result['price'] - bs_price)/bs_price:.4%})")
    print(f"  Tree error:  {abs(eu_price - bs_price):.4f} ({abs(eu_price - bs_price)/bs_price:.4%})")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
