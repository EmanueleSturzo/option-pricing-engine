# Option Pricing Engine

A Python library for pricing European and American options using multiple quantitative models, with full Greeks calculation, implied volatility solving, and model convergence comparison.

## Models

| Model | Type | Method |
|---|---|---|
| **Black-Scholes-Merton** | European | Closed-form analytical solution with 1st/2nd/3rd order Greeks |
| **Monte Carlo Simulation** | European | Geometric Brownian Motion with 100,000 path simulation |
| **Longstaff-Schwartz Monte Carlo** | American | Least-squares regression for optimal early exercise |
| **Binomial Tree (CRR)** | European + American | Cox-Ross-Rubinstein lattice with configurable steps |

## Additional Features

- **Full Greeks**: Delta, Gamma, Theta, Vega, Rho, Vanna, Charm, Speed
- **Implied Volatility Solver**: Newton-Raphson method to reverse-engineer IV from market prices
- **Historical Volatility**: Rolling realized volatility from price data
- **Put-Call Parity Validation**: Automated parity check across all parameter sets
- **American Option Early Exercise Premium**: Quantifies the value of early exercise rights
- **Model Convergence Analysis**: Side-by-side comparison with error metrics vs analytical BSM

## Quick Start

# Clone the repo
git clone https://github.com/EmanueleSturzo/DCF-Valuation-Model.git
(download zip file and extract)

for Mac:
cd ~/Downloads/DCF-Valuation-Model-main

for windows:
cd C:\Users\YourName\Downloads\DCF-Valuation-Model-main

# Install dependencies
```bash
pip install -r requirements.txt

# Launch web app
streamlit run streamlit_app.py

or, if it doesn't work : python -m streamlit run streamlit_app.py

# Or use CLI
python price_option.py --spot 100 --strike 105 --maturity 0.5 --rate 0.05 --vol 0.25
python price_option.py --spot 100 --strike 110 --maturity 1.0 --rate 0.05 --vol 0.30 --type put --american
python price_option.py --spot 150 --strike 155 --maturity 0.5 --rate 0.05 --market-price 8.50

# Run tests
python tests.py
```

## Web App

The Streamlit web app provides an interactive interface with:

- **Sidebar controls** for spot price, strike, maturity, rate, volatility, dividend yield, and model settings
- **Price cards** showing results from all three models side by side
- **Greeks dashboard** with Delta, Gamma, Theta, Vega, Rho, Vanna, and Charm
- **Call vs Put comparison table** with put-call parity validation
- **Model convergence table** showing absolute and percentage error vs BSM
- **Monte Carlo paths chart** visualizing simulated GBM price trajectories
- **Delta & Gamma surface plots** across a range of spot prices
- **Volatility sensitivity chart** showing how call and put prices change with volatility
- **Implied volatility solver** from a market-observed option price
- **American option toggle** with early exercise premium calculation

## Sample Output

```
=================================================================
  OPTION PRICING ENGINE
=================================================================
  Underlying Spot:    $100.00
  Strike Price:       $110.00
  Time to Maturity:   1.0000 years (365 days)
  Risk-Free Rate:     5.00%
  Volatility (σ):     30.00%
  Option Type:        PUT
  Style:              AMERICAN

  BLACK-SCHOLES MODEL
  ─────────────────────────────────────────────
  Call Price:         $   10.0201
  Put Price:          $   14.6553

  GREEKS (PUT)
  ─────────────────────────────────────────────
  Delta (Δ):           -0.500412
  Gamma (Γ):            0.013298
  Theta (Θ):           -0.007532  (per day)
  Vega (ν):             0.398942  (per 1% σ)
  Rho (ρ):             -0.646966  (per 1% r)

  LONGSTAFF-SCHWARTZ AMERICAN MC
  ─────────────────────────────────────────────
  American PUT Price: $   15.5649
  Early Exercise Premium: $    0.9212

  MODEL COMPARISON
  ─────────────────────────────────────────────
  Black-Scholes (analytical)     $   14.6553
  Monte Carlo (European)         $   14.6437
  Binomial Tree (European)       $   14.6606
  Monte Carlo (American, LS)     $   15.5649
  Binomial Tree (American)       $   15.6222
```

## CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--spot` | Current spot price (required) | — |
| `--strike` | Strike price (required) | — |
| `--maturity` | Time to maturity in years (required) | — |
| `--rate` | Risk-free interest rate | 0.05 |
| `--vol` | Annualized volatility | 0.20 |
| `--div` | Continuous dividend yield | 0.0 |
| `--type` | Option type: `call` or `put` | call |
| `--american` | Enable American option pricing | False |
| `--market-price` | Market price (triggers IV solver) | None |
| `--mc-sims` | Number of Monte Carlo simulations | 100,000 |
| `--tree-steps` | Binomial tree time steps | 500 |

## Use as a Python Library

```python
from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel
from option_pricing import implied_volatility, AmericanMonteCarlo

# Black-Scholes with Greeks
bsm = BlackScholesModel(S=100, K=105, T=0.5, r=0.05, sigma=0.25)
print(f"Call: ${bsm.call_price():.4f}")
print(f"Delta: {bsm.delta('call'):.4f}")
print(f"Gamma: {bsm.gamma():.6f}")
print(f"All Greeks: {bsm.greeks('call')}")

# Monte Carlo
mc = MonteCarloPricing(S=100, K=105, T=0.5, r=0.05, sigma=0.25)
result = mc.price("call")
print(f"MC Price: ${result['price']:.4f} ± {result['std_error']:.4f}")

# American put (Longstaff-Schwartz)
amc = AmericanMonteCarlo(S=100, K=110, T=1.0, r=0.05, sigma=0.30)
result = amc.price("put")
print(f"American Put: ${result['price']:.4f}")

# Binomial tree
bt = BinomialTreeModel(S=100, K=105, T=0.5, r=0.05, sigma=0.25, american=True)
print(f"American Call: ${bt.price('call'):.4f}")

# Implied volatility
iv = implied_volatility(market_price=8.50, S=150, K=155, T=0.5, r=0.05)
print(f"Implied Vol: {iv:.2%}")
```

## Methodology

### Black-Scholes-Merton

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Monte Carlo (GBM)

$$S_T = S_0 \exp\left[(r - q - \tfrac{\sigma^2}{2})T + \sigma\sqrt{T}\,Z\right], \quad Z \sim N(0,1)$$

Price = discounted average payoff across simulated terminal prices.

### Binomial Tree (CRR)

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

Backward induction from terminal payoffs with optional early exercise check at each node.

### Longstaff-Schwartz (American MC)

At each time step, for in-the-money paths, regresses discounted future cashflows on current stock prices using polynomial basis functions. Exercises early when immediate payoff exceeds the estimated continuation value.

### Implied Volatility

Newton-Raphson iteration using BSM vega as the derivative:

$$\sigma_{n+1} = \sigma_n - \frac{C_{BSM}(\sigma_n) - C_{market}}{\text{vega}(\sigma_n)}$$

## Project Structure

```
option-pricing-engine/
├── option_pricing/
│   ├── __init__.py           # Package exports
│   ├── black_scholes.py      # BSM model with full Greeks
│   ├── monte_carlo.py        # European MC + American LS-MC
│   ├── binomial_tree.py      # CRR binomial lattice
│   └── volatility.py         # IV solver + historical vol
├── streamlit_app.py          # Web app (interactive UI)
├── price_option.py           # CLI application
├── tests.py                  # Test suite (6 tests)
├── requirements.txt
├── LICENSE
└── README.md
```

## References

- Hull, J.C. — *Options, Futures, and Other Derivatives*
- Longstaff, F.A. & Schwartz, E.S. (2001) — *Valuing American Options by Simulation*
- Cox, Ross & Rubinstein (1979) — *Option Pricing: A Simplified Approach*
- Black, F. & Scholes, M. (1973) — *The Pricing of Options and Corporate Liabilities*

## License

[MIT](LICENSE)
