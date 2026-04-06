"""
Option Pricing Engine — Streamlit Web App
==========================================
Run: python -m streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from option_pricing import (
    BlackScholesModel,
    MonteCarloPricing,
    AmericanMonteCarlo,
    BinomialTreeModel,
    implied_volatility,
)

st.set_page_config(page_title="Option Pricing Engine", page_icon="📈", layout="wide")
st.title("📈 Option Pricing Engine")

# ─── Top Navigation ──────────────────────────────────────────
tab_bs, tab_mc, tab_bt = st.tabs(["Black-Scholes Model", "Monte Carlo Simulation", "Binomial Model"])


# ══════════════════════════════════════════════════════════════
#  TAB 1: BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════
with tab_bs:
    st.header("Black-Scholes Model")
    st.markdown("Closed-form analytical solution for European options with full Greeks calculation.")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("Parameters")
        bs_spot = st.number_input("Spot Price ($)", min_value=0.01, value=100.0, step=1.0, key="bs_spot")
        bs_strike = st.number_input("Strike Price ($)", min_value=0.01, value=105.0, step=1.0, key="bs_strike")
        bs_maturity = st.number_input("Time to Maturity (years)", min_value=0.01, max_value=10.0, value=0.50, step=0.05, key="bs_mat")
        bs_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.25, key="bs_rate") / 100
        bs_sigma = st.number_input("Volatility (%)", min_value=0.1, max_value=200.0, value=25.0, step=1.0, key="bs_vol") / 100
        bs_div = st.number_input("Dividend Yield (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="bs_div") / 100

        st.markdown("---")
        st.subheader("Implied Volatility")
        bs_use_iv = st.checkbox("Solve IV from market price", key="bs_iv")
        bs_market = st.number_input("Market Option Price ($)", min_value=0.01, value=6.0, step=0.1, key="bs_mkt", disabled=not bs_use_iv)
        bs_iv_type = st.selectbox("Option type for IV", ["Call", "Put"], key="bs_iv_type", disabled=not bs_use_iv)

    with col_result:
        if bs_use_iv:
            try:
                iv = implied_volatility(bs_market, bs_spot, bs_strike, bs_maturity, bs_rate, bs_iv_type.lower(), bs_div)
                st.success(f"**Implied Volatility: {iv:.4%}** (Newton-Raphson, from market price ${bs_market:.2f})")
                bs_sigma = iv
            except Exception as e:
                st.error(f"IV solver failed: {e}")

        bsm = BlackScholesModel(bs_spot, bs_strike, bs_maturity, bs_rate, bs_sigma, bs_div)

        st.subheader("Option Prices")
        pc1, pc2 = st.columns(2)
        pc1.metric("Call Price", f"${bsm.call_price():.4f}")
        pc2.metric("Put Price", f"${bsm.put_price():.4f}")

        parity = bsm.put_call_parity_check()
        st.caption(f"Put-Call Parity Error: {parity:.2e} {'✅' if parity < 1e-8 else '❌'}")

        st.subheader("Greeks")
        greeks_type = st.radio("Show Greeks for:", ["Call", "Put"], horizontal=True, key="bs_greek_type")
        g = bsm.greeks(greeks_type.lower())

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta (Δ)", f"{g['delta']:.4f}")
        g2.metric("Gamma (Γ)", f"{g['gamma']:.6f}")
        g3.metric("Theta (Θ)", f"{g['theta']:.6f}")
        g4.metric("Vega (ν)", f"{g['vega']:.4f}")
        g5.metric("Rho (ρ)", f"{g['rho']:.4f}")

        with st.expander("2nd Order Greeks"):
            g6, g7 = st.columns(2)
            g6.metric("Vanna", f"{g['vanna']:.6f}")
            g7.metric("Charm", f"{g['charm']:.6f}")

        st.subheader("Greeks Across Spot Prices")
        spot_range = np.linspace(bs_spot * 0.5, bs_spot * 1.5, 100)
        deltas_c, deltas_p, gammas = [], [], []
        for s in spot_range:
            b = BlackScholesModel(s, bs_strike, bs_maturity, bs_rate, bs_sigma, bs_div)
            deltas_c.append(b.delta("call"))
            deltas_p.append(b.delta("put"))
            gammas.append(b.gamma())

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Delta**")
            st.line_chart(pd.DataFrame({"Call Delta": deltas_c, "Put Delta": deltas_p}, index=np.round(spot_range, 1)))
        with ch2:
            st.markdown("**Gamma**")
            st.line_chart(pd.DataFrame({"Gamma": gammas}, index=np.round(spot_range, 1)))

        st.subheader("Price Sensitivity to Volatility")
        vol_range = np.linspace(0.05, 0.80, 50)
        cv, pv = [], []
        for v in vol_range:
            b = BlackScholesModel(bs_spot, bs_strike, bs_maturity, bs_rate, v, bs_div)
            cv.append(b.call_price())
            pv.append(b.put_price())
        st.line_chart(pd.DataFrame({"Call": cv, "Put": pv}, index=[f"{v:.0%}" for v in vol_range]))


# ══════════════════════════════════════════════════════════════
#  TAB 2: MONTE CARLO
# ══════════════════════════════════════════════════════════════
with tab_mc:
    st.header("Monte Carlo Simulation")
    st.markdown("Prices options by simulating random price paths using Geometric Brownian Motion.")

    col_input2, col_result2 = st.columns([1, 2])

    with col_input2:
        st.subheader("Parameters")
        mc_spot = st.number_input("Spot Price ($)", min_value=0.01, value=100.0, step=1.0, key="mc_spot")
        mc_strike = st.number_input("Strike Price ($)", min_value=0.01, value=105.0, step=1.0, key="mc_strike")
        mc_maturity = st.number_input("Time to Maturity (years)", min_value=0.01, max_value=10.0, value=0.50, step=0.05, key="mc_mat")
        mc_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.25, key="mc_rate") / 100
        mc_sigma = st.number_input("Volatility (%)", min_value=0.1, max_value=200.0, value=25.0, step=1.0, key="mc_vol") / 100
        mc_div = st.number_input("Dividend Yield (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="mc_div") / 100
        mc_sims = st.number_input("Number of Simulations", min_value=1000, max_value=500000, value=100000, step=10000, key="mc_sims")
        mc_steps = st.number_input("Time Steps", min_value=10, max_value=1000, value=252, step=10, key="mc_steps")
        mc_american = st.checkbox("American Option (Longstaff-Schwartz)", key="mc_am")

    with col_result2:
        mc = MonteCarloPricing(mc_spot, mc_strike, mc_maturity, mc_rate, mc_sigma, mc_div,
                               n_simulations=mc_sims, n_steps=mc_steps)
        mc_call = mc.price("call")
        mc_put = mc.price("put")
        bsm_ref = BlackScholesModel(mc_spot, mc_strike, mc_maturity, mc_rate, mc_sigma, mc_div)

        st.subheader("European Option Prices")
        ec1, ec2 = st.columns(2)
        ec1.metric("Call Price", f"${mc_call['price']:.4f}",
                   delta=f"BSM: ${bsm_ref.call_price():.4f}", delta_color="off")
        ec2.metric("Put Price", f"${mc_put['price']:.4f}",
                   delta=f"BSM: ${bsm_ref.put_price():.4f}", delta_color="off")

        st.caption(f"Std Error — Call: ±${mc_call['std_error']:.4f} | Put: ±${mc_put['std_error']:.4f}")
        st.caption(f"95% CI (Call): [${mc_call['price'] - 1.96*mc_call['std_error']:.4f}, ${mc_call['price'] + 1.96*mc_call['std_error']:.4f}]")

        st.subheader("Convergence vs Black-Scholes")
        conv_data = {
            "": ["Call", "Put"],
            "Monte Carlo": [f"${mc_call['price']:.4f}", f"${mc_put['price']:.4f}"],
            "Black-Scholes": [f"${bsm_ref.call_price():.4f}", f"${bsm_ref.put_price():.4f}"],
            "Error": [f"${abs(mc_call['price'] - bsm_ref.call_price()):.4f}",
                      f"${abs(mc_put['price'] - bsm_ref.put_price()):.4f}"],
        }
        st.table(pd.DataFrame(conv_data).set_index(""))

        if mc_american:
            st.subheader("American Option (Longstaff-Schwartz)")
            amc = AmericanMonteCarlo(mc_spot, mc_strike, mc_maturity, mc_rate, mc_sigma, mc_div,
                                     n_simulations=min(mc_sims, 50000))
            am_call = amc.price("call")
            am_put = amc.price("put")
            ac1, ac2 = st.columns(2)
            ac1.metric("American Call", f"${am_call['price']:.4f}",
                       delta=f"Premium: ${am_call['price'] - mc_call['price']:.4f}")
            ac2.metric("American Put", f"${am_put['price']:.4f}",
                       delta=f"Premium: ${am_put['price'] - mc_put['price']:.4f}")

        st.subheader("Simulated Price Paths")
        n_display = st.slider("Paths to display", 10, 200, 50, key="mc_paths")
        paths = mc.get_paths(n_display)
        time_ax = np.linspace(0, mc_maturity, paths.shape[1])
        path_df = pd.DataFrame(paths.T, index=np.round(time_ax, 4))
        path_df.index.name = "Time (years)"
        st.line_chart(path_df, use_container_width=True)
        st.caption(f"Showing {n_display} of {mc_sims:,} simulated GBM paths")

        st.subheader("Terminal Price Distribution")
        all_paths = mc.get_paths(min(mc_sims, 10000))
        terminal = all_paths[:, -1]
        hist_df = pd.DataFrame({"Terminal Price": terminal})
        st.bar_chart(hist_df["Terminal Price"].value_counts(bins=80).sort_index())
        st.caption(f"Mean: ${np.mean(terminal):.2f} | Median: ${np.median(terminal):.2f} | Std: ${np.std(terminal):.2f}")


# ══════════════════════════════════════════════════════════════
#  TAB 3: BINOMIAL MODEL
# ══════════════════════════════════════════════════════════════
with tab_bt:
    st.header("Binomial Model")
    st.markdown("Cox-Ross-Rubinstein lattice model. Supports both European and American options.")

    col_input3, col_result3 = st.columns([1, 2])

    with col_input3:
        st.subheader("Parameters")
        bt_spot = st.number_input("Spot Price ($)", min_value=0.01, value=100.0, step=1.0, key="bt_spot")
        bt_strike = st.number_input("Strike Price ($)", min_value=0.01, value=105.0, step=1.0, key="bt_strike")
        bt_maturity = st.number_input("Time to Maturity (years)", min_value=0.01, max_value=10.0, value=0.50, step=0.05, key="bt_mat")
        bt_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.25, key="bt_rate") / 100
        bt_sigma = st.number_input("Volatility (%)", min_value=0.1, max_value=200.0, value=25.0, step=1.0, key="bt_vol") / 100
        bt_div = st.number_input("Dividend Yield (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="bt_div") / 100
        bt_steps = st.number_input("Number of Steps", min_value=10, max_value=5000, value=500, step=50, key="bt_steps")
        bt_american = st.checkbox("American Option", key="bt_am")

    with col_result3:
        bt_eu = BinomialTreeModel(bt_spot, bt_strike, bt_maturity, bt_rate, bt_sigma, bt_div,
                                  n_steps=bt_steps, american=False)
        bsm_ref2 = BlackScholesModel(bt_spot, bt_strike, bt_maturity, bt_rate, bt_sigma, bt_div)

        eu_call = bt_eu.price("call")
        eu_put = bt_eu.price("put")

        st.subheader("European Option Prices")
        bc1, bc2 = st.columns(2)
        bc1.metric("Call Price", f"${eu_call:.4f}",
                   delta=f"BSM: ${bsm_ref2.call_price():.4f}", delta_color="off")
        bc2.metric("Put Price", f"${eu_put:.4f}",
                   delta=f"BSM: ${bsm_ref2.put_price():.4f}", delta_color="off")

        st.caption(f"Error vs BSM — Call: ${abs(eu_call - bsm_ref2.call_price()):.4f} | Put: ${abs(eu_put - bsm_ref2.put_price()):.4f}")

        if bt_american:
            st.subheader("American Option Prices")
            bt_am = BinomialTreeModel(bt_spot, bt_strike, bt_maturity, bt_rate, bt_sigma, bt_div,
                                      n_steps=bt_steps, american=True)
            am_call_bt = bt_am.price("call")
            am_put_bt = bt_am.price("put")

            ba1, ba2 = st.columns(2)
            ba1.metric("American Call", f"${am_call_bt:.4f}",
                       delta=f"Premium: ${am_call_bt - eu_call:.4f}")
            ba2.metric("American Put", f"${am_put_bt:.4f}",
                       delta=f"Premium: ${am_put_bt - eu_put:.4f}")
            st.caption("For non-dividend-paying stocks, American call = European call (no early exercise benefit).")

        st.subheader("Tree Parameters")
        dt = bt_maturity / bt_steps
        u = np.exp(bt_sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((bt_rate - bt_div) * dt) - d) / (u - d)

        tp1, tp2, tp3, tp4 = st.columns(4)
        tp1.metric("Δt", f"{dt:.6f}")
        tp2.metric("Up (u)", f"{u:.6f}")
        tp3.metric("Down (d)", f"{d:.6f}")
        tp4.metric("Prob (p)", f"{p:.6f}")

        st.subheader("Convergence as Steps Increase")
        step_range = [10, 25, 50, 100, 200, 300, 500, 750, 1000]
        conv_calls, conv_puts = [], []
        for n in step_range:
            bt_temp = BinomialTreeModel(bt_spot, bt_strike, bt_maturity, bt_rate, bt_sigma, bt_div, n_steps=n)
            conv_calls.append(bt_temp.price("call"))
            conv_puts.append(bt_temp.price("put"))

        conv_df = pd.DataFrame({
            "Binomial Call": conv_calls,
            "Binomial Put": conv_puts,
            "BSM Call": [bsm_ref2.call_price()] * len(step_range),
            "BSM Put": [bsm_ref2.put_price()] * len(step_range),
        }, index=step_range)
        conv_df.index.name = "Steps"

        cv1, cv2 = st.columns(2)
        with cv1:
            st.markdown("**Call Price Convergence**")
            st.line_chart(conv_df[["Binomial Call", "BSM Call"]])
        with cv2:
            st.markdown("**Put Price Convergence**")
            st.line_chart(conv_df[["Binomial Put", "BSM Put"]])

        st.caption("The binomial model converges to the Black-Scholes solution as steps increase.")

        with st.expander("Convergence Data"):
            show_df = conv_df.copy()
            show_df["Call Error"] = abs(show_df["Binomial Call"] - show_df["BSM Call"])
            show_df["Put Error"] = abs(show_df["Binomial Put"] - show_df["BSM Put"])
            st.dataframe(show_df[["Binomial Call", "BSM Call", "Call Error", "Binomial Put", "BSM Put", "Put Error"]].round(4))


# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption("Black-Scholes · Monte Carlo · Binomial Tree · Longstaff-Schwartz · Greeks · Implied Volatility")
