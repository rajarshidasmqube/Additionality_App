
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------

def sample_activity_based(n, mu, sigma, rng):
    x = rng.normal(loc=mu, scale=sigma, size=n)
    return np.clip(x, 0, None)  # no negative emissions

def sample_lognormal(median, gsd, size, rng):
    # Convert median & geometric std to log-space params
    mu = np.log(median)
    sigma = np.log(gsd)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

def sample_direct_measurement(n, p_super, base_median, base_gsd, super_median, super_gsd, rng):
    # Mixture of two lognormals: background + super-emitters
    k_super = rng.binomial(n=n, p=p_super)
    k_base = n - k_super
    base = sample_lognormal(base_median, base_gsd, k_base, rng)
    sup = sample_lognormal(super_median, super_gsd, k_super, rng)
    x = np.concatenate([base, sup])
    rng.shuffle(x)
    return x, k_super

def ci_mean(x, alpha=0.05):
    m = np.mean(x)
    s = np.std(x, ddof=1)
    n = len(x)
    if n <= 1 or s == 0:
        return m, m, m
    z = 1.959963984540054  # ~ N(0,1) 97.5th
    half = z * s / np.sqrt(n)
    return m, m - half, m + half

def summarize(name, baseline, scenario, gwp=None, leakage_rate=0.0, permanence_years=0, annual_reversal=0.0):
    # Returns a dict of summary metrics for a scenario relative to baseline
    bl_mean, bl_lo, bl_hi = ci_mean(baseline)
    sc_mean, sc_lo, sc_hi = ci_mean(scenario)

    bl_total = np.sum(baseline)
    sc_total = np.sum(scenario)

    gross_add_mean = bl_mean - sc_mean
    gross_add_total = bl_total - sc_total

    # Convert to CO2e if GWP provided
    factor = 1.0 if (gwp is None or gwp <= 0) else gwp
    bl_mean_e = bl_mean * factor
    sc_mean_e = sc_mean * factor
    bl_total_e = bl_total * factor
    sc_total_e = sc_total * factor
    gross_add_mean_e = gross_add_mean * factor
    gross_add_total_e = gross_add_total * factor

    # Leakage and permanence
    retained_fraction = (1 - leakage_rate) * ((1 - annual_reversal) ** permanence_years)
    retained_fraction = max(0.0, min(1.0, retained_fraction))
    net_add_mean_e = gross_add_mean_e * retained_fraction
    net_add_total_e = gross_add_total_e * retained_fraction

    return {
        "Method": name,
        "Baseline Mean": bl_mean_e,
        "Scenario Mean": sc_mean_e,
        "Baseline Total": bl_total_e,
        "Scenario Total": sc_total_e,
        "Gross Additionality (Mean)": gross_add_mean_e,
        "Gross Additionality (Total)": gross_add_total_e,
        "Net Additionality (Mean)": net_add_mean_e,
        "Net Additionality (Total)": net_add_total_e,
        "Mean 95% CI (Baseline)": f"[{bl_lo*factor:.3g}, {bl_hi*factor:.3g}]",
        "Mean 95% CI (Scenario)": f"[{sc_lo*factor:.3g}, {sc_hi*factor:.3g}]",
        "Retained Fraction (Leakage & Permanence)": retained_fraction,
    }

def plot_histogram(baseline, scenario, title, bins=50):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(baseline, bins=bins, alpha=0.5, label='Baseline', density=True)
    ax.hist(scenario, bins=bins, alpha=0.5, label='Scenario', density=True)
    ax.set_title(title)
    ax.set_xlabel("Emissions per event (units)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="LNG Methane Project: Additionality (Ex-Ante vs Ex-Post)", layout="wide")
st.title("LNG Methane Credit Additionality — Ex‑Ante vs Ex‑Post")
st.caption("Compare activity‑based inventory (Normal) vs direct measurement (mixture Lognormal with super‑emitters). "
           "Model interventions that reduce **(1)** emission sizes, **(2)** super‑emission frequency, or **(1 & 2)** both. "
           "Adjust for **leakage** and check **permanence**.")

with st.sidebar:
    st.header("Simulation Controls")
    seed = st.number_input("Random seed", value=42, min_value=0, step=1)
    rng = np.random.default_rng(int(seed))

    n_events = st.number_input("Number of emission events (per period)", value=5000, min_value=100, step=100)
    periods = st.number_input("Number of periods to aggregate", value=1, min_value=1, step=1)
    total_events = int(n_events * periods)

    st.markdown("---")
    st.subheader("Units & Conversion")
    unit = st.text_input("Emission unit (e.g., kg CH₄)", value="kg CH₄")
    use_co2e = st.checkbox("Convert to CO₂e using GWP", value=True)
    gwp = st.number_input("Methane GWP (100-year)", value=28.0, min_value=0.0, step=1.0) if use_co2e else None

    st.markdown("---")
    st.subheader("Activity‑Based Inventory (Normal)")
    abi_mu = st.number_input("Baseline mean (Normal μ)", value=5.0, min_value=0.0, step=0.5)
    abi_sigma = st.number_input("Baseline std dev (Normal σ)", value=3.0, min_value=0.0, step=0.5)

    st.markdown("---")
    st.subheader("Direct Measurement — Mixture Lognormal")
    p_super = st.slider("Baseline super‑emitter probability", min_value=0.0, max_value=0.2, value=0.02, step=0.005)
    base_median = st.number_input("Background median (lognormal)", value=3.0, min_value=0.0, step=0.5)
    base_gsd = st.number_input("Background geometric std (lognormal)", value=1.6, min_value=1.0, step=0.1)
    super_median = st.number_input("Super‑emitter median (lognormal)", value=120.0, min_value=0.0, step=5.0)
    super_gsd = st.number_input("Super‑emitter geometric std (lognormal)", value=2.2, min_value=1.0, step=0.1)

    st.markdown("---")
    st.subheader("Interventions")
    st.caption("Choose the intervention(s) implemented by the project. Ex‑ante assumes these will be achieved; ex‑post samples represent realized outcomes.")
    size_reduction = st.slider("Reduce emission **size** (all non‑super sources)", min_value=0, max_value=100, value=30, step=5, help="Applies to ABI and the **background** component of Direct Measurement.")
    super_reduction = st.slider("Reduce **frequency** of super‑emitters", min_value=0, max_value=100, value=40, step=5, help="Reduces the probability of super‑emitter events in Direct Measurement.")

    use_different_seed_post = st.checkbox("Use a different random seed for ex‑post", value=True)

    st.markdown("---")
    st.subheader("Leakage & Permanence")
    leakage_rate = st.slider("Leakage rate (share of reductions leaked elsewhere)", min_value=0.0, max_value=0.9, value=0.1, step=0.05)
    permanence_years = st.number_input("Permanence horizon (years)", value=10, min_value=0, step=1)
    annual_reversal = st.slider("Annual reversal risk (re‑emission)", min_value=0.0, max_value=0.2, value=0.02, step=0.01)
    permanence_threshold = st.slider("Permanence pass threshold (retained fraction)", min_value=0.5, max_value=1.0, value=0.9, step=0.05)

# -----------------------------
# Baseline sampling (pre‑project)
# -----------------------------
baseline_abi = sample_activity_based(total_events, abi_mu, abi_sigma, rng)
baseline_dm, baseline_k_super = sample_direct_measurement(
    total_events, p_super, base_median, base_gsd, super_median, super_gsd, rng
)

# -----------------------------
# Ex‑ante scenarios (expected after project)
# -----------------------------
# Scenario parameters
size_factor = 1.0 - (size_reduction / 100.0)
p_super_post = p_super * (1.0 - (super_reduction / 100.0))

# ABI: reduce sizes by size_factor
ex_ante_abi = baseline_abi * size_factor

# DM: reduce background sizes by size_factor, reduce super frequency by p_super_post
# Build a fresh ex‑ante sample from modified parameters
ex_ante_dm_bg, _ = sample_direct_measurement(
    total_events, 0.0, base_median * size_factor, base_gsd, super_median, super_gsd, rng
)
ex_ante_dm_sup, _ = sample_direct_measurement(
    total_events, p_super_post, base_median * size_factor, base_gsd, super_median, super_gsd, rng
)
# Combine by taking the super‑emitter draw for p_super_post and background for the rest:
# The call above with p_super_post already includes background + super, so we use that as ex‑ante DM.
ex_ante_dm, _ = sample_direct_measurement(
    total_events, p_super_post, base_median * size_factor, base_gsd, super_median, super_gsd, rng
)

# -----------------------------
# Ex‑post sampling (realized after project)
# -----------------------------
if use_different_seed_post:
    rng_post = np.random.default_rng(int(seed) + 1)
else:
    rng_post = rng

ex_post_abi = sample_activity_based(total_events, abi_mu * size_factor, abi_sigma * size_factor, rng_post)

# For DM ex‑post: reduce background intensities, reduce super frequency
ex_post_dm, ex_post_k_super = sample_direct_measurement(
    total_events, p_super_post, base_median * size_factor, base_gsd, super_median, super_gsd, rng_post
)

# -----------------------------
# Summaries
# -----------------------------
# Gross & net additionality (with leakage/permanence)
sum_s1 = summarize("Activity‑Based (Normal)", baseline_abi, ex_post_abi, gwp, leakage_rate, permanence_years, annual_reversal)
sum_s2 = summarize("Direct Meas. (Mixture Lognormal)", baseline_dm, ex_post_dm, gwp, leakage_rate, permanence_years, annual_reversal)

df_summary = pd.DataFrame([sum_s1, sum_s2])
df_summary_display = df_summary.copy()
cols_fmt = ["Baseline Mean", "Scenario Mean", "Baseline Total", "Scenario Total",
            "Gross Additionality (Mean)", "Gross Additionality (Total)",
            "Net Additionality (Mean)", "Net Additionality (Total)"]
for c in cols_fmt:
    df_summary_display[c] = df_summary_display[c].map(lambda v: f"{v:,.3f}")

# -----------------------------
# Layout
# -----------------------------
st.subheader("Baseline vs Project (Ex‑Post Realized) — Summary")
unit_label = ("t CO₂e" if (gwp and gwp > 0) else unit)
st.caption(f"All values shown in **{unit_label}**. Net additionality reflects **leakage** and **permanence** settings.")
st.dataframe(df_summary_display, use_container_width=True)

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Activity‑Based Inventory (Normal)")
    plot_histogram(baseline_abi, ex_post_abi, "ABI: Baseline vs Ex‑Post")
    st.write(f"**Baseline mean:** {np.mean(baseline_abi) * (gwp if gwp else 1):,.3f} {unit_label}")
    st.write(f"**Ex‑Post mean:** {np.mean(ex_post_abi) * (gwp if gwp else 1):,.3f} {unit_label}")
with c2:
    st.markdown("#### Direct Measurement (Mixture Lognormal)")
    plot_histogram(baseline_dm, ex_post_dm, "Direct Measurement: Baseline vs Ex‑Post (fat tail)")
    st.write(f"**Baseline mean:** {np.mean(baseline_dm) * (gwp if gwp else 1):,.3f} {unit_label}")
    st.write(f"**Ex‑Post mean:** {np.mean(ex_post_dm) * (gwp if gwp else 1):,.3f} {unit_label}")

st.markdown("---")
st.subheader("Ex‑Ante vs Ex‑Post (Expected vs Realized)")
ante1 = summarize("Activity‑Based (Normal)", baseline_abi, ex_ante_abi, gwp, leakage_rate, permanence_years, annual_reversal)
ante2 = summarize("Direct Meas. (Mixture Lognormal)", baseline_dm, ex_ante_dm, gwp, leakage_rate, permanence_years, annual_reversal)

df_ante_post = pd.DataFrame([
    {"Method": "Activity‑Based (Normal)", "Phase": "Ex‑Ante Net Add (Mean)", "Value": ante1["Net Additionality (Mean)"]},
    {"Method": "Activity‑Based (Normal)", "Phase": "Ex‑Post Net Add (Mean)", "Value": sum_s1["Net Additionality (Mean)"]},
    {"Method": "Direct Meas. (Mixture Lognormal)", "Phase": "Ex‑Ante Net Add (Mean)", "Value": ante2["Net Additionality (Mean)"]},
    {"Method": "Direct Meas. (Mixture Lognormal)", "Phase": "Ex‑Post Net Add (Mean)", "Value": sum_s2["Net Additionality (Mean)"]},
])

# Table view
st.dataframe(df_ante_post.pivot(index="Method", columns="Phase", values="Value").applymap(lambda v: f"{v:,.3f}"), use_container_width=True)

# Permanence check
retained_fraction = (1 - leakage_rate) * ((1 - annual_reversal) ** permanence_years)
passes = retained_fraction >= permanence_threshold
st.markdown("---")
st.subheader("Leakage & Permanence Check")
st.write(f"**Retained fraction** = (1 − leakage) × (1 − annual reversal)^{permanence_years} = **{retained_fraction:.3f}**")
if passes:
    st.success(f"Permanence is **maintained** (≥ {permanence_threshold:.2f}).")
else:
    st.warning(f"Permanence **not** maintained (< {permanence_threshold:.2f}). Consider lowering leakage or reversal risk.")

# -----------------------------
# Downloads
# -----------------------------
# Detailed export with both gross & net add
detailed_rows = []
for label, bl, sc in [
    ("Activity‑Based (Normal)", baseline_abi, ex_post_abi),
    ("Direct Meas. (Mixture Lognormal)", baseline_dm, ex_post_dm),
]:
    s = summarize(label, bl, sc, gwp, leakage_rate, permanence_years, annual_reversal)
    detailed_rows.append(s)

df_export = pd.DataFrame(detailed_rows)
csv = df_export.to_csv(index=False).encode("utf-8")
st.download_button("Download results (CSV)", data=csv, file_name="lng_additionality_results.csv", mime="text/csv")

st.markdown("---")
with st.expander("Notes & Assumptions"):
    st.markdown(
        """
        - **Activity‑based inventory (ABI)** is modeled as a Normal distribution with non‑negative clipping; this typically **underestimates** true emissions.
        - **Direct measurement (DM)** uses a **mixture of lognormal distributions** to capture rare, high‑magnitude **super‑emissions**, yielding a **fat tail**.
        - **Intervention (1)** reduces the **size** of emissions in ABI and the **background** (non‑super) component of DM by the same factor.
        - **Intervention (2)** reduces the **probability** of super‑emitter events (DM only). ABI generally does **not** capture these rare events.
        - **Combined** applies both (1) and (2).
        - **Ex‑ante** reflects expected outcomes under modified parameters; **Ex‑post** reflects realized samples (optionally with a different random seed).
        - **Leakage** reduces credited reductions; **Permanence** applies an expected retention over the specified horizon via annual reversal risk.
        - Convert to **CO₂e** via a user‑specified **GWP** (default 28 for CH₄, 100‑yr). Units are otherwise reported in the chosen base unit.
        """
    )
