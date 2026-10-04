#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pricing_visualised.py - Interactive risk-based pricing dashboard (Part C).

====================================================================
 MANUAL
====================================================================
Purpose
-------
Demonstrates risk-sensitive loan pricing for a 10,000 EUR
unsecured consumer instalment loan, combining

  * the A-IRB capital requirement (Part A, CRR Art. 154),
  * the expected-loss/provisioning view (Part B, IFRS 9),
  * a RAROC break-even calculation against a cost-of-equity
    hurdle (Part C).

The implementation follows the companion textbook
"Understanding RWA, Impairment, and Risk-Sensitive Pricing",
Part C.

Requirements
------------
Python 3.12 with the packages listed in requirements_py312.txt
(at minimum: flask, matplotlib, numpy).

How to run
----------
1. Spyder:  open this file and press F5 (Run File).  The Flask
   development server starts and the dashboard opens automatically
   in your browser (see console output for the URL).
2. Terminal: python Pricing_visualised.py
Then open http://127.0.0.1:5003 in a browser if it does not open
automatically.

How to use
----------
Set PD, LGD, EAD, the funding rate, the one-time operating cost,
the hurdle rate (cost of equity), the capital multiplier (1.0 =
regulatory capital; higher values approximate economic capital),
the EL pricing basis (12-month recurring or amortised lifetime),
and a candidate customer rate.  The dashboard shows the break-even
rate, its decomposition, and the RAROC curve of the candidate rate
against the hurdle.

Notes
-----
- The RAROC uses the average outstanding balance of the 3-year
  annuity at the candidate rate.
- Regulatory capital for low-PD retail loans is small; raising the
  capital multiplier is the standard way to approximate economic
  capital (see textbook C.3).
- All parameters are illustrative, not a bank's approved model.

Author: generated with AI assistance (see textbook preface).
"""

import base64
import io
import logging
import math
import webbrowser

import matplotlib

matplotlib.use("Agg")  # headless rendering; safe inside Spyder
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template_string, request

# ---------------------------------------------------------------------
# Constants (textbook baseline)
# ---------------------------------------------------------------------

CAPITAL_RATIO = 0.08
SCALING = 1.0 / CAPITAL_RATIO
CONFIDENCE = 0.999
CORR_FLOOR = 0.03
CORR_CEIL = 0.16
K_FACTOR = 35.0
PRINCIPAL = 10_000.0
YEARS = 3

DEFAULTS = {
    "pd": 0.02,
    "lgd": 0.45,
    "ead": 10_000.0,
    "funding": 0.02,
    "opex": 500.0,        # one-time, amortised over YEARS
    "hurdle": 0.12,       # cost of equity
    "cap_mult": 1.0,      # regulatory capital; >1 approximates economic
    "el_basis": "12m",    # "12m" or "lifetime"
    "rate": 0.085,        # candidate customer rate
}

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Risk-Based Pricing Dashboard</title>
<style>
 body { font-family: sans-serif; margin: 2em; background: #fafafa; }
 h1 { color: #1a355e; } h2 { color: #1a355e; border-bottom: 1px solid #ccc; }
 .kpis { display: flex; flex-wrap: wrap; gap: 1em; }
 .card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
         padding: 0.8em 1.2em; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
 .card b { font-size: 1.3em; color: #1a355e; }
 img { max-width: 100%; height: auto; margin: 1em 0;
       border: 1px solid #eee; border-radius: 6px; background: #fff; }
 form span { margin-right: 1.0em; }
 input[type=number] { width: 5.5em; }
 .note { color: #666; font-size: 0.9em; }
</style></head><body>
<h1>Risk-Sensitive Pricing - RAROC Break-Even</h1>
<form method="get">
 <span>PD: <input type="number" step="0.005" min="0" max="1"
      name="pd" value="{{ pd }}"></span>
 <span>LGD: <input type="number" step="0.05" min="0" max="1"
      name="lgd" value="{{ lgd }}"></span>
 <span>EAD (EUR): <input type="number" step="500" min="1"
      name="ead" value="{{ ead }}"></span>
 <span>Funding rate: <input type="number" step="0.005" min="0" max="0.5"
      name="funding" value="{{ funding }}"></span>
 <span>Opex one-time (EUR): <input type="number" step="50" min="0"
      name="opex" value="{{ opex }}"></span><br><br>
 <span>Hurdle (COE): <input type="number" step="0.01" min="0" max="0.5"
      name="hurdle" value="{{ hurdle }}"></span>
 <span>Capital multiplier: <input type="number" step="0.5" min="1" max="10"
      name="cap_mult" value="{{ cap_mult }}"></span>
 <span>EL basis:
   <select name="el_basis">
     <option value="12m" {{ "selected" if el_basis == "12m" }}>12-month</option>
     <option value="lifetime" {{ "selected" if el_basis == "lifetime" }}>lifetime</option>
   </select></span>
 <span>Candidate rate: <input type="number" step="0.005" min="0" max="0.5"
      name="rate" value="{{ rate }}"></span>
 <button type="submit">Apply</button>
</form>
<h2>Key figures</h2>
<div class="kpis">
 {% for label, value in kpis %}
 <div class="card">{{ label }}<br><b>{{ value }}</b></div>
 {% endfor %}
</div>
<h2>Break-even rate decomposition</h2>
<img src="data:image/png;base64,{{ chart_decomp }}">
<h2>RAROC vs customer rate (hurdle line and candidate)</h2>
<img src="data:image/png;base64,{{ chart_raroc }}">
<p class="note">Model: RAROC = ((r - funding) x avg balance - opex p.a.
- EL p.a.) / capital; capital = capital multiplier x 8% x RWA
(A-IRB, CRR Art. 154).  Illustrative parameters only - not a
bank's approved model and not a loan offer.</p>
</body></html>
"""


# ---------------------------------------------------------------------
# Model functions (pure, unit-testable)
# ---------------------------------------------------------------------

def _norm_cdf(x_value):
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x_value / math.sqrt(2.0)))


def _norm_cdf_inv(probability):
    """Standard normal inverse CDF via bisection (accurate ~1e-12)."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    low, high = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _norm_cdf(mid) < probability:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def retail_correlation(pd_value):
    """Asset correlation R(PD) for non-revolving retail (CRR Art. 154)."""
    share = (1.0 - math.exp(-K_FACTOR * pd_value)) / (1.0 - math.exp(-K_FACTOR))
    return CORR_FLOOR * share + CORR_CEIL * (1.0 - share)


def capital_factor(pd_value, lgd_value):
    """Regulatory capital requirement K per unit of EAD (A-IRB)."""
    correl = retail_correlation(pd_value)
    root = math.sqrt(1.0 - correl)
    argument = (_norm_cdf_inv(pd_value) / root
                + math.sqrt(correl / (1.0 - correl))
                * _norm_cdf_inv(CONFIDENCE))
    conditional_pd = _norm_cdf(argument)
    return lgd_value * (conditional_pd - pd_value), conditional_pd, correl


def annuity_schedule(principal, rate, years):
    """Amortisation schedule and average outstanding balance."""
    payment = principal * rate / (1.0 - (1.0 + rate) ** -years)
    balance = principal
    balances = []
    for _ in range(years):
        balances.append(balance)
        balance -= (payment - balance * rate)
    return payment, balances


def lifetime_weighted_ecl(pd_value, lgd_value, ead_value):
    """Probability-weighted lifetime ECL (textbook Part B, EUR).

    Marginal PDs step linearly (pd, 1.25*pd, 1.5*pd - the textbook
    baseline 2.0%/2.5%/3.0%), exposures follow the 3-year annuity
    at the 8.5% EIR, losses are discounted at that EIR, and the
    scenario weighting is 20/60/20 with PD scales 0.5/1/2
    (weighted scale 1.1).  At the defaults this reproduces the
    textbook value of 206.10 EUR.
    """
    payment, balances = annuity_schedule(ead_value, 0.085, YEARS)
    ecl = 0.0
    for index, balance in enumerate(balances):
        marginal_pd = pd_value * (1.0 + 0.25 * index)
        ecl += marginal_pd * lgd_value * balance / (1.085 ** (index + 1))
    return 1.1 * ecl


def compute_pricing(pd_value, lgd_value, ead_value, funding_rate, opex_total,
                    hurdle_rate, capital_multiplier, el_basis):
    """Break-even rate and decomposition for the parameter set.

    EL basis "12m" charges PD x LGD x EAD per year (conservative,
    ignores run-off); "lifetime" amortises the probability-weighted
    lifetime ECL over the loan life (scaled with the same scenario
    weights as the textbook: 20/60/20 optimistic/base/adverse).
    """
    k_value, conditional_pd, correl = capital_factor(pd_value, lgd_value)
    capital = capital_multiplier * CAPITAL_RATIO * SCALING * k_value * ead_value
    if el_basis == "lifetime":
        ecl_weighted = lifetime_weighted_ecl(pd_value, lgd_value, ead_value)
        el_per_year = ecl_weighted / YEARS
    else:
        el_per_year = pd_value * lgd_value * ead_value
    # Textbook convention: average balance from the 3-year annuity
    # at the 8.5% EIR used in Parts B and C of the textbook.
    payment, balances = annuity_schedule(ead_value, 0.085, YEARS)
    average_balance = sum(balances) / YEARS
    opex_per_year = opex_total / YEARS
    margin_needed = opex_per_year + el_per_year + hurdle_rate * capital
    required_rate = funding_rate + margin_needed / average_balance
    return {
        "k": k_value,
        "pd_stress": conditional_pd,
        "correlation": correl,
        "rw": SCALING * k_value,
        "rwa": SCALING * k_value * ead_value,
        "capital": capital,
        "el_per_year": el_per_year,
        "average_balance": average_balance,
        "opex_per_year": opex_per_year,
        "capital_charge": hurdle_rate * capital,
        "required_rate": required_rate,
        "margin": margin_needed / average_balance,
    }


def raroc(customer_rate, funding_rate, average_balance, opex_per_year,
          el_per_year, capital):
    """RAROC of a candidate customer rate (accounting approximation)."""
    if capital <= 0:
        return float("inf")
    net_income = ((customer_rate - funding_rate) * average_balance
                  - opex_per_year - el_per_year)
    return net_income / capital


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def _fig_to_base64(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_charts(result, pd_value, lgd_value, ead_value, funding_rate,
                opex_per_year, el_per_year, capital, hurdle_rate,
                candidate_rate, required_rate):
    """Build the two dashboard PNGs as base64 strings."""
    labels = ["operating cost", "expected loss", "cost of capital"]
    values = [opex_per_year, el_per_year, result["capital_charge"]]
    figure_d, axis_d = plt.subplots(figsize=(6.4, 3.6))
    bars = axis_d.bar(labels, values, color=["#5b8db8", "#c0704d", "#7fa87f"])
    for bar in bars:
        axis_d.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height() / result['average_balance']:.2%}",
                    ha="center", va="bottom")
    axis_d.set_ylabel("EUR per year")
    axis_d.set_title("Margin components (per average balance)")
    axis_d.grid(axis="y", alpha=0.3)

    grid = np.linspace(0.0, 0.25, 200)
    curve = [raroc(r, funding_rate, result["average_balance"], opex_per_year,
                   el_per_year, capital) for r in grid]
    figure_r, axis_r = plt.subplots(figsize=(6.4, 3.6))
    axis_r.plot(grid * 100, curve, lw=2, label="RAROC")
    axis_r.axhline(hurdle_rate, color="black", ls="--", lw=1,
                   label=f"hurdle = {hurdle_rate:.0%}")
    axis_r.axvline(candidate_rate * 100, color="red", ls=":",
                   label=f"candidate = {candidate_rate:.2%}")
    axis_r.axvline(required_rate * 100, color="green", ls="--",
                   label=f"break-even = {required_rate:.2%}")
    axis_r.set_xlabel("customer rate (%)"); axis_r.set_ylabel("RAROC")
    axis_r.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis_r.set_title("RAROC vs customer rate")
    axis_r.legend(); axis_r.grid(alpha=0.3)

    return _fig_to_base64(figure_d), _fig_to_base64(figure_r)


# ---------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------

app = Flask(__name__)


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


@app.route("/")
def index():
    """Render the pricing dashboard for the requested parameters."""
    pd_value = min(max(_to_float(request.args.get("pd"), DEFAULTS["pd"]),
                       0.0001), 0.9999)
    lgd_value = min(max(_to_float(request.args.get("lgd"), DEFAULTS["lgd"]),
                        0.0), 1.0)
    ead_value = max(_to_float(request.args.get("ead"), DEFAULTS["ead"]), 1.0)
    funding_rate = min(max(_to_float(request.args.get("funding"),
                                  DEFAULTS["funding"]), 0.0), 0.9)
    opex_total = max(_to_float(request.args.get("opex"), DEFAULTS["opex"]),
                     0.0)
    hurdle_rate = min(max(_to_float(request.args.get("hurdle"),
                                   DEFAULTS["hurdle"]), 0.0), 1.0)
    capital_multiplier = max(_to_float(request.args.get("cap_mult"),
                                       DEFAULTS["cap_mult"]), 0.1)
    el_basis = request.args.get("el_basis", DEFAULTS["el_basis"])
    if el_basis not in ("12m", "lifetime"):
        el_basis = "12m"
    candidate_rate = min(max(_to_float(request.args.get("rate"),
                                      DEFAULTS["rate"]), 0.0), 1.0)

    result = compute_pricing(pd_value, lgd_value, ead_value, funding_rate,
                             opex_total, hurdle_rate, capital_multiplier,
                             el_basis)
    current_raroc = raroc(candidate_rate, funding_rate,
                           result["average_balance"], result["opex_per_year"],
                           result["el_per_year"], result["capital"])
    logging.info("Break-even rate: %s", result["required_rate"])
    chart_d, chart_r = make_charts(result, pd_value, lgd_value, ead_value,
                                   funding_rate, result["opex_per_year"],
                                   result["el_per_year"], result["capital"],
                                   hurdle_rate, candidate_rate,
                                   result["required_rate"])
    kpis = [
        ("Capital factor K", f"{result['k']:.2%}"),
        ("Risk weight", f"{result['rw']:.1%}"),
        ("RWA", f"{result['rwa']:,.0f} EUR"),
        ("Allocated capital", f"{result['capital']:,.2f} EUR"),
        ("Average balance", f"{result['average_balance']:,.2f} EUR"),
        ("Break-even rate", f"{result['required_rate']:.2%}"),
        ("RAROC at candidate",
         f"{current_raroc:.1%} (vs hurdle {hurdle_rate:.0%})"),
    ]
    return render_template_string(
        PAGE_TEMPLATE,
        pd=f"{pd_value:.3f}", lgd=f"{lgd_value:.2f}", ead=f"{ead_value:.0f}",
        funding=f"{funding_rate:.3f}", opex=f"{opex_total:.0f}",
        hurdle=f"{hurdle_rate:.2f}", cap_mult=f"{capital_multiplier:.1f}",
        el_basis=el_basis, rate=f"{candidate_rate:.3f}",
        kpis=kpis, chart_decomp=chart_d, chart_raroc=chart_r)


def main():
    """Start the local dashboard server."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    url = "http://127.0.0.1:5003"
    print(f"Pricing dashboard running at {url} (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - headless environments
        pass
    # use_reloader=False: required in Spyder/IDE environments.
    app.run(host="127.0.0.1", port=5003, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
