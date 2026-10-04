#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWA_visualised.py - Interactive A-IRB RWA dashboard (Part A).

====================================================================
 MANUAL
====================================================================
Purpose
-------
Demonstrates how the risk-weighted assets (RWA) of a single unsecured
retail loan depend on the risk parameters PD, LGD and EAD under the
Advanced Internal Ratings-Based (A-IRB) approach of Regulation (EU)
No 575/2013 (CRR), Articles 153/154 (Basel/Gordy one-factor model).

The implementation follows the companion textbook
"Understanding RWA, Impairment, and Risk-Sensitive Pricing",
Part A.

Requirements
------------
Python 3.12 with the packages listed in requirements_py312.txt
(at minimum: flask, matplotlib, numpy).

How to run
----------
1. Spyder:  open this file and press F5 (Run File).  The Flask
   development server starts and the dashboard opens automatically
   in your browser (see console output for the URL).
2. Terminal: python RWA_visualised.py
Then open http://127.0.0.1:5001 in a browser if it does not open
automatically.

How to use
----------
Use the sliders/fields on the page to change PD, LGD and EAD.
The KPI panel and the three charts (risk weight vs PD, capital
factor vs LGD, RWA vs EAD) update on every "Apply" click.
Every number shown can be traced to the formulas printed in the
console log.

Notes
-----
- The 99.9% stress quantile, the correlation function and the
  12.5 scaling follow CRR Art. 154 (retail, non-revolving,
  unsecured).  Mortgage (R=0.15) and QRRE (R=0.04) sub-classes
  are not modelled.
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
# Constants (CRR Part A baseline)
# ---------------------------------------------------------------------

CAPITAL_RATIO = 0.08          # minimum own funds ratio (8%)
SCALING = 1.0 / CAPITAL_RATIO  # 12.5
CONFIDENCE = 0.999             # supervisory stress quantil
CORR_FLOOR = 0.03             # retail correlation floor
CORR_CEIL = 0.16              # retail correlation ceiling
K_FACTOR = 35.0               # steepness of the correlation function

DEFAULTS = {"pd": 0.02, "lgd": 0.45, "ead": 10_000.0}

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>A-IRB RWA Dashboard</title>
<style>
 body { font-family: sans-serif; margin: 2em; background: #fafafa; }
 h1 { color: #1a355e; } h2 { color: #1a355e; border-bottom: 1px solid #ccc; }
 .kpis { display: flex; flex-wrap: wrap; gap: 1em; }
 .card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
         padding: 0.8em 1.2em; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
 .card b { font-size: 1.3em; color: #1a355e; }
 img { max-width: 100%; height: auto; margin: 1em 0;
       border: 1px solid #eee; border-radius: 6px; background: #fff; }
 form span { margin-right: 1.2em; }
 input[type=number] { width: 6.5em; }
 .note { color: #666; font-size: 0.9em; }
</style></head><body>
<h1>Risk-Weighted Assets - A-IRB (Gordy formula, CRR Art. 154)</h1>
<form method="get">
 <span>PD (0-1): <input type="number" step="0.001" min="0" max="1"
      name="pd" value="{{ pd }}"></span>
 <span>LGD (0-1): <input type="number" step="0.01" min="0" max="1"
      name="lgd" value="{{ lgd }}"></span>
 <span>EAD (EUR): <input type="number" step="100" min="1"
      name="ead" value="{{ ead }}"></span>
 <button type="submit">Apply</button>
</form>
<h2>Key figures for the current loan</h2>
<div class="kpis">
 {% for label, value in kpis %}
 <div class="card">{{ label }}<br><b>{{ value }}</b></div>
 {% endfor %}
</div>
<h2>Risk weight as a function of PD (current LGD)</h2>
<img src="data:image/png;base64,{{ chart_rw }}">
<h2>Capital factor K and correlation R as functions of PD</h2>
<img src="data:image/png;base64,{{ chart_k }}">
<h2>RWA vs EAD (current PD, LGD)</h2>
<img src="data:image/png;base64,{{ chart_ead }}">
<p class="note">Model: CRR Art. 154 retail risk-weight function;
stress quantile 99.9%; RWA = 12.5 x K x EAD; capital = 8% x RWA.
Illustrative parameters only - not a bank's approved model.</p>
</body></html>
"""


# ---------------------------------------------------------------------
# Model functions (pure, unit-testable)
# ---------------------------------------------------------------------

def retail_correlation(pd_value):
    """Asset correlation R(PD) for non-revolving retail (CRR Art. 154).

    R = 0.03*A + 0.16*(1-A),  A = (1-exp(-35*PD))/(1-exp(-35)).
    """
    share = (1.0 - math.exp(-K_FACTOR * pd_value)) / (1.0 - math.exp(-K_FACTOR))
    return CORR_FLOOR * share + CORR_CEIL * (1.0 - share)


def stress_pd(pd_value):
    """Conditional PD in the 99.9% adverse systematic state (Vasicek)."""
    correl = retail_correlation(pd_value)
    root = math.sqrt(1.0 - correl)
    argument = (
        _norm_cdf_inv(pd_value) / root
        + math.sqrt(correl / (1.0 - correl)) * _norm_cdf_inv(CONFIDENCE)
    )
    return _norm_cdf(argument), correl


def capital_factor(pd_value, lgd_value):
    """Regulatory capital requirement K per unit of EAD."""
    conditional_pd, correl = stress_pd(pd_value)
    return lgd_value * (conditional_pd - pd_value), conditional_pd, correl


def risk_weight(pd_value, lgd_value):
    """Risk weight RW = 12.5 x K (a percentage-like fraction)."""
    k_value, _, _ = capital_factor(pd_value, lgd_value)
    return SCALING * k_value


def compute_rwa(pd_value, lgd_value, ead_value):
    """Return a result dictionary for one exposure."""
    k_value, conditional_pd, correl = capital_factor(pd_value, lgd_value)
    rw_value = SCALING * k_value
    rwa_value = rw_value * ead_value
    return {
        "pd": pd_value,
        "lgd": lgd_value,
        "ead": ead_value,
        "correlation": correl,
        "pd_stress": conditional_pd,
        "k": k_value,
        "rw": rw_value,
        "rwa": rwa_value,
        "capital": CAPITAL_RATIO * rwa_value,
        "expected_loss": pd_value * lgd_value * ead_value,
    }


def _norm_cdf(x_value):
    """Standard normal CDF via math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x_value / math.sqrt(2.0)))


def _norm_cdf_inv(probability):
    """Standard normal inverse CDF via bisection on the CDF.

    Accurate to ~1e-12 in probability; sufficient for the dashboard
    and consistent across all three companion scripts.
    """
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


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def _fig_to_base64(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_charts(pd_value, lgd_value, ead_value):
    """Build the three dashboard PNGs as base64 strings."""
    grid = np.linspace(0.0005, 0.20, 200)
    rw_curve = [risk_weight(pd_i, lgd_value) for pd_i in grid]
    k_curve = [capital_factor(pd_i, lgd_value)[0] for pd_i in grid]
    corr_curve = [retail_correlation(pd_i) for pd_i in grid]
    ead_grid = np.linspace(0.0, 50_000.0, 100)
    rwa_curve = [risk_weight(pd_value, lgd_value) * e for e in ead_grid]

    figure_rw, axis_rw = plt.subplots(figsize=(6.4, 3.6))
    axis_rw.plot(grid, rw_curve, lw=2, label="risk weight 12.5*K")
    axis_rw.axvline(pd_value, color="red", ls="--", lw=1,
                    label=f"current PD = {pd_value:.2%}")
    axis_rw.set_xlabel("PD"); axis_rw.set_ylabel("risk weight")
    axis_rw.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis_rw.set_title(f"Risk weight vs PD (LGD = {lgd_value:.0%})")
    axis_rw.legend(); axis_rw.grid(alpha=0.3)

    figure_k, axis_k = plt.subplots(figsize=(6.4, 3.6))
    axis_k.plot(grid, k_curve, lw=2, label="capital factor K")
    axis_k.plot(grid, corr_curve, lw=2, ls="--", label="correlation R(PD)")
    axis_k.axvline(pd_value, color="red", ls=":", lw=1)
    axis_k.set_xlabel("PD"); axis_k.set_ylabel("fraction")
    axis_k.set_title("K and asset correlation vs PD")
    axis_k.legend(); axis_k.grid(alpha=0.3)

    figure_ead, axis_ead = plt.subplots(figsize=(6.4, 3.6))
    axis_ead.plot(ead_grid, rwa_curve, lw=2, color="green")
    axis_ead.scatter([ead_value], [risk_weight(pd_value, lgd_value)
                                   * ead_value], color="red", zorder=5)
    axis_ead.set_xlabel("EAD (EUR)"); axis_ead.set_ylabel("RWA (EUR)")
    axis_ead.set_title(f"RWA vs EAD (RW = {risk_weight(pd_value, lgd_value):.1%})")
    axis_ead.grid(alpha=0.3)

    return (_fig_to_base64(figure_rw), _fig_to_base64(figure_k),
            _fig_to_base64(figure_ead))


# ---------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------

app = Flask(__name__)


def _to_float(request_value, fallback):
    try:
        return float(request_value)
    except (TypeError, ValueError):
        return fallback


@app.route("/")
def index():
    """Render the RWA dashboard for the requested parameters."""
    pd_value = min(max(_to_float(request.args.get("pd"), DEFAULTS["pd"]),
                       0.0001), 0.9999)
    lgd_value = min(max(_to_float(request.args.get("lgd"), DEFAULTS["lgd"]),
                        0.0), 1.0)
    ead_value = max(_to_float(request.args.get("ead"), DEFAULTS["ead"]), 0.0)
    result = compute_rwa(pd_value, lgd_value, ead_value)
    logging.info("RWA result: %s", result)
    chart_rw, chart_k, chart_ead = make_charts(pd_value, lgd_value, ead_value)
    kpis = [
        ("Correlation R", f"{result['correlation']:.2%}"),
        ("Stress PD (99.9%)", f"{result['pd_stress']:.2%}"),
        ("Capital factor K", f"{result['k']:.2%}"),
        ("Risk weight", f"{result['rw']:.1%}"),
        ("RWA (EUR)", f"{result['rwa']:,.2f}"),
        ("Own funds (8% RWA)", f"{result['capital']:,.2f} EUR"),
        ("Expected loss", f"{result['expected_loss']:,.2f} EUR"),
    ]
    return render_template_string(
        PAGE_TEMPLATE,
        pd=f"{pd_value:.4f}", lgd=f"{lgd_value:.4f}",
        ead=f"{ead_value:.0f}", kpis=kpis,
        chart_rw=chart_rw, chart_k=chart_k, chart_ead=chart_ead)


def main():
    """Start the local dashboard server."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    url = "http://127.0.0.1:5001"
    print(f"RWA dashboard running at {url} (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - headless environments
        pass
    # use_reloader=False: required in Spyder/IDE environments.
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
