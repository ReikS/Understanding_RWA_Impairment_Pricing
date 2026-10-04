#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Impairment_visualised.py - Interactive IFRS 9 ECL dashboard (Part B).

====================================================================
 MANUAL
====================================================================
Purpose
-------
Demonstrates how the impairment (loss allowance) of a 10,000 EUR
unsecured consumer instalment loan is computed under IFRS 9:

  * 3-stage model (12-month ECL vs lifetime ECL),
  * scenario-weighted, forward-looking probability weights,
  * discounting at the effective interest rate (EIR),
  * marginal (conditional-on-survival) default probabilities.

The implementation follows the companion textbook
"Understanding RWA, Impairment, and Risk-Sensitive Pricing",
Part B.

Requirements
------------
Python 3.12 with the packages listed in requirements_py312.txt
(at minimum: flask, matplotlib, numpy).

How to run
----------
1. Spyder:  open this file and press F5 (Run File).  The Flask
   development server starts and the dashboard opens automatically
   in your browser (see console output for the URL).
2. Terminal: python Impairment_visualised.py
Then open http://127.0.0.1:5002 in a browser if it does not open
automatically.

How to use
----------
Choose the stage (1 or 2), the base-scenario first-year PD, the
PD trend factor (multiplier for later years' marginal PDs), LGD,
the EIR, and the three scenario weights (they are normalised
automatically).  The KPI panel, the per-year loss table and the
charts update on every "Apply" click.

Notes
-----
- Scenario ECLs are computed by scaling the base marginal PDs
  (optimistic: x0.5, adverse: x2.0) as in the textbook example.
- Stage 3 (credit-impaired, interest on net carrying amount) is
  not modelled.
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
# Loan and model constants (textbook baseline)
# ---------------------------------------------------------------------

PRINCIPAL = 10_000.0
YEARS = 3
BASE_PDS = (0.02, 0.025, 0.03)   # marginal PDs, base scenario
SCENARIO_SCALES = {"optimistic": 0.5, "base": 1.0, "adverse": 2.0}
DEFAULTS = {
    "stage": 1,
    "pd1": 0.02,          # first-year marginal PD (base scenario)
    "trend": 0.005,       # absolute PD step per year (0.005 = +0.5 pp/yr)
    "lgd": 0.45,
    "eir": 0.085,
    "w_optimistic": 0.2,
    "w_base": 0.6,
    "w_adverse": 0.2,
}

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>IFRS 9 Impairment Dashboard</title>
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
 table.data { border-collapse: collapse; background: #fff; }
 table.data th, table.data td { border: 1px solid #ccc; padding: 4px 10px; }
 .note { color: #666; font-size: 0.9em; }
</style></head><body>
<h1>IFRS 9 Expected Credit Losses - {{ "Stage 1 (12-month ECL)" if stage == 1
     else "Stage 2 (lifetime ECL)" }}</h1>
<form method="get">
 <span>Stage:
   <select name="stage">
     <option value="1" {{ "selected" if stage == 1 }}>1</option>
     <option value="2" {{ "selected" if stage == 2 }}>2</option>
   </select></span>
 <span>PD year 1: <input type="number" step="0.005" min="0" max="1"
      name="pd1" value="{{ pd1 }}"></span>
 <span>PD step/yr: <input type="number" step="0.001" min="0" max="0.1"
      name="trend" value="{{ trend }}"></span>
 <span>LGD: <input type="number" step="0.05" min="0" max="1"
      name="lgd" value="{{ lgd }}"></span>
 <span>EIR: <input type="number" step="0.005" min="0" max="0.5"
      name="eir" value="{{ eir }}"></span><br><br>
 <span>Weight optimistic: <input type="number" step="0.05" min="0" max="1"
      name="w_optimistic" value="{{ w_optimistic }}"></span>
 <span>Weight base: <input type="number" step="0.05" min="0" max="1"
      name="w_base" value="{{ w_base }}"></span>
 <span>Weight adverse: <input type="number" step="0.05" min="0" max="1"
      name="w_adverse" value="{{ w_adverse }}"></span>
 <button type="submit">Apply</button>
</form>
<h2>Key figures</h2>
<div class="kpis">
 {% for label, value in kpis %}
 <div class="card">{{ label }}<br><b>{{ value }}</b></div>
 {% endfor %}
</div>
<h2>Per-year expected loss decomposition (base scenario)</h2>
<table class="data">
 <tr><th>Year</th><th>Balance BoY</th><th>Marginal PD</th>
     <th>Loss (undiscounted)</th><th>PV at EIR</th></tr>
 {% for row in rows %}
 <tr><td>{{ row.t }}</td><td>{{ row.balance }}</td><td>{{ row.pd }}</td>
     <td>{{ row.loss }}</td><td>{{ row.pv }}</td></tr>
 {% endfor %}
</table>
<h2>Scenario comparison (probability-weighted ECL)</h2>
<img src="data:image/png;base64,{{ chart_scenario }}">
<h2>ECL vs first-year PD (stage 1 vs stage 2)</h2>
<img src="data:image/png;base64,{{ chart_pd }}">
<p class="note">Model: IFRS 9 paras 5.5.15-5.5.20; marginal PDs,
scenario weighting, discounting at EIR.  Illustrative parameters
only - not a bank's approved model.</p>
</body></html>
"""


# ---------------------------------------------------------------------
# Model functions (pure, unit-testable)
# ---------------------------------------------------------------------

def annuity_schedule(principal, rate, years):
    """Return the amortisation schedule as a list of dicts."""
    payment = principal * rate / (1.0 - (1.0 + rate) ** -years)
    balance = principal
    schedule = []
    for year in range(1, years + 1):
        interest = balance * rate
        principal_part = payment - interest
        schedule.append({
            "year": year,
            "balance_boY": balance,
            "interest": interest,
            "principal": principal_part,
            "balance_eoY": balance - principal_part,
        })
        balance -= principal_part
    return schedule, payment


def marginal_pds(pd_first, step, years):
    """Marginal (conditional-on-survival) one-year PDs, base scenario.

    Linear stepping: pd_t = pd_first + (t - 1) * step.  The default
    (2%, +0.5 pp per year) reproduces the textbook example
    (2.0% / 2.5% / 3.0%).
    """
    return [pd_first + (year - 1) * step for year in range(1, years + 1)]


def ecl_schedule(pds, lgd, schedule, eir):
    """Discounted ECL per year for given marginal PDs.

    ECL_t = m_t * LGD * balance_t-1 * (1+EIR)^-t.
    """
    rows, total = [], 0.0
    for year, (marginal_pd, row) in enumerate(zip(pds, schedule), start=1):
        loss = marginal_pd * lgd * row["balance_boY"]
        present_value = loss / (1.0 + eir) ** year
        rows.append({"t": year, "pd": marginal_pd, "loss": loss,
                     "pv": present_value,
                     "balance": row["balance_boY"]})
        total += present_value
    return rows, total


def compute_ecl(stage, pd_first, trend, lgd, eir, weights):
    """Full result dictionary for the current parameter set.

    Weights are (optimistic, base, adverse) and are normalised.
    Stage 1 keeps only the first year of each scenario; stage 2
    uses the full remaining life.
    """
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    schedule, payment = annuity_schedule(PRINCIPAL, eir, YEARS)
    base_pds = marginal_pds(pd_first, trend, YEARS)
    horizon = 1 if stage == 1 else YEARS
    scenario_totals = {}
    for name, scale in SCENARIO_SCALES.items():
        pds = [pd * scale for pd in base_pds]
        rows, total = ecl_schedule(pds[:horizon], lgd,
                                   schedule[:horizon], eir)
        scenario_totals[name] = total
    weighted_total = float(sum(weights[i] * scenario_totals[key]
                               for i, key in enumerate(
                                   ["optimistic", "base", "adverse"])))
    base_rows, base_total = ecl_schedule(base_pds[:horizon], lgd,
                                         schedule[:horizon], eir)
    return {
        "payment": payment,
        "schedule": schedule,
        "scenario_totals": scenario_totals,
        "weighted_ecl": weighted_total,
        "base_rows": base_rows,
        "base_total": base_total,
        "weights": weights.tolist(),
        "cumulative_pd": 1.0 - math.prod(1.0 - pd for pd in base_pds),
    }


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def _fig_to_base64(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_charts(result, stage, pd_first, trend, lgd, eir):
    """Build the two dashboard PNGs as base64 strings."""
    labels = list(result["scenario_totals"].keys())
    values = [result["scenario_totals"][key] for key in labels]

    figure_sc, axis_sc = plt.subplots(figsize=(6.4, 3.6))
    bars = axis_sc.bar(labels, values, color=["#7fa87f", "#5b8db8", "#c0704d"])
    for bar, weight in zip(bars, result["weights"]):
        axis_sc.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 3,
                     f"w={weight:.0%}", ha="center")
    axis_sc.axhline(result["weighted_ecl"], color="black", ls="--", lw=1,
                    label=f"weighted ECL = {result['weighted_ecl']:.2f} EUR")
    axis_sc.set_ylabel("ECL (EUR)")
    axis_sc.set_title("Scenario ECLs and probability weighting")
    axis_sc.legend(); axis_sc.grid(axis="y", alpha=0.3)

    grid = np.linspace(0.002, 0.15, 100)
    ecl_stage1, ecl_stage2 = [], []
    for pd_i in grid:
        sub = compute_ecl(1, pd_i, trend, lgd, eir, (0.2, 0.6, 0.2))
        ecl_stage1.append(sub["weighted_ecl"])
        sub = compute_ecl(2, pd_i, trend, lgd, eir, (0.2, 0.6, 0.2))
        ecl_stage2.append(sub["weighted_ecl"])
    figure_pd, axis_pd = plt.subplots(figsize=(6.4, 3.6))
    axis_pd.plot(grid * 100, ecl_stage1, lw=2, label="Stage 1 (12m ECL)")
    axis_pd.plot(grid * 100, ecl_stage2, lw=2, label="Stage 2 (lifetime ECL)")
    axis_pd.axvline(pd_first * 100, color="red", ls="--", lw=1,
                    label=f"current PD1 = {pd_first:.1%}")
    axis_pd.set_xlabel("first-year PD (%)"); axis_pd.set_ylabel("ECL (EUR)")
    axis_pd.set_title("Allowance vs PD, by stage")
    axis_pd.legend(); axis_pd.grid(alpha=0.3)

    return _fig_to_base64(figure_sc), _fig_to_base64(figure_pd)


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
    """Render the impairment dashboard for the requested parameters."""
    stage = int(_to_float(request.args.get("stage"), DEFAULTS["stage"]))
    stage = 1 if stage < 2 else 2
    pd_first = min(max(_to_float(request.args.get("pd1"), DEFAULTS["pd1"]),
                       0.0), 0.99)
    trend = max(_to_float(request.args.get("trend"), DEFAULTS["trend"]), 0.1)
    lgd_value = min(max(_to_float(request.args.get("lgd"), DEFAULTS["lgd"]),
                        0.0), 1.0)
    eir = min(max(_to_float(request.args.get("eir"), DEFAULTS["eir"]),
                  0.0), 1.0)
    weights = (
        _to_float(request.args.get("w_optimistic"), DEFAULTS["w_optimistic"]),
        _to_float(request.args.get("w_base"), DEFAULTS["w_base"]),
        _to_float(request.args.get("w_adverse"), DEFAULTS["w_adverse"]),
    )
    result = compute_ecl(stage, pd_first, trend, lgd_value, eir, weights)
    logging.info("ECL result: %s", result["weighted_ecl"])
    chart_sc, chart_pd = make_charts(result, stage, pd_first, trend,
                                     lgd_value, eir)
    kpis = [
        ("Annuity payment", f"{result['payment']:,.2f} EUR"),
        ("Stage", str(stage)),
        ("Lifetime PD (base)", f"{result['cumulative_pd']:.2%}"),
        ("ECL 12-month view",
         f"{compute_ecl(1, pd_first, trend, lgd_value, eir, weights)['weighted_ecl']:,.2f} EUR"),
        ("ECL lifetime view",
         f"{compute_ecl(2, pd_first, trend, lgd_value, eir, weights)['weighted_ecl']:,.2f} EUR"),
        ("Allowance (this stage)",
         f"{result['weighted_ecl']:,.2f} EUR"),
        ("Coverage ratio",
         f"{result['weighted_ecl'] / PRINCIPAL:.2%}"),
    ]
    rows = [{"t": row["t"], "balance": f"{row['balance']:,.2f}",
             "pd": f"{row['pd']:.2%}", "loss": f"{row['loss']:,.2f}",
             "pv": f"{row['pv']:,.2f}"} for row in result["base_rows"]]
    return render_template_string(
        PAGE_TEMPLATE,
        stage=stage, pd1=f"{pd_first:.3f}", trend=f"{trend:.3f}",
        lgd=f"{lgd_value:.2f}", eir=f"{eir:.3f}",
        w_optimistic=f"{weights[0]:.2f}", w_base=f"{weights[1]:.2f}",
        w_adverse=f"{weights[2]:.2f}",
        kpis=kpis, rows=rows,
        chart_scenario=chart_sc, chart_pd=chart_pd)


def main():
    """Start the local dashboard server."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    url = "http://127.0.0.1:5002"
    print(f"IFRS 9 impairment dashboard running at {url} (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - headless environments
        pass
    # use_reloader=False: required in Spyder/IDE environments.
    app.run(host="127.0.0.1", port=5002, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
