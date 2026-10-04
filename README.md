# Understanding RWA, Impairment, and Risk-Sensitive Pricing

A small study project that explains, on one consistent numerical
example, how a bank measures and prices the credit risk of a single
loan under three frameworks:

- **Part A — Regulation (CRR/Basel, A-IRB):** the risk-weight function
  ("Gordy formula"), from the Vasicek single-factor model to
  risk-weighted assets (RWA) and the 8% own-funds requirement.
- **Part B — Accounting (IFRS 9):** the three-stage model and expected
  credit loss (ECL) impairment, from 12-month to lifetime ECL with
  scenario weighting.
- **Part C — Management (pricing):** risk-based pricing via RAROC; the
  break-even customer rate as funding + opex + expected loss + cost of
  capital.

The running example throughout is an unsecured consumer instalment
loan of EUR 10,000 (PD 2%, LGD 45%, 3-year annuity at 8.5%). All three
frameworks are computed on identical parameters, so the numbers are
directly comparable: EUR 463.89 regulatory capital, EUR 82.95–206.10
IFRS 9 allowance, 6.56% break-even rate.

## Repository contents

| File | Purpose |
|---|---|
| `Understanding_RWA_Impairment_Pricing.tex` | The textbook (LaTeX, ~27 pages) |
| `RWA_visualised.py` | Flask dashboard, port 5001 — Part A: K, correlation, risk weight vs PD, RWA vs EAD |
| `Impairment_visualised.py` | Flask dashboard, port 5002 — Part B: stage-1 vs stage-2 allowance, scenario weighting, per-year ECL |
| `Pricing_visualised.py` | Flask dashboard, port 5003 — Part C: break-even decomposition, RAROC vs hurdle, capital multiplier, EL basis |
| `requirements_py312.txt` | Python 3.12 environment (the scripts need only Flask, numpy, scipy) |
| `old_chat.txt` | Prior worked derivation of the RWA example (source material) |
| `README.md` | This file |

## Tooling

**Dashboards.** Python 3.12. The scripts are self-contained Flask
apps; no build step. Run from Spyder (F5) or a terminal:

```
pip install flask numpy scipy
python RWA_visualised.py        # then open http://127.0.0.1:5001
python Impairment_visualised.py # http://127.0.0.1:5002
python Pricing_visualised.py    # http://127.0.0.1:5003
```

Each dashboard prints its parameter set and results on startup, so
every figure in the book can be reproduced by hand or by slider.

**Book.** LaTeX, compiled with MiKTeX:

```
pdflatex Understanding_RWA_Impairment_Pricing.tex   # run twice
```

`lualatex` also works (and is recommended for the fonts). Two runs are
needed for the table of contents and cross-references. The document
class is `report` 11pt a4paper with TeX Gyre Pagella text and
newpxmath math fonts, TikZ diagrams, colored hyperlinks; all required
packages are standard MiKTeX/TeX Live and install on demand.

## How it was created

Drafted with AI assistance in an iterative process. The mathematical
content follows published regulatory and academic sources (CRR
Arts. 153–154, BIS explanatory note, Gordy 2003, Vasicek 2002, IFRS 9
paras 5.5.15–5.5.20; citations and credibility ratings are in the
book's References and Source Notes). Every numerical example was
computed with executable code rather than asserted, and two errors
found in an earlier draft (an incorrect risk weight for the PD = 8%
comparison and an inconsistent lifetime-EL pricing basis) were
corrected against computation before publication. The LaTeX version
supersedes an earlier Markdown/PDF draft; its preamble was fixed
against a real MiKTeX compile (package load order, `\EUR` redefinition
instead of `\newcommand`, missing `\Var`/`\PV` macros).

The three Python scripts predate the book and are treated as frozen;
the book documents them in its "Companion Dashboards" chapter.

## How to update it

- **Change example parameters** (PD, LGD, EAD, EIR, hurdle, scenario
  weights): the dashboards expose all of them as sliders, so explore
  there first. To make a change permanent in the book, edit the worked
  examples in the `.tex`, recompute the derived numbers (the formulas
  and every intermediate step are typeset in the chapters), and keep
  the tables consistent — the summary numbers cited above
  (463.89 / 82.95–206.10 / 6.56%) appear in several chapters and the
  Synthesis chapter.
- **Extend the book:** the `.tex` uses numbered equations, shared
  theorem counters per chapter, and `\label`/`\ref` throughout, so new
  sections integrate cleanly. Recompile twice. Do not load
  `amssymb`/`amsthm` after `newpxmath` (symbol clashes: `\Bbbk`,
  `\openbox`); the current preamble order must stay.
- **Update the scripts:** they are intentionally frozen. If you change
  one, mirror the change in the corresponding book chapter so numbers
  stay consistent.
- **Regulatory changes** (CRR3, output floor, IFRS 9 practice): the
  book states these qualitatively; check EUR-Lex and the IFRS
  Foundation before treating them as current.

Educational treatment with didactic simplifications; not legal,
regulatory, accounting, or investment advice.
