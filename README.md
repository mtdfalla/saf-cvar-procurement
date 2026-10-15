# saf-cvar-procurement

Code and data for the paper:

> M. Abdelmagid, A. Quran, M. Maalouf. *Risk-Aware Sustainable Aviation Fuel Procurement under Regulatory Uncertainty: A Two-Stage Stochastic CVaR Model for Airline Networks*. 53rd International Conference on Computers and Industrial Engineering (CIE53), Abu Dhabi, October 2026.

MIT license for the code. Data sources and attributions are listed in `data/README.md`.

An airline signs sustainable aviation fuel (SAF) forward contracts before fuel prices, carbon prices and SAF availability are known. It then blends, buys spot SAF and buys book-and-claim certificates in each of 100 scenarios. The model is a two-stage stochastic mixed-integer program that minimizes expected net compliance cost plus λ times its CVaR at 95%. Net compliance cost is the total fuel, carbon and penalty cost minus the all-fossil cost in the same scenario.

The case study is a stylized network of 13 long-haul routes from a Gulf hub over 24 quarters (2025 to 2030). Regulatory exposures are stylized: ReFuelEU Aviation blending shares and EU ETS allowance costs on EU routes, UK SAF Mandate shares on UK routes, and CORSIA offsetting on the other routes. Mandate costs are modelled as passed through to the airline. Certificates count only toward a voluntary network SAF pledge (2% in 2025 rising to 8% in 2030), not toward the mandates.

## Environment

Python 3.11 or later. Tested with Python 3.13, numpy 2.5, pandas 3.0 and matplotlib 3.11.

    pip install -r requirements.txt

The HiGHS solver is installed with `highspy`. No commercial solver is needed.

## Running

    python src/build_dataset.py     # data/inputs -> data/model
    python src/scenarios.py         # 100 scenarios -> data/model/scenarios.npz
    python src/run_experiments.py   # eight runs of Table 2 -> results/experiments.json
    python src/make_figures.py      # results/table2.csv and results/figures/
    python src/sensitivity.py       # robustness checks -> results/sensitivity.json
    python src/test_model.py        # model checks

The first two steps reproduce the files in `data/model/` exactly (scenario seed 53), so the experiments can also be run directly from step 3. Each solve takes 10 to 20 s on a laptop. A single run can be repeated with `python src/run_experiments.py lam1`.

## Results

`results/table2.csv` reproduces Table 2 of the paper. `results/experiments.json` holds every run with its net cost in each scenario, and a summary (value of the stochastic solution, CVaR reduction and risk premium of the most risk-averse plan, value of book-and-claim). `results/first_stage.json` holds the forward schedules of the five λ runs.

## Notes

* **CVaR.** Table 2 reports the probability-weighted mean net cost over the scenarios at or above VaR (`cvar95_M`). Scenario probabilities are unequal, so this set covers about 6.4% of probability (7 of 100 scenarios). The exact CVaR at α = 0.95 is also reported (`cvar95_exact_M`). It is 1.4 to 1.8 M EUR higher, and the comparisons in the paper hold. For example, certificates lower the exact tail cost at λ = 1 by 6.4 M EUR (219.98 to 213.58).
* **Mandates.** The main EU and UK shares are met in all 100 scenarios. The 2030 EU synthetic sub-share (about 2.2 ML in expectation) is settled through the penalty.
* **EU penalty price.** EU shortfalls are priced at 2 × (SAF spot − jet). The code uses the HEFA spot price for both the main share and the synthetic sub-share. `src/sensitivity.py` repeats the runs with the PtL price for the sub-share. The sub-share is then met in full, expected net cost at λ = 0 is 191.4 M EUR, and certificates still lower the exact tail cost at λ = 1 by 6.2 M EUR.
* **Certificate price.** The certificate value depends on the certificate price relative to the SAF premium (base case 0.60 EUR/L). At λ = 1 the exact tail-cost saving from certificates is 6.4, 1.3, 0.6, 0.4 and 0.1 M EUR at 0.60, 0.70, 0.80, 0.90 and 1.20 EUR/L (`results/sensitivity.json`).

## Contents

* `src/` dataset build, scenario generation, model, experiments, figures, robustness checks and tests.
* `data/inputs/` source tables (prices, mandates, SAF pathways and projections, routes, fuel burn).
* `data/model/` model inputs built from `data/inputs/`, including the 100 scenarios.
* `results/` result files, Table 2 and figures.

## Citation

See `CITATION.cff`.
