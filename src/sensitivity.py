"""Robustness checks, written to results/sensitivity.json.

1. Certificate price: the book-and-claim certificate price is varied from EUR 0.60/L
   (base case) to 1.20/L at lambda 0 and 1, with and without certificates.
2. EU synthetic sub-mandate penalty: the shortfall is priced at 2 x (PtL - jet)
   instead of 2 x (HEFA - jet).

Run: python src/sensitivity.py   (15 solves)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pyomo.environ as pyo  # noqa: E402
from model import build_model, extract, load_data, solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sensitivity.json"
KEEP = ["expected_M", "cvar95_M", "cvar95_exact_M", "fwd_ML", "bc_expected_ML",
        "spot_expected_ML", "mandate_compliance_rate"]


def run(d, label, **kw):
    m = build_model(d, **kw)
    solve(m)
    r = {k: v for k, v in extract(m, d, label).items() if k in KEEP}
    pi, S, T = d.probs, range(d.S), range(d.T)
    r["eu_ptl_shortfall_ML"] = float(sum(pi[s] * sum(pyo.value(m.Useu[t, s]) for t in T)
                                         for s in S))
    print(label, {k: round(v, 2) for k, v in r.items()}, flush=True)
    return r


def main():
    out = {"certificate_price": {}, "eu_ptl_penalty_on_ptl": {}}
    base = load_data()
    for lam in (0.0, 1.0):
        out["certificate_price"][f"lam{lam:g}_no_bc"] = run(base, f"lam{lam:g} no B&C",
                                                            lam=lam, no_bc=True)
        for price in (0.60, 0.70, 0.80, 0.90, 1.20):
            d = load_data()
            d.cen["bc_cert_eur_per_L"] = price
            out["certificate_price"][f"lam{lam:g}_p{price:.2f}"] = run(
                d, f"lam{lam:g} cert {price:.2f}", lam=lam)
    for lam, no_bc in ((0.0, False), (1.0, False), (1.0, True)):
        key = f"lam{lam:g}" + ("_no_bc" if no_bc else "")
        out["eu_ptl_penalty_on_ptl"][key] = run(base, f"PtL penalty {key}", lam=lam,
                                                no_bc=no_bc, eu_ptl_penalty_on="ptl")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
