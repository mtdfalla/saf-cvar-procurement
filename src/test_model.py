"""Checks: CVaR linearization against a hand calculation, and ordering properties on a
10-scenario instance.

Run: python src/test_model.py
"""
import sys
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, extract, load_data, solve  # noqa: E402

# ---- check 1: CVaR linearization on a fixed 5-scenario cost vector
costs = np.array([100.0, 120.0, 90.0, 200.0, 150.0])
probs = np.full(5, 0.2)
alpha = 0.75
m = pyo.ConcreteModel()
m.eta = pyo.Var()
m.V = pyo.Var(range(5), domain=pyo.NonNegativeReals)
m.c = pyo.Constraint(range(5), rule=lambda m, s: m.V[s] >= costs[s] - m.eta)
m.obj = pyo.Objective(
    expr=m.eta + 1 / (1 - alpha) * sum(probs[s] * m.V[s] for s in range(5)))
pyo.SolverFactory("appsi_highs").solve(m)
cvar_lp = pyo.value(m.obj)
# hand: alpha=0.75 -> tail is worst 25% = worst 1.25 scenarios. Sorted desc: 200,150,...
# CVaR = (0.2*200 + 0.05*150)/0.25 = (40+7.5)/0.25 = 190
assert abs(cvar_lp - 190.0) < 1e-6, f"CVaR check FAILED: {cvar_lp} != 190"
print(f"check 1 passed: CVaR linearization = {cvar_lp:.4f} (hand-computed 190.0000)")

# ---- check 2: 10-scenario instance, objective orderings
d = load_data(n_scen=10)
res = {}
for lam in [0.0, 1.0]:
    mm = build_model(d, lam=lam)
    solve(mm)
    res[lam] = extract(mm, d, label=f"lam={lam}")
    print(f"  lam={lam}: E={res[lam]['expected_M']:.1f}M  CVaR95={res[lam]['cvar95_M']:.1f}M  "
          f"fwd={res[lam]['fwd_ML']:.1f}ML bc={res[lam]['bc_expected_ML']:.1f}ML "
          f"comply={res[lam]['mandate_compliance_rate']:.2f}")

assert res[1.0]["cvar95_exact_M"] <= res[0.0]["cvar95_exact_M"] + 1e-3, "CVaR should fall with risk aversion"
assert res[0.0]["expected_M"] <= res[1.0]["expected_M"] + 1e-3, "E should rise with risk aversion"
print("check 2 passed: CVaR(lam=1) <= CVaR(lam=0) and E(lam=0) <= E(lam=1)")

# removing certificates cannot lower the optimal objective (lambda = 0: expected cost)
mb = build_model(d, lam=0.0, no_bc=True)
solve(mb)
rb = extract(mb, d, "no_bc")
print(f"  no-BC: E={rb['expected_M']:.1f}M (full {res[0.0]['expected_M']:.1f}M)")
assert rb["expected_M"] >= res[0.0]["expected_M"] - 1e-3, "removing an option cannot reduce cost"
print("check 3 passed: removing certificates weakly increases expected cost at lam=0")
