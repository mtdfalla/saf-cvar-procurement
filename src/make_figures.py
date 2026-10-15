"""Paper figures and Table 2, from results/ and the scenario file.

Run: python src/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)
exp = json.loads((RES / "experiments.json").read_text())
fs = json.loads((RES / "first_stage.json").read_text())
z = np.load(ROOT / "data" / "model" / "scenarios.npz")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
qlab = [f"{y}Q{q}" for y in range(2025, 2031) for q in (1, 2, 3, 4)]
xq = np.arange(24)

# ---- Fig 1: scenario fan (jet + HEFA)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, key, ttl, ylab in [(axes[0], "jet", "Fossil jet price", "EUR/L"),
                           (axes[1], "hefa", "HEFA-SPK spot price", "EUR/L")]:
    v = z[key]
    for q in [(5, 95), (25, 75)]:
        lo, hi = np.percentile(v, q[0], axis=0), np.percentile(v, q[1], axis=0)
        ax.fill_between(xq, lo, hi, alpha=0.25, color="tab:blue", lw=0)
    ax.fill_between([], [], [], alpha=0.25, color="tab:blue", lw=0, label="5-95% and 25-75% (unweighted)")
    ax.plot(xq, np.median(v, axis=0), color="tab:blue", lw=1.5, label="median")
    for i in range(0, 100, 25):
        ax.plot(xq, v[i], color="gray", lw=0.5, alpha=0.6,
                label="sample scenarios" if i == 0 else None)
    ax.set_xticks(xq[::4]); ax.set_xticklabels(qlab[::4], rotation=45)
    ax.set_title(ttl); ax.set_ylabel(ylab)
axes[0].legend(fontsize=7, loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig1_scenarios.png", dpi=300); plt.close(fig)

# ---- Fig 2: cost-risk frontier
lams = ["lam0", "lam025", "lam05", "lam1", "lam2"]
lam_v = [0, 0.25, 0.5, 1.0, 2.0]
E = [exp[f"cvar_{k}"]["expected_M"] for k in lams]
C = [exp[f"cvar_{k}"]["cvar95_M"] for k in lams]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(E, C, "-o", color="tab:blue", ms=5, label="mean-CVaR frontier")
for e, c, l in zip(E, C, lam_v):
    if l not in (0, 2.0):
        continue
    ax.annotate(f"$\\lambda$={l}", (e, c), textcoords="offset points",
                xytext=(8, -2), fontsize=8)
b0 = exp["no_bc_0"]; b1 = exp["no_bc_1"]; ev = exp["eev"]
ax.scatter([b0["expected_M"], b1["expected_M"]], [b0["cvar95_M"], b1["cvar95_M"]],
           marker="s", color="tab:red", label="no book-and-claim")
ax.annotate("no B&C, $\\lambda$=0", (b0["expected_M"], b0["cvar95_M"]),
            textcoords="offset points", xytext=(-65, 5), fontsize=8)
ax.annotate("no B&C, $\\lambda$=1", (b1["expected_M"], b1["cvar95_M"]),
            textcoords="offset points", xytext=(-65, -10), fontsize=8)
ax.scatter([ev["expected_M"]], [ev["cvar95_M"]], marker="^", color="tab:green",
           label="deterministic EV plan", zorder=5)
ax.set_xlabel("Expected net compliance cost (EUR M)")
ax.set_ylabel("CVaR$_{0.95}$ (EUR M)")
ax.legend(fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(FIG / "fig2_frontier.png", dpi=300); plt.close(fig)

# ---- Fig 3: first-stage HEFA forward schedule by lambda
fig, ax = plt.subplots(figsize=(7.0, 2.8))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(lams)))
for k, lv, c in zip(lams, lam_v, colors):
    xs = [fs[k][f"HEFA-SPK|{t}"] for t in range(24)]
    ax.step(xq, xs, where="mid", color=c, lw=1.4, label=f"$\\lambda$={lv}")
ax.set_xticks(xq[::4]); ax.set_xticklabels(qlab[::4], rotation=45)
ax.set_ylabel("HEFA forwards (ML/quarter)")
ax.legend(fontsize=8, ncol=5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig3_forwards.png", dpi=300); plt.close(fig)

# ---- Table 2
rows = [["Model", "E[net cost] (EUR M)", "CVaR95 (EUR M)", "Forward (ML)",
         "E[spot] (ML)", "E[B&C] (ML)", "Compliance rate"]]
label_map = [("cvar_lam0", "Risk-neutral stochastic (lam=0)"),
             ("cvar_lam025", "Mean-CVaR lam=0.25"), ("cvar_lam05", "Mean-CVaR lam=0.5"),
             ("cvar_lam1", "Mean-CVaR lam=1"), ("cvar_lam2", "Mean-CVaR lam=2"),
             ("eev", "Deterministic EV plan (EEV)"),
             ("no_bc_0", "No B&C (lam=0)"), ("no_bc_1", "No B&C (lam=1)")]
for k, name in label_map:
    r = exp[k]
    rows.append([name, f"{r['expected_M']:.2f}", f"{r['cvar95_M']:.2f}",
                 f"{r['fwd_ML']:.1f}", f"{r['spot_expected_ML']:.1f}",
                 f"{r['bc_expected_ML']:.1f}", f"{r['mandate_compliance_rate']:.3f}"])
with open(RES / "table2.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("figures written to", FIG, "| table written to", RES / "table2.csv")
for k, name in label_map:
    print(name, "->", {kk: exp[k][kk] for kk in ["expected_M", "cvar95_M", "solve_seconds"]})
