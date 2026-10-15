"""Two-stage stochastic MILP with mean-CVaR objective for airline SAF procurement.

Pyomo model solved with HiGHS. Money in EUR million, volumes in ML (million liters).
Risk is measured on the net compliance cost: total cost minus the all-fossil baseline
of the same scenario.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "model"

EPS = 2.528e-3          # tCO2 per liter fossil jet (3.16 kg/kg * 0.80 kg/L / 1000)
P_VOL = 1.50            # EUR/L voluntary SAF-pledge shortfall penalty (parametric)
FWD_PREMIUM = 1.02      # forward price = central spot projection * (1 + 2% term premium)
ALPHA = 0.95
ASTM_CAP = 0.50
PHI = {"HEFA-SPK": 1.0, "PtL-FT": 3.0}        # EUR M contract activation cost
XMIN = {"HEFA-SPK": 1.0, "PtL-FT": 0.5}       # ML minimum total if activated
FWD_CAP = {"HEFA-SPK": 12.0, "PtL-FT": 2.0}   # ML per quarter
PTL_FWD_START = 8                              # PtL forwards deliverable from 2027Q1 (t=8)
BC_CAP_SHARE = 0.05                            # B&C cap: 5% of network fuel per quarter
OMEGA = {2025: .02, 2026: .032, 2027: .044, 2028: .056, 2029: .068, 2030: .08}


@dataclass
class Data:
    routes: pd.DataFrame
    mand: pd.DataFrame
    cen: pd.DataFrame
    sc: dict          # scenario arrays
    probs: np.ndarray
    T: int
    S: int
    P: list


def load_data(n_scen=None) -> Data:
    routes = pd.read_csv(MODEL / "routes.csv")
    mand = pd.read_csv(MODEL / "mandates.csv").sort_values("t")
    cen = pd.read_csv(MODEL / "central_prices.csv").sort_values("t")
    z = np.load(MODEL / "scenarios.npz")
    sc = {k: z[k] for k in ["jet", "eua", "hefa", "ptl", "corsia", "a_hefa", "a_ptl"]}
    probs = z["probs"]
    if n_scen is not None:                      # scenario subset for tests
        sc = {k: v[:n_scen] for k, v in sc.items()}
        probs = probs[:n_scen] / probs[:n_scen].sum()
    return Data(routes, mand, cen, sc, probs, T=24, S=len(probs), P=["HEFA-SPK", "PtL-FT"])


def expected_value_data(d: Data) -> Data:
    """Single-scenario Data with probability-weighted expected stochastic parameters (expected-value benchmark)."""
    sc = {k: (d.sc[k] * d.probs[:, None]).sum(axis=0, keepdims=True) for k in d.sc}
    return Data(d.routes, d.mand, d.cen, sc, np.array([1.0]), d.T, 1, d.P)


def build_model(d: Data, lam=1.0, alpha=ALPHA, no_bc=False, fix_first_stage=None,
                eu_ptl_penalty_on="hefa"):
    m = pyo.ConcreteModel()
    R = list(d.routes.route_id)
    P, T, S = d.P, range(d.T), range(d.S)
    rt = d.routes.set_index("route_id")
    L = {r: rt.loc[r, "quarterly_fuel_L"] / 1e6 for r in R}            # ML/quarter
    Lup = {r: L[r] * rt.loc[r, "uplift_share"] for r in R}             # mandate-relevant uplift
    region = {r: rt.loc[r, "reg_region"] for r in R}
    R_EU = [r for r in R if region[r] == "EU"]
    R_UK = [r for r in R if region[r] == "UK"]
    R_NE = [r for r in R if region[r] == "NONE"]
    Ltot = {t: sum(L[r] for r in R) for t in T}
    md = d.mand.set_index("t")
    year = {t: int(md.loc[t, "year"]) for t in T}
    pi = d.probs

    # scenario parameters: EUR/L x ML = EUR M, EUR/t x tCO2 / 1e6 = EUR M
    jet, eua, hefa, ptl, cors = (d.sc[k] for k in ["jet", "eua", "hefa", "ptl", "corsia"])
    a = {"HEFA-SPK": d.sc["a_hefa"] / 1e6, "PtL-FT": d.sc["a_ptl"] / 1e6}  # ML
    cs = {"HEFA-SPK": hefa, "PtL-FT": ptl}
    cf = {("HEFA-SPK", t): d.cen.iloc[t]["hefa_spot_eur_per_L"] * FWD_PREMIUM for t in T}
    cf.update({("PtL-FT", t): d.cen.iloc[t]["ptl_spot_eur_per_L"] * FWD_PREMIUM for t in T})
    cb = {t: d.cen.iloc[t]["bc_cert_eur_per_L"] for t in T}
    pw = pd.read_csv(MODEL / "pathways.csv")
    delta = dict(zip(pw.pathway, pw.delta))

    # first stage
    m.X = pyo.Var(P, T, domain=pyo.NonNegativeReals)
    m.Rc = pyo.Var(P, domain=pyo.Binary)
    # second stage
    m.W = pyo.Var(P, T, S, domain=pyo.NonNegativeReals)
    m.Y = pyo.Var(T, S, domain=pyo.NonNegativeReals)
    m.Z = pyo.Var(R, P, T, S, bounds=(0, 1))
    m.Ueu = pyo.Var(T, S, domain=pyo.NonNegativeReals)
    m.Uuk = pyo.Var(T, S, domain=pyo.NonNegativeReals)
    m.Useu = pyo.Var(T, S, domain=pyo.NonNegativeReals)   # EU PtL sub-mandate shortfall
    m.Usuk = pyo.Var(T, S, domain=pyo.NonNegativeReals)   # UK PtL sub-mandate shortfall
    m.Uv = pyo.Var(T, S, domain=pyo.NonNegativeReals)     # voluntary-pledge shortfall (ML)
    m.eta = pyo.Var(domain=pyo.Reals)
    m.V = pyo.Var(S, domain=pyo.NonNegativeReals)

    if no_bc:
        for t in T:
            for s in S:
                m.Y[t, s].fix(0.0)
    if fix_first_stage is not None:
        Xf, Rf = fix_first_stage
        for p in P:
            m.Rc[p].fix(round(Rf[p]))
            for t in T:
                m.X[p, t].fix(max(0.0, Xf[p, t]))

    # ---- constraints
    def eu_mandate(m, t, s):
        mu = md.loc[t, "eu_saf_pct"] / 100
        lhs = sum(m.Z[r, p, t, s] * Lup[r] for r in R_EU for p in P)
        return lhs >= mu * sum(Lup[r] for r in R_EU) - m.Ueu[t, s]
    m.c_eu = pyo.Constraint(T, S, rule=eu_mandate)

    def uk_mandate(m, t, s):
        mu = md.loc[t, "uk_saf_pct"] / 100
        lhs = sum(m.Z[r, p, t, s] * Lup[r] for r in R_UK for p in P)
        return lhs >= mu * sum(Lup[r] for r in R_UK) - m.Uuk[t, s]
    m.c_uk = pyo.Constraint(T, S, rule=uk_mandate)

    def eu_ptl(m, t, s):
        sg = md.loc[t, "eu_ptl_pct"] / 100
        if sg <= 0:
            return pyo.Constraint.Skip
        lhs = sum(m.Z[r, "PtL-FT", t, s] * Lup[r] for r in R_EU)
        return lhs >= sg * sum(Lup[r] for r in R_EU) - m.Useu[t, s]
    m.c_eu_ptl = pyo.Constraint(T, S, rule=eu_ptl)

    def uk_ptl(m, t, s):
        sg = md.loc[t, "uk_ptl_pct"] / 100
        if sg <= 0:
            return pyo.Constraint.Skip
        lhs = sum(m.Z[r, "PtL-FT", t, s] * Lup[r] for r in R_UK)
        return lhs >= sg * sum(Lup[r] for r in R_UK) - m.Usuk[t, s]
    m.c_uk_ptl = pyo.Constraint(T, S, rule=uk_ptl)

    def voluntary(m, t, s):
        lhs = sum(m.Z[r, p, t, s] * L[r] for r in R for p in P) + m.Y[t, s] + m.Uv[t, s]
        return lhs >= OMEGA[year[t]] * Ltot[t]
    m.c_vol = pyo.Constraint(T, S, rule=voluntary)

    def balance(m, p, t, s):
        return m.X[p, t] + m.W[p, t, s] >= sum(m.Z[r, p, t, s] * L[r] for r in R)
    m.c_bal = pyo.Constraint(P, T, S, rule=balance)

    def avail(m, p, t, s):
        return m.W[p, t, s] <= a[p][s, t]
    m.c_av = pyo.Constraint(P, T, S, rule=avail)

    def fwd_cap(m, p, t):
        cap = FWD_CAP[p] if (p != "PtL-FT" or t >= PTL_FWD_START) else 0.0
        return m.X[p, t] <= cap * m.Rc[p]
    m.c_fc = pyo.Constraint(P, T, rule=fwd_cap)

    def fwd_min(m, p):
        return sum(m.X[p, t] for t in T) >= XMIN[p] * m.Rc[p]
    m.c_fm = pyo.Constraint(P, rule=fwd_min)

    def bc_cap(m, t, s):
        return m.Y[t, s] <= BC_CAP_SHARE * Ltot[t]
    m.c_bc = pyo.Constraint(T, S, rule=bc_cap)

    def blend(m, r, t, s):
        return sum(m.Z[r, p, t, s] for p in P) <= ASTM_CAP
    m.c_bl = pyo.Constraint(R, T, S, rule=blend)

    # ---- scenario cost expression (EUR M)
    def cost_s(m, s):
        fwd = sum(cf[p, t] * m.X[p, t] for p in P for t in T)
        fix = sum(PHI[p] * m.Rc[p] for p in P)
        bc = sum(cb[t] * m.Y[t, s] for t in T)
        spot = sum(cs[p][s, t] * m.W[p, t, s] for p in P for t in T)
        fossil = sum(jet[s, t] * (1 - sum(m.Z[r, p, t, s] for p in P)) * L[r]
                     for r in R for t in T)
        ets = sum(eua[s, t] * EPS * sum(Lup[r] * (1 - sum(delta[p] * m.Z[r, p, t, s] for p in P))
                                        for r in R_EU) for t in T)
        corsia = sum(cors[s, t] * md.loc[t, "corsia_offset_share"] * EPS *
                     sum(L[r] * (1 - sum(delta[p] * m.Z[r, p, t, s] for p in P))
                         for r in R_NE) for t in T)
        # EU shortfall penalty: multiplier x (SAF price - jet price). The synthetic
        # sub-mandate shortfall uses the HEFA gap by default (paper results) or the
        # PtL gap with eu_ptl_penalty_on="ptl" (robustness check)
        syn = hefa if eu_ptl_penalty_on == "hefa" else ptl
        pen_eu = sum(md.loc[t, "eu_penalty_multiplier"] *
                     (max(0.0, hefa[s, t] - jet[s, t]) * m.Ueu[t, s] +
                      max(0.0, syn[s, t] - jet[s, t]) * m.Useu[t, s]) for t in T)
        pen_uk = sum(md.loc[t, "uk_buyout_eur_per_L"] * m.Uuk[t, s] +
                     md.loc[t, "uk_buyout_ptl_eur_per_L"] * m.Usuk[t, s] for t in T)
        pen_vol = sum(P_VOL * m.Uv[t, s] for t in T)
        return fwd + fix + bc + spot + fossil + ets + corsia + pen_eu + pen_uk + pen_vol
    m.Cost = pyo.Expression(S, rule=cost_s)

    # all-fossil counterfactual baseline (decision-independent, per scenario, EUR M)
    base = np.array([
        sum(jet[s, t] * Ltot[t] for t in T)
        + sum(eua[s, t] * EPS * sum(Lup[r] for r in R_EU) for t in T)
        + sum(cors[s, t] * md.loc[t, "corsia_offset_share"] * EPS *
              sum(L[r] for r in R_NE) for t in T)
        for s in S])
    m._base = base

    def net_cost(m, s):
        return m.Cost[s] - float(base[s])
    m.NetCost = pyo.Expression(S, rule=net_cost)

    def cvar_link(m, s):
        return m.V[s] >= m.NetCost[s] - m.eta
    m.c_cvar = pyo.Constraint(S, rule=cvar_link)

    expected = sum(pi[s] * m.NetCost[s] for s in S)
    cvar = m.eta + (1.0 / (1.0 - alpha)) * sum(pi[s] * m.V[s] for s in S)
    m.obj = pyo.Objective(expr=expected + lam * cvar, sense=pyo.minimize)
    m._meta = dict(lam=lam, alpha=alpha, no_bc=no_bc, eu_ptl_penalty_on=eu_ptl_penalty_on,
                   R_EU=R_EU, R_UK=R_UK, R_NE=R_NE,
                   L=L, Lup=Lup, Ltot=Ltot, pi=pi)
    return m


def solve(m, mip_gap=1e-4, time_limit=1800, log=False):
    opt = Highs()
    opt.config.mip_gap = mip_gap
    opt.config.time_limit = time_limit
    opt.config.stream_solver = log
    res = opt.solve(m)
    if res.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(f"solver stopped with {res.termination_condition}")
    return res


def cvar_exact(costs, pi, alpha=ALPHA):
    """Exact CVaR_alpha of a discrete distribution (the scenario at VaR enters fractionally)."""
    rem, acc = 1.0 - alpha, 0.0
    for i in np.argsort(costs)[::-1]:
        w = min(pi[i], rem)
        acc += w * costs[i]
        rem -= w
        if rem <= 1e-15:
            break
    return float(acc / (1.0 - alpha))


def extract(m, d: Data, label=""):
    """Pull headline metrics from a solved model. Costs on NET compliance-cost basis."""
    pi, alpha = m._meta["pi"], m._meta["alpha"]
    S = range(d.S)
    costs = np.array([pyo.value(m.NetCost[s]) for s in S])
    total = np.array([pyo.value(m.Cost[s]) for s in S])
    exp = float((pi * costs).sum())
    exp_total = float((pi * total).sum())
    # CVaR as reported in the paper: probability-weighted mean of net cost over the
    # scenarios at or above VaR (cvar95_exact_M gives the exact value at alpha)
    order = np.argsort(costs)
    cw = np.cumsum(pi[order])
    var_idx = order[min(np.searchsorted(cw, alpha), len(order) - 1)]
    var95 = costs[var_idx]
    tail = costs >= var95 - 1e-12
    cvar95 = float((pi[tail] * costs[tail]).sum() / pi[tail].sum())
    T = range(d.T)
    fwd_total = sum(pyo.value(m.X[p, t]) for p in d.P for t in T)
    spot_total = float(np.sum([pi[s] * sum(pyo.value(m.W[p, t, s]) for p in d.P for t in T)
                               for s in S]))
    bc_total = float(np.sum([pi[s] * sum(pyo.value(m.Y[t, s]) for t in T) for s in S]))
    short_eu = float(np.sum([pi[s] * sum(pyo.value(m.Ueu[t, s]) for t in T) for s in S]))
    short_uk = float(np.sum([pi[s] * sum(pyo.value(m.Uuk[t, s]) for t in T) for s in S]))
    short_v = float(np.sum([pi[s] * sum(pyo.value(m.Uv[t, s]) for t in T) for s in S]))
    S_ok = [s for s in S
            if max(pyo.value(m.Ueu[t, s]) for t in T) <= 1e-6
            and max(pyo.value(m.Uuk[t, s]) for t in T) <= 1e-6]
    return dict(label=label, expected_M=exp, expected_total_M=exp_total,
                cvar95_M=cvar95, cvar95_exact_M=cvar_exact(costs, pi, alpha), var95_M=float(var95), fwd_ML=float(fwd_total),
                spot_expected_ML=spot_total, bc_expected_ML=bc_total,
                shortfall_eu_ML=short_eu, shortfall_uk_ML=short_uk, shortfall_vol_ML=short_v,
                mandate_compliance_rate=len(S_ok) / d.S, costs=costs.tolist())


def first_stage_solution(m, d: Data):
    """Forward volumes X[p, t] (ML) and contract activations Rc[p] of a solved model."""
    Xf = {(p, t): float(pyo.value(m.X[p, t])) for p in d.P for t in range(d.T)}
    Rf = {p: float(pyo.value(m.Rc[p])) for p in d.P}
    return Xf, Rf
