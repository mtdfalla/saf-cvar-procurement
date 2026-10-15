"""Scenario generation.

Risk factors (6): jet price, EUA price, HEFA spot, PtL spot, CORSIA credit, SAF availability.
Process: stationary AR(1) in log-deviations around central trajectories, innovations drawn
from a Gaussian copula (multivariate normal). AR(1) persistence estimated from jet/EUA history.
500 raw paths -> 100 representative scenarios via Heitsch–Römisch fast-forward selection.

Run: python src/scenarios.py  (writes data/model/scenarios.npz and scenario_summary.csv)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "model"
INP = ROOT / "data" / "inputs"
SEED = 53
N_RAW, N_RED = 500, 100
FACTORS = ["jet", "eua", "hefa", "ptl", "corsia", "avail"]


def estimate_phi():
    """AR(1) persistence of quarterly log prices around trend, from history."""
    jet = pd.read_csv(INP / "jet_fuel_price_monthly.csv", parse_dates=["date"])
    jq = np.log(jet[jet.date >= "2010-01-01"].set_index("date")["usd_per_L"]
                .resample("QE").mean()).dropna()
    eua = pd.read_csv(INP / "eua_price_monthly.csv")
    eua["date"] = pd.to_datetime(eua["date"], format="%Y-%m")
    eq = np.log(eua[eua.date >= "2018-01-01"].set_index("date")["eur_per_tCO2"]
                .astype(float).resample("QE").mean()).dropna()
    phis = {}
    for name, s in [("jet", jq), ("eua", eq)]:
        x = (s - s.rolling(8, min_periods=4).mean()).dropna()  # deviation from local trend
        phis[name] = float(np.clip(np.corrcoef(x[1:], x[:-1])[0, 1], 0.3, 0.9))
    return phis


def corr_matrix(c):
    """Assemble 6x6 correlation matrix from pairwise entries, with nearest-PD repair if needed."""
    M = np.eye(6)
    ix = {f: i for i, f in enumerate(FACTORS)}
    pairs = {("jet", "eua"): c["jet_eua"], ("jet", "hefa"): c["jet_hefa"],
             ("jet", "ptl"): c["jet_ptl"], ("eua", "corsia"): c["eua_corsia"],
             ("hefa", "avail"): c["hefa_avail"], ("ptl", "avail"): c["ptl_avail"],
             ("hefa", "ptl"): 0.4}
    for (a, b), v in pairs.items():
        M[ix[a], ix[b]] = M[ix[b], ix[a]] = v
    w, V = np.linalg.eigh(M)
    if w.min() < 1e-8:                      # nearest-PD repair
        w = np.clip(w, 1e-8, None)
        M = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(M)); M = M / np.outer(d, d)
    return M


def simulate(central, calib, phis, n=N_RAW, T=24, seed=SEED):
    rng = np.random.default_rng(seed)
    sig = np.array([calib["quarterly_sigma"][f] for f in FACTORS])
    phi = np.array([phis.get(f, np.mean(list(phis.values()))) for f in FACTORS])
    Mcorr = corr_matrix(calib["correlations"])
    Lc = np.linalg.cholesky(Mcorr)
    # stationary AR(1): x_t = phi x_{t-1} + eps, sd(eps) chosen so sd(x) = target sigma path-wise
    eps_sd = sig * np.sqrt(1 - phi**2)
    x = np.zeros((n, T, 6))
    x0 = (rng.standard_normal((n, 6)) @ Lc.T) * sig          # stationary start
    prev = x0
    for t in range(T):
        eps = (rng.standard_normal((n, 6)) @ Lc.T) * eps_sd
        prev = phi * prev + eps
        x[:, t, :] = prev
    # lognormal, mean-unbiased: factor = central * exp(x - 0.5*Var_stationary)
    adj = 0.5 * sig**2
    mult = np.exp(x - adj)                                   # (n, T, 6)
    out = {}
    cen = central.sort_values("t")
    out["jet"] = cen["jet_eur_per_L"].values * mult[:, :, 0]
    out["eua"] = cen["eua_eur_per_t"].values * mult[:, :, 1]
    out["hefa"] = cen["hefa_spot_eur_per_L"].values * mult[:, :, 2]
    out["ptl"] = cen["ptl_spot_eur_per_L"].values * mult[:, :, 3]
    out["corsia"] = cen["corsia_eur_per_t"].values * mult[:, :, 4]
    out["avail_mult"] = mult[:, :, 5]                        # applied to accessible-supply path
    return out


def fast_forward_select(paths_std, n_keep, probs=None):
    """Heitsch–Römisch fast-forward scenario selection. paths_std: (n, dim)."""
    n = paths_std.shape[0]
    p = np.full(n, 1.0 / n) if probs is None else probs
    D = np.linalg.norm(paths_std[:, None, :] - paths_std[None, :, :], axis=2)
    selected, remaining = [], list(range(n))
    d = D.copy()
    for _ in range(n_keep):
        # pick scenario minimizing sum of prob-weighted distances of remaining to nearest selected
        costs = [(p[remaining] * d[remaining, u]).sum() for u in remaining]
        u = remaining[int(np.argmin(costs))]
        selected.append(u)
        remaining.remove(u)
        d = np.minimum(d, D[:, [u]])                          # update nearest-selected distance
    # redistribute probabilities to nearest selected scenario
    sel = np.array(selected)
    nearest = sel[np.argmin(D[:, sel], axis=1)]
    new_p = np.array([p[nearest == u].sum() for u in sel])
    return sel, new_p / new_p.sum()


def main():
    central = pd.read_csv(MODEL / "central_prices.csv")
    calib = json.loads((MODEL / "calibration.json").read_text())
    phis = estimate_phi()
    print("AR(1) persistence estimated:", phis)

    raw = simulate(central, calib, phis)

    # availability central path: accessible share of EU supply, quarterly, split by pathway
    avail = pd.read_csv(INP / "saf_availability_projections.csv")
    eu = avail[avail.region == "EU"].set_index("year")["saf_total_billion_L_central"]
    share = calib["availability"]["accessible_share_of_eu_supply"]
    years = central.sort_values("t")["year"].values
    acc_q = np.array([eu[y] * 1e9 * share / 4 for y in years])          # L per quarter
    hefa_share = np.array([0.95 if y < 2028 else 0.85 for y in years])
    a_hefa = acc_q * hefa_share * raw["avail_mult"]
    a_ptl = acc_q * (1 - hefa_share) * raw["avail_mult"]

    # stack standardized paths for reduction
    stack = []
    for k in ["jet", "eua", "hefa", "ptl", "corsia", "avail_mult"]:
        v = raw[k]
        stack.append((v - v.mean(0)) / (v.std(0) + 1e-12))
    flat = np.concatenate([s.reshape(N_RAW, -1) for s in stack], axis=1)
    sel, probs = fast_forward_select(flat, N_RED)
    print(f"Reduced {N_RAW} -> {N_RED}, prob range [{probs.min():.4f}, {probs.max():.4f}]")

    np.savez_compressed(
        MODEL / "scenarios.npz",
        jet=raw["jet"][sel], eua=raw["eua"][sel], hefa=raw["hefa"][sel],
        ptl=raw["ptl"][sel], corsia=raw["corsia"][sel],
        a_hefa=a_hefa[sel], a_ptl=a_ptl[sel], probs=probs,
        raw_jet_mean=raw["jet"].mean(0), raw_jet_std=raw["jet"].std(0),
    )

    # summary for sanity checks + paper table
    s = pd.DataFrame({
        "factor": ["jet EUR/L", "EUA EUR/t", "HEFA EUR/L", "PtL EUR/L",
                   "CORSIA EUR/t", "HEFA avail ML/q (2025)", "HEFA avail ML/q (2030)"],
        "mean": [raw["jet"][sel].mean(), raw["eua"][sel].mean(), raw["hefa"][sel].mean(),
                 raw["ptl"][sel].mean(), raw["corsia"][sel].mean(),
                 a_hefa[sel][:, :4].mean() / 1e6, a_hefa[sel][:, -4:].mean() / 1e6],
        "p5": [np.quantile(raw["jet"][sel], .05), np.quantile(raw["eua"][sel], .05),
               np.quantile(raw["hefa"][sel], .05), np.quantile(raw["ptl"][sel], .05),
               np.quantile(raw["corsia"][sel], .05),
               np.quantile(a_hefa[sel][:, :4], .05) / 1e6, np.quantile(a_hefa[sel][:, -4:], .05) / 1e6],
        "p95": [np.quantile(raw["jet"][sel], .95), np.quantile(raw["eua"][sel], .95),
                np.quantile(raw["hefa"][sel], .95), np.quantile(raw["ptl"][sel], .95),
                np.quantile(raw["corsia"][sel], .95),
                np.quantile(a_hefa[sel][:, :4], .95) / 1e6, np.quantile(a_hefa[sel][:, -4:], .95) / 1e6],
    }).round(3)
    s.to_csv(MODEL / "scenario_summary.csv", index=False)
    print(s.to_string(index=False))

    # reduction quality check: moment preservation
    for k in ["jet", "eua", "hefa"]:
        full_m = raw[k].mean()
        red_m = float((raw[k][sel].mean(axis=1) * probs).sum())
        print(f"moment check {k}: raw mean {full_m:.3f} vs reduced weighted {red_m:.3f}")


if __name__ == "__main__":
    main()
