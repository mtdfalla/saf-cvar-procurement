"""Build the model inputs in data/model/ from the source tables in data/inputs/.

Outputs:
  routes.csv          13 routes with quarterly fuel and regulatory region (EU, UK, NONE)
  mandates.csv        quarterly EU and UK blending shares, penalty and CORSIA parameters
  pathways.csv        HEFA-SPK and PtL-FT parameters
  central_prices.csv  central quarterly price paths (jet, EUA, CORSIA, HEFA, PtL, certificate)
  calibration.json    volatilities and correlations (estimated from history or assumed)

Run: python src/build_dataset.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "inputs"
OUT = ROOT / "data" / "model"
OUT.mkdir(parents=True, exist_ok=True)

EUR_PER_USD = 1 / 1.08      # fixed reference exchange rates
EUR_PER_GBP = 1.17
JET_DENSITY = 0.80          # kg/L
UPLIFT_SHARE = 0.5          # share of route fuel uplifted at the regulated airport
FOSSIL_GHG = 89.0           # gCO2e/MJ, CORSIA fossil jet baseline

YEARS = list(range(2025, 2031))
QUARTERS = [(y, q) for y in YEARS for q in (1, 2, 3, 4)]

# ------------------------------------------------------------------ routes
routes = pd.read_csv(INP / "hub_routes.csv")
burn = pd.read_csv(INP / "aircraft_fuel_burn.csv").set_index("aircraft_type")
kg_per_km = routes["aircraft"].map(burn["fuel_burn_kg_per_km_cruise"])
assert kg_per_km.notna().all(), "aircraft type missing from aircraft_fuel_burn.csv"
# outbound flights only: distance x cruise burn x weekly departures x 52 weeks
routes["annual_fuel_L"] = routes.distance_km * kg_per_km * routes.weekly_frequency * 52 / JET_DENSITY
routes["quarterly_fuel_L"] = routes["annual_fuel_L"] / 4.0          # no seasonality
routes["reg_region"] = routes["region"].replace({"Other": "NONE"})
routes["uplift_share"] = np.where(routes.reg_region != "NONE", UPLIFT_SHARE, 0.0)
routes = routes.rename(columns={"aircraft": "primary_aircraft"})
routes = routes[["route_id", "destination", "destination_country", "reg_region",
                 "distance_km", "primary_aircraft", "weekly_frequency",
                 "annual_fuel_L", "quarterly_fuel_L", "uplift_share"]]
routes.to_csv(OUT / "routes.csv", index=False)

# ---------------------------------------------------------------- mandates
refueleu = pd.read_csv(INP / "refueleu_mandates.csv").set_index("year")
uk_total = {2025: 2.0, 2026: 3.6, 2027: 5.2, 2028: 6.8, 2029: 8.4, 2030: 10.0}  # UK SAF Mandate
uk_ptl = {2025: 0.0, 2026: 0.0, 2027: 0.0, 2028: 0.2, 2029: 0.35, 2030: 0.5}    # UK PtL obligation

rows = []
for (y, q) in QUARTERS:
    rows.append({
        "year": y, "quarter": q, "t": (y - 2025) * 4 + q - 1,
        "eu_saf_pct": refueleu.loc[y, "saf_total_pct"],
        "eu_ptl_pct": refueleu.loc[y, "efuel_subshare_pct"],
        "uk_saf_pct": uk_total[y],
        "uk_ptl_pct": uk_ptl[y],
        "eu_penalty_multiplier": 2.0,                         # ReFuelEU Art. 12 minimum
        "uk_buyout_eur_per_L": 4.70 * EUR_PER_GBP,            # GBP 4.70/L main buy-out
        "uk_buyout_ptl_eur_per_L": 5.00 * EUR_PER_GBP,        # GBP 5.00/L PtL buy-out
        "corsia_offset_share": 0.10 + 0.10 * (y - 2025) / 5,  # offset share 10% to 20%
    })
pd.DataFrame(rows).to_csv(OUT / "mandates.csv", index=False)

# ---------------------------------------------------------------- pathways
pw = pd.read_csv(INP / "saf_pathways.csv")
pw = pw[pw.pathway.isin(["HEFA-SPK", "PtL-FT"])].copy()
# life-cycle reduction factor from central intensity against the fossil baseline
pw["delta"] = (1 - pw["ghg_intensity_g_CO2e_per_MJ_central"] / FOSSIL_GHG).round(3)
pw[["pathway", "mjsp_eur_per_L_central", "delta", "max_blend_pct_ASTM_D7566"]].to_csv(
    OUT / "pathways.csv", index=False)

# ---------------------------------------------------------- central prices
proj = pd.read_csv(INP / "saf_price_projections.csv")
jet = pd.read_csv(INP / "jet_fuel_price_monthly.csv", parse_dates=["date"])
eua = pd.read_csv(INP / "eua_price_monthly.csv")

jet_base_eur = jet.tail(12)["usd_per_L"].mean() * EUR_PER_USD    # mean of last 12 months
eua_base = eua.tail(12)["eur_per_tCO2"].astype(float).mean()

central = []
for (y, q) in QUARTERS:
    hefa = proj[(proj.pathway == "HEFA-SPK") & (proj.year == y)]["price_central_eur_per_L"].iloc[0]
    ptl = proj[(proj.pathway == "PtL-FT") & (proj.year == y)]["price_central_eur_per_L"].iloc[0]
    central.append({
        "year": y, "quarter": q, "t": (y - 2025) * 4 + q - 1,
        "jet_eur_per_L": round(jet_base_eur, 4),      # flat central path
        "eua_eur_per_t": round(eua_base, 2),
        "corsia_eur_per_t": 20.0,
        "hefa_spot_eur_per_L": hefa,
        "ptl_spot_eur_per_L": ptl,
        "bc_cert_eur_per_L": 0.60,                    # book-and-claim certificate price
    })
pd.DataFrame(central).to_csv(OUT / "central_prices.csv", index=False)

# ------------------------------------------------------------- calibration
jet_q = jet[jet.date >= "2010-01-01"].set_index("date")["usd_per_L"].resample("QE").mean()
jet_lr = np.log(jet_q).diff().dropna()
eua["date"] = pd.to_datetime(eua["date"], format="%Y-%m")
eua_q = (eua[eua.date >= "2018-01-01"].set_index("date")["eur_per_tCO2"]
         .astype(float).resample("QE").mean())
eua_lr = np.log(eua_q).diff().dropna()
rho_jet_eua = float(pd.concat([jet_lr.rename("jet"), eua_lr.rename("eua")], axis=1)
                    .dropna().corr().iloc[0, 1])

calib = {
    "quarterly_sigma": {
        "jet": round(float(jet_lr.std()), 4),     # estimated, 2010 onward
        "eua": round(float(eua_lr.std()), 4),     # estimated, 2018 onward
        "hefa": 0.075,                            # assumed
        "ptl": 0.125,                             # assumed
        "corsia": 0.20,                           # assumed
        "avail": 0.15,                            # assumed
    },
    "correlations": {
        "jet_eua": round(rho_jet_eua, 3),         # estimated
        "jet_hefa": 0.50, "jet_ptl": 0.30,        # assumed
        "eua_corsia": 0.30,                       # assumed
        "hefa_avail": -0.40, "ptl_avail": -0.40,  # assumed: scarcity raises price
    },
    "availability": {
        "accessible_share_of_eu_supply": 0.015,
        "note": "Airline-accessible spot SAF as a share of central EU supply",
    },
    "history": {
        "jet_base_eur_per_L": round(float(jet_base_eur), 4),
        "jet_post2010_quarterly_sigma": round(float(jet_lr.std()), 4),
        "eua_base_eur_per_t": round(float(eua_base), 2),
        "eua_post2018_quarterly_sigma": round(float(eua_lr.std()), 4),
        "rho_jet_eua_estimated": round(rho_jet_eua, 3),
        "n_quarters_jet": int(len(jet_lr)), "n_quarters_eua": int(len(eua_lr)),
    },
}
(OUT / "calibration.json").write_text(json.dumps(calib, indent=2))

ann = routes.groupby("reg_region")["annual_fuel_L"].sum() / 1e9
print("Routes:", len(routes), routes.reg_region.value_counts().to_dict())
print("Annual fuel (billion L):", ann.round(3).to_dict(), "total", round(ann.sum(), 3))
print("Jet base EUR/L:", round(jet_base_eur, 4), "| EUA base EUR/t:", round(eua_base, 2))
print("Files written to", OUT)
