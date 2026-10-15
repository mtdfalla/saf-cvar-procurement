# Data

All inputs come from public sources. Prices are in EUR unless stated (1 EUR = 1.08 USD = 0.855 GBP, fixed).

## data/inputs

| File | Content | Source |
|---|---|---|
| `jet_fuel_price_monthly.csv` | U.S. Gulf Coast kerosene-type jet fuel spot price, monthly, April 1990 to April 2026 (USD/gal and USD/L) | U.S. Energy Information Administration, via FRED series [MJFUELUSGULF](https://fred.stlouisfed.org/series/MJFUELUSGULF). Public domain |
| `eua_price_monthly.csv` | EU ETS allowance price, monthly, January 2013 to April 2026 (EUR/tCO2), rounded to 0.1 | Compiled from published series: [Sandbag Carbon Price Viewer](https://sandbag.be/carbon-price-viewer/), [ICAP Allowance Price Explorer](https://icapcarbonaction.com/en/ets-prices) |
| `hub_routes.csv` | 13 routes from Dubai (DXB): destination, regulatory region, great-circle distance, aircraft type, weekly departures | Illustrative frequencies and aircraft, informed by published 2026 schedules |
| `aircraft_fuel_burn.csv` | Cruise fuel burn per km by aircraft type | ICAO Aircraft Engine Emissions Databank, EUROCONTROL Small Emitters Tool |
| `refueleu_mandates.csv` | EU minimum SAF share and synthetic sub-share, 2025 to 2050 | Regulation (EU) 2023/2405 (ReFuelEU Aviation), Annex I |
| `saf_pathways.csv` | Cost range, life-cycle intensity, blend limit and maturity of five SAF pathways and a fossil jet reference | Reviews and agency reports listed in the `sources` column |
| `saf_price_projections.csv` | SAF price by pathway and year, 2025 to 2050 (low, central, high) | Constant learning-rate projection from the sources in the `sources` column |
| `saf_availability_projections.csv` | SAF supply, EU and global, 2025 to 2050 (billion L) | IATA SAF output estimates, IEA Net Zero by 2050 (2023 update), ICAO long-term aspirational goal |

UK mandate shares (2% in 2025 rising to 10% in 2030) and buy-out prices (GBP 4.70/L main, 5.00/L PtL) are set in `src/build_dataset.py` from the UK Department for Transport SAF Mandate guidance (2024). The same file sets these assumptions: UK PtL shares of 0.2%, 0.35% and 0.5% in 2028 to 2030, a CORSIA credit price of 20 EUR/t, and a CORSIA offset share rising from 10% to 20%.

Full references for the `sources` columns are given in the paper.

## data/model

Built by `src/build_dataset.py` and `src/scenarios.py`.

| File | Content |
|---|---|
| `routes.csv` | Quarterly fuel per route (outbound flights: distance × cruise fuel burn × weekly departures × 52 / 0.80 kg/L / 4) and the share uplifted at the regulated airport (0.5 on EU and UK routes) |
| `mandates.csv` | Quarterly EU and UK shares, penalty multiplier, buy-out prices, CORSIA offset share |
| `pathways.csv` | HEFA-SPK and PtL-FT, with life-cycle reduction factor 1 − (central intensity / 89 gCO2e/MJ) |
| `central_prices.csv` | Central quarterly paths: jet and EUA (mean of the last 12 months, flat), CORSIA credit, HEFA and PtL spot, certificate price |
| `calibration.json` | Quarterly volatilities and correlations. Jet and EUA values are estimated from the price histories, the others are assumptions. `src/scenarios.py` adds the HEFA-PtL correlation (0.4) and estimates the AR(1) persistence |
| `scenarios.npz` | 100 scenarios × 24 quarters: prices (`jet`, `eua`, `hefa`, `ptl`, `corsia`), spot availability in L per quarter (`a_hefa`, `a_ptl`) and `probs`. `raw_jet_mean` and `raw_jet_std` summarize the 500 simulated jet paths before reduction |
| `scenario_summary.csv` | Mean, 5th and 95th percentile of each factor, unweighted and pooled over all quarters |

