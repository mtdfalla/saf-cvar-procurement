"""Solve the experiments reported in the paper and write results/.

Usage: python src/run_experiments.py [step ...]
Steps: lam0 lam025 lam05 lam1 lam2 eev nobc0 nobc1 (default: all, then the summary)

Each step is one solve. Results are added to results/experiments.json, so steps can
be run separately. The summary is recomputed after every call once all steps exist.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import (build_model, expected_value_data, extract,  # noqa: E402
                   first_stage_solution, load_data, solve)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
EXP = RES / "experiments.json"
FS = RES / "first_stage.json"

LAMBDAS = {"lam0": 0.0, "lam025": 0.25, "lam05": 0.5, "lam1": 1.0, "lam2": 2.0}
STEPS = list(LAMBDAS) + ["eev", "nobc0", "nobc1"]


def run_step(step, d, out, fs_out):
    t0 = time.time()
    if step in LAMBDAS:
        m = build_model(d, lam=LAMBDAS[step])
        solve(m)
        r = extract(m, d, label=f"cvar_{step}")
        Xf, _ = first_stage_solution(m, d)
        fs_out[step] = {f"{p}|{t}": round(v, 4) for (p, t), v in Xf.items()}
    elif step == "eev":
        # expected-value plan, evaluated against all scenarios
        dev = expected_value_data(d)
        mev = build_model(dev, lam=0.0)
        solve(mev)
        m = build_model(d, lam=0.0, fix_first_stage=first_stage_solution(mev, dev))
        solve(m)
        r = extract(m, d, label="eev")
    elif step in ("nobc0", "nobc1"):
        m = build_model(d, lam=0.0 if step == "nobc0" else 1.0, no_bc=True)
        solve(m)
        r = extract(m, d, label=f"no_bc_{step[-1]}")
    else:
        raise SystemExit(f"unknown step {step}, choose from {STEPS}")
    r["solve_seconds"] = round(time.time() - t0, 1)
    out[r["label"]] = r
    print(f"{r['label']}: E={r['expected_M']:.2f} CVaR95={r['cvar95_M']:.2f} "
          f"(exact {r['cvar95_exact_M']:.2f}) fwd={r['fwd_ML']:.1f} "
          f"spot={r['spot_expected_ML']:.1f} bc={r['bc_expected_ML']:.1f} "
          f"compliance={r['mandate_compliance_rate']:.3f} ({r['solve_seconds']} s)", flush=True)


def summary(out):
    rn, eev = out["cvar_lam0"], out["eev"]
    lam_runs = [out[f"cvar_{k}"] for k in LAMBDAS if k != "lam0"]
    best = min(lam_runs, key=lambda r: r["cvar95_M"])
    vss = eev["expected_M"] - rn["expected_M"]
    vra = rn["cvar95_M"] - best["cvar95_M"]
    return dict(
        value_of_stochastic_solution_M=round(vss, 2),
        value_of_stochastic_solution_pct=round(100 * vss / rn["expected_M"], 2),
        cvar_reduction_M=round(vra, 2), cvar_reduction_run=best["label"],
        cvar_reduction_pct=round(100 * vra / rn["cvar95_M"], 2),
        risk_premium_M=round(best["expected_M"] - rn["expected_M"], 2),
        bc_value_expected_M=round(out["no_bc_0"]["expected_M"] - rn["expected_M"], 2),
        bc_value_cvar_M=round(out["no_bc_1"]["cvar95_M"] - out["cvar_lam1"]["cvar95_M"], 2),
        bc_value_cvar_exact_M=round(out["no_bc_1"]["cvar95_exact_M"]
                                    - out["cvar_lam1"]["cvar95_exact_M"], 2),
    )


def main(steps):
    out = json.loads(EXP.read_text()) if EXP.exists() else {}
    fs_out = json.loads(FS.read_text()) if FS.exists() else {}
    out.pop("summary", None)
    d = load_data()
    for step in steps:
        run_step(step, d, out, fs_out)
        FS.write_text(json.dumps(fs_out, indent=1))
        EXP.write_text(json.dumps(out, indent=1))
    needed = [f"cvar_{k}" for k in LAMBDAS] + ["eev", "no_bc_0", "no_bc_1"]
    if all(k in out for k in needed):
        out["summary"] = summary(out)
        print("summary", json.dumps(out["summary"]))
    EXP.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or STEPS)
