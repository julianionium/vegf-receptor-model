"""
Analyses built on the binding model. Each returns a pandas DataFrame.

validate          ODE time course + checks against the equilibrium solution
dose_response     VEGFR2 binding vs VEGF dose, VEGFR-low vs VEGFR-high cells
competition_map   VEGFR1 decoy effect across dose and VEGFR1:VEGFR2 ratio
sensitivity       one-at-a-time +/-50% changes to each input
"""

import numpy as np
import pandas as pd
from .binding import simulate, equilibrium, vegfr2_bound, vegfr1_effect, mass_balance_error


def validate(p, dose_pM=None, cells=None):
    """Solve the ODE for VEGFR-low cells and compare its end state to equilibrium."""
    dose = p.dose_baseline if dose_pM is None else dose_pM
    r1, r2 = p.low if cells is None else cells
    t, y = simulate(p, dose, r1, r2)
    L, R1, R2, C1, C2 = y
    R1_tot, R2_tot = p.to_molar(r1), p.to_molar(r2)

    _, _, C2_eq = equilibrium(p, dose, r1, r2)
    checks = {
        "mass_balance_error": mass_balance_error(p, dose, r1, r2, y),
        "vegfr2_bound_ode": C2[-1] / R2_tot,
        "vegfr2_bound_equilibrium": C2_eq / R2_tot,
    }
    timecourse = pd.DataFrame({
        "time_hours": t, "free_VEGF_M": L, "free_VEGFR1_M": R1, "free_VEGFR2_M": R2,
        "VEGF_VEGFR1_M": C1, "VEGF_VEGFR2_M": C2,
        "VEGFR1_bound_fraction": C1 / R1_tot, "VEGFR2_bound_fraction": C2 / R2_tot,
    })
    return timecourse, checks


def dose_response(p, doses_pM=np.logspace(0, 3, 40)):
    rows = []
    for name, (r1, r2) in {"VEGFR-low": p.low, "VEGFR-high": p.high}.items():
        for d in doses_pM:
            a = vegfr2_bound(p, d, r1, r2)
            b = vegfr2_bound(p, d, 0, r2)
            rows.append({"cells": name, "VEGFR1_per_cell": r1, "VEGFR2_per_cell": r2,
                         "dose_pM": d, "VEGFR2_bound_with_R1": a,
                         "VEGFR2_bound_no_R1": b, "percent_drop": 100 * (1 - a / b)})
    return pd.DataFrame(rows)


def competition_map(p, doses_pM=np.logspace(0, 3, 50), ratios=np.logspace(-1, 1, 41)):
    rows = []
    for r2 in (p.vegfr2_low, p.vegfr2_high):
        for ratio in ratios:
            for d in doses_pM:
                rows.append({"VEGFR2_per_cell": r2, "VEGFR1_to_VEGFR2_ratio": ratio,
                             "dose_pM": d,
                             "percent_drop": vegfr1_effect(p, d, ratio * r2, r2)})
    return pd.DataFrame(rows)


SENSITIVITY_INPUTS = {
    "dose": "VEGF dose", "vegfr1": "VEGFR1 per cell", "vegfr2": "VEGFR2 per cell",
    "kon1": "kon VEGFR1", "koff1": "koff VEGFR1", "kon2": "kon VEGFR2", "koff2": "koff VEGFR2",
}


def sensitivity(p, change=0.5):
    """Start from VEGFR-high cells at the baseline dose and change one input at a time."""
    base = {"dose": p.dose_baseline, "vegfr1": p.vegfr1_high, "vegfr2": p.vegfr2_high}

    def run(scale_key=None, factor=1.0):
        inputs = {k: v * (factor if k == scale_key else 1) for k, v in base.items()}
        params = p
        if scale_key in ("kon1", "koff1", "kon2", "koff2"):
            params = p.with_changes(**{scale_key: getattr(p, scale_key) * factor})
        return vegfr2_bound(params, inputs["dose"], inputs["vegfr1"], inputs["vegfr2"])

    baseline = run()
    rows = []
    for key, name in SENSITIVITY_INPUTS.items():
        rows.append({"input": name,
                     "percent_change_at_minus": 100 * (run(key, 1 - change) / baseline - 1),
                     "percent_change_at_plus": 100 * (run(key, 1 + change) / baseline - 1)})
    df = pd.DataFrame(rows)
    df.attrs["baseline_vegfr2_bound"] = baseline
    return df
