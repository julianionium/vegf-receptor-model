"""
Binding model.

    VEGF + VEGFR1 <-> C1    (kon1, koff1)
    VEGF + VEGFR2 <-> C2    (kon2, koff2)

simulate()     solves the 5 mass-action ODEs over time
equilibrium()  solves the steady state directly (much faster, used for sweeps)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def _rhs(t, y, p):
    L, R1, R2, C1, C2 = y
    bind1 = p.kon1 * L * R1 - p.koff1 * C1
    bind2 = p.kon2 * L * R2 - p.koff2 * C2
    return [-bind1 - bind2, -bind1, -bind2, bind1, bind2]


def simulate(p, dose_pM, r1_per_cell, r2_per_cell, hours=48):
    """
    Solve the ODEs from an unbound start.
    Returns time (h) and concentrations (M) of L, R1, R2, C1, C2.
    """
    y0 = [dose_pM * 1e-12, p.to_molar(r1_per_cell), p.to_molar(r2_per_cell), 0.0, 0.0]
    sol = solve_ivp(_rhs, (0, hours * 3600), y0, args=(p,), method="LSODA",
                    rtol=1e-8, atol=1e-22)
    return sol.t / 3600, sol.y


def equilibrium(p, dose_pM, r1_per_cell, r2_per_cell):
    """
    Steady state. At equilibrium each complex is C = R_total * L / (Kd + L),
    so solve for the free VEGF (L) that conserves total VEGF.
    Returns free L, C1, C2 (M).
    """
    L_total = dose_pM * 1e-12
    R1, R2 = p.to_molar(r1_per_cell), p.to_molar(r2_per_cell)

    def leftover(L):
        return L + R1 * L / (p.kd1 + L) + R2 * L / (p.kd2 + L) - L_total

    # tolerance must be far below pM-scale values (brentq's default is 2e-12)
    L = brentq(leftover, 0, L_total, xtol=1e-26, rtol=1e-14)
    return L, R1 * L / (p.kd1 + L), R2 * L / (p.kd2 + L)


def vegfr2_bound(p, dose_pM, r1_per_cell, r2_per_cell):
    """Fraction of VEGFR2 bound at equilibrium."""
    L, _, _ = equilibrium(p, dose_pM, r1_per_cell, r2_per_cell)
    return L / (p.kd2 + L)


def vegfr1_effect(p, dose_pM, r1_per_cell, r2_per_cell):
    """Percent drop in VEGFR2 binding caused by VEGFR1 (vs. the same cell with no VEGFR1)."""
    with_r1 = vegfr2_bound(p, dose_pM, r1_per_cell, r2_per_cell)
    no_r1 = vegfr2_bound(p, dose_pM, 0, r2_per_cell)
    return 100 * (1 - with_r1 / no_r1)


def mass_balance_error(p, dose_pM, r1_per_cell, r2_per_cell, y):
    """Largest relative drift in total VEGF, VEGFR1 and VEGFR2 over a simulation."""
    L, R1, R2, C1, C2 = y
    totals = [(L + C1 + C2, dose_pM * 1e-12),
              (R1 + C1, p.to_molar(r1_per_cell)),
              (R2 + C2, p.to_molar(r2_per_cell))]
    return max(np.max(np.abs(x - t)) / t for x, t in totals if t > 0)
