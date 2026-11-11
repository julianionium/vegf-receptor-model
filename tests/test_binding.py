"""
Tests for the binding model. Run with:  pytest
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vegfr_model import load_parameters, simulate, equilibrium, vegfr2_bound, vegfr1_effect
from vegfr_model.binding import mass_balance_error


@pytest.fixture(scope="module")
def p():
    return load_parameters()


def test_mass_is_conserved(p):
    """Binding only moves molecules between states, so totals must not drift."""
    r1, r2 = p.low
    _, y = simulate(p, p.dose_baseline, r1, r2)
    assert mass_balance_error(p, p.dose_baseline, r1, r2, y) < 1e-10


def test_ode_reaches_equilibrium(p):
    """The ODE end state should match the directly solved steady state."""
    r1, r2 = p.low
    _, y = simulate(p, p.dose_baseline, r1, r2)
    _, _, C2_eq = equilibrium(p, p.dose_baseline, r1, r2)
    assert y[4, -1] == pytest.approx(C2_eq, rel=1e-6)


def test_single_receptor_limit(p):
    """With no VEGFR1 and VEGF in large excess, binding follows L / (Kd + L)."""
    dose = 50
    expected = dose * 1e-12 / (p.kd2 + dose * 1e-12)
    assert vegfr2_bound(p, dose, 0, 10) == pytest.approx(expected, rel=1e-4)


def test_no_vegfr1_means_no_effect(p):
    assert vegfr1_effect(p, 50, 0, p.vegfr2_high) == pytest.approx(0, abs=1e-9)


def test_more_vegfr1_never_helps_vegfr2(p):
    """Adding a competitor can only lower VEGFR2 binding."""
    values = [vegfr2_bound(p, 10, r1, p.vegfr2_high) for r1 in (0, 1e4, 1e5, 1e6)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_only_kd_matters_at_equilibrium(p):
    """Doubling kon and koff together leaves Kd, and so the equilibrium, unchanged."""
    q = p.with_changes(kon1=2 * p.kon1, koff1=2 * p.koff1, kon2=2 * p.kon2, koff2=2 * p.koff2)
    r1, r2 = p.high
    assert vegfr2_bound(q, 50, r1, r2) == pytest.approx(vegfr2_bound(p, 50, r1, r2), rel=1e-10)


def test_decoy_effect_bigger_in_high_cells(p):
    assert vegfr1_effect(p, 5, *p.high) > vegfr1_effect(p, 5, *p.low)
