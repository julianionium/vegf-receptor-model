"""Load model parameters from data/vegf_parameters.csv."""

import csv
import os
from dataclasses import dataclass, replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
PARAMETER_FILE = os.path.join(DATA_DIR, "vegf_parameters.csv")

N_A = 6.022e23  # Avogadro's number (molecules/mol)


@dataclass(frozen=True)
class Parameters:
    kon1: float          # VEGFR1 association rate (1/M/s)
    koff1: float         # VEGFR1 dissociation rate (1/s)
    kon2: float          # VEGFR2 association rate (1/M/s)
    koff2: float         # VEGFR2 dissociation rate (1/s)
    cells: float         # cells in the well
    volume: float        # medium volume (L)
    vegfr1_low: float    # receptors per cell
    vegfr2_low: float
    vegfr1_high: float
    vegfr2_high: float
    dose_baseline: float  # pM

    @property
    def kd1(self):
        return self.koff1 / self.kon1

    @property
    def kd2(self):
        return self.koff2 / self.kon2

    @property
    def low(self):
        return (self.vegfr1_low, self.vegfr2_low)

    @property
    def high(self):
        return (self.vegfr1_high, self.vegfr2_high)

    def with_changes(self, **changes):
        """Copy of these parameters with some values changed."""
        return replace(self, **changes)

    def to_molar(self, per_cell):
        """Receptors per cell -> molar concentration in the well."""
        return per_cell * self.cells / (N_A * self.volume)


def load_parameters(path=PARAMETER_FILE):
    """Read the parameter CSV. Rows whose value isn't a single number are skipped."""
    raw = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                raw[row["parameter"]] = float(row["value"])
            except ValueError:
                pass
    return Parameters(
        kon1=raw["kon_VEGFR1"], koff1=raw["koff_VEGFR1"],
        kon2=raw["kon_VEGFR2"], koff2=raw["koff_VEGFR2"],
        cells=raw["cells"], volume=raw["volume"],
        vegfr1_low=raw["VEGFR1_low"], vegfr2_low=raw["VEGFR2_low"],
        vegfr1_high=raw["VEGFR1_high"], vegfr2_high=raw["VEGFR2_high"],
        dose_baseline=raw["VEGF_dose_baseline"],
    )


def data_path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def fig_path(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)
