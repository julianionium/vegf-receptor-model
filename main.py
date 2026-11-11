"""
Run the VEGF receptor model analyses.

    python main.py                    run everything
    python main.py validate           ODE time course + checks
    python main.py dose-response      VEGFR-low vs VEGFR-high cells
    python main.py competition-map    decoy effect across dose and receptor ratio
    python main.py sensitivity        +/-50% one-at-a-time sensitivity
    python main.py surrogate          generate training data and train the neural network

Numbers go to data/, figures go to figures/.
"""

import argparse
import pandas as pd
from vegfr_model import load_parameters
from vegfr_model import analysis, surrogate, plots
from vegfr_model.parameters import data_path, fig_path


def run_validate(p):
    tc, checks = analysis.validate(p)
    tc.to_csv(data_path("timecourse.csv"), index=False)
    plots.timecourse(tc, fig_path("binding_timecourse.png"),
                     f"VEGF ({p.dose_baseline:.0f} pM) binding, "
                     f"{p.vegfr1_low:,.0f} of each receptor per cell")
    print(f"  mass balance error         {checks['mass_balance_error']:.1e}")
    print(f"  VEGFR2 bound, ODE          {checks['vegfr2_bound_ode']:.6f}")
    print(f"  VEGFR2 bound, equilibrium  {checks['vegfr2_bound_equilibrium']:.6f}")


def run_dose_response(p):
    df = analysis.dose_response(p)
    df.to_csv(data_path("dose_response.csv"), index=False)
    plots.dose_response(df, fig_path("dose_response.png"))
    small = analysis.dose_response(p, doses_pM=[5, 50, 500])
    print(small[["cells", "dose_pM", "percent_drop"]].round(1).to_string(index=False))


def run_competition_map(p):
    df = analysis.competition_map(p)
    df.to_csv(data_path("competition_map.csv"), index=False)
    plots.competition_map(df, fig_path("competition_map.png"))
    print(df.groupby("VEGFR2_per_cell").percent_drop.max().round(0).to_string())


def run_sensitivity(p):
    df = analysis.sensitivity(p)
    df.to_csv(data_path("sensitivity.csv"), index=False)
    plots.sensitivity(df, fig_path("sensitivity.png"),
                      f"VEGFR-high cells, {p.dose_baseline:.0f} pM")
    print(f"  baseline VEGFR2 bound {df.attrs['baseline_vegfr2_bound']:.3f}")
    print(df.round(1).to_string(index=False))


def run_surrogate(p):
    data = surrogate.generate_dataset(p)
    data.to_csv(data_path("training_data.csv"), index=False)
    data = pd.read_csv(data_path("training_data.csv"))  # train from the saved file
    scores, test = surrogate.train_and_evaluate(data)
    scores.to_csv(data_path("surrogate_scores.csv"), index=False)
    test.to_csv(data_path("surrogate_test_predictions.csv"), index=False)
    plots.surrogate_fit(test, scores, fig_path("surrogate_fit.png"))
    print(scores.round(4).to_string(index=False))


COMMANDS = {
    "validate": run_validate,
    "dose-response": run_dose_response,
    "competition-map": run_competition_map,
    "sensitivity": run_sensitivity,
    "surrogate": run_surrogate,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", *COMMANDS])
    args = parser.parse_args()

    p = load_parameters()
    todo = COMMANDS if args.command == "all" else {args.command: COMMANDS[args.command]}
    for name, fn in todo.items():
        print(f"\n== {name}")
        fn(p)
    print("\nDone. Results in data/ and figures/.")


if __name__ == "__main__":
    main()
