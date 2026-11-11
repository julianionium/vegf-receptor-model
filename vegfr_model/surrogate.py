"""
Neural network surrogate: learn to predict VEGFR2 binding from dose and receptor counts.

generate_dataset   simulate random cells/doses with the binding model
train_and_evaluate train an MLP and a linear-regression baseline, score on held-out data

Inputs are log10 values because doses and receptor counts span several orders of magnitude.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from .binding import vegfr2_bound

FEATURES = ["log10_dose_pM", "log10_vegfr1", "log10_vegfr2"]
TARGET = "vegfr2_bound"


def generate_dataset(p, n=10_000, seed=42,
                     log_dose=(0, 3), log_receptors=(3, 5.5)):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "log10_dose_pM": rng.uniform(*log_dose, n),        # 1 to 1000 pM
        "log10_vegfr1": rng.uniform(*log_receptors, n),    # 1,000 to ~300,000 per cell
        "log10_vegfr2": rng.uniform(*log_receptors, n),
    })
    df[TARGET] = [vegfr2_bound(p, 10**d, 10**a, 10**b) for d, a, b in df[FEATURES].values]
    return df


def train_and_evaluate(df, test_size=0.2, seed=0, hidden=(64, 64)):
    X, y = df[FEATURES].values, df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "Linear regression": LinearRegression(),
        "Neural network": MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000,
                                       early_stopping=True, random_state=seed),
    }
    scores, test = [], pd.DataFrame(X_test, columns=FEATURES)
    test["true_vegfr2_bound"] = y_test
    for name, model in models.items():
        pred = model.fit(X_train_s, y_train).predict(X_test_s)
        test[f"{name.lower().replace(' ', '_')}_prediction"] = pred
        scores.append({"model": name, "test_MAE": mean_absolute_error(y_test, pred),
                       "test_R2": r2_score(y_test, pred)})
    return pd.DataFrame(scores), test
