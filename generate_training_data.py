"""
Make the training data for the neural network (step 5).

Each row = one random cell + VEGF dose, and the VEGFR2 bound fraction the model predicts.
A fixed random seed means you get the exact same file every time.
Output:
    data/training_data.csv
"""

import numpy as np
import pandas as pd
from vegf_model import vegfr2_bound, data_path

rng = np.random.default_rng(42)
n = 10_000

df = pd.DataFrame({
    "log10_dose_pM": rng.uniform(0, 3, n),    # 1 to 1000 pM
    "log10_vegfr1": rng.uniform(3, 5.5, n),   # 1,000 to ~300,000 per cell
    "log10_vegfr2": rng.uniform(3, 5.5, n),
})
df["vegfr2_bound"] = [
    vegfr2_bound(10**d, 10**a, 10**b)
    for d, a, b in df[["log10_dose_pM", "log10_vegfr1", "log10_vegfr2"]].values
]

df.to_csv(data_path("training_data.csv"), index=False)
print(f"Saved {len(df)} rows to data/training_data.csv")
print(df.describe().round(3))
