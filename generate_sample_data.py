"""Create a SMALL SYNTHETIC (made-up) dataset so the project runs out of the box.

This is NOT real market data. For real results, download the Hyderabad house
price dataset (Hyderbad_House_price.csv) and put it in backend/data/ -
train_model.py will pick it up automatically.

Columns match the real file: Unnamed: 0, title, location, price(L),
rate_persqft, area_insqft, building_status.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "sample_hyderabad_house_price.csv"

# Rough, invented base rates (Rs per sq ft) just to give the model a pattern to learn.
LOCATIONS = {
    "Banjara Hills": 12000, "Gachibowli": 8500, "Hitech City": 8000,
    "Madhapur": 8200, "Kondapur": 7000, "Kukatpally": 5800,
    "Miyapur": 5200, "Nizampet": 5600, "Bachupally": 5000,
    "Kollur": 4800, "Pocharam": 4200, "Dundigal": 4500,
}
TYPES = {"Apartment": 1.0, "Independent House": 1.15, "Villa": 1.35, "Penthouse": 1.5}
STATUS = {"Ready to move": 1.05, "New": 1.0, "Resale": 0.92, "Under Construction": 0.9}


def main(n_rows: int = 1500, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    locs = rng.choice(list(LOCATIONS), n_rows)
    types = rng.choice(list(TYPES), n_rows, p=[0.7, 0.12, 0.13, 0.05])
    status = rng.choice(list(STATUS), n_rows)
    bhk = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.08, 0.35, 0.42, 0.12, 0.03])
    area = (bhk * 450 + rng.normal(250, 120, n_rows)).clip(350, 6000).round()
    rate = np.array([LOCATIONS[x] for x in locs]) * np.array([TYPES[x] for x in types]) \
        * np.array([STATUS[x] for x in status]) * rng.normal(1.0, 0.08, n_rows)
    rate = rate.round()
    price_lakhs = (rate * area / 100000).round(2)

    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n_rows),
        "title": [f"{b} BHK {t}" for b, t in zip(bhk, types)],
        "location": locs,
        "price(L)": price_lakhs,
        "rate_persqft": rate.astype(int),
        "area_insqft": area.astype(int),
        "building_status": status,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} synthetic rows to {OUT}")


if __name__ == "__main__":
    main()
