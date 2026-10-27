# House Price Prediction using XGBoost

A web app that estimates house prices in **Hyderabad**. You enter a location, property type,
size (sq ft), BHK and building status, and a trained **XGBoost** model returns an estimated price
(in Lakhs / Crores) with a likely range.

- **Backend:** Python + Flask (REST API) + XGBoost
- **Frontend:** React (Vite)
- **Extras:** register / log in (demo, stored in the browser), save and review past predictions

> B.Tech (CSE - AI & ML) project, Malla Reddy College of Engineering and Technology, 2024-25.
> Built by Kshitij Kodati, M Nandu Kumar and M Naga Manoj Varma. Guide: Dr. M. Gayatri.

## Project layout

```
backend/    Flask API, training script, tests, sample data
frontend/   React website
```

## 1. Run the backend

You need **Python 3.10+**.

```bash
cd backend
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train_model.py     # trains the model (about a minute) and saves it to backend/model/
python app.py             # starts the API at http://127.0.0.1:5000
```

Run the tests any time with `pytest` (inside `backend/`).

### About the data

The real dataset (`Hyderbad_House_price.csv`, columns `title, location, price(L), rate_persqft,
area_insqft, building_status`) is **not included**. To use it, put the file in `backend/data/`
and run `python train_model.py` again.

Until then, the project uses a small **made-up sample dataset**
(`backend/data/sample_hyderabad_house_price.csv`, created by `generate_sample_data.py`) so everything
runs. Predictions from it are **not real prices**, and the website shows a warning when the sample data is used.

## 2. Run the frontend

You need **Node.js 18+**. In a second terminal:

```bash
cd frontend
npm install
npm run dev               # opens at http://localhost:5173
```

If your API runs somewhere else, set `VITE_API_URL` (for example `VITE_API_URL=http://127.0.0.1:5000`).

## API

| Method | URL        | What it does                                              |
|--------|------------|-----------------------------------------------------------|
| GET    | `/health`  | Returns `{"status": "ok"}`                                |
| GET    | `/options` | Locations / property types / statuses the model knows     |
| POST   | `/predict` | Takes the property details, returns the predicted price   |

Example request body for `/predict`:

```json
{ "location": "Gachibowli", "propertyType": "Apartment", "buildingStatus": "New", "area": 1200, "bedrooms": 2 }
```

## How the model works

1. Clean the data (leave out plots, remove duplicates, missing values and extreme price-per-sq-ft outliers).
2. Read BHK and property type from the listing `title`; label-encode location, type and status.
3. Split 80 / 20 into train and test sets.
4. Tune XGBoost with 5-fold cross-validated random search on **log(price)** (prices are very skewed), then score on the test set in Lakhs (RMSE, MAE, R²).

`rate_persqft` is deliberately **not** used as an input: price = rate x area, so using it would leak the answer.

## Limitations

- Login is a browser-only demo (no real security).
- Bathrooms, parking and amenities are not inputs because the dataset has no such columns.
- Prices depend entirely on the training data; retrain regularly with fresh data.
