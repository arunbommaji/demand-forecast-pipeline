"""
serve.py
--------
FastAPI inference service for the demand forecasting LSTM.
Loads the trained model + scalers and exposes POST /predict.

Run locally:   uvicorn serve:app --host 0.0.0.0 --port 8000
Docker:        see Dockerfile in this directory
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

FEATURES = ["total_units", "any_promo", "dow", "month", "lag_1", "lag_7", "roll_mean_7", "roll_mean_28"]
LOOKBACK = 14


class DemandLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


app = FastAPI(title="Demand Forecast API")

device = torch.device("cpu")
model = DemandLSTM(n_features=len(FEATURES))
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

x_scaler = joblib.load("scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")


class DayRecord(BaseModel):
    total_units: float
    any_promo: int
    dow: int
    month: int
    lag_1: float
    lag_7: float
    roll_mean_7: float
    roll_mean_28: float


class PredictRequest(BaseModel):
    history: List[DayRecord] = Field(..., description=f"Exactly {LOOKBACK} most recent daily records, oldest first")


class PredictResponse(BaseModel):
    predicted_units_next_day: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.history) != LOOKBACK:
        raise HTTPException(
            status_code=422,
            detail=f"history must contain exactly {LOOKBACK} records, got {len(req.history)}",
        )

    rows = [[getattr(r, f) for f in FEATURES] for r in req.history]
    arr = np.array(rows, dtype=np.float32)
    scaled = x_scaler.transform(arr)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(tensor).numpy()

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    return {"predicted_units_next_day": float(pred)}
