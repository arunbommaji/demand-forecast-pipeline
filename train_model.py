"""
train_model.py
---------------
Trains an LSTM demand-forecasting model on the Spark-engineered features.
Predicts next-day total_units per store from a lookback window of
recent daily demand + calendar/lag/rolling features.
Saves the trained model + reports real accuracy and latency metrics.
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

LOOKBACK = 14  # days of history fed to the LSTM
FEATURES = ["total_units", "any_promo", "dow", "month", "lag_1", "lag_7", "roll_mean_7", "roll_mean_28"]

df = pd.read_csv("data/features.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "date"])
print(f"Loaded {len(df):,} store-day feature rows across {df.store_id.nunique()} stores")

# Chronological split: last 60 days per store held out for testing
cutoff_date = df["date"].max() - pd.Timedelta(days=60)
train_df = df[df["date"] <= cutoff_date].copy()
test_df = df[df["date"] > cutoff_date].copy()
print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} (chronological split, no leakage)")

scaler = StandardScaler()
scaler.fit(train_df[FEATURES])

y_scaler = StandardScaler()
y_scaler.fit(train_df[["total_units"]])


def build_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for store_id, g in data.groupby("store_id"):
        g = g.sort_values("date")
        vals = scaler.transform(g[FEATURES])
        target = y_scaler.transform(g[["total_units"]]).flatten()
        for i in range(len(g) - lookback):
            X.append(vals[i:i + lookback])
            y.append(target[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


t0 = time.time()
X_train, y_train = build_sequences(train_df)
X_test, y_test = build_sequences(pd.concat([train_df.tail(LOOKBACK * df.store_id.nunique()), test_df]))
print(f"Built sequences in {time.time()-t0:.2f}s -> X_train {X_train.shape}, X_test {X_test.shape}")


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=256, shuffle=True)
test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=512, shuffle=False)


class DemandLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DemandLSTM(n_features=len(FEATURES)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print(f"\nTraining on device: {device}")
EPOCHS = 12
t_train_start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(xb)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"  epoch {epoch:2d}/{EPOCHS}  train_mse={avg_loss:.3f}")
train_time = time.time() - t_train_start

# --- Evaluation ---
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        p = model(xb).cpu().numpy()
        preds.extend(p)
        actuals.extend(yb.numpy())

preds = y_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
actuals = y_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
mape = np.mean(np.abs((actuals - preds) / np.clip(actuals, 1, None))) * 100

# naive baseline: predict "same as 7 days ago" (lag_7 feature, unscaled -> need raw)
# NOTE: actuals[i] corresponds to test_df.iloc[i] directly (build_sequences() walks
# each store's [train_tail(LOOKBACK) + test_df] window and its first prediction
# target is test_df's first row for that store), so no LOOKBACK offset belongs here.
# The previous version sliced test_df[LOOKBACK : len(actuals)+LOOKBACK], which both
# misaligned the comparison and, once test_df ran out of rows, produced a shorter
# array than `actuals` -- causing sklearn's mean_absolute_error to raise
# "Found input variables with inconsistent numbers of samples" before printing
# any results.
baseline_mae = mean_absolute_error(test_df["total_units"].values[:len(actuals)],
                                    test_df["lag_7"].values[:len(actuals)])

# --- Inference latency benchmark ---
sample = torch.tensor(X_test[:1]).to(device)
n_reps = 200
with torch.no_grad():
    for _ in range(10):  # warmup
        model(sample)
    t0 = time.time()
    for _ in range(n_reps):
        model(sample)
    single_latency_ms = (time.time() - t0) / n_reps * 1000

batch = torch.tensor(X_test[:256]).to(device)
with torch.no_grad():
    t0 = time.time()
    for _ in range(20):
        model(batch)
    batch_time = (time.time() - t0) / 20
throughput = 256 / batch_time

torch.save(model.state_dict(), "model.pt")
import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

print("\n--- Model results (held-out chronological test set, last 60 days/store) ---")
print(f"Test samples:              {len(actuals):,}")
print(f"MAE:                       {mae:.2f} units/day")
print(f"RMSE:                      {rmse:.2f} units/day")
print(f"MAPE:                      {mape:.2f}%")
print(f"Naive 7-day-lag baseline MAE: {baseline_mae:.2f} units/day")
print(f"Improvement over baseline: {(1 - mae/baseline_mae)*100:.1f}%")
print(f"Training time ({EPOCHS} epochs): {train_time:.2f}s on {device}")
print(f"Single-sample inference latency: {single_latency_ms:.3f} ms")
print(f"Batch throughput (batch=256):     {throughput:.0f} samples/sec")
