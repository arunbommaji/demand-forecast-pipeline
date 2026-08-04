import csv, json

FEATURES = ["total_units", "any_promo", "dow", "month", "lag_1", "lag_7", "roll_mean_7", "roll_mean_28"]
STORE_ID = "0"

rows = []
with open("data/features.csv", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["store_id"] == STORE_ID:
            rows.append(row)

rows.sort(key=lambda r: r["date"])
last14 = rows[-14:]
assert len(last14) == 14, f"expected 14 rows, got {len(last14)}"

history = []
for row in last14:
    rec = {}
    for f_name in FEATURES:
        v = row[f_name]
        rec[f_name] = int(v) if f_name in ("any_promo", "dow", "month") else float(v)
    history.append(rec)

payload = {"history": history}
with open("predict_payload.json", "w") as f:
    json.dump(payload, f, indent=2)

print(f"Wrote predict_payload.json for store_id={STORE_ID}, dates {last14[0]['date']} to {last14[-1]['date']}")
