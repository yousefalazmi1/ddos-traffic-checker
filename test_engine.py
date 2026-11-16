"""Checks the three steps: lookup first, ML fallback, live frequency update.
Run:  python test_engine.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import engine as eng

eng.DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else eng.BASE / "test_stats.sqlite3"
eng.DB_PATH.unlink(missing_ok=True)
e = eng.Engine()

# A real CICIoT2023 training row must be a lookup hit with its true label
entry = next(v for v in e.samples["CICIoT2023"].values() if v["samples"].get("train"))
r = e.classify(entry["samples"]["train"][0])
assert r["route"] == "lookup" and r["label"] == entry["label"] and r["frequency"] == 1, r

# An unseen vector goes to the model, then is remembered, and its frequency grows
new = [6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 54.123, 54.123, 54.123]
r1 = e.classify(new)
r2 = e.classify(new)
assert r1["route"] == "ml" and r1["frequency"] == 1, r1
assert r2["route"] == "learned" and r2["label"] == r1["label"] and r2["frequency"] == 2, r2

# The Booster gives the same probability as the notebook's XGBClassifier
clf = XGBClassifier()
clf.load_model(str(eng.MODEL_DIR / "xgboost.json"))
p = clf.predict_proba(pd.DataFrame([new], columns=eng.FEATURES))[0, 1]
assert np.isclose(p, r1["p_ddos"], atol=1e-6), (p, r1["p_ddos"])

s = e.stats()
assert s["total"] == 3, s

# Real rows: every CICIoT2023 training row is a lookup hit with its true label,
# and rows from the other sources are classified without error
for _ in range(100):
    row = e.sample("CICIoT2023", "train")
    r = e.classify(list(row["features"].values()))
    assert r["route"] == "lookup" and r["label_name"] == row["label_name"], (row, r)
for dataset, split in [("CICIoT2023", "test"), ("CICIoMT2024", "test")]:
    hits = [e.classify(list(row["features"].values()))["label_name"] == row["label_name"]
            for row in (e.sample(dataset, split) for _ in range(300))]
    print(f"{dataset} {split}: {sum(hits)}/300 random real rows classified correctly")
e.reset()
e.classify(entry["samples"]["train"][0]); e.classify(new); e.classify(new)
s = e.stats()
assert s["total"] == 3 and s["routes"] == {"lookup": 1, "learned": 1, "ml": 1}, s
assert s["top"][0]["hits"] == 2
e.reset()
assert e.stats()["total"] == 0
e.db.close()
eng.DB_PATH.unlink()
print("All checks passed. P(DDoS) for the unseen vector:", round(float(p), 4))
