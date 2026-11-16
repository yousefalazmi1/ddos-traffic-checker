"""
Detection engine:  1) lookup first  ->  2) ML fallback  ->  3) live frequency update.

1. Lookup   : the 22-feature vector is hashed and searched in the lookup table built from
              the CICIoT2023 training split. A hit returns the stored label right away.
              Answers the model gave earlier (the "learned" table) are searched next.
2. ML       : on a miss, XGBoost classifies the vector and the answer is stored, so the
              same vector is a lookup hit next time.
3. Frequency: every check updates how often that vector was seen and the live counters.
"""
import hashlib
import json
import random
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
DB_PATH = BASE / "live_stats.sqlite3"

FEATURES = [
    "Protocol Type",
    "fin_flag_number", "syn_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "HTTP", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP",
    "Min", "Max", "AVG",
]
LABELS = {0: "Benign", 1: "DDoS"}


def vector_key(row):
    """Same hash that built lookup_table.npz: rounded float64 bytes -> 64-bit blake2b."""
    v = np.round(np.asarray(row, dtype="<f8"), 6) + 0.0
    return int.from_bytes(hashlib.blake2b(v.tobytes(), digest_size=8).digest(), "little")


class Engine:
    def __init__(self):
        table = np.load(MODEL_DIR / "lookup_table.npz")
        self.keys, self.benign_n, self.ddos_n = table["keys"], table["benign_n"], table["ddos_n"]
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_DIR / "xgboost.json"))
        # Real rows from both datasets, with their true labels (built by the notebook export)
        self.samples = json.loads((MODEL_DIR / "dataset_samples.json").read_text())["datasets"]
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS learned (key TEXT PRIMARY KEY, label INTEGER, confidence REAL);
        CREATE TABLE IF NOT EXISTS freq (
            key TEXT PRIMARY KEY, hits INTEGER, label INTEGER, vector TEXT, last_seen REAL);
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, route TEXT, label INTEGER,
            confidence REAL);
        """)
        self.db.commit()

    def sample(self, dataset, split):
        """Random real row: Benign or DDoS with equal chance, then a random traffic type."""
        types = self.samples[dataset]
        label = random.randint(0, 1)
        name = random.choice([t for t, v in types.items() if v["label"] == label])
        row = random.choice(types[name]["samples"][split])
        return {"features": dict(zip(FEATURES, row)), "label_name": LABELS[label],
                "type": name, "dataset": dataset, "split": split}

    def _lookup(self, key):
        i = int(np.searchsorted(self.keys, np.uint64(key)))
        if i < len(self.keys) and int(self.keys[i]) == key:
            b, d = int(self.benign_n[i]), int(self.ddos_n[i])
            return {"route": "lookup", "label": 1 if d >= b else 0,
                    "confidence": max(b, d) / (b + d), "benign_n": b, "ddos_n": d}
        return None

    def classify(self, vector):
        x = np.asarray(vector, dtype="float64")
        key = vector_key(x)
        ks = str(key)
        t0 = time.perf_counter()
        with self.lock:
            # 1) lookup first
            res = self._lookup(key)
            if res is None:
                row = self.db.execute("SELECT label, confidence FROM learned WHERE key=?",
                                      (ks,)).fetchone()
                if row:
                    res = {"route": "learned", "label": row[0], "confidence": row[1]}
            # 2) ML fallback
            if res is None:
                p = float(self.model.inplace_predict(x.reshape(1, -1))[0])
                label = int(p >= 0.5)
                res = {"route": "ml", "label": label, "p_ddos": p,
                       "confidence": p if label else 1 - p}
                self.db.execute("INSERT OR IGNORE INTO learned VALUES (?,?,?)",
                                (ks, label, res["confidence"]))
            res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            # 3) live frequency update
            now = time.time()
            self.db.execute("""INSERT INTO freq VALUES (?,1,?,?,?)
                               ON CONFLICT(key) DO UPDATE SET hits = hits + 1,
                               label = excluded.label, last_seen = excluded.last_seen""",
                            (ks, res["label"], json.dumps([float(v) for v in x]), now))
            res["frequency"] = self.db.execute("SELECT hits FROM freq WHERE key=?",
                                               (ks,)).fetchone()[0]
            self.db.execute("INSERT INTO events (ts, route, label, confidence) VALUES (?,?,?,?)",
                            (now, res["route"], res["label"], res["confidence"]))
            self.db.commit()
        res["label_name"] = LABELS[res["label"]]
        return res

    def stats(self):
        with self.lock:
            q = lambda s: self.db.execute(s).fetchall()  # noqa: E731
            routes = dict(q("SELECT route, COUNT(*) FROM events GROUP BY route"))
            labels = dict(q("SELECT label, COUNT(*) FROM events GROUP BY label"))
            top = q("SELECT hits, label, vector FROM freq ORDER BY hits DESC, last_seen DESC LIMIT 6")
            recent = q("SELECT ts, route, label, confidence FROM events ORDER BY id DESC LIMIT 8")
            distinct = q("SELECT COUNT(*) FROM freq")[0][0]
        return {
            "total": sum(routes.values()),
            "distinct": distinct,
            "known_records": int(len(self.keys)),
            "routes": {r: routes.get(r, 0) for r in ("lookup", "learned", "ml")},
            "labels": {"Benign": labels.get(0, 0), "DDoS": labels.get(1, 0)},
            "top": [{"hits": h, "label": LABELS[lab], "vector": dict(zip(FEATURES, json.loads(v)))}
                    for h, lab, v in top],
            "recent": [{"ts": ts, "route": rt, "label": LABELS[lab], "confidence": c}
                       for ts, rt, lab, c in recent],
        }

    def reset(self):
        with self.lock:
            self.db.executescript("DELETE FROM events; DELETE FROM freq; DELETE FROM learned;")
            self.db.commit()
