"""DDoS detector website.  Run:  python app.py  ->  http://127.0.0.1:5000"""
import math

from flask import Flask, jsonify, request

from engine import FEATURES, Engine

app = Flask(__name__, static_folder="static", static_url_path="")
engine = Engine()


@app.get("/")
def home():
    return app.send_static_file("index.html")


@app.post("/api/classify")
def api_classify():
    data = (request.get_json(silent=True) or {}).get("features", {})
    missing = [f for f in FEATURES if data.get(f) in ("", None)]
    if missing:
        return jsonify(error="Missing value(s): " + ", ".join(missing)), 400
    try:
        vector = [float(data[f]) for f in FEATURES]
    except (TypeError, ValueError):
        return jsonify(error="Every value must be a number."), 400
    if not all(math.isfinite(v) and v >= 0 for v in vector):
        return jsonify(error="Values must be zero or positive numbers."), 400
    return jsonify(engine.classify(vector))


@app.get("/api/sample")
def api_sample():
    try:
        return jsonify(engine.sample(request.args["dataset"], request.args["split"]))
    except KeyError:
        return jsonify(error="Unknown dataset or split."), 404


@app.get("/api/stats")
def api_stats():
    return jsonify(engine.stats())


@app.post("/api/reset")
def api_reset():
    engine.reset()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(port=5000, threaded=True)
