# DDoS traffic checker

One-page demo for the graduation project "AI-Based Detection of Cloud DDoS Attacks Using IoT Datasets".
Enter the 22 traffic features and the site answers Benign or DDoS in three steps:

1. **Lookup first**: the record is searched in a table of the CICIoT2023 training split
   (1,312,987 records) and in the answers the model gave earlier.
2. **ML fallback**: a record that is not found is classified by XGBoost, and the answer is remembered.
3. **Live frequency**: every check updates how often the record was seen and the live counters.

"Load a real row" fills the form with a real row from CICIoT2023 or CICIoMT2024 and compares
the prediction with the dataset's true answer.

## Run locally

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. `python test_engine.py` checks the three steps (it also needs pandas and scikit-learn).

## Hosting (Render)

Build command: `pip install -r requirements.txt`
Start command: `gunicorn app:app --threads 4`

The live counters are stored in `live_stats.sqlite3` next to the app, so on free hosting they
start from zero whenever the service restarts.
