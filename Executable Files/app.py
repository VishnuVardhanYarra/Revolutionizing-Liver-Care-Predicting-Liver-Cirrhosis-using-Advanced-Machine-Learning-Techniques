from flask import Flask, render_template, request
import pickle, numpy as np

app = Flask(__name__)

# ── Load calibrated model + scaler ────────────────────────────
model  = pickle.load(open("rf_small.pkl", "rb"))
scaler = pickle.load(open("scaler_small.pkl", "rb"))

# Inputs MUST arrive in this exact order
INPUT_ORDER = [
    "age",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "albumin",
    "sgot",          # SGOT / AST (U/L)
    "sgpt",          # SGPT / ALT (U/L)
]

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1 ▸ Collect & scale
    features = [float(request.form[f]) for f in INPUT_ORDER]
    features_scaled = scaler.transform([features])

    # 2 ▸ Predict class & calibrated probability
    prob_disease = model.predict_proba(features_scaled)[0][1] * 100
    pred         = 1 if prob_disease >= 50 else 0   # threshold 50 %

    # 3 ▸ Craft message
    outcome = (
        f"Liver Disease Detected (probability {prob_disease:.1f} %)"
        if pred == 1
        else f"No Liver Disease (probability {100 - prob_disease:.1f} %)"
    )

    # 4 ▸ Return page
    return render_template("index.html", prediction_text=outcome)

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
