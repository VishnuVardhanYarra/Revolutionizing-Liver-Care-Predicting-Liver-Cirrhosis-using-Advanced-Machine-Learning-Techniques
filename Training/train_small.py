# train_small_model.py  (7-feature, SMOTE-balanced, isotonic-calibrated RF)
import sys, pickle, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# ── CONFIG ──────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]        # …/Liver_app
CSV_PATH    = PROJECT_DIR / "Data" / "liver_data.csv"
TARGET_COL  = "Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"

USE_COLS = [
    "Age",
    "Total Bilirubin    (mg/dl)",
    "Direct    (mg/dl)",
    "AL.Phosphatase      (U/L)",
    "Albumin   (g/dl)",
    "SGOT/AST      (U/L)",
    "SGPT/ALT (U/L)",
]
# ────────────────────────────────────────────────────────────

# 1 ▸ Load & encode labels
df = pd.read_csv(CSV_PATH)
labels = df[TARGET_COL].astype(str).str.strip().str.lower()
df["target"] = labels.map({"yes": 1, "no": 0})
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

# 2 ▸ Select features & coerce numeric
missing = [c for c in USE_COLS if c not in df.columns]
if missing:
    sys.exit(f"Missing columns: {missing}")

features = df[USE_COLS].apply(pd.to_numeric, errors="coerce")
data = pd.concat([features, df["target"]], axis=1).dropna()

X, y = data[USE_COLS], data["target"]

# 3 ▸ Train-test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4 ▸ Scale
scaler = MinMaxScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# 5 ▸ Balance with SMOTE
X_tr_bal, y_tr_bal = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)
print("After SMOTE:", y_tr_bal.value_counts().to_dict())

# 6 ▸ Train RF + isotonic calibration
rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced")
rf.fit(X_tr_bal, y_tr_bal)

model = CalibratedClassifierCV(rf, cv=5, method="isotonic")
model.fit(X_tr_bal, y_tr_bal)

# 7 ▸ Evaluate
print("\nReport:\n",
      classification_report(y_te, model.predict(X_te_s)))
print("ROC-AUC:",
      roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1]))

# 8 ▸ Save artifacts next to app.py
pickle.dump(model,  open(PROJECT_DIR / "rf_small.pkl", "wb"))
pickle.dump(scaler, open(PROJECT_DIR / "scaler_small.pkl", "wb"))
print("\n✅ Saved rf_small.pkl & scaler_small.pkl to project root")
