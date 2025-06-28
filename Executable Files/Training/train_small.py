# train_small_model.py  (with screenshot helpers)
import sys, pickle, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ============ CONFIG =========================================
PROJECT_DIR = Path(__file__).resolve().parents[1]       # …/Liver_app
CSV_PATH = PROJECT_DIR / "Data" / "liver_data.csv"
TARGET_COL = "Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"

USE_COLS = [
    "Age",
    "Total Bilirubin    (mg/dl)",
    "Direct    (mg/dl)",
    "AL.Phosphatase      (U/L)",
    "Albumin   (g/dl)",
    "SGOT/AST      (U/L)",
    "SGPT/ALT (U/L)",
]

# Folder for screenshots
SS_DIR = PROJECT_DIR / "assets" / "screenshots"
SS_DIR.mkdir(parents=True, exist_ok=True)
# =============================================================

# 1 ▸ Load & encode
df = pd.read_csv(CSV_PATH)
labels = df[TARGET_COL].astype(str).str.strip().str.lower()
df["target"] = labels.map({"yes": 1, "no": 0})
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

# 2 ▸ Feature selection
missing = [c for c in USE_COLS if c not in df.columns]
if missing:
    sys.exit(f"Missing columns: {missing}")
data = pd.concat(
    [df[USE_COLS].apply(pd.to_numeric, errors="coerce"), df["target"]], axis=1
).dropna()

X, y = data[USE_COLS], data["target"]

# 3 ▸ Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4 ▸ Scale
scaler = MinMaxScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# 5 ▸ SMOTE
X_tr_bal, y_tr_bal = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)
print("After SMOTE:", y_tr_bal.value_counts().to_dict())

# 6 ▸ Train calibrated RF
rf = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced"
)
rf.fit(X_tr_bal, y_tr_bal)
model = CalibratedClassifierCV(rf, cv=5, method="isotonic")
model.fit(X_tr_bal, y_tr_bal)

# 7 ▸ Evaluate & screenshots -------------------------------------------------
y_pred = model.predict(X_te_s)
y_prob = model.predict_proba(X_te_s)[:, 1]

print("\nReport:\n", classification_report(y_te, y_pred))
print("ROC-AUC:", roc_auc_score(y_te, y_prob))

# 7a Confusion-matrix PNG
disp = ConfusionMatrixDisplay.from_predictions(
    y_te, y_pred, cmap="Blues", xticks_rotation=45
)
plt.title("Confusion Matrix")
plt.tight_layout()
cm_path = SS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"📸 Saved {cm_path}")

# 7b Classification-report PNG (as heatmap)
report_dict = classification_report(y_te, y_pred, output_dict=True)
rep_df = pd.DataFrame(report_dict).iloc[:3, :].T  # precision/recall/f1 rows
plt.figure(figsize=(6, 2))
sns.heatmap(rep_df, annot=True, fmt=".2f", cmap="PuBu")
plt.title("Classification Report")
plt.tight_layout()
cr_path = SS_DIR / "classification_report.png"
plt.savefig(cr_path, dpi=120)
plt.close()
print(f"📸 Saved {cr_path}")


# ──────────────────────────────────────────────────────────────
# ✨ OPTIONAL REGRESSION MODULE  (Albumin as example target)
#   1) Pick a numeric column to predict
#   2) Train RandomForestRegressor
#   3) Save regression metrics screenshots
# ----------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from regression_perf import regression_performance  # helper file

# Choose a regression target (example: Albumin)
REG_TARGET = "Albumin   (g/dl)"

if REG_TARGET in df.columns:
    # Prepare data (reuse same USE_COLS but exclude the target)
    reg_features = df[USE_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    y_reg = df.loc[reg_features.index, REG_TARGET].astype(float)
    
    # Train-test split
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        reg_features, y_reg, test_size=0.2, random_state=42
    )
    
    # (Optional) scale numeric features
    reg_scaler = MinMaxScaler()
    Xr_tr_s = reg_scaler.fit_transform(Xr_tr)
    Xr_te_s = reg_scaler.transform(Xr_te)
    
    # Train regressor
    rfr = RandomForestRegressor(n_estimators=300, random_state=42)
    rfr.fit(Xr_tr_s, yr_tr)

    # Predictions
    yr_pred = rfr.predict(Xr_te_s)

    # Save screenshots & print scores
    reg_scores = regression_performance(
        yr_te, yr_pred, plot_dir=str(SS_DIR), tag="rf_reg"
    )
    print("Regression scores:", reg_scores)
    
    # Persist regressor & scaler if you want to reuse later
    pickle.dump(rfr,        open(PROJECT_DIR / "rf_reg.pkl", "wb"))
    pickle.dump(reg_scaler, open(PROJECT_DIR / "scaler_reg.pkl", "wb"))
    print("✅ Saved rf_reg.pkl & scaler_reg.pkl")
else:
    print(f"[Skip Regression] Column '{REG_TARGET}' not found in dataset.")


# ---------------------------------------------------------------------------

# OPTIONAL 8 ▸ Hyperparameter tuning screenshots (comment-out if not needed)
DO_GRID = False
if DO_GRID:
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 20, None],
    }
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced"),
        param_grid,
        cv=5,
        scoring="accuracy",
    )
    grid.fit(X_tr_s, y_tr)
    print("Best params:", grid.best_params_)
    # simple bar chart of mean scores
    scores = grid.cv_results_["mean_test_score"]
    labels = [f"{p['n_estimators']},{p['max_depth']}" for p in grid.cv_results_["params"]]
    plt.figure(figsize=(6,3))
    sns.barplot(x=scores, y=labels)
    plt.xlabel("Mean CV Accuracy")
    plt.title("Grid Search Results")
    plt.tight_layout()
    gs_path = SS_DIR / "grid_search_results.png"
    plt.savefig(gs_path, dpi=120)
    plt.close()
    print(f"📸 Saved {gs_path}")

# 9 ▸ Save model & scaler
pickle.dump(model,  open(PROJECT_DIR / "rf_small.pkl",     "wb"))
pickle.dump(scaler, open(PROJECT_DIR / "scaler_small.pkl", "wb"))
print("\n✅ Saved rf_small.pkl & scaler_small.pkl to project root")

# ───────── OPTIONAL HYPERPARAMETER TUNING + CV PLOTS ──────────
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import numpy as np

DO_TUNE_PLOTS = True
if DO_TUNE_PLOTS:
    print("\n🔧 Running GridSearchCV for hyper-parameter tuning…")

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_tr_s, y_tr)

    print("Best parameters :", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)

    # 1️⃣  Bar-chart of mean CV scores  ─────────────────────────
    scores = grid.cv_results_["mean_test_score"]
    labels = [
        f"{p['n_estimators']}/{p['max_depth']}/{p['min_samples_split']}"
        for p in grid.cv_results_["params"]
    ]

    plt.figure(figsize=(8, 3))
    sns.barplot(x=scores, y=labels, orient="h")
    plt.xlabel("Mean CV Accuracy")
    plt.ylabel("Param combo  (n_estim / depth / min_split)")
    plt.title("Grid Search Results")
    plt.tight_layout()
    gs_path = SS_DIR / "grid_search_results.png"
    plt.savefig(gs_path, dpi=120)
    plt.close()
    print(f"📸 Saved {gs_path}")

    # 2️⃣  Per-fold validation accuracy of BEST estimator ──────
    best_model = grid.best_estimator_
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = cross_val_score(best_model, X_tr_s, y_tr, cv=cv, scoring="accuracy")

    plt.figure(figsize=(4,2.5))
    sns.barplot(x=np.arange(1, 6), y=fold_scores, palette="crest")
    plt.ylim(0, 1)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title(f"5-Fold CV (mean={fold_scores.mean():.3f})")
    plt.tight_layout()
    cv_path = SS_DIR / "cv_fold_accuracy.png"
    plt.savefig(cv_path, dpi=120)
    plt.close()
    print(f"📸 Saved {cv_path}")

