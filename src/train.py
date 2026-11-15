"""
Train and evaluate the commit risk model.

Usage:
    python src/train.py data/faststore_commits.csv

Outputs:
    models/model.joblib            trained model + metadata (used by predict.py and the app)
    results/metrics.json           evaluation numbers
    results/effort_curve.png       "% of bugs caught vs % of commits reviewed"
    results/feature_importance.png which features the model relies on
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # render charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, to_model_input  # noqa: E402

# Commits newer than this have not had time to "reveal" their bugs yet
# (the fix may not exist yet), so their labels are not trustworthy.
LABEL_CUTOFF = "2026-05-01"
TEST_FRACTION = 0.2
RANDOM_STATE = 42


def bugs_caught(y_true, risk, fraction):
    """Share of buggy commits found when reviewing only the top `fraction` riskiest commits."""
    top = np.argsort(-risk)[: int(len(risk) * fraction)]
    return float(y_true.iloc[top].sum() / y_true.sum())


def build_models():
    # Baseline: simple, fast, explainable. Any fancier model must beat it.
    baseline = make_pipeline(
        FunctionTransformer(np.log1p),   # compress large counts (1 vs 5000 lines)
        StandardScaler(),                # put every feature on the same scale
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    forest = RandomForestClassifier(
        n_estimators=500,          # number of trees voting
        min_samples_leaf=5,        # stops trees from memorising single commits
        class_weight="balanced",   # buggy commits are the minority class
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return baseline, forest


def main(csv_path):
    results_dir, models_dir = Path("results"), Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    df = df[df["date"] < LABEL_CUTOFF].reset_index(drop=True)
    X, y = to_model_input(df), df[TARGET]

    # Time-based split: train on the past, test on the future (never shuffle!)
    cut = int(len(df) * (1 - TEST_FRACTION))
    X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    print(f"Train: {cut} commits ({df.date.iloc[0]} to {df.date.iloc[cut - 1]})")
    print(f"Test:  {len(df) - cut} commits ({df.date.iloc[cut]} to {df.date.iloc[-1]})\n")

    baseline, forest = build_models()
    baseline.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    risk_baseline = baseline.predict_proba(X_test)[:, 1]
    risk_forest = forest.predict_proba(X_test)[:, 1]

    metrics = {
        "n_commits": len(df),
        "bug_rate": round(float(y.mean()), 3),
        "train_period": [df.date.iloc[0], df.date.iloc[cut - 1]],
        "test_period": [df.date.iloc[cut], df.date.iloc[-1]],
        "roc_auc_logistic_regression": round(roc_auc_score(y_test, risk_baseline), 3),
        "roc_auc_random_forest": round(roc_auc_score(y_test, risk_forest), 3),
        "bugs_caught_reviewing_top_20pct": round(bugs_caught(y_test, risk_forest, 0.2), 3),
        "bugs_caught_reviewing_top_30pct": round(bugs_caught(y_test, risk_forest, 0.3), 3),
    }
    print(json.dumps(metrics, indent=2))
    print("\nRandom Forest at threshold 0.5:")
    print(classification_report(y_test, (risk_forest >= 0.5).astype(int),
                                target_names=["no bug", "bug"]))

    # --- Chart 1: effort curve ------------------------------------------------
    fractions = np.linspace(0.05, 1, 20)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fractions * 100, [bugs_caught(y_test, risk_forest, f) * 100 for f in fractions],
             lw=2, label="Random Forest")
    plt.plot(fractions * 100, [bugs_caught(y_test, risk_baseline, f) * 100 for f in fractions],
             ls="--", label="Logistic Regression")
    plt.plot([0, 100], [0, 100], ls=":", color="gray", label="Random")
    plt.xlabel("% of commits reviewed (riskiest first)")
    plt.ylabel("% of bug-introducing commits caught")
    plt.title("Effort curve on the test period")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(results_dir / "effort_curve.png", dpi=150); plt.close()

    # --- Chart 2: permutation importance -------------------------------------
    # Shuffle one feature at a time and measure how much the AUC drops.
    imp = permutation_importance(forest, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values()
    importance.tail(10).plot(kind="barh", figsize=(6, 4.5), color="#4f81bd")
    plt.title("Top features (AUC drop when shuffled)"); plt.tight_layout()
    plt.savefig(results_dir / "feature_importance.png", dpi=150); plt.close()
    metrics["top_features"] = importance.sort_values(ascending=False).head(5).round(4).to_dict()

    # --- Final model: retrain on ALL trustworthy data -------------------------
    forest.fit(X, y)
    joblib.dump({"model": forest, "columns": list(X.columns)}, models_dir / "model.joblib")
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved models/model.joblib, results/metrics.json and charts in results/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
