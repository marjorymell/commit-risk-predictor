"""Train and evaluate the commit risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from commit_risk.export import export_logistic, write_json
from commit_risk.features import TARGET, to_model_input

# A bug usually surfaces months after it is introduced, so the newest commits
# still look "clean" only because their fixes have not happened yet.
LABEL_MATURITY_DAYS = 180
TEST_FRACTION = 0.2
RANDOM_STATE = 42


def drop_immature_labels(df: pd.DataFrame, days: int = LABEL_MATURITY_DAYS) -> pd.DataFrame:
    """Remove commits from the last `days` of history, whose labels are not settled yet."""
    cutoff = pd.to_datetime(df["date"]).max() - pd.Timedelta(days=days)
    return df[pd.to_datetime(df["date"]) < cutoff]


def temporal_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Oldest commits train, newest commits test: the model must predict the future."""
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def build_baseline() -> Pipeline:
    return make_pipeline(
        FunctionTransformer(np.log1p),  # line counts span 1..10k; log keeps them comparable
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def build_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def bugs_caught(y_true: pd.Series, risk: np.ndarray, review_fraction: float) -> float:
    """Share of bug-introducing commits found by reviewing only the riskiest ones."""
    top = np.argsort(-risk)[: int(len(risk) * review_fraction)]
    return float(y_true.iloc[top].sum() / y_true.sum())


def plot_effort_curve(y_test: pd.Series, risks: dict[str, np.ndarray], path: Path) -> None:
    fractions = np.linspace(0.05, 1, 20)
    plt.figure(figsize=(6, 4.5))
    for (name, risk), style in zip(risks.items(), ["-", "--"], strict=True):
        caught = [bugs_caught(y_test, risk, f) * 100 for f in fractions]
        plt.plot(fractions * 100, caught, ls=style, lw=2, label=name)
    plt.plot([0, 100], [0, 100], ls=":", color="gray", label="Random")
    plt.xlabel("% of commits reviewed (riskiest first)")
    plt.ylabel("% of bug-introducing commits caught")
    plt.title("Effort curve on the test period")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def feature_importance(model: ClassifierMixin, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """AUC drop when each feature is shuffled (model-agnostic, measured on unseen data)."""
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def plot_feature_importance(importance: pd.Series, path: Path, top: int = 10) -> None:
    importance.head(top).iloc[::-1].plot(kind="barh", figsize=(6, 4.5), color="#4f81bd")
    plt.title("Top features (AUC drop when shuffled)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def pick_examples(
    commits: pd.DataFrame, risk: np.ndarray, quantiles=(0.15, 0.5, 0.9)
) -> pd.DataFrame:
    """Real commits at low, medium and high predicted risk, for the demo."""
    commits = commits.assign(_risk=risk).reset_index(drop=True)
    readable = commits[commits["subject"].str.len() <= 80]
    picks = [(readable["_risk"] - readable["_risk"].quantile(q)).abs().idxmin() for q in quantiles]
    return commits.loc[picks].drop(columns="_risk")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV produced by commit-risk-extract")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--web-model",
        type=Path,
        default=Path("space/model.json"),
        help="where to export the logistic regression for the browser demo",
    )
    args = parser.parse_args(argv)
    args.models_dir.mkdir(exist_ok=True)
    args.results_dir.mkdir(exist_ok=True)

    df = drop_immature_labels(pd.read_csv(args.dataset))
    train, test = temporal_split(df)
    X_train, y_train = to_model_input(train), train[TARGET]
    X_test, y_test = to_model_input(test, list(X_train.columns)), test[TARGET]

    baseline, forest = build_baseline(), build_forest()
    baseline.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    risk_baseline = baseline.predict_proba(X_test)[:, 1]
    risk_forest = forest.predict_proba(X_test)[:, 1]
    importance = feature_importance(forest, X_test, y_test)

    metrics = {
        "n_commits": len(df),
        "bug_rate": round(float(df[TARGET].mean()), 3),
        "train_period": [train["date"].iloc[0], train["date"].iloc[-1]],
        "test_period": [test["date"].iloc[0], test["date"].iloc[-1]],
        "roc_auc_logistic_regression": round(roc_auc_score(y_test, risk_baseline), 3),
        "roc_auc_random_forest": round(roc_auc_score(y_test, risk_forest), 3),
        "bugs_caught_reviewing_top_20pct": round(bugs_caught(y_test, risk_forest, 0.2), 3),
        "bugs_caught_reviewing_top_30pct": round(bugs_caught(y_test, risk_forest, 0.3), 3),
        "logistic_bugs_caught_reviewing_top_30pct": round(
            bugs_caught(y_test, risk_baseline, 0.3), 3
        ),
        "top_features": importance.head(5).round(4).to_dict(),
    }
    print(json.dumps(metrics, indent=2))

    plot_effort_curve(
        y_test,
        {"Random Forest": risk_forest, "Logistic Regression": risk_baseline},
        args.results_dir / "effort_curve.png",
    )
    plot_feature_importance(importance, args.results_dir / "feature_importance.png")
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Evaluation is done; the deployed model learns from every labeled commit,
    # including the most recent ones, which resemble future commits the most.
    X_all = to_model_input(df)
    forest.fit(X_all, df[TARGET])
    joblib.dump({"model": forest, "columns": list(X_all.columns)}, args.models_dir / "model.joblib")
    print(f"Saved {args.models_dir / 'model.joblib'} and results in {args.results_dir}/")

    examples = pick_examples(test, risk_baseline)  # out-of-sample risk, before refitting
    baseline.fit(X_all, df[TARGET])
    web_model = export_logistic(baseline, list(X_all.columns), df, examples, metrics)
    write_json(web_model, args.web_model)
    print(f"Exported browser model to {args.web_model}")


if __name__ == "__main__":
    main()
