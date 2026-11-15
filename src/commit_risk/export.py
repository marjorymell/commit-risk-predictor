"""
Export the logistic regression baseline as JSON so it can run in a browser.

The pipeline is log1p -> standardisation -> logistic regression, so a prediction is:

    z_i  = (log1p(x_i) - mean_i) / scale_i
    risk = sigmoid(intercept + sum(coef_i * z_i))

Every number needed for that formula goes into the JSON; `space/risk.js`
implements the same formula in JavaScript.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from commit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, to_model_input

FORMAT_VERSION = 1


def export_logistic(
    pipeline: Pipeline,
    columns: list[str],
    dataset: pd.DataFrame,
    examples: pd.DataFrame,
    metrics: dict,
) -> dict:
    scaler = pipeline.named_steps["standardscaler"]
    logistic = pipeline.named_steps["logisticregression"]
    risks = pipeline.predict_proba(to_model_input(dataset, columns))[:, 1]
    return {
        "format_version": FORMAT_VERSION,
        "columns": columns,
        "mean": scaler.mean_.round(8).tolist(),
        "scale": scaler.scale_.round(8).tolist(),
        "coef": logistic.coef_[0].round(8).tolist(),
        "intercept": round(float(logistic.intercept_[0]), 8),
        "defaults": dataset[NUMERIC_FEATURES].median().round(2).to_dict(),
        "commit_types": sorted(
            c.removeprefix("commit_type_") for c in columns if c.startswith("commit_type_")
        ),
        # Percentiles of the historical risk, to say "riskier than X% of commits".
        "risk_percentiles": np.percentile(risks, range(101)).round(6).tolist(),
        "examples": [
            {"subject": row["subject"], "features": _features_of(row)}
            for _, row in examples.iterrows()
        ],
        "metrics": metrics,
    }


def write_json(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=1) + "\n")


def _features_of(row: pd.Series) -> dict:
    values = {name: float(row[name]) for name in NUMERIC_FEATURES}
    return values | {name: row[name] for name in CATEGORICAL_FEATURES}
