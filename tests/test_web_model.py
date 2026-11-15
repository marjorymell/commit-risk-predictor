"""The browser demo must give exactly the same risk as the Python model."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from commit_risk.export import export_logistic
from commit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, to_model_input
from commit_risk.train import build_baseline

ROOT = Path(__file__).resolve().parents[1]
RISK_JS = ROOT / "space" / "risk.js"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    df = pd.read_csv(ROOT / "data" / "faststore_commits.csv")
    X = to_model_input(df)
    pipeline = build_baseline().fit(X, df[TARGET])
    model = export_logistic(pipeline, list(X.columns), df, df.head(3), metrics={})

    sample = df.sample(200, random_state=0)
    expected = pipeline.predict_proba(to_model_input(sample, list(X.columns)))[:, 1]
    features = sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES].to_dict(orient="records")
    return model, features, expected


def _formula(model, features):
    """The JSON formula, written in Python: what risk.js does."""
    x = np.array(
        [
            float(features["commit_type"] == c.removeprefix("commit_type_"))
            if c.startswith("commit_type_")
            else float(features[c])
            for c in model["columns"]
        ]
    )
    z = (np.log1p(x) - np.array(model["mean"])) / np.array(model["scale"])
    return 1 / (1 + np.exp(-(model["intercept"] + np.dot(model["coef"], z))))


def test_exported_formula_matches_scikit_learn(exported):
    model, features, expected = exported

    actual = [_formula(model, f) for f in features]

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_risk_js_matches_scikit_learn(exported, tmp_path):
    model, features, expected = exported
    (tmp_path / "input.json").write_text(json.dumps({"model": model, "features": features}))
    script = tmp_path / "run.mjs"
    script.write_text(
        f"import {{ predictRisk }} from {json.dumps(RISK_JS.as_uri())};\n"
        "import { readFileSync } from 'node:fs';\n"
        "const { model, features } = JSON.parse(readFileSync(process.argv[2], 'utf8'));\n"
        "console.log(JSON.stringify(features.map((f) => predictRisk(model, f))));\n"
    )

    output = subprocess.run(
        ["node", str(script), str(tmp_path / "input.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    np.testing.assert_allclose(json.loads(output), expected, atol=1e-6)


def test_committed_web_model_is_well_formed():
    model = json.loads((ROOT / "space" / "model.json").read_text())

    n = len(model["columns"])
    assert len(model["mean"]) == len(model["scale"]) == len(model["coef"]) == n
    assert len(model["risk_percentiles"]) == 101
    assert set(NUMERIC_FEATURES) <= set(model["defaults"])
    assert model["examples"], "the demo needs example commits"
