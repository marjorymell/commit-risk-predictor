import pandas as pd
import pytest

from commit_risk.features import NUMERIC_FEATURES, build_dataset, entropy, to_model_input
from commit_risk.git_history import read_commits


def test_entropy_bounds():
    assert entropy([100]) == 0.0
    assert entropy([10, 10, 10, 10]) == pytest.approx(1.0)
    assert 0 < entropy([90, 5, 5]) < 1
    assert entropy([0, 0]) == 0.0


def test_model_input_aligns_unseen_commit_types():
    row = {f: 1 for f in NUMERIC_FEATURES} | {"commit_type": "brand-new-type"}
    training_columns = NUMERIC_FEATURES + ["commit_type_feat", "commit_type_fix"]

    X = to_model_input(pd.DataFrame([row]), training_columns)

    assert list(X.columns) == training_columns
    assert X[["commit_type_feat", "commit_type_fix"]].sum().sum() == 0


def test_commits_without_code_are_skipped(repo):
    df = build_dataset(read_commits(repo), culprits=())

    assert "docs: add readme" not in set(df["subject"])
    assert len(df) == 3


def test_features_never_look_into_the_future(repo):
    """The first commit has no past; the fix must not count itself."""
    df = build_dataset(read_commits(repo), culprits=())
    first, fix = df.iloc[0], df.iloc[2]

    assert first["prior_changes"] == 0
    assert first["prior_fixes"] == 0
    assert first["author_experience"] == 0
    assert first["new_files"] == 1

    assert fix["prior_changes"] == 1
    assert fix["prior_fixes"] == 0
    assert fix["author_experience"] == 2
    assert fix["new_files"] == 0
