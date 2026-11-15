"""Unit tests: small pure functions, no git or files involved."""

import pandas as pd
import pytest

from extract_dataset import entropy, is_code, is_fix, is_test, subsystem
from features import NUMERIC_FEATURES, to_model_input
from predict import risk_level


@pytest.mark.parametrize("subject, expected", [
    ("fix: null price on PDP", True),
    ("fix(core): handle empty cart", True),
    ("hotfix!: rollback checkout", True),
    ("feat: add wishlist", False),
    ("chore: fix typo in docs", False),     # "fix" in the middle is not a fix commit
    ("prefix: something", False),           # must not match "fix" inside a word
])
def test_is_fix_follows_conventional_commits(subject, expected):
    assert is_fix(subject) is expected


@pytest.mark.parametrize("path, expected", [
    ("packages/core/src/Cart.tsx", True),
    ("packages/ui/styles/button.scss", True),
    ("README.md", False),
    ("yarn.lock", False),
    ("packages/core/CHANGELOG.md", False),
    ("packages/api/src/types.d.ts", False),  # generated type declarations
])
def test_is_code(path, expected):
    assert is_code(path) is expected


@pytest.mark.parametrize("path, expected", [
    ("packages/core/src/Cart.test.tsx", True),
    ("packages/core/__tests__/cart.ts", True),
    ("cypress/e2e/checkout.cy.js", True),
    ("packages/core/src/Cart.tsx", False),
])
def test_is_test(path, expected):
    assert is_test(path) is expected


def test_subsystem_groups_monorepo_packages():
    assert subsystem("packages/core/src/a/b.ts") == "packages/core"
    assert subsystem("apps/site/page.tsx") == "apps/site"
    assert subsystem("src/index.ts") == "src"


def test_entropy_bounds():
    assert entropy([100]) == 0.0                      # one file: fully concentrated
    assert entropy([10, 10, 10, 10]) == pytest.approx(1.0)  # evenly spread
    assert 0 < entropy([90, 5, 5]) < 1
    assert entropy([0, 0]) == 0.0                     # no changed lines


@pytest.mark.parametrize("risk, level", [
    (0.10, "LOW"), (0.344, "LOW"), (0.35, "MEDIUM"),
    (0.349, "MEDIUM"),  # displayed as 35%, so it must be MEDIUM, not LOW
    (0.598, "HIGH"),    # displayed as 60%, so it must be HIGH, not MEDIUM
    (0.95, "HIGH"),
])
def test_risk_level_matches_displayed_percentage(risk, level):
    assert risk_level(risk) == level


def test_model_input_aligns_unseen_commit_types():
    """A commit type never seen in training must not break or shift the columns."""
    row = {f: 1 for f in NUMERIC_FEATURES} | {"commit_type": "brand-new-type"}
    training_columns = NUMERIC_FEATURES + ["commit_type_feat", "commit_type_fix"]

    X = to_model_input(pd.DataFrame([row]), training_columns)

    assert list(X.columns) == training_columns
    assert X[["commit_type_feat", "commit_type_fix"]].sum().sum() == 0
