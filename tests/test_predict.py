import pytest

from commit_risk.git_history import read_commits
from commit_risk.predict import ScoringError, commit_features, risk_level


@pytest.mark.parametrize(
    "risk, level",
    [
        (0.10, "LOW"),
        (0.344, "LOW"),
        (0.349, "MEDIUM"),  # displayed as 35%
        (0.35, "MEDIUM"),
        (0.598, "HIGH"),  # displayed as 60%
        (0.95, "HIGH"),
    ],
)
def test_risk_level_matches_displayed_percentage(risk, level):
    assert risk_level(risk) == level


def test_default_scores_latest_code_commit(repo):
    features = commit_features(repo)

    assert features["subject"] == "fix: cart total was subtracting"


def test_features_of_a_specific_commit(repo):
    first = read_commits(repo)[0]["hash"]

    features = commit_features(repo, first[:7])

    assert features["hash"] == first
    assert features["prior_changes"] == 0


def test_unknown_commit_is_rejected(repo):
    with pytest.raises(ScoringError, match="not found"):
        commit_features(repo, "does-not-exist")


def test_docs_only_commit_is_rejected(repo):
    docs = read_commits(repo)[-1]["hash"]

    with pytest.raises(ScoringError, match="no code"):
        commit_features(repo, docs)
