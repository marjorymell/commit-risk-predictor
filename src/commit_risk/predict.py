"""Estimate how likely a commit is to introduce a bug."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from commit_risk.classify import is_release
from commit_risk.features import build_dataset, to_model_input
from commit_risk.git_history import read_commits, resolve_commit

HIGH_RISK = 0.6
MEDIUM_RISK = 0.35


class ScoringError(Exception):
    """The requested commit cannot be scored."""


def risk_level(risk: float) -> str:
    # Classify the value as displayed (whole %), so "60%" is never shown as MEDIUM.
    shown = round(risk, 2)
    if shown >= HIGH_RISK:
        return "HIGH"
    return "MEDIUM" if shown >= MEDIUM_RISK else "LOW"


def commit_features(repo: str, ref: str | None = None) -> pd.Series:
    """
    Features of `ref` computed exactly as in training.

    History features depend on everything before the commit, so the history is
    replayed up to it. Labels are not needed, so SZZ is skipped.
    Without `ref`, scores the most recent commit that changes code.
    """
    commits = [c for c in read_commits(repo) if not is_release(c["subject"])]

    if ref is not None:
        full_hash = resolve_commit(repo, ref)
        if full_hash is None:
            raise ScoringError(f"Commit '{ref}' not found in {repo}")
        hashes = [c["hash"] for c in commits]
        if full_hash not in hashes:
            raise ScoringError("Merge and release commits are not scored.")
        commits = commits[: hashes.index(full_hash) + 1]

    rows = build_dataset(commits, culprits=())
    if rows.empty or (ref is not None and rows.iloc[-1]["hash"] != full_hash):
        raise ScoringError("That commit changes no code files (only docs/config).")
    return rows.iloc[-1]


def predict_risk(features: pd.Series, model_path: Path) -> float:
    bundle = joblib.load(model_path)
    X = to_model_input(features.to_frame().T.infer_objects(), bundle["columns"])
    return float(bundle["model"].predict_proba(X)[0, 1])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="path to a local git repository")
    parser.add_argument("commit", nargs="?", help="hash or ref (default: latest code commit)")
    parser.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    args = parser.parse_args(argv)

    if not args.model.exists():
        parser.exit(1, f"Model not found at {args.model}. Run commit-risk-train first.\n")
    try:
        f = commit_features(args.repo, args.commit)
    except ScoringError as error:
        parser.exit(1, f"{error}\n")
    risk = predict_risk(f, args.model)

    print(f"\nCommit:  {f['hash'][:10]}  {f['subject'][:70]}")
    print(f"Risk of introducing a bug: {risk:.0%}  [{risk_level(risk)}]\n")
    print(
        f"  Change:   +{f['lines_added']} / -{f['lines_deleted']} lines in {f['n_files']} files "
        f"({f['n_test_files']} test files), {f['n_subsystems']} subsystem(s)"
    )
    print(
        f"  History:  touched files had {f['prior_changes']} previous changes "
        f"and {f['prior_fixes']} previous fixes"
    )
    print(f"  Author:   {f['author_experience']} previous commits in this repo\n")


if __name__ == "__main__":
    main()
