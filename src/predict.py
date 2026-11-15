"""
Estimate how likely a commit is to introduce a bug.

Usage:
    python src/predict.py <repo_path> [commit]
    (without a commit, scores the most recent commit that changes code)

Examples:
    python src/predict.py ../faststore
    python src/predict.py ../faststore 3f2a1b9

Requires a trained model (run src/train.py first).
"""

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))
from extract_dataset import RELEASE_PATTERN, build_dataset, git, read_commits  # noqa: E402
from features import to_model_input  # noqa: E402

MODEL_PATH = Path("models/model.joblib")


def risk_level(risk):
    risk = round(risk, 2)  # same precision as displayed, avoids "60% [MEDIUM]"
    return "HIGH" if risk >= 0.6 else "MEDIUM" if risk >= 0.35 else "LOW"


def commit_features(repo, commit):
    """
    Rebuild the features of `commit` exactly as in training.

    History features (prior changes, prior fixes, author experience...) depend
    on everything that happened BEFORE the commit, so we replay the history up
    to it. The label is not needed here, so SZZ is skipped (no culprits).
    """
    commits = [c for c in read_commits(repo) if not RELEASE_PATTERN.search(c["subject"])]

    if commit is None:   # default: the most recent commit that touches code
        rows = build_dataset(commits, culprits={})
        return rows.tail(1)

    full_hash = git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}").strip()
    if not full_hash:
        raise SystemExit(f"Commit '{commit}' not found in {repo}")
    hashes = [c["hash"] for c in commits]
    if full_hash not in hashes:
        raise SystemExit("That commit is a merge or release commit, which the model does not score.")

    history = commits[: hashes.index(full_hash) + 1]   # everything up to the commit
    rows = build_dataset(history, culprits={})
    row = rows[rows["hash"] == full_hash]
    if row.empty:
        raise SystemExit("That commit changes no code files (only docs/config), nothing to score.")
    return row


def main(repo, commit=None):
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: python src/train.py data/faststore_commits.csv")
    bundle = joblib.load(MODEL_PATH)

    row = commit_features(repo, commit)
    X = to_model_input(row, bundle["columns"])
    risk = float(bundle["model"].predict_proba(X)[0, 1])

    r = row.iloc[0]
    print(f"\nCommit:  {r['hash'][:10]}  {r['subject'][:70]}")
    print(f"Risk of introducing a bug: {risk:.0%}  [{risk_level(risk)}]\n")
    print(f"  Change:   +{r['lines_added']} / -{r['lines_deleted']} lines in {r['n_files']} files "
          f"({r['n_test_files']} test files), {r['n_subsystems']} subsystem(s)")
    print(f"  History:  touched files had {r['prior_changes']} previous changes "
          f"and {r['prior_fixes']} previous fixes")
    print(f"  Author:   {r['author_experience']} previous commits in this repo\n")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
