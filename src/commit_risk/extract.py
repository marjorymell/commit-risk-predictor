"""Build the labeled commit dataset from a git repository."""

from __future__ import annotations

import argparse

from commit_risk.classify import is_fix, is_release
from commit_risk.features import TARGET, build_dataset
from commit_risk.git_history import read_commits
from commit_risk.szz import find_bug_introducing_commits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="path to a local git repository")
    parser.add_argument("output", help="CSV file to write, e.g. data/faststore_commits.csv")
    args = parser.parse_args(argv)

    print("Reading history...")
    commits = [c for c in read_commits(args.repo) if not is_release(c["subject"])]
    fixes = [c for c in commits if is_fix(c["subject"])]
    print(f"  {len(commits)} commits, {len(fixes)} fixes (merges and releases excluded)")

    print("Running SZZ (a few minutes on large repositories)...")
    culprits = find_bug_introducing_commits(args.repo, fixes)

    df = build_dataset(commits, culprits)
    df.to_csv(args.output, index=False)
    print(f"Saved {args.output}: {len(df)} commits, {df[TARGET].mean():.1%} introduced a bug")


if __name__ == "__main__":
    main()
