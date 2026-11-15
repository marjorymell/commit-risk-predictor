"""Turn commits into feature rows, and feature rows into model input."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from commit_risk.classify import commit_type, is_code, is_fix, is_test, subsystem
from commit_risk.git_history import Commit

NUMERIC_FEATURES = [
    "lines_added", "lines_deleted", "n_files", "n_dirs", "n_subsystems", "entropy",
    "n_test_files", "new_files", "n_prior_devs", "prior_changes", "prior_fixes",
    "avg_file_age_days", "author_experience", "author_subsystem_experience",
    "is_fix", "hour", "weekday",
]  # fmt: skip
CATEGORICAL_FEATURES = ["commit_type"]
TARGET = "introduced_bug"

SECONDS_PER_DAY = 86_400


def entropy(changed_lines_per_file: list[int]) -> float:
    """Normalised Shannon entropy: 0 = change in a single file, 1 = evenly spread."""
    total = sum(changed_lines_per_file)
    if total == 0 or len(changed_lines_per_file) < 2:
        return 0.0
    h = -sum((x / total) * math.log2(x / total) for x in changed_lines_per_file if x > 0)
    return h / math.log2(len(changed_lines_per_file))


@dataclass
class RepoHistory:
    """What the repository looked like up to a point in time."""

    commits_by_author: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    commits_by_author_subsystem: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    last_change: dict[str, int] = field(default_factory=dict)
    devs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    changes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fixes: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def describe(self, commit: Commit, code_files: list[tuple[str, int, int]]) -> dict:
        paths = [path for path, _, _ in code_files]
        subsystems = {subsystem(p) for p in paths}
        directories = {p.rsplit("/", 1)[0] if "/" in p else "." for p in paths}
        # Rebases and cherry-picks can put commit dates out of order.
        ages = [
            max(0, commit["timestamp"] - self.last_change[p]) / SECONDS_PER_DAY
            for p in paths
            if p in self.last_change
        ]
        date = datetime.fromtimestamp(commit["timestamp"], tz=timezone.utc)
        author = commit["author"]

        return {
            "hash": commit["hash"],
            "date": date.strftime("%Y-%m-%d"),
            "subject": commit["subject"],
            "lines_added": sum(added for _, added, _ in code_files),
            "lines_deleted": sum(deleted for _, _, deleted in code_files),
            "n_files": len(paths),
            "n_dirs": len(directories),
            "n_subsystems": len(subsystems),
            "entropy": round(entropy([a + d for _, a, d in code_files]), 4),
            "n_test_files": sum(1 for p in paths if is_test(p)),
            "new_files": sum(1 for p in paths if p not in self.changes),
            "n_prior_devs": len(set().union(*(self.devs[p] for p in paths))),
            "prior_changes": sum(self.changes[p] for p in paths),
            "prior_fixes": sum(self.fixes[p] for p in paths),
            "avg_file_age_days": round(sum(ages) / len(ages), 1) if ages else 0.0,
            "author_experience": self.commits_by_author[author],
            "author_subsystem_experience": sum(
                self.commits_by_author_subsystem[(author, s)] for s in subsystems
            ),
            "is_fix": int(is_fix(commit["subject"])),
            "commit_type": commit_type(commit["subject"]),
            "hour": date.hour,
            "weekday": date.weekday(),
        }

    def record(self, commit: Commit, paths: list[str]) -> None:
        author, fix = commit["author"], is_fix(commit["subject"])
        self.commits_by_author[author] += 1
        for s in {subsystem(p) for p in paths}:
            self.commits_by_author_subsystem[(author, s)] += 1
        for p in paths:
            self.last_change[p] = commit["timestamp"]
            self.devs[p].add(author)
            self.changes[p] += 1
            if fix:
                self.fixes[p] += 1


def build_dataset(commits: list[Commit], culprits: Collection[str]) -> pd.DataFrame:
    """
    One row per commit that touches code, in history order.

    Each row is described BEFORE the commit is recorded into the history, so its
    features only use the past. Recording first would leak the commit into its
    own features (data leakage).
    """
    history = RepoHistory()
    rows = []
    for commit in commits:
        code_files = [f for f in commit["files"] if is_code(f[0])]
        if not code_files:
            continue
        row = history.describe(commit, code_files)
        row[TARGET] = int(commit["hash"] in culprits)
        rows.append(row)
        history.record(commit, [path for path, _, _ in code_files])
    return pd.DataFrame(rows)


def to_model_input(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Numeric matrix for the model, with `commit_type` one-hot encoded.

    Pass the training `columns` at prediction time: unseen commit types are
    dropped and missing ones filled with 0, so the column order always matches.
    """
    X = pd.get_dummies(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, dtype=int
    )
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X
