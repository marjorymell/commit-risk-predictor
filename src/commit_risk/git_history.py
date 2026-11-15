"""Read commit history from a local git repository."""

from __future__ import annotations

import subprocess
from typing import TypedDict

FileChange = tuple[str, int, int]  # (path, lines_added, lines_deleted)

# Non-printable separators that never appear in commit messages.
_COMMIT_MARKER = "@@@"
_FIELD_SEP = "\x1f"


class Commit(TypedDict):
    hash: str
    author: str
    timestamp: int
    subject: str
    files: list[FileChange]


def git(repo: str, *args: str) -> str:
    """Run a git command inside `repo` and return its stdout."""
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def resolve_commit(repo: str, ref: str) -> str | None:
    """Full hash of `ref`, or None if it does not exist."""
    return git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip() or None


def read_commits(repo: str) -> list[Commit]:
    """Non-merge commits, oldest first, with per-file line counts."""
    output = git(
        repo,
        "log",
        "--no-merges",
        "--no-renames",
        "--reverse",
        "--numstat",
        f"--format={_COMMIT_MARKER}%H{_FIELD_SEP}%ae{_FIELD_SEP}%at{_FIELD_SEP}%s",
    )
    commits: list[Commit] = []
    current: Commit | None = None
    for line in output.splitlines():
        if line.startswith(_COMMIT_MARKER):
            commit_hash, author, timestamp, subject = line[len(_COMMIT_MARKER) :].split(
                _FIELD_SEP, 3
            )
            current = Commit(
                hash=commit_hash,
                author=author.lower(),
                timestamp=int(timestamp),
                subject=subject,
                files=[],
            )
            commits.append(current)
        elif line.strip() and current is not None:
            added, deleted, path = line.split("\t", 2)
            if added == "-":  # binary file: git reports no line counts
                continue
            current["files"].append((path, int(added), int(deleted)))
    return commits
