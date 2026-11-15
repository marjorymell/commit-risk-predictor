"""
SZZ algorithm (Sliwerski, Zimmermann & Zeller, 2005).

A commit is considered bug-introducing when a later fix commit changed or
deleted lines that it wrote: we diff each fix against its parent and blame
the touched lines on the parent revision.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from commit_risk.classify import is_code, is_test
from commit_risk.git_history import Commit, git

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+")
_BLAME_HEADER = re.compile(r"^[0-9a-f]{40} \d+ \d+")


def changed_line_ranges(diff: str) -> list[str]:
    """`-L start,+count` arguments for every hunk that removes or changes lines."""
    ranges: list[str] = []
    for line in diff.splitlines():
        match = _HUNK.match(line)
        if not match:
            continue
        start, count = match.group(1), match.group(2) or "1"
        if int(count) > 0:  # count 0 = pure addition, nothing to blame
            ranges += ["-L", f"{start},+{count}"]
    return ranges


def find_bug_introducing_commits(
    repo: str, fix_commits: Iterable[Commit], progress_every: int = 100
) -> dict[str, set[str]]:
    """Map of bug-introducing commit hash -> hashes of the fixes that blamed it."""
    culprits: dict[str, set[str]] = defaultdict(set)
    for i, fix in enumerate(fix_commits, 1):
        if progress_every and i % progress_every == 0:
            print(f"  SZZ: {i} fixes analysed", flush=True)

        parent = f"{fix['hash']}^"
        for path, _, deleted in fix["files"]:
            # Only removed/changed production code can point at an earlier commit.
            if deleted == 0 or not is_code(path) or is_test(path):
                continue
            # -w: whitespace-only changes are formatting, not bug fixes.
            diff = git(repo, "diff", "-U0", "-w", parent, fix["hash"], "--", path)
            ranges = changed_line_ranges(diff)
            if not ranges:
                continue
            blame = git(repo, "blame", "-w", "--porcelain", *ranges, parent, "--", path)
            for line in blame.splitlines():
                if _BLAME_HEADER.match(line):
                    culprits[line[:40]].add(fix["hash"])
    return culprits
