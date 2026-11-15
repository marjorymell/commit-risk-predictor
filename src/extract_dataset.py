"""
Extract a commit dataset from a git repository for
Just-in-Time defect prediction.
"""

import subprocess
import sys


def git(repo, *args):
    """Run a git command inside `repo` and return its output as text."""
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def read_commits(repo):
    """Read the history (oldest first) with the files changed by each commit."""
    output = git(
        repo, "log", "--no-merges", "--no-renames", "--reverse", "--numstat",
        "--format=@@@%H\x1f%ae\x1f%at\x1f%s",
    )
    commits = []
    current = None
    for line in output.splitlines():
        if line.startswith("@@@"):
            commit_hash, author, timestamp, subject = line[3:].split("\x1f", 3)
            current = {
                "hash": commit_hash,
                "author": author.lower(),
                "timestamp": int(timestamp),
                "subject": subject,
                "files": [],
            }
            commits.append(current)
        elif line.strip() and current is not None:
            added, deleted, path = line.split("\t", 2)
            if added == "-":  # binary file, no line counts
                continue
            current["files"].append((path, int(added), int(deleted)))
    return commits


if __name__ == "__main__":
    commits = read_commits(sys.argv[1])
    print(f"{len(commits)} commits read")
    print("First commit:", commits[0]["subject"])
    print("Last commit:", commits[-1]["subject"])
    print("Files in last commit:", commits[-1]["files"][:3])