"""
Extract a commit dataset from a git repository for
Just-in-Time defect prediction.

Usage:
    python src/extract_dataset.py <repo_path> <output_csv>

Example:
    python src/extract_dataset.py ../faststore data/faststore_commits.csv

For every commit (excluding merges and release commits) the script computes
a set of features describing the change, plus a label:

    introduced_bug = 1  -> some line written by this commit was later changed
                           or deleted by a bug-fixing commit
    introduced_bug = 0  -> otherwise

The label follows the SZZ algorithm (Sliwerski, Zimmermann & Zeller, 2005).
"""

import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Only these files count as "code". Docs, lockfiles and configs are ignored.
CODE_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
                   ".scss", ".css", ".graphql", ".gql")
IGNORED_PATHS = ("CHANGELOG", "yarn.lock", "pnpm-lock", "package-lock",
                 "__snapshots__", ".d.ts", "/generated/", "/dist/")

# Conventional Commits: "fix: ...", "fix(core): ...", "hotfix!: ..."
FIX_PATTERN = re.compile(r"^(fix|bugfix|hotfix)(\(|:|!|\s)", re.I)
TYPE_PATTERN = re.compile(r"^(\w+)(\([^)]*\))?!?:")
RELEASE_PATTERN = re.compile(r"^\[no ci\]|^chore\(release\)|^Release", re.I)
TEST_PATTERN = re.compile(r"(\.test\.|\.spec\.|__tests__|/test/|/tests/|cypress/)")
HUNK_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+")
BLAME_LINE_PATTERN = re.compile(r"^[0-9a-f]{40} \d+ \d+")


# ---------------------------------------------------------------------------
# Part 1 - Reading the git history
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_code(path):
    return path.endswith(CODE_EXTENSIONS) and not any(p in path for p in IGNORED_PATHS)


def is_test(path):
    return bool(TEST_PATTERN.search(path))


def is_fix(subject):
    return bool(FIX_PATTERN.search(subject))


def subsystem(path):
    """packages/core/src/a.ts -> 'packages/core' ; src/a.ts -> 'src'"""
    parts = path.split("/")
    if parts[0] in ("packages", "apps") and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]


def entropy(changed_lines_per_file):
    """How spread out a change is across files (0 = all in one file, 1 = evenly spread)."""
    total = sum(changed_lines_per_file)
    if total == 0 or len(changed_lines_per_file) < 2:
        return 0.0
    h = -sum((x / total) * math.log2(x / total) for x in changed_lines_per_file if x > 0)
    return h / math.log2(len(changed_lines_per_file))


# ---------------------------------------------------------------------------
# Part 2 - SZZ: which commits introduced the bugs that fixes later repaired?
# ---------------------------------------------------------------------------

def find_bug_introducing_commits(repo, fix_commits):
    """
    For each fix commit:
      1. look at the lines it deleted or changed (`git diff -U0`)
      2. ask `git blame` who wrote those lines BEFORE the fix
      3. that author commit is marked as "bug-introducing"
    Returns {bug_introducing_hash: {fix_hash, ...}}.
    """
    culprits = defaultdict(set)
    for i, fix in enumerate(fix_commits, 1):
        if i % 100 == 0:
            print(f"  SZZ: {i}/{len(fix_commits)} fixes analysed", flush=True)

        for path, _, deleted in fix["files"]:
            # only production code with removed/changed lines can be blamed
            if deleted == 0 or not is_code(path) or is_test(path):
                continue

            diff = git(repo, "diff", "-U0", "-w", f"{fix['hash']}^", fix["hash"], "--", path)
            line_ranges = []
            for line in diff.splitlines():
                match = HUNK_PATTERN.match(line)
                if match and int(match.group(2) or 1) > 0:
                    start, count = match.group(1), match.group(2) or "1"
                    line_ranges += ["-L", f"{start},+{count}"]
            if not line_ranges:
                continue

            # blame the parent of the fix: "who wrote these lines before they were fixed?"
            blame = git(repo, "blame", "-w", "--porcelain", *line_ranges,
                        f"{fix['hash']}^", "--", path)
            for line in blame.splitlines():
                if BLAME_LINE_PATTERN.match(line):
                    culprits[line[:40]].add(fix["hash"])
    return culprits


# ---------------------------------------------------------------------------
# Part 3 - Features: describe each commit using ONLY information from its past
# ---------------------------------------------------------------------------

def build_dataset(commits, culprits):
    # Running state of the repository, updated commit by commit (oldest first).
    commits_by_author = defaultdict(int)
    commits_by_author_subsystem = defaultdict(int)
    last_change_of_file = {}
    devs_of_file = defaultdict(set)
    changes_of_file = defaultdict(int)
    fixes_of_file = defaultdict(int)

    rows = []
    for c in commits:
        files = [f for f in c["files"] if is_code(f[0])]
        if not files:
            continue  # docs/config-only commit

        paths = [f[0] for f in files]
        prod_paths = [p for p in paths if not is_test(p)]
        subsystems = {subsystem(p) for p in paths}
        directories = {p.rsplit("/", 1)[0] if "/" in p else "." for p in paths}
        # max(0, ...) because commit dates can be out of order (rebase, cherry-pick)
        ages = [max(0, c["timestamp"] - last_change_of_file[p]) / 86400
                for p in paths if p in last_change_of_file]
        type_match = TYPE_PATTERN.match(c["subject"])
        date = datetime.fromtimestamp(c["timestamp"], tz=timezone.utc)

        rows.append({
            "hash": c["hash"],
            "date": date.strftime("%Y-%m-%d"),
            "subject": c["subject"],
            # --- size of the change ---
            "lines_added": sum(f[1] for f in files),
            "lines_deleted": sum(f[2] for f in files),
            "n_files": len(files),
            "n_dirs": len(directories),
            "n_subsystems": len(subsystems),
            "entropy": round(entropy([f[1] + f[2] for f in files]), 4),
            "n_test_files": len(paths) - len(prod_paths),
            "new_files": sum(1 for p in paths if p not in changes_of_file),
            # --- history of the touched files ---
            "n_prior_devs": len(set().union(*(devs_of_file[p] for p in paths))),
            "prior_changes": sum(changes_of_file[p] for p in paths),
            "prior_fixes": sum(fixes_of_file[p] for p in paths),
            "avg_file_age_days": round(sum(ages) / len(ages), 1) if ages else 0.0,
            # --- who made it ---
            "author_experience": commits_by_author[c["author"]],
            "author_subsystem_experience": sum(
                commits_by_author_subsystem[(c["author"], s)] for s in subsystems),
            # --- context ---
            "is_fix": int(is_fix(c["subject"])),
            "commit_type": type_match.group(1).lower() if type_match else "other",
            "hour": date.hour,
            "weekday": date.weekday(),
            # --- label ---
            "introduced_bug": int(c["hash"] in culprits),
        })

        # Update the state AFTER recording the row, so a commit never "sees" itself
        # or the future (that would be data leakage).
        commits_by_author[c["author"]] += 1
        for s in subsystems:
            commits_by_author_subsystem[(c["author"], s)] += 1
        for p in paths:
            last_change_of_file[p] = c["timestamp"]
            devs_of_file[p].add(c["author"])
            changes_of_file[p] += 1
            if is_fix(c["subject"]):
                fixes_of_file[p] += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(repo, output_csv):
    print("Reading history...")
    commits = [c for c in read_commits(repo) if not RELEASE_PATTERN.search(c["subject"])]
    print(f"  {len(commits)} commits (merges and releases excluded)")

    fixes = [c for c in commits if is_fix(c["subject"])]
    print(f"  {len(fixes)} fix commits. Running SZZ (this takes a few minutes)...")
    culprits = find_bug_introducing_commits(repo, fixes)

    df = build_dataset(commits, culprits)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {output_csv}: {len(df)} commits, "
          f"{df['introduced_bug'].mean():.1%} introduced a bug")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
