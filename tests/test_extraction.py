"""
Integration tests: build a tiny real git repository in a temp folder and check
that SZZ labels and history features come out right.
"""

import os
import subprocess

import pytest

from extract_dataset import build_dataset, find_bug_introducing_commits, read_commits


def run_git(repo, *args, date=None):
    env = {**os.environ,
           "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "dev@test.com",
           "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "dev@test.com"}
    if date:  # fixed dates keep the history order deterministic
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                   capture_output=True)


def commit_file(repo, path, content, message, date):
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    run_git(repo, "add", path)
    run_git(repo, "commit", "-m", message, date=date)


@pytest.fixture
def repo(tmp_path):
    """
    History of the fake repository:
      1. feat: writes cart.ts with a buggy line          <- should be labeled as bug-introducing
      2. feat: adds an unrelated file                     <- should NOT be labeled
      3. fix:  corrects the buggy line written in (1)
    """
    run_git(tmp_path, "init", "-q")
    commit_file(tmp_path, "src/cart.ts",
                "export const total = (a, b) => a - b;\nexport const x = 1;\n",
                "feat: add cart total", "2024-01-01T10:00:00")
    commit_file(tmp_path, "src/wishlist.ts",
                "export const wishlist = [];\n",
                "feat: add wishlist", "2024-01-02T10:00:00")
    commit_file(tmp_path, "src/cart.ts",
                "export const total = (a, b) => a + b;\nexport const x = 1;\n",
                "fix: cart total was subtracting", "2024-01-03T10:00:00")
    return tmp_path


def test_read_commits_returns_history_oldest_first(repo):
    commits = read_commits(str(repo))

    assert [c["subject"] for c in commits] == [
        "feat: add cart total", "feat: add wishlist", "fix: cart total was subtracting"]
    assert commits[0]["files"] == [("src/cart.ts", 2, 0)]
    assert commits[2]["files"] == [("src/cart.ts", 1, 1)]


def test_szz_blames_only_the_commit_that_wrote_the_fixed_line(repo):
    commits = read_commits(str(repo))
    fixes = [c for c in commits if c["subject"].startswith("fix")]

    culprits = find_bug_introducing_commits(str(repo), fixes)

    assert set(culprits) == {commits[0]["hash"]}   # the cart commit, not the wishlist one


def test_dataset_labels_and_history_features(repo):
    commits = read_commits(str(repo))
    culprits = find_bug_introducing_commits(str(repo), [commits[2]])

    df = build_dataset(commits, culprits).set_index("subject")

    assert df.loc["feat: add cart total", "introduced_bug"] == 1
    assert df.loc["feat: add wishlist", "introduced_bug"] == 0
    assert df.loc["fix: cart total was subtracting", "is_fix"] == 1


def test_features_never_look_into_the_future(repo):
    """
    Each row may only use information from BEFORE its commit (no data leakage).
    The first commit has no past, so its history features must all be zero.
    """
    df = build_dataset(read_commits(str(repo)), culprits={})
    first, fix = df.iloc[0], df.iloc[2]

    assert first["prior_changes"] == 0
    assert first["prior_fixes"] == 0
    assert first["author_experience"] == 0
    assert first["new_files"] == 1

    # the fix touches cart.ts, which was changed once before (by commit 1)
    assert fix["prior_changes"] == 1
    assert fix["prior_fixes"] == 0          # the fix itself must not count yet
    assert fix["author_experience"] == 2    # two commits before it
    assert fix["new_files"] == 0
