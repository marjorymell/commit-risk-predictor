from commit_risk.classify import is_fix
from commit_risk.features import build_dataset
from commit_risk.git_history import read_commits
from commit_risk.szz import changed_line_ranges, find_bug_introducing_commits


def test_read_commits_returns_history_oldest_first(repo):
    commits = read_commits(repo)

    assert [c["subject"] for c in commits] == [
        "feat: add cart total",
        "feat: add wishlist",
        "fix: cart total was subtracting",
        "docs: add readme",
    ]
    assert commits[0]["files"] == [("src/cart.ts", 2, 0)]
    assert commits[2]["files"] == [("src/cart.ts", 1, 1)]


def test_changed_line_ranges_ignores_pure_additions():
    diff = "@@ -3,2 +3,2 @@\n-a\n+b\n@@ -10,0 +11,4 @@\n+new\n@@ -20 +24 @@\n-x\n+y\n"

    assert changed_line_ranges(diff) == ["-L", "3,+2", "-L", "20,+1"]


def test_szz_blames_only_the_commit_that_wrote_the_fixed_line(repo):
    commits = read_commits(repo)
    fixes = [c for c in commits if is_fix(c["subject"])]

    culprits = find_bug_introducing_commits(repo, fixes)

    assert set(culprits) == {commits[0]["hash"]}


def test_dataset_labels(repo):
    commits = read_commits(repo)
    culprits = find_bug_introducing_commits(repo, [commits[2]])

    df = build_dataset(commits, culprits).set_index("subject")

    assert df.loc["feat: add cart total", "introduced_bug"] == 1
    assert df.loc["feat: add wishlist", "introduced_bug"] == 0
    assert df.loc["fix: cart total was subtracting", "is_fix"] == 1
