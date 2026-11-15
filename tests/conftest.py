import os
import subprocess

import pytest


def _git(repo, *args, date=None):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "dev@test.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "dev@test.com",
    }
    if date:  # fixed dates keep the history order deterministic
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)


def _commit(repo, path, content, message, date):
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message, date=date)


@pytest.fixture
def repo(tmp_path):
    """
    A tiny real repository:
      1. feat: writes cart.ts with a buggy line   -> bug-introducing
      2. feat: adds an unrelated file              -> clean
      3. fix:  corrects the line written in (1)
      4. docs: README only                         -> no code, never scored
    """
    _git(tmp_path, "init", "-q")
    _commit(
        tmp_path,
        "src/cart.ts",
        "export const total = (a, b) => a - b;\nexport const x = 1;\n",
        "feat: add cart total",
        "2024-01-01T10:00:00",
    )
    _commit(
        tmp_path,
        "src/wishlist.ts",
        "export const wishlist = [];\n",
        "feat: add wishlist",
        "2024-01-02T10:00:00",
    )
    _commit(
        tmp_path,
        "src/cart.ts",
        "export const total = (a, b) => a + b;\nexport const x = 1;\n",
        "fix: cart total was subtracting",
        "2024-01-03T10:00:00",
    )
    _commit(tmp_path, "README.md", "# shop\n", "docs: add readme", "2024-01-04T10:00:00")
    return str(tmp_path)
