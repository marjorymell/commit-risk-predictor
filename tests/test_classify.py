import pytest

from commit_risk.classify import commit_type, is_code, is_fix, is_release, is_test, subsystem


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("fix: null price on PDP", True),
        ("fix(core): handle empty cart", True),
        ("hotfix!: rollback checkout", True),
        ("feat: add wishlist", False),
        ("chore: fix typo in docs", False),
        ("prefix: something", False),
    ],
)
def test_is_fix_follows_conventional_commits(subject, expected):
    assert is_fix(subject) is expected


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("[no ci] Release: 4.11.4", True),
        ("chore(release): publish", True),
        ("feat: release notes page", False),
    ],
)
def test_is_release(subject, expected):
    assert is_release(subject) is expected


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("feat(ui): add button", "feat"),
        ("Fix!: urgent", "fix"),
        ("Update readme", "other"),
    ],
)
def test_commit_type(subject, expected):
    assert commit_type(subject) == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("packages/core/src/Cart.tsx", True),
        ("packages/ui/styles/button.scss", True),
        ("README.md", False),
        ("yarn.lock", False),
        ("packages/core/CHANGELOG.md", False),
        ("packages/api/src/types.d.ts", False),
    ],
)
def test_is_code(path, expected):
    assert is_code(path) is expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("packages/core/src/Cart.test.tsx", True),
        ("packages/core/__tests__/cart.ts", True),
        ("cypress/e2e/checkout.cy.js", True),
        ("packages/core/src/Cart.tsx", False),
    ],
)
def test_is_test(path, expected):
    assert is_test(path) is expected


def test_subsystem_groups_monorepo_packages():
    assert subsystem("packages/core/src/a/b.ts") == "packages/core"
    assert subsystem("apps/site/page.tsx") == "apps/site"
    assert subsystem("src/index.ts") == "src"
