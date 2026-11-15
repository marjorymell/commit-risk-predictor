"""Rules that classify files and commit messages."""

from __future__ import annotations

import re

CODE_EXTENSIONS = (
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".scss", ".css", ".graphql", ".gql",
)  # fmt: skip
IGNORED_PATHS = (
    "CHANGELOG", "yarn.lock", "pnpm-lock", "package-lock",
    "__snapshots__", ".d.ts", "/generated/", "/dist/",
)  # fmt: skip

_FIX = re.compile(r"^(fix|bugfix|hotfix)(\(|:|!|\s)", re.IGNORECASE)
_TYPE = re.compile(r"^(\w+)(\([^)]*\))?!?:")
_RELEASE = re.compile(r"^\[no ci\]|^chore\(release\)|^Release", re.IGNORECASE)
_TEST = re.compile(r"(\.test\.|\.spec\.|__tests__|/test/|/tests/|cypress/)")


def is_code(path: str) -> bool:
    return path.endswith(CODE_EXTENSIONS) and not any(p in path for p in IGNORED_PATHS)


def is_test(path: str) -> bool:
    return bool(_TEST.search(path))


def is_fix(subject: str) -> bool:
    """Conventional Commits fix: `fix: ...`, `fix(scope): ...`, `hotfix!: ...`."""
    return bool(_FIX.search(subject))


def is_release(subject: str) -> bool:
    return bool(_RELEASE.search(subject))


def commit_type(subject: str) -> str:
    match = _TYPE.match(subject)
    return match.group(1).lower() if match else "other"


def subsystem(path: str) -> str:
    """Top-level unit of a monorepo: `packages/core/src/a.ts` -> `packages/core`."""
    parts = path.split("/")
    if parts[0] in ("packages", "apps") and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]
