"""
Interactive demo: describe a commit and get its estimated bug risk.

Run locally:   python app/app.py
On Hugging Face Spaces the same file runs as the Space entry point.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from commit_risk.features import NUMERIC_FEATURES, TARGET, to_model_input
from commit_risk.predict import risk_level
from commit_risk.train import build_forest, drop_immature_labels

DATA_URL = (
    "https://raw.githubusercontent.com/marjorymell/commit-risk-predictor/main/"
    "data/faststore_commits.csv"
)
LOCAL_DATA = Path(__file__).resolve().parents[1] / "data" / "faststore_commits.csv"
REPO_URL = "https://github.com/marjorymell/commit-risk-predictor"

# Real FastStore commits shown as clickable examples.
EXAMPLE_COMMITS = {
    "a55fe24dac70ccfedc6296001bd5c18225affeda": "Large feature, no tests",
    "3e8fdaa4bb728d2bde038ac96a2045f5d8670744": "Medium feature",
    "c11edcf5ac93b6e042e665f3e9e90bdf9a4b3a04": "Small one-file fix",
}
COMMIT_TYPES = ["feat", "fix", "chore", "refactor", "test", "docs", "style", "perf", "other"]

# Features the user edits; the rest are filled with the dataset median.
EDITABLE = [
    ("lines_added", "Lines added"),
    ("lines_deleted", "Lines deleted"),
    ("n_files", "Files changed"),
    ("n_test_files", "Test files changed"),
    ("n_subsystems", "Packages touched"),
    ("new_files", "New files"),
    ("prior_changes", "Previous changes to these files"),
    ("prior_fixes", "Previous bug fixes in these files"),
    ("n_prior_devs", "Developers who touched these files"),
    ("author_experience", "Author's previous commits"),
]


def load_dataset() -> pd.DataFrame:
    # The Space has no copy of the repo, so it reads the dataset from GitHub.
    return pd.read_csv(LOCAL_DATA if LOCAL_DATA.exists() else DATA_URL)


def train_model(df: pd.DataFrame):
    # Training at startup (~20s) avoids shipping a pickled model whose
    # scikit-learn version could differ from the one installed in the Space.
    labeled = drop_immature_labels(df)
    X = to_model_input(labeled)
    model = build_forest().fit(X, labeled[TARGET])
    historical_risk = model.predict_proba(X)[:, 1]
    return model, list(X.columns), pd.Series(historical_risk).sort_values().to_numpy()


DATASET = load_dataset()
MODEL, COLUMNS, HISTORICAL_RISK = train_model(DATASET)
MEDIANS = DATASET[NUMERIC_FEATURES].median()


def score(commit_type: str, *values: float) -> str:
    row = MEDIANS.to_dict() | {name: v for (name, _), v in zip(EDITABLE, values, strict=True)}
    row["commit_type"] = commit_type
    row["is_fix"] = int(commit_type == "fix")
    row["n_dirs"] = max(row["n_subsystems"], min(row["n_files"], row["n_dirs"]))
    row["entropy"] = 0.0 if row["n_files"] <= 1 else row["entropy"]

    risk = float(MODEL.predict_proba(to_model_input(pd.DataFrame([row]), COLUMNS))[0, 1])
    percentile = (risk > HISTORICAL_RISK).mean()
    level = risk_level(risk)
    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[level]
    return (
        f"## {icon} {risk:.0%} risk of introducing a bug ({level})\n\n"
        f"Riskier than **{percentile:.0%}** of the {len(HISTORICAL_RISK):,} historical commits "
        "the model learned from."
    )


def example_rows() -> list[list]:
    rows = []
    for commit_hash, label in EXAMPLE_COMMITS.items():
        match = DATASET[DATASET["hash"] == commit_hash]
        if match.empty:
            continue
        r = match.iloc[0]
        values = [int(r[name]) for name, _ in EDITABLE]
        rows.append([f"{label}: {r['subject'][:60]}", r["commit_type"], *values])
    return rows


with gr.Blocks(title="Commit Risk Predictor") as demo:
    gr.Markdown(
        "# Commit Risk Predictor\n"
        "Estimates how likely a commit is to introduce a bug, using a Random Forest trained on "
        "the history of [vtex/faststore](https://github.com/vtex/faststore) "
        "(bug-introducing commits labeled with the SZZ algorithm).\n\n"
        "Use it to **prioritise testing**, not to block merges. "
        f"Code and evaluation: [GitHub]({REPO_URL})."
    )
    with gr.Row():
        with gr.Column():
            description = gr.Textbox(label="Example", interactive=False, visible=False)
            commit_type = gr.Dropdown(COMMIT_TYPES, value="feat", label="Commit type")
            inputs = [
                gr.Number(value=float(MEDIANS[name]), label=label, minimum=0, precision=0)
                for name, label in EDITABLE
            ]
            button = gr.Button("Estimate risk", variant="primary")
        with gr.Column():
            result = gr.Markdown()
            gr.Markdown(
                "**How to read it.** On unseen commits the model reaches ROC AUC ≈ 0.76: "
                "reviewing the 30% riskiest commits catches about half of the bug-introducing "
                "ones. Size of the change and the history of the touched files weigh the most."
            )

    gr.Examples(
        examples=example_rows(),
        inputs=[description, commit_type, *inputs],
        label="Real FastStore commits (click to load)",
    )
    button.click(score, inputs=[commit_type, *inputs], outputs=result)

if __name__ == "__main__":
    demo.launch()
