"""
Feature definitions shared by training, prediction and the demo app.
Keeping them in one place guarantees the model always receives
the same columns, in the same order.
"""

import pandas as pd

NUMERIC_FEATURES = [
    "lines_added", "lines_deleted", "n_files", "n_dirs", "n_subsystems", "entropy",
    "n_test_files", "new_files", "n_prior_devs", "prior_changes", "prior_fixes",
    "avg_file_age_days", "author_experience", "author_subsystem_experience",
    "is_fix", "hour", "weekday",
]
CATEGORICAL_FEATURES = ["commit_type"]
TARGET = "introduced_bug"


def to_model_input(df, columns=None):
    """
    Turn raw feature rows into the numeric matrix the model expects.

    `commit_type` (feat, fix, chore...) is one-hot encoded: each type becomes
    a 0/1 column, because models only understand numbers.
    When `columns` is given (at prediction time), the result is aligned to the
    columns seen during training, filling missing ones with 0.
    """
    X = pd.get_dummies(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                       columns=CATEGORICAL_FEATURES, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X
