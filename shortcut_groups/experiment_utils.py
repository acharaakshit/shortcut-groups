import os
from pathlib import Path

import numpy as np
import pandas as pd


def artifact_prefix():
    """Return the required root for datasets and artifacts."""
    return Path(os.environ["PREFIX"])


def require_exact_keys(frame, columns, expected_keys, context):
    # check for missing columns and throw an error because it would be a corrupt file
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{context} is missing required columns: {missing_columns}")

    # observed keys to be compared with expected keys
    observed = list(frame[columns].itertuples(index=False, name=None))
    observed_set = set(observed)
    expected = set(expected_keys)
    if len(observed) != len(observed_set):
        raise ValueError(f"{context} contains duplicate candidate keys")
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        unexpected = sorted(observed_set - expected)
        raise ValueError(
            f"{context} has an incomplete candidate set: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )

def merge_csv_rows(path, rows, key_columns):
    """Atomically replace completed result keys while preserving all other rows."""
    path = Path(path)
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return

    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty:
            incoming = pd.concat([existing, incoming])
    # keep is last to keep the latest computation
    incoming = incoming.drop_duplicates(key_columns, keep="last")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    incoming.to_csv(temporary, index=False)
    temporary.replace(path)


# check which experiments are already completed, this is to avoid redundancy in computation
def completed_csv_keys(path, key_columns):
    """Return settings written atomically by merge_csv_rows."""
    path = Path(path)
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=key_columns)
    return set(frame[key_columns].itertuples(index=False, name=None))

def top_ranked_indices(scores, count, secondary_ranking=None):
    """Select the highest scores, using a secondary ranking only for exact ties."""
    if secondary_ranking is None:
        return np.argsort(scores, kind="mergesort")[-count:]
    secondary_ranking = np.asarray(secondary_ranking)
    if secondary_ranking.shape != np.asarray(scores).shape:
        raise ValueError("Secondary ranking and scores must have the same shape")
    return np.lexsort((secondary_ranking, scores))[-count:]


def errors_found_at_coverage(errors, scores, coverage, secondary_ranking=None):
    # count based on the coverage which is like 20% of test set for most experiments
    count = max(1, int(np.ceil(len(errors) * coverage)))
    # sort the scores in ascending order and pick last such that higher error is higher rank
    top_idx = top_ranked_indices(scores, count, secondary_ranking)
    # total number of errors in test set
    total_errors = errors.sum()
    # capture is the ratio of count of errors in top-20% sorted by risk score and the total errors in the test set
    return errors[top_idx].sum() / (total_errors + 1e-12)
