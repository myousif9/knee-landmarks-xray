from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def make_diversity_strata(
    df: pd.DataFrame,
    columns: Iterable[str],
    numeric_bins: int = 4,
) -> pd.Series:
    """Build a stable stratum key from categorical and numeric metadata columns.

    Numeric columns are quantile-binned before being combined with categorical
    values. Missing values are kept as their own level so they can still be
    represented in the sample.
    """

    columns = list(columns)
    if not columns:
        return pd.Series(["all"] * len(df), index=df.index, dtype="object")

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing diversity columns: {', '.join(missing)}")

    parts = []
    for col in columns:
        values = df[col]
        if pd.api.types.is_numeric_dtype(values):
            non_null_unique = values.dropna().nunique()
            if non_null_unique <= 1:
                binned = values.astype("object").where(values.notna(), "missing")
            else:
                bins = min(numeric_bins, non_null_unique)
                binned = pd.qcut(values, q=bins, duplicates="drop").astype("object")
                binned = binned.where(values.notna(), "missing")
            parts.append(col + "=" + binned.astype(str))
        else:
            cleaned = values.astype("object").where(values.notna(), "missing")
            parts.append(col + "=" + cleaned.astype(str))

    strata = parts[0]
    for part in parts[1:]:
        strata = strata + "|" + part
    return strata


def sample_diverse_rows(
    df: pd.DataFrame,
    n: int | None,
    diversity_columns: Iterable[str] | None = None,
    numeric_bins: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows as evenly as possible across metadata-defined strata.

    The sampler first assigns each row to a stratum using ``diversity_columns``.
    It then repeatedly draws one row from each stratum until ``n`` rows are
    selected or the dataframe is exhausted. This favors broad coverage without
    discarding rare strata.
    """

    if n is None or n >= len(df):
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if n <= 0:
        raise ValueError("n must be positive when provided.")

    diversity_columns = list(diversity_columns or [])
    strata = make_diversity_strata(df, diversity_columns, numeric_bins=numeric_bins)
    rng = np.random.default_rng(random_state)

    groups = defaultdict(list)
    for idx, stratum in strata.items():
        groups[stratum].append(idx)

    for indices in groups.values():
        rng.shuffle(indices)

    selected = []
    strata_order = list(groups.keys())
    rng.shuffle(strata_order)

    while len(selected) < n and strata_order:
        next_order = []
        for stratum in strata_order:
            indices = groups[stratum]
            if indices:
                selected.append(indices.pop())
                if len(selected) == n:
                    break
            if indices:
                next_order.append(stratum)
        strata_order = next_order

    return df.loc[selected].reset_index(drop=True)


def stratify_labels_for_split(
    df: pd.DataFrame,
    diversity_columns: Iterable[str] | None = None,
    numeric_bins: int = 4,
) -> pd.Series | None:
    """Return stratification labels only when every label has at least 2 rows."""

    diversity_columns = list(diversity_columns or [])
    if not diversity_columns:
        return None

    labels = make_diversity_strata(df, diversity_columns, numeric_bins=numeric_bins)
    if labels.value_counts().min() < 2:
        return None
    return labels
