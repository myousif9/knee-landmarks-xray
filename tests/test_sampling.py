import pandas as pd

from src.data.sampling import (
    make_diversity_strata,
    sample_diverse_rows,
    stratify_labels_for_split,
)


def test_make_diversity_strata_bins_numeric_columns():
    df = pd.DataFrame(
        {
            "laterality": ["L", "R", "L", "R"],
            "age": [20, 40, 60, 80],
        }
    )

    strata = make_diversity_strata(df, ["laterality", "age"], numeric_bins=2)

    assert strata.str.contains("laterality=").all()
    assert strata.str.contains("age=").all()
    assert strata.nunique() == 4


def test_sample_diverse_rows_preserves_rare_categories():
    df = pd.DataFrame(
        {
            "dicom_path": [f"img_{i}.dcm" for i in range(10)],
            "manufacturer": ["A"] * 8 + ["B", "C"],
        }
    )

    sampled = sample_diverse_rows(
        df,
        n=3,
        diversity_columns=["manufacturer"],
        random_state=0,
    )

    assert set(sampled["manufacturer"]) == {"A", "B", "C"}


def test_sample_diverse_rows_is_reproducible():
    df = pd.DataFrame(
        {
            "dicom_path": [f"img_{i}.dcm" for i in range(20)],
            "site": ["A", "B"] * 10,
        }
    )

    sampled_1 = sample_diverse_rows(df, n=8, diversity_columns=["site"], random_state=7)
    sampled_2 = sample_diverse_rows(df, n=8, diversity_columns=["site"], random_state=7)

    assert sampled_1["dicom_path"].tolist() == sampled_2["dicom_path"].tolist()


def test_stratify_labels_for_split_skips_singletons():
    df = pd.DataFrame({"site": ["A", "A", "B"]})

    labels = stratify_labels_for_split(df, ["site"])

    assert labels is None
