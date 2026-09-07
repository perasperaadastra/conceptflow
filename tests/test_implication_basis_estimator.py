import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from conceptflow.rules import ImplicationBasisEstimator


def _two_of_three_context():
    # Only single attributes or all three ever co-occur, so any two
    # attributes together force the third.
    return np.array([
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])


def test_fits_numpy_array_and_exposes_implications():
    est = ImplicationBasisEstimator().fit(_two_of_three_context())

    check_is_fitted(est, "implications_")
    assert len(est.get_implications()) == 3

    rules = {
        (imp.premise, imp.conclusion) for imp in est.get_implications()
    }
    assert (frozenset({0, 1}), frozenset({2})) in rules
    assert (frozenset({0, 2}), frozenset({1})) in rules
    assert (frozenset({1, 2}), frozenset({0})) in rules


def test_transform_marks_premise_satisfaction():
    X = _two_of_three_context()
    est = ImplicationBasisEstimator(output="dataframe").fit(X)

    out = est.transform(X)

    assert list(out.columns) == list(est.get_feature_names_out())
    # Row 0 has all three attributes, so every implication's premise holds.
    assert out.iloc[0].all()
    # Rows 1-3 have only one attribute each, so no two-attribute premise holds.
    assert not out.iloc[1:].any().any()


def test_transform_generalizes_to_unseen_object_pattern():
    # Fit on the 4 rows above, then transform a 5th, never-seen row that
    # only satisfies the {a, b} premise -- transform must be recomputed from
    # this object's own attributes, not looked up by fit-time row position.
    X = _two_of_three_context()
    est = ImplicationBasisEstimator(output="array").fit(X)

    out = est.transform(np.array([[1, 1, 0]]))

    labels = list(est.get_feature_names_out())
    ab_to_c = labels.index("{m0, m1}->{m2}")
    assert out[0, ab_to_c]

    other_indices = [i for i in range(len(labels)) if i != ab_to_c]
    assert not out[0, other_indices].any()


def test_transform_before_fit_raises():
    est = ImplicationBasisEstimator()
    with pytest.raises(Exception):
        est.transform(_two_of_three_context())


def test_get_implications_before_fit_raises():
    est = ImplicationBasisEstimator()
    with pytest.raises(Exception):
        est.get_implications()


def test_output_formats_agree():
    X = _two_of_three_context()

    array_out = ImplicationBasisEstimator(output="array").fit_transform(X)
    df_out = ImplicationBasisEstimator(output="dataframe").fit_transform(X)
    sparse_out = ImplicationBasisEstimator(output="sparse").fit_transform(X)

    assert array_out.tolist() == df_out.to_numpy(dtype=bool).tolist()
    assert array_out.tolist() == sparse_out.toarray().tolist()


def test_works_in_sklearn_pipeline():
    X = _two_of_three_context()

    pipe = Pipeline([("rules", ImplicationBasisEstimator())])
    pipe.fit(X)

    assert len(pipe.named_steps["rules"].get_implications()) == 3


def test_clone_preserves_params_and_refits_independently():
    est = ImplicationBasisEstimator(output="dataframe")
    cloned = clone(est)

    assert cloned.get_params() == est.get_params()

    cloned.fit(_two_of_three_context())
    with pytest.raises(Exception):
        check_is_fitted(est, "implications_")


def test_fits_dataframe_with_named_columns():
    X = pd.DataFrame(
        _two_of_three_context(),
        index=["g1", "g2", "g3", "g4"],
        columns=["a", "b", "c"],
    )
    est = ImplicationBasisEstimator().fit(X)

    labels = set(est.get_feature_names_out())
    assert "{a, b}->{c}" in labels
    assert "{a, c}->{b}" in labels
    assert "{b, c}->{a}" in labels
