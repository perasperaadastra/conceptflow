"""
Formal sklearn check_estimator / parametrize_with_checks tests.

ConceptMembershipEncoder, ConceptLatticeEstimator, and ImplicationBasisEstimator
pass all checks except check_transformer_preserve_dtypes — all three always
output bool by design.

ConceptualScaler uses NominalScale(0), which resolves to source_attribute="x0".
Generic numpy arrays are converted to ManyValuedContexts with synthesised column
names x0, x1, ... so NominalScale(0) works with any 2D array.  The xfails are:
  - check_transformer_preserve_dtypes: always outputs bool (same as above)
  - check_fit_idempotent: transforms a second, independently-generated random
    dataset through the same fitted estimator. NominalScale has a fixed,
    fit-time-derived vocabulary by FCA design (unlike a numeric scaler), so it
    correctly raises ValueError when the check's second dataset contains
    values outside that vocabulary -- this is not a bug, see
    NominalScale.encode_value.

NaN is treated as a valid category value by design (not an error condition),
so ConceptualScaler declares tags.input_tags.allow_nan = True; this makes
check_estimators_nan_inf pass rather than xfail.
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.feature_extraction import ConceptMembershipEncoder
from conceptflow.preprocessing import ConceptualScaler, NominalScale
from conceptflow.rules import ImplicationBasisEstimator

_BOOL_OUTPUT_XFAIL = (
    "Always outputs bool dtype regardless of input dtype — concept membership "
    "indicators are inherently boolean."
)

_CME_EXPECTED_FAILURES = {
    "check_transformer_preserve_dtypes": _BOOL_OUTPUT_XFAIL,
}

_CLE_EXPECTED_FAILURES = {
    "check_transformer_preserve_dtypes": _BOOL_OUTPUT_XFAIL,
}

_IBE_EXPECTED_FAILURES = {
    "check_transformer_preserve_dtypes": _BOOL_OUTPUT_XFAIL,
}

_SCALER_EXPECTED_FAILURES = {
    "check_transformer_preserve_dtypes": _BOOL_OUTPUT_XFAIL,
    "check_fit_idempotent": (
        "NominalScale has a fixed, fit-time vocabulary by design; transforming "
        "a second, independently-generated dataset containing out-of-vocabulary "
        "values correctly raises ValueError instead of silently producing "
        "wrong output."
    ),
}


@parametrize_with_checks(
    [ConceptMembershipEncoder()],
    expected_failed_checks=lambda est: _CME_EXPECTED_FAILURES,
)
def test_concept_membership_encoder_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ConceptLatticeEstimator()],
    expected_failed_checks=lambda est: _CLE_EXPECTED_FAILURES,
)
def test_concept_lattice_estimator_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ConceptualScaler(scales=[NominalScale(0)])],
    expected_failed_checks=lambda est: _SCALER_EXPECTED_FAILURES,
)
def test_conceptual_scaler_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ImplicationBasisEstimator()],
    expected_failed_checks=lambda est: _IBE_EXPECTED_FAILURES,
)
def test_implication_basis_estimator_sklearn_checks(estimator, check):
    check(estimator)
