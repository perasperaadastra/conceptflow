import numpy as np
import pytest

from conceptflow import FormalContext
from conceptflow.algorithms import Implication, compute_canonical_basis
from conceptflow.algorithms.enumeration import enumerate_concepts_bruteforce


def test_boolean_lattice_context_has_no_implications():
    # Every subset of {a, b, c} is already closed (2^3 = 8 concepts), so
    # there is nothing left to force -- the canonical basis is empty.
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]),
        objects=["g1", "g2", "g3", "g4"],
        attributes=["a", "b", "c"],
    )

    assert compute_canonical_basis(ctx) == []


def test_two_of_three_implies_the_third():
    # Only single attributes or all three ever co-occur, so any two
    # attributes together force the third.
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        objects=["g1", "g2", "g3", "g4"],
        attributes=["a", "b", "c"],
    )

    basis = compute_canonical_basis(ctx)

    rules = {
        (imp.premise_names(ctx), imp.conclusion_names(ctx))
        for imp in basis
    }

    assert rules == {
        (("a", "b"), ("c",)),
        (("a", "c"), ("b",)),
        (("b", "c"), ("a",)),
    }


def test_basis_is_sound_and_complete_against_bruteforce_concepts():
    """
    A basis is sound and complete iff the set of attribute sets closed under
    it (context closure plus repeatedly applying every implication) is
    exactly the set of real concept intents -- no more, no less. Verify this
    directly against independently-enumerated concepts (bruteforce) on a
    context with a nontrivial basis, rather than trusting the basis
    computation's own closure operator.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2, size=(8, 5))
    ctx = FormalContext.from_array(data)

    concepts = enumerate_concepts_bruteforce(ctx)
    real_intents = {c.intent for c in concepts}

    basis = compute_canonical_basis(ctx)

    def close_under_basis(attrs: frozenset[int]) -> frozenset[int]:
        changed = True
        while changed:
            changed = False
            for imp in basis:
                if imp.premise <= attrs and not (imp.conclusion <= attrs):
                    attrs = attrs | imp.conclusion
                    changed = True
        return attrs

    # Every real intent must be a fixpoint of the basis closure (soundness:
    # the basis must not forbid anything that is actually possible).
    for intent in real_intents:
        assert close_under_basis(intent) == intent

    # Every subset, once closed under the basis, must land on a real
    # intent (completeness: the basis must forbid everything that both the
    # bruteforce enumeration and the basis agree is impossible).
    all_attribute_sets = [
        frozenset(s) for r in range(ctx.n_attributes + 1)
        for s in __import__("itertools").combinations(range(ctx.n_attributes), r)
    ]
    for attrs in all_attribute_sets:
        assert close_under_basis(attrs) in real_intents


def test_implication_premise_and_conclusion_are_disjoint():
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        attributes=["a", "b", "c"],
    )

    for imp in compute_canonical_basis(ctx):
        assert imp.premise.isdisjoint(imp.conclusion)


def test_implication_is_frozen_and_deduplicates_input():
    imp = Implication(premise=[0, 0, 1], conclusion=[2])
    assert imp.premise == frozenset({0, 1})
    with pytest.raises(Exception):
        imp.premise = frozenset()  # frozen dataclass
