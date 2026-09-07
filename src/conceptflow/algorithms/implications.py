"""
Duquenne-Guigues (stem) basis computation for ConceptFlow.

The canonical basis is the smallest set of implications (attribute-set
"premise forces conclusion" rules) that is logically equivalent to every
implication that holds in a formal context. It is built from the context's
*pseudo-intents*.

Definition (Ganter & Wille, "Formal Concept Analysis", Prop. 2.15)
-------------------------------------------------------------------
For a formal context K = (G, M, I) with derivation/closure operator
``''`` (``attribute_closure``), a set P subseteq M is a **pseudo-intent** if:

1. P'' != P              (P is not itself a concept intent), and
2. for every pseudo-intent Q that is a *proper subset* of P: Q'' subseteq P.

Condition 2 is recursive, but well-founded: since Q must be strictly
smaller than P, processing candidate sets in increasing size order (ties
broken by any fixed rule, here lectic/lexicographic) guarantees every
pseudo-intent smaller than the current candidate has already been decided
before it is needed.

The canonical basis is then exactly:

    { P -> (P'' \\ P) : P is a pseudo-intent }

This is the same style of "clear, correct, exponential-worst-case"
algorithm as ``enumerate_concepts_bruteforce``: it tries every subset of
attributes (there is no way around this in general -- deciding whether a
given set is a pseudo-intent is already as hard as the underlying concept
lattice can be large), but it is straightforward to verify against the
definition above.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations

from conceptflow.algorithms.derivation import attribute_closure
from conceptflow.core.context import FormalContext


@dataclass(frozen=True)
class Implication:
    """
    A single exact implication premise -> conclusion.

    Parameters
    ----------
    premise:
        Attribute indices forming the "if" side.

    conclusion:
        Attribute indices forced by the premise, *beyond* the premise
        itself (i.e. ``conclusion`` and ``premise`` are disjoint).
    """

    premise: frozenset[int]
    conclusion: frozenset[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "premise", frozenset(self.premise))
        object.__setattr__(self, "conclusion", frozenset(self.conclusion))

    def premise_names(self, context: FormalContext) -> tuple[str, ...]:
        """Return the premise as attribute names, in a fixed order."""
        return tuple(context.attributes[i] for i in sorted(self.premise))

    def conclusion_names(self, context: FormalContext) -> tuple[str, ...]:
        """Return the conclusion as attribute names, in a fixed order."""
        return tuple(context.attributes[i] for i in sorted(self.conclusion))

    def __repr__(self) -> str:
        return (
            f"Implication(premise={sorted(self.premise)}, "
            f"conclusion={sorted(self.conclusion)})"
        )


def _powerset(n_attributes: int):
    """Yield every subset of range(n_attributes), smallest first."""
    attributes = range(n_attributes)
    return chain.from_iterable(
        combinations(attributes, size) for size in range(n_attributes + 1)
    )


def compute_canonical_basis(context: FormalContext) -> list[Implication]:
    """
    Compute the Duquenne-Guigues (stem) basis of a formal context.

    Step by step
    ------------
    1. Generate every candidate attribute subset, sorted by
       ``(size, sorted(subset))`` -- smallest first. This order is what
       makes step 3 well-defined: any pseudo-intent that could matter for
       a minimality check on the current candidate is strictly smaller,
       so it was already visited earlier in this same loop.
    2. Close the candidate with the context's own derivation operator
       (``attribute_closure``, i.e. B''). If the candidate is already
       equal to its own closure, it is a genuine concept intent, not a
       pseudo-intent -- skip it.
    3. Otherwise, check the minimality condition against every
       already-found pseudo-intent Q that is a proper subset of this
       candidate: all of their closures must already fit inside the
       candidate. If so, the candidate is itself a new pseudo-intent;
       record it together with its closure.
    4. Once every candidate has been classified, turn each pseudo-intent
       P (with closure P'') into one implication P -> (P'' \\ P) -- the
       attributes P forces *beyond itself*.

    Parameters
    ----------
    context:
        The formal context to compute the basis for.

    Returns
    -------
    list[Implication]
        The canonical basis, ordered by ``(premise size, premise)``.
    """
    pseudo_intents: dict[frozenset[int], frozenset[int]] = {}

    for subset in _powerset(context.n_attributes):
        candidate = frozenset(subset)
        closure = attribute_closure(context, candidate)

        if closure == candidate:
            # Already a concept intent -- nothing new to force.
            continue

        is_minimal = all(
            known_closure <= candidate
            for known_premise, known_closure in pseudo_intents.items()
            if known_premise < candidate
        )

        if is_minimal:
            pseudo_intents[candidate] = closure

    ordered_premises = sorted(
        pseudo_intents,
        key=lambda premise: (len(premise), sorted(premise)),
    )

    return [
        Implication(premise=premise, conclusion=pseudo_intents[premise] - premise)
        for premise in ordered_premises
    ]
