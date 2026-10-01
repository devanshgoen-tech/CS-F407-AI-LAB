"""Reference Sprinkler Bayesian network and exact inference.

Worksheet: llm_bn.ipynb -- *Building and Learning a Bayesian Network*.

Structure:

    Cloudy -> Rain
    Cloudy -> Sprinkler
    Rain, Sprinkler -> WetGrass

Joint factorisation:

    P(C, R, S, W) = P(C) P(R | C) P(S | C) P(W | R, S)

All variables are binary (0 = False, 1 = True).
"""

from __future__ import annotations

import itertools

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


def build_reference_model() -> DiscreteBayesianNetwork:
    """Construct the trusted Sprinkler Bayesian network."""
    model = DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])

    cpd_cloudy = TabularCPD(
        variable="Cloudy",
        variable_card=2,
        values=[[0.5], [0.5]],
    )

    # Columns are indexed by Cloudy = 0, 1.
    cpd_rain = TabularCPD(
        variable="Rain",
        variable_card=2,
        values=[
            [0.8, 0.2],   # P(Rain=0 | Cloudy)
            [0.2, 0.8],   # P(Rain=1 | Cloudy)
        ],
        evidence=["Cloudy"],
        evidence_card=[2],
    )

    cpd_sprinkler = TabularCPD(
        variable="Sprinkler",
        variable_card=2,
        values=[
            [0.5, 0.9],   # P(S=0 | Cloudy)
            [0.5, 0.1],   # P(S=1 | Cloudy)
        ],
        evidence=["Cloudy"],
        evidence_card=[2],
    )

    # Column order follows pgmpy: evidence rightmost varies fastest.
    # Columns: (R=0,S=0), (R=0,S=1), (R=1,S=0), (R=1,S=1).
    cpd_wetgrass = TabularCPD(
        variable="WetGrass",
        variable_card=2,
        values=[
            [0.99, 0.10, 0.10, 0.01],   # P(W=0 | R, S)
            [0.01, 0.90, 0.90, 0.99],   # P(W=1 | R, S)
        ],
        evidence=["Rain", "Sprinkler"],
        evidence_card=[2, 2],
    )

    model.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)
    assert model.check_model()
    return model


# ---------------------------------------------------------------------------
# Independent brute-force oracle
# ---------------------------------------------------------------------------

def _p_cloudy(c: int) -> float:
    return 0.5


def _p_rain(r: int, c: int) -> float:
    p_true = {0: 0.2, 1: 0.8}[c]
    return p_true if r == 1 else 1 - p_true


def _p_sprinkler(s: int, c: int) -> float:
    p_true = {0: 0.5, 1: 0.1}[c]
    return p_true if s == 1 else 1 - p_true


def _p_wetgrass(w: int, r: int, s: int) -> float:
    p_true = {
        (0, 0): 0.01,
        (0, 1): 0.90,
        (1, 0): 0.90,
        (1, 1): 0.99,
    }[(r, s)]
    return p_true if w == 1 else 1 - p_true


def enumerate_posterior(query: str, evidence: dict[str, int]) -> dict[int, float]:
    """P(query | evidence) by brute-force enumeration of all 16 states."""
    totals: dict[int, float] = {0: 0.0, 1: 0.0}
    denom = 0.0
    for c, r, s, w in itertools.product([0, 1], repeat=4):
        assignment = {"Cloudy": c, "Rain": r, "Sprinkler": s, "WetGrass": w}
        if any(assignment[k] != v for k, v in evidence.items()):
            continue
        p = (
            _p_cloudy(c)
            * _p_rain(r, c)
            * _p_sprinkler(s, c)
            * _p_wetgrass(w, r, s)
        )
        totals[assignment[query]] += p
        denom += p
    return {k: v / denom for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    model = build_reference_model()

    print("Nodes:", sorted(model.nodes()))
    print("Edges:", sorted(model.edges()))
    print("check_model():", model.check_model())
    print()

    print("=== CPDs ===")
    for cpd in model.get_cpds():
        print(cpd)
        print()

    inf = VariableElimination(model)

    print("=== Exact inference via Variable Elimination ===")
    for query, evidence in [
        ("Rain", {"WetGrass": 1}),
        ("Sprinkler", {"WetGrass": 1}),
        ("Cloudy", {"WetGrass": 1}),
        ("Rain", {"WetGrass": 1, "Sprinkler": 0}),
    ]:
        posterior = inf.query([query], evidence=evidence, show_progress=False)
        p1 = float(posterior.values[1])
        brute = enumerate_posterior(query, evidence)[1]
        tag = " ".join(f"{k}={v}" for k, v in evidence.items())
        print(f"  P({query}=1 | {tag}) = {p1:.6f}   (brute force: {brute:.6f})")


if __name__ == "__main__":
    main()
