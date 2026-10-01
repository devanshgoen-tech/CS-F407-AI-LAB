"""Reusable tests for the Sprinkler BN lab (notebook `turn validation
into reusable tests` section)."""

from __future__ import annotations

import math

from pgmpy.inference import VariableElimination

from sprinkler_bn import build_reference_model, enumerate_posterior
from estimation import fit_mle, rain_c1


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_reference_structure() -> None:
    print("test_reference_structure")
    m = build_reference_model()
    check("check_model()", m.check_model())
    check(
        "nodes",
        set(m.nodes()) == {"Cloudy", "Rain", "Sprinkler", "WetGrass"},
    )
    check(
        "edges",
        set(m.edges()) == {
            ("Cloudy", "Rain"),
            ("Cloudy", "Sprinkler"),
            ("Rain", "WetGrass"),
            ("Sprinkler", "WetGrass"),
        },
    )


def test_posterior_matches_oracle() -> None:
    print("test_posterior_matches_oracle")
    m = build_reference_model()
    inf = VariableElimination(m)
    for query, evidence in [
        ("Rain", {"WetGrass": 1}),
        ("Sprinkler", {"WetGrass": 1}),
        ("Cloudy", {"WetGrass": 1}),
        ("Rain", {"WetGrass": 1, "Sprinkler": 0}),
    ]:
        pgmpy_val = float(
            inf.query([query], evidence=evidence, show_progress=False).values[1]
        )
        oracle_val = enumerate_posterior(query, evidence)[1]
        check(
            f"P({query}=1 | {evidence}) matches brute-force oracle",
            math.isclose(pgmpy_val, oracle_val, rel_tol=1e-9, abs_tol=1e-9),
        )


def test_mle_converges_to_truth() -> None:
    print("test_mle_converges_to_truth")
    ref = build_reference_model()
    data_small = ref.simulate(n_samples=50, seed=7, show_progress=False)
    data_large = ref.simulate(n_samples=5000, seed=7, show_progress=False)
    err_small = abs(rain_c1(fit_mle(data_small)) - 0.8)
    err_large = abs(rain_c1(fit_mle(data_large)) - 0.8)
    check("large-sample MLE closer to truth than small-sample", err_large <= err_small)
    check("large-sample MLE within 0.05 of truth", err_large < 0.05)


def test_cpds_normalised() -> None:
    print("test_cpds_normalised")
    m = build_reference_model()
    for cpd in m.get_cpds():
        values = cpd.values.reshape(cpd.variable_card, -1)
        column_sums = values.sum(axis=0)
        check(
            f"{cpd.variable}: columns sum to 1",
            all(math.isclose(s, 1.0, rel_tol=1e-9) for s in column_sums),
        )


if __name__ == "__main__":
    for t in (
        test_reference_structure,
        test_cpds_normalised,
        test_posterior_matches_oracle,
        test_mle_converges_to_truth,
    ):
        t()
    print("\nAll tests passed.")
