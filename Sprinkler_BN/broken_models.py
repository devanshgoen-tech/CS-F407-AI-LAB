"""Failure-mode demonstrations.

Worksheet: llm_bn.ipynb -- *A deliberately broken Bayesian network* and
*A subtler failure: valid numbers, wrong semantics*.

1. A CPD whose columns do not sum to 1 should be rejected by
   `check_model()`.
2. A CPD whose columns DO sum to 1 but whose parent-state ordering is
   wrong will pass `check_model()` and still represent the wrong
   distribution.  This is why semantic tests (independent inference,
   known queries) are also needed.
"""

from __future__ import annotations

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from sprinkler_bn import build_reference_model, enumerate_posterior


def demo_structural_break() -> None:
    print("=== Structural break: columns do not sum to 1 ===")
    broken = DiscreteBayesianNetwork([("Cloudy", "Rain")])
    cpd_cloudy = TabularCPD("Cloudy", 2, [[0.5], [0.5]])
    cpd_rain_bad = TabularCPD(
        "Rain",
        2,
        values=[
            [0.9, 0.2],   # column for Cloudy=0 sums to 0.9+0.3 = 1.2
            [0.3, 0.8],
        ],
        evidence=["Cloudy"],
        evidence_card=[2],
    )
    broken.add_cpds(cpd_cloudy, cpd_rain_bad)
    try:
        broken.check_model()
        print("  check_model() PASSED (should not have!)")
    except Exception as exc:  # noqa: BLE001
        print(f"  check_model() rejected the model: {type(exc).__name__}: {exc}")


def demo_wrong_semantics() -> None:
    print("\n=== Subtler break: numbers normalise but ordering is wrong ===")
    wrong = build_reference_model()
    wrong.remove_cpds(wrong.get_cpds("WetGrass"))

    # Permute the four parent-state columns from the correct ordering.
    wrong_wetgrass = TabularCPD(
        variable="WetGrass",
        variable_card=2,
        values=[
            [0.99, 0.10, 0.01, 0.10],
            [0.01, 0.90, 0.99, 0.90],
        ],
        evidence=["Rain", "Sprinkler"],
        evidence_card=[2, 2],
    )
    wrong.add_cpds(wrong_wetgrass)
    print("  check_model():", wrong.check_model())

    inf = VariableElimination(wrong)
    posterior = inf.query(["Rain"], evidence={"WetGrass": 1}, show_progress=False)
    p_wrong = float(posterior.values[1])
    p_right = enumerate_posterior("Rain", {"WetGrass": 1})[1]

    print(f"  wrong-semantics model gives P(Rain=1 | WetGrass=1) = {p_wrong:.6f}")
    print(f"  trusted  oracle      gives P(Rain=1 | WetGrass=1) = {p_right:.6f}")
    print("  difference:", round(abs(p_wrong - p_right), 6))
    print("  -> check_model() is necessary but not sufficient.")


if __name__ == "__main__":
    demo_structural_break()
    demo_wrong_semantics()
