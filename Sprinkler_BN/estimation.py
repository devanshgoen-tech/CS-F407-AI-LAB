"""Parameter estimation from data.

Worksheet: llm_bn.ipynb -- *Parameter estimation from data* and
*Sparse data and Bayesian parameter estimation*.

Keeps the graph fixed and compares:
  * maximum-likelihood (MLE) estimates with the true parameters;
  * MLE stability as the sample size increases;
  * MLE vs Bayesian (BDeu) estimates on a small sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import (
    DiscreteBayesianEstimator,
    DiscreteMLE,
)

from sprinkler_bn import build_reference_model


def make_empty_structure() -> DiscreteBayesianNetwork:
    return DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])


def fit_mle(data: pd.DataFrame) -> DiscreteBayesianNetwork:
    model = make_empty_structure()
    model.fit(data, estimator=DiscreteMLE())
    return model


def fit_bdeu(data: pd.DataFrame, equivalent_sample_size: int = 10) -> DiscreteBayesianNetwork:
    model = make_empty_structure()
    model.fit(
        data,
        estimator=DiscreteBayesianEstimator(
            prior_type="BDeu",
            equivalent_sample_size=equivalent_sample_size,
        ),
    )
    return model


def rain_c1(model: DiscreteBayesianNetwork) -> float:
    """P_hat(Rain=1 | Cloudy=1)."""
    return float(model.get_cpds("Rain").values[1, 1])


def main() -> None:
    np.random.seed(7)
    ref = build_reference_model()

    print("=== 1000-sample MLE fit ===")
    data_1000 = ref.simulate(n_samples=1000, seed=7, show_progress=False)
    print(data_1000.head())
    print("Shape:", data_1000.shape)

    mle_1000 = fit_mle(data_1000)
    print("\nValid fitted model:", mle_1000.check_model())
    for cpd in mle_1000.get_cpds():
        print(cpd)
        print()

    cloudy_true = data_1000[data_1000["Cloudy"] == 1]
    manual_mle = (cloudy_true["Rain"] == 1).mean()
    print(f"Manual estimate  P(R=1|C=1) = {manual_mle:.6f}")
    print(f"pgmpy  estimate  P(R=1|C=1) = {rain_c1(mle_1000):.6f}")
    print(f"True             P(R=1|C=1) = 0.800000")

    print("\n=== Sample-size sweep for P(R=1 | C=1) ===")
    rows = []
    for n in [20, 50, 100, 500, 1000, 5000]:
        data = ref.simulate(n_samples=n, seed=7, show_progress=False)
        est = rain_c1(fit_mle(data))
        rows.append({"N": n, "P_hat(R=1|C=1)": est, "abs_error": abs(est - 0.8)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Sampling variability across seeds (N = 100) ===")
    rows = []
    for seed in [1, 2, 3, 4, 5]:
        data = ref.simulate(n_samples=100, seed=seed, show_progress=False)
        est = rain_c1(fit_mle(data))
        rows.append({"seed": seed, "P_hat(R=1|C=1)": est})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Small sample: MLE vs Bayesian (BDeu, ess=10) at N = 30 ===")
    data_small = ref.simulate(n_samples=30, seed=11, show_progress=False)
    mle_small = fit_mle(data_small)
    bayes_small = fit_bdeu(data_small, equivalent_sample_size=10)
    print(pd.DataFrame({
        "method": ["True parameter", "MLE, N=30", "BDeu, N=30"],
        "P(Rain=1 | Cloudy=1)": [
            0.8,
            rain_c1(mle_small),
            rain_c1(bayes_small),
        ],
    }).to_string(index=False))

    print("\nMLE Rain CPD (N=30):")
    print(mle_small.get_cpds("Rain"))
    print("\nBDeu Rain CPD (N=30):")
    print(bayes_small.get_cpds("Rain"))


if __name__ == "__main__":
    main()
