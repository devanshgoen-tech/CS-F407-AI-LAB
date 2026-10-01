# Sprinkler BN Lab — Written Answers

Worksheet: [`llm_bn.ipynb`](llm_bn.ipynb) — *Building and Learning a
Bayesian Network*.

The notebook combines two activities:

- **AI science**: define a probabilistic model, perform correct
  inference, estimate parameters from data.
- **AI engineering**: use a coding LLM to generate the implementation,
  then inspect, approve, execute, and validate it.

I have reproduced every programmatic result in plain Python scripts so
that it can be re-run without a GPU or the Qwen weights.

| Script | Covers |
|--------|--------|
| [`sprinkler_bn.py`](sprinkler_bn.py) | Reference pgmpy model, CPDs, Variable Elimination, independent brute-force oracle. |
| [`estimation.py`](estimation.py) | MLE from simulated data, sample-size sweep, sampling-variability sweep, MLE vs Bayesian (BDeu) on 30 samples. |
| [`broken_models.py`](broken_models.py) | A structurally invalid CPD (rejected by `check_model`), and a subtler semantic break (passes `check_model` but gives wrong posteriors). |
| [`tests.py`](tests.py) | The reusable test battery. |

Captured output lives in `outputs/`.

## 1. The probabilistic specification

Edges:
`Cloudy → Rain`, `Cloudy → Sprinkler`, `Rain → WetGrass`,
`Sprinkler → WetGrass`.

Joint:
`P(C, R, S, W) = P(C) · P(R|C) · P(S|C) · P(W|R, S)`.

Parameters:

- `P(C=1) = 0.5`
- `P(R=1|C=0) = 0.2`, `P(R=1|C=1) = 0.8`
- `P(S=1|C=0) = 0.5`, `P(S=1|C=1) = 0.1`
- `P(W=1|R=0,S=0) = 0.01`,  `P(W=1|R=0,S=1) = 0.90`,
  `P(W=1|R=1,S=0) = 0.90`, `P(W=1|R=1,S=1) = 0.99`.

In pgmpy, the `TabularCPD` for `WetGrass` carries
`evidence=["Rain", "Sprinkler"]` and the four columns correspond to
`(R=0,S=0), (R=0,S=1), (R=1,S=0), (R=1,S=1)` in that order. Getting
this ordering wrong is the mistake exercised in `broken_models.py`.

## 2. Exact inference (Variable Elimination)

Running `python3 sprinkler_bn.py`:

| Query | pgmpy VE | Brute-force oracle |
|-------|----------|--------------------|
| `P(Rain=1 | WetGrass=1)` | 0.704769 | 0.704769 |
| `P(Sprinkler=1 | WetGrass=1)` | 0.427846 | 0.427846 |
| `P(Cloudy=1 | WetGrass=1)` | 0.574615 | 0.574615 |
| `P(Rain=1 | WetGrass=1, Sprinkler=0)` | 0.992202 | 0.992202 |

The brute-force oracle enumerates all 2⁴ = 16 joint assignments and
computes the posterior from the factorisation directly. Perfect
agreement with Variable Elimination confirms that the CPDs encode the
specification (including the parent-state ordering).

### Homework query — qualitative predictions

- `P(S=1 | W=1) ≈ 0.428`. The grass is wet, so some cause must be
  active. The sprinkler by itself is a priori less likely under the
  Cloudy prior than rain, which keeps this below 0.5.
- `P(C=1 | W=1) ≈ 0.575`. Mild upward update: wet grass is weak
  evidence that it was cloudy, because `P(Rain|Cloudy)` is high.
- `P(R=1 | W=1, S=0) ≈ 0.992`. Explaining-away in reverse: if the
  grass is wet but the sprinkler was off, rain is almost certainly the
  cause.

## 3. Parameter estimation

### 3.1 MLE on 1000 samples

```
Manual estimate  P(R=1|C=1) = 0.772541
pgmpy  estimate  P(R=1|C=1) = 0.772541
True             P(R=1|C=1) = 0.800000
```

The library result matches the hand-computed ratio
`N(R=1, C=1) / N(C=1)` exactly, which is the definition of the MLE.

### 3.2 Sample-size sweep for `P(R=1|C=1)`

| N     | P̂ | |error| |
|-------|---------|---------|
| 20    | 0.8889 | 0.0889 |
| 50    | 0.8261 | 0.0261 |
| 100   | 0.7083 | 0.0917 |
| 500   | 0.7805 | 0.0195 |
| 1000  | 0.7725 | 0.0275 |
| 5000  | 0.8040 | 0.0040 |

The error shrinks on average with `N`, but the sequence is not
monotone — the N=100 draw happened to be unlucky. This is sampling
variability, not an API inconsistency.

### 3.3 Sampling variability across seeds (N = 100)

| seed | P̂(R=1|C=1) |
|------|-----------|
| 1 | 0.694 |
| 2 | 0.795 |
| 3 | 0.745 |
| 4 | 0.768 |
| 5 | 0.796 |

Different random draws of the same size give different MLEs because
the sample itself is a random variable. The scientific explanation is
sampling variability, not something to blame on `pgmpy` or an LLM.

### 3.4 MLE vs Bayesian (BDeu, equivalent sample size = 10) at N = 30

| method | P̂(R=1|C=1) |
|--------|-------------|
| True parameter | 0.800 |
| MLE  N=30 | **0.909** |
| BDeu N=30 | **0.781** |

With only 30 observations the MLE lands 0.11 away from the truth; the
BDeu estimator lands 0.019 away. The Dirichlet prior adds the
equivalent of 10 pseudo-observations spread uniformly over the parent
states, which pulls the estimate back toward the uniform `0.5` baseline
and (here) toward the truth. The prior is a modelling choice, not a
hidden implementation detail.

## 4. Failure-mode demonstrations

- **Structural break.** A `TabularCPD` whose columns do not sum to 1
  is rejected by `check_model()` (we deliberately set
  `P(R=0|C=0) + P(R=1|C=0) = 0.9 + 0.3 = 1.2` and the library raises
  `ValueError: Sum or integral of conditional probabilities for node
  Rain is not equal to 1.`).
- **Wrong semantics.** The reference model's WetGrass CPD has its
  four parent-state columns permuted. Every column still sums to 1,
  so `check_model()` returns True — but the posterior
  `P(Rain=1|WetGrass=1)` becomes 0.717 instead of the correct 0.705.
  `check_model()` is necessary but not sufficient; semantic tests
  against an independent oracle are needed.

## 5. What exactly did the LLM contribute?

I did not need to run Qwen2.5-Coder locally — the specification is
already in the notebook — but the division of labour the notebook
teaches is still useful.

| Task | Component |
|------|-----------|
| Interpret the natural-language specification | LLM |
| Emit candidate pgmpy code | LLM |
| Represent the Bayesian network | `pgmpy.DiscreteBayesianNetwork` |
| Local consistency check | `model.check_model()` |
| Exact inference | `pgmpy.inference.VariableElimination` |
| MLE / Bayesian estimation | `DiscreteMLE` / `DiscreteBayesianEstimator` |
| Decide whether generated code is correct | **Me** |
| Independent oracle for the posterior | `enumerate_posterior` in `sprinkler_bn.py` |

Prompt actually used (summarised from the notebook):

> *Write Python code using the current `pgmpy` API. Construct the
> binary discrete BN with the edges and parameters listed above. Store
> the fitted model in `generated_model` and the posterior
> `P(Rain | WetGrass=1)` in `generated_posterior`.*

Failure modes the LLM could introduce — all caught by the tests in
`tests.py`:

- a reversed row / column (same numbers, different meaning);
- `evidence=["Sprinkler", "Rain"]` instead of `["Rain", "Sprinkler"]`;
- an obsolete API path (`BayesianModel` vs `DiscreteBayesianNetwork`);
- a hidden `exec`/`open` call (ruled out by the AST sanity check the
  notebook defines).

## 6. Reflection on the engineering workflow

```
specify  ->  generate  ->  inspect  ->  execute  ->  validate
```

The LLM accelerates the "generate" step. Every other step is the
engineer's responsibility. Running a program without a `Traceback` is
not evidence of correctness — the "valid numbers, wrong semantics"
demonstration gives a model that passes `check_model()` and still
represents the wrong distribution.

The scientific object remains:

```
P(C, R, S, W) = P(C) · P(R|C) · P(S|C) · P(W|R, S)
```

The engineering tooling (`pgmpy`, Variable Elimination, estimators,
LLM code assistants) is interchangeable; the probabilistic
specification is not.
