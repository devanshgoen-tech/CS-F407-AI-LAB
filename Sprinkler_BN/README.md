# Sprinkler BN Lab — pgmpy + LLM Code Assistant

Worksheet: [`llm_bn.ipynb`](llm_bn.ipynb).

The notebook builds the classic Sprinkler Bayesian network with
`pgmpy`, runs exact inference by Variable Elimination, fits CPTs from
simulated data (MLE and BDeu Bayesian estimation), and uses a local
Qwen2.5-Coder LLM to generate the pgmpy code and parameter-estimation
code. This folder reproduces every programmatic result as standalone
Python scripts.

## Files

| File | Description |
|------|-------------|
| `llm_bn.ipynb` | Original notebook (the "worksheet"). |
| `sprinkler_bn.py` | Reference BN + CPDs + Variable Elimination + brute-force oracle. |
| `estimation.py` | MLE and Bayesian (BDeu) parameter estimation; sample-size and sampling-variability sweeps. |
| `broken_models.py` | Structural and semantic failure demos. |
| `tests.py` | Reusable test battery (structure, normalisation, posteriors vs oracle, MLE convergence). |
| `answers.md` | Written answers, result tables, reflection on the LLM workflow. |
| `outputs/` | Captured runs. |

## Run

```bash
pip install "pgmpy>=1.0"
python3 sprinkler_bn.py
python3 estimation.py
python3 broken_models.py
python3 tests.py
```
