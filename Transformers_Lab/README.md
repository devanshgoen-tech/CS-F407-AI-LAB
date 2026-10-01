# Transformers Lab — Attention Primitives + HuggingFace / Ollama Demos

Worksheet sources:
[`Transformer.pdf`](Transformer.pdf),
[`AI_lab_transformers.ipynb`](AI_lab_transformers.ipynb),
[`Run_Ollama.ipynb`](Run_Ollama.ipynb).

The hands-on concepts from the PDF (attention, self-attention,
cross-attention, multi-head attention, positional encoding, causal
mask) are implemented from scratch in PyTorch and verified with tests.
The HuggingFace and Ollama cells from the notebooks are reproduced as
runnable scripts.

## Files

| File | Description |
|------|-------------|
| `Transformer.pdf` / `AI_lab_transformers.ipynb` / `Run_Ollama.ipynb` | Original lab sources. |
| `attention.py` | Scaled dot-product, self-, cross- and multi-head attention; causal mask. |
| `positional_encoding.py` | Sinusoidal positional encoding + property demo. |
| `tests.py` | Shape, softmax, identity, causal, multi-head, and permutation-invariance tests. |
| `hf_demos.py` | T5 translation, GPT-2 generation, BERT sentiment classification. |
| `ollama_demo.py` | Query a local Ollama server over HTTP. |
| `answers.md` | Written answers to the lab concepts + reflection on the three Transformer families. |
| `outputs/` | Captured runs. |

## Run

The from-scratch demos only need PyTorch:

```bash
python3 attention.py
python3 positional_encoding.py
python3 tests.py
```

The HuggingFace demos download pre-trained weights:

```bash
pip install "transformers[torch]"
python3 hf_demos.py
```

For the Ollama demo:

```bash
# install Ollama from https://ollama.com/download, then
ollama pull mistral
python3 ollama_demo.py "Who was Sir Isaac Newton?"
```
