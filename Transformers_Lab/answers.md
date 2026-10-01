# Transformers Lab — Written Answers

Worksheet: [`Transformer.pdf`](Transformer.pdf) and the two notebooks
[`AI_lab_transformers.ipynb`](AI_lab_transformers.ipynb) and
[`Run_Ollama.ipynb`](Run_Ollama.ipynb).

The PDF gives the conceptual overview; the two notebooks demonstrate
pre-trained Transformer models with HuggingFace and Ollama. For the
hands-on concepts I built the pieces from scratch in PyTorch so that
the formulas can be checked directly, and I kept the HuggingFace /
Ollama demos as runnable scripts.

| File | Covers |
|------|--------|
| [`attention.py`](attention.py) | Scaled dot-product, self-, cross-, multi-head, causal mask. |
| [`positional_encoding.py`](positional_encoding.py) | Sinusoidal PE and its properties. |
| [`tests.py`](tests.py) | Shape / softmax / causal / permutation-invariance tests. |
| [`hf_demos.py`](hf_demos.py) | T5 translation, GPT-2 generation, BERT sentiment (pipeline). |
| [`ollama_demo.py`](ollama_demo.py) | Query a local Ollama server over HTTP. |
| [`outputs/`](outputs) | Captured runs. |

## 1. What was the Transformer initially introduced for?

Machine translation — a sequence-to-sequence task. The original paper
(*Attention Is All You Need*, 2017) replaced the recurrent layers in a
seq2seq architecture with stacks of multi-head self- and
cross-attention, eliminating the sequential bottleneck of RNNs and
enabling training on much larger corpora.

## 2. Three Transformer families (per the PDF)

| Family | Example | Typical use |
|--------|---------|-------------|
| Encoder-only | BERT | Semantic search, retrieval, clustering, classification, NER — understanding existing text. |
| Decoder-only | GPT-style models | Text generation, chatbots, code generation, QA, summarisation. |
| Encoder-decoder | T5, BART, original Transformer | Translation and general seq2seq (source sequence → target sequence). |

## 3. Hands-on concepts

### 3.1 Attention mechanism

Scaled dot-product attention is

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V.

The `sqrt(d_k)` scaling keeps the logits in a range where the softmax
doesn't saturate as `d_k` grows. The output is a weighted average of
the value rows; the weights come from the compatibility between each
query and each key.

See `scaled_dot_product_attention` in [`attention.py`](attention.py).
The toy run (`outputs/attention.txt`) shows that the weight rows sum
to 1 and the output has the expected `(L_q, d_v)` shape.

### 3.2 Self-attention

Q, K, V are three linear projections of the same input sequence `X`.
Each position produces its new representation as a weighted sum of all
positions' values — the model learns how much each position should
"look at" every other position.

**Why we need positional encoding.** Without positional information,
self-attention is permutation-invariant: shuffling the input shuffles
the output in exactly the same way. The test
`test_self_attention_is_permutation_invariant_without_PE` in
[`tests.py`](tests.py) demonstrates this empirically. Adding sinusoidal
positional encodings before the first attention block breaks the
symmetry and lets the model distinguish "the dog chased the cat" from
"the cat chased the dog".

### 3.3 Cross-attention

Q comes from the decoder, K and V come from the encoder's output. This
is what lets a decoder "look at" the encoded source sentence when it
produces each target token (e.g. a French token looking at the English
source). See `CrossAttention` in [`attention.py`](attention.py); the
run shows a decoder of length 5 attending over an encoder of length 9
and producing a length-5 output.

### 3.4 Multi-head attention

Instead of one attention operation of dimension `d_model`, run `h`
operations in parallel, each on `d_model / h` dimensions, then
concatenate and linearly project. Different heads can learn to focus on
different kinds of relationships (short-range syntactic, long-range
coreference, etc.) at the same depth.

See `MultiHeadAttention` in [`attention.py`](attention.py) and the test
`test_multihead_decomposition`:

- input `(2, 6, 16)` with `num_heads=4` → output `(2, 6, 16)`;
- attention weights `(B, heads, L_q, L_k) = (2, 4, 6, 6)`;
- each head's rows still sum to 1.

### 3.5 Positional encoding

Sinusoidal PE (Vaswani et al., 2017):

    PE[pos, 2k]   = sin(pos / 10000^{2k / d_model})
    PE[pos, 2k+1] = cos(pos / 10000^{2k / d_model})

Properties verified in [`positional_encoding.py`](positional_encoding.py):

- `PE[0, even] = 0`, `PE[0, odd] = 1` (as sin(0)=0, cos(0)=1);
- the L2 norm of each row is `sqrt(d_model / 2)` and does not depend on
  the position (here, `sqrt(8) ≈ 2.828`);
- adjacent positions have higher cosine similarity than distant ones
  (0.94 at distance 1 vs 0.49 at distance 40 for `d_model = 16`).

### 3.6 Masked (causal) attention

In the decoder of a seq2seq model, and throughout a decoder-only model
like GPT, token `t` must not be allowed to look at tokens `t+1`,
`t+2`, …; otherwise training would leak the targets through
self-attention. The fix is to add a lower-triangular keep-mask to the
attention scores (set the upper triangle to `-inf` before softmax).

`causal_mask(n)` returns the mask; `test_causal_mask_blocks_future`
checks that the upper triangle of the resulting attention matrix is
exactly zero and that position 0's attention row is `[1, 0, 0, 0, 0]`
— it can only attend to itself.

## 4. HuggingFace hands-on blocks

Reproducing the three cells of `AI_lab_transformers.ipynb` in
[`hf_demos.py`](hf_demos.py):

```python
# Encoder-decoder translation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
inputs = tok("translate English to French: ...", return_tensors="pt")
model.generate(**inputs, max_length=30)

# Decoder-only generation
from transformers import pipeline
pipeline("text-generation", model="gpt2")("The future of AI is", max_length=30)

# Encoder-only classification
pipeline("sentiment-analysis")("I love machine learning")
```

Running these requires `pip install "transformers[torch]"`, after
which the models are cached locally. The pipeline API hides the
tokenise → model → decode plumbing; under the hood each call still
runs the same Transformer blocks whose primitives are in `attention.py`.

## 5. Ollama (`Run_Ollama.ipynb`)

Ollama ships local open-weight LLMs behind a simple HTTP server. The
workflow is:

1. Install Ollama (`https://ollama.com/download`).
2. Pull a model: `ollama pull mistral` (or `llama3`, `qwen`, …).
3. Start the server (`ollama serve`; starts automatically on macOS).
4. Query from Python:

```bash
python3 ollama_demo.py "Who was Sir Isaac Newton?"
```

`ollama_demo.py` posts to `http://localhost:11434/api/generate` with a
plain-Python `urllib` call — no LangChain required for a one-off query.
The notebook's LangChain version is equivalent; it just wraps the HTTP
call in `OllamaLLM` and chains it with a `ChatPromptTemplate`.

**Comparing with hosted services.** The same prompt sent to a hosted
model (GPT, Claude, Gemini) will typically give a more polished answer,
because those models are larger and more heavily instruction-tuned —
but local Ollama models can be queried offline, do not send data to a
third party, and are small enough to run on a laptop.

## 6. Takeaway

```
Encoder-decoder : seq2seq (translation, summarisation)
Encoder-only    : representation (search, classification)
Decoder-only    : generation (chat, completion, code)
```

The common engine in all three is the same attention primitive. The
Python in [`attention.py`](attention.py) and
[`positional_encoding.py`](positional_encoding.py) is enough to assemble
each of the three families — pre-trained models from HuggingFace or
Ollama give the full-scale version, but the maths they implement is
exactly what the tests here already check.
