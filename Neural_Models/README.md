# Neural Models: Learning, Depth, Activations, and Output Layers

AI Laboratory worksheet on the smallest interesting neural model — a
2-2-1 network trained on XOR — used to make representation,
backpropagation, initialisation, activation choice, and output-layer
selection concrete and testable.

## Files

- [neur_models_lab_ex.pdf](neur_models_lab_ex.pdf) — the original
  worksheet.
- [xor_binary.py](xor_binary.py) — Task 3 + Task 4A/4B: 2-2-1 network,
  BCEWithLogitsLoss, initial vs final loss, four probabilities, and the
  first-layer weight gradient after `backward()`.
- [symmetry.py](symmetry.py) — Task 4C: all-zero weight initialisation,
  showing that the two hidden units stay identical and the network
  cannot break out of the symmetric solution.
- [activation.py](activation.py) — Task 4D: sigmoid vs tanh vs ReLU
  under otherwise identical settings, with the early first-layer
  gradient norm as diagnostic.
- [three_class.py](three_class.py) — Task 5: 2 → 2 → 3 logits, softmax +
  cross-entropy, four-input predictions, softmax-sum check, and the
  additive-constant invariance that motivates the max-subtract trick.
- [answers.md](answers.md) — Tasks 1–5 writeup, LLM prompt used and
  corrections applied, activation result table, and answers to the
  seven reflection questions.
- [outputs/](outputs/) — captured stdout of each script.

## Run

```bash
python3 xor_binary.py
python3 symmetry.py
python3 activation.py
python3 three_class.py
```

Requires PyTorch (CPU is sufficient). No GPU needed.
