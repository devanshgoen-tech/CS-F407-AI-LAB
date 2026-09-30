"""Task 4D — Activation experiment.

Train the random-initialised 2-2-1 network three times, changing only
the hidden activation (sigmoid / tanh / ReLU). For each run report:

    - the final loss,
    - whether all four examples are classified correctly,
    - the Euclidean norm of the first-layer weight gradient at an early
      training step (measured at step 5).

The lab explicitly cautions against declaring a universal winner from
four data points; the paragraph in `answers.md` interprets the numbers
in terms of gradient behaviour, not activation "quality".
"""

import torch
import torch.nn as nn

from xor_binary import build_model, xor_data, STEPS, LR


SEED = 0
EARLY_STEP = 5


def run(activation: str) -> dict:
    torch.manual_seed(SEED)
    X, y = xor_data()
    model = build_model(activation)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    W1 = model[0].weight

    early_norm = None
    initial_loss = None
    for step in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        if step == EARLY_STEP:
            early_norm = W1.grad.norm().item()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).flatten()
        pred = (probs >= 0.5).long()
        final_loss = loss_fn(model(X), y).item()
    correct = int((pred == y.flatten().long()).sum().item())

    return {
        "activation": activation,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "correct": correct,
        "all_correct": correct == 4,
        "early_grad_norm": early_norm,
        "probs": probs.tolist(),
    }


def main() -> None:
    print("=== Task 4D: activation experiment (2-2-1 XOR, seed=0) ===\n")
    results = [run(a) for a in ("sigmoid", "tanh", "relu")]

    print(f"early gradient norm measured at training step {EARLY_STEP}\n")
    print("hidden activation | init loss | final loss | 4/4? | ||∇W(1) L||_2 (step 5)")
    print("------------------+-----------+------------+------+-----------------------")
    for r in results:
        print(f"  {r['activation']:<15} | {r['initial_loss']:9.6f} | {r['final_loss']:10.6f} | "
              f" {r['correct']}/4 | {r['early_grad_norm']:.6f}")

    print("\nfinal probabilities per input:")
    for r in results:
        p = r["probs"]
        print(f"  {r['activation']:<8} -> (0,0)={p[0]:.4f}  (0,1)={p[1]:.4f}  "
              f"(1,0)={p[2]:.4f}  (1,1)={p[3]:.4f}")


if __name__ == "__main__":
    main()
