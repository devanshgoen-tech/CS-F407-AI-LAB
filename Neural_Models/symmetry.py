"""Task 4C — Symmetry experiment.

Start from *identical* weights (all zero) instead of random weights, keep
everything else the same, and inspect the two rows of the hidden-layer
weight matrix as training progresses.

The scientific claim being tested:
    if two hidden units start with the same weights and biases, they
    receive the same pre-activation on every input, so they compute the
    same output and receive the same gradient. Gradient descent then
    keeps them identical forever — the "symmetry" that random
    initialisation is designed to break.
"""

import torch
import torch.nn as nn

from xor_binary import build_model, xor_data


SEED = 0
LR = 0.1
STEPS_TO_SHOW = [0, 1, 5, 20, 100, 500, 2000]


def zero_all_parameters(model: nn.Module) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()


def main() -> None:
    torch.manual_seed(SEED)
    X, y = xor_data()

    model = build_model("sigmoid")
    zero_all_parameters(model)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    W1 = model[0].weight

    print("=== Task 4C: symmetry experiment (all weights initialised to 0) ===\n")
    print("hidden-layer weight matrix W(1) at selected steps:")
    print("(row 0 and row 1 are the parameter vectors of hidden units 0 and 1)\n")

    for step in range(max(STEPS_TO_SHOW) + 1):
        if step in STEPS_TO_SHOW:
            with torch.no_grad():
                row0 = W1[0].tolist()
                row1 = W1[1].tolist()
                identical = torch.allclose(W1[0], W1[1])
            print(f"step {step:>4}: row 0 = {row0},  row 1 = {row1},  identical={identical}")

        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).flatten()
        pred = (probs >= 0.5).long()
        final_loss = loss_fn(model(X), y).item()

    print(f"\nfinal loss = {final_loss:.6f}")
    print("final predictions:")
    for (x1, x2), t, p, yh in zip(X.tolist(), y.flatten().tolist(), probs.tolist(), pred.tolist()):
        print(f"  x=({int(x1)},{int(x2)})  target={int(t)}  prob={p:.4f}  pred={yh}")

    correct = int((pred == y.flatten().long()).sum().item())
    print(f"\n{correct}/4 correct — the network cannot break out of the symmetric solution.")


if __name__ == "__main__":
    main()
