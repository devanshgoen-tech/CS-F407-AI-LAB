"""Task 5 — three-class extension of the sensor task.

Same 2-input inputs, but instead of a single yes/no output we predict
one of three classes:

    class 0: both sensors inactive     -> (0,0)
    class 1: sensors disagree          -> (0,1) or (1,0)
    class 2: both sensors active       -> (1,1)

Only the output layer and loss change relative to Task 4:

    2 inputs -> 2 hidden (sigmoid) -> 3 logits -> softmax + cross-entropy

Predictions to make BEFORE running (see answers.md for the writeup):
    - final weight matrix shape:      W(2) is 3x2  (hidden_dim=2 -> 3 classes)
    - number of logits per example:   3
    - softmax probabilities sum to 1: they are normalised by the sum of
      exponentials by construction
    - the logit gradient p - y arises from the derivative of softmax +
      cross-entropy telescoping to `probabilities minus one-hot target`

The optional stability check adds a constant of 100 to every logit and
verifies the softmax vector is unchanged (up to floating-point noise),
because softmax is invariant to additive constants — which is exactly
why stable implementations subtract the maximum logit before exp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 0
STEPS = 4000
LR = 0.1


def data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([0, 1, 1, 2])   # class indices
    return X, y


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 2),      # hidden
        nn.Sigmoid(),         # hidden nonlinearity
        nn.Linear(2, 3),      # 3 logits
    )


def main() -> None:
    torch.manual_seed(SEED)
    X, y = data()
    model = build_model()

    loss_fn = nn.CrossEntropyLoss()   # softmax + NLL, on logits
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    print("=== Task 5: three-class sensor decision (seed=0) ===\n")

    for step in range(STEPS):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = loss.item()

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        final_loss = loss_fn(logits, y).item()

    print(f"initial loss = {initial_loss:.6f}")
    print(f"final   loss = {final_loss:.6f}\n")

    print("output-layer weight matrix shape W(2) =", tuple(model[2].weight.shape),
          " (rows=classes=3, cols=hidden_dim=2)")
    print("logits per example =", logits.shape[1], "\n")

    print("x1 x2 | target | probabilities [c0, c1, c2]        | pred | correct?")
    print("------+--------+-----------------------------------+------+---------")
    for (x1, x2), t, p, yh in zip(X.tolist(), y.tolist(), probs.tolist(), pred.tolist()):
        mark = "yes" if int(t) == int(yh) else "NO"
        p_str = "[" + ", ".join(f"{v:.4f}" for v in p) + "]"
        print(f" {int(x1)}  {int(x2)} |   {int(t)}    | {p_str} |  {int(yh)}   |  {mark}")

    correct = int((pred == y).sum().item())
    print(f"\n{correct}/4 correct")

    # Softmax normalisation check for one example.
    row = probs[1]                    # the (0,1) example
    print(f"\nsoftmax check on the (0,1) example:")
    print(f"  probabilities = {row.tolist()}")
    print(f"  sum           = {row.sum().item():.10f}")

    # Optional diagnostic: add 100 to every logit; softmax must be
    # invariant. This is why the stable form is exp(z - max(z)).
    with torch.no_grad():
        shifted = F.softmax(logits + 100.0, dim=1)
    max_abs_diff = (shifted - probs).abs().max().item()
    print(f"\nsoftmax(logits + 100) == softmax(logits)? "
          f"max |Δ| across all 4 x 3 entries = {max_abs_diff:.3e}")


if __name__ == "__main__":
    main()
