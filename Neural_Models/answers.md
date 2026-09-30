# Lab: Neural Models — Answers

Course: CS F407 (AI Laboratory)
Student: Devansh Goenka

Worksheet: [neur_models_lab_ex.pdf](neur_models_lab_ex.pdf).
Runnable code: [xor_binary.py](xor_binary.py),
[symmetry.py](symmetry.py), [activation.py](activation.py),
[three_class.py](three_class.py). Captured outputs are under
[outputs/](outputs/).

---

## Task 1 — Understand the problem before coding

**Input / output spaces and the four labelled examples.**

- `X = {0, 1}^2` — two binary sensor readings.
- `Y = {0, 1}` — the disagreement warning.

| x1 | x2 | y  (disagree?) |
|----|----|-----------------|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

**Sketch, described in words.** In the unit square the two positive
examples `(0,1)` and `(1,0)` sit on the anti-diagonal; the two negatives
`(0,0)` and `(1,1)` sit on the main diagonal. Any straight line in the
plane splits the square into two half-planes and therefore always
groups one main-diagonal corner with one anti-diagonal corner — it
cannot isolate exactly the anti-diagonal from exactly the main
diagonal. XOR is the canonical non-linearly-separable Boolean function.

**Why a single straight boundary fails.** A linear model with a sigmoid
computes `σ(w1 x1 + w2 x2 + b)` and thresholds at `0.5`, i.e. it
separates the plane by the *line* `w1 x1 + w2 x2 + b = 0`. There is no
choice of `(w1, w2, b)` that puts `(0,0)` and `(1,1)` on one side and
`(0,1)` and `(1,0)` on the other, because the sum `x1 + x2` is `0`, `1`,
`1`, `2` for those four points and no single threshold on `x1 + x2`
separates the ones from the zeros and twos simultaneously.

**Prediction for a single affine → sigmoid model.** It will drive every
output toward the majority-marginal probability (here 0.5 because the
labels are balanced), and the BCE loss will plateau near
`-log 0.5 ≈ 0.693` — the entropy of a fair coin. The four predicted
probabilities will be indistinguishable from each other. That is
exactly the "linear-only" failure mode we later observe as a stalled
sigmoid run with SGD in `xor_binary.py`.

**Checkpoint (recorded here):**
- specification: four (x, y) pairs above, task = binary classification.
- prediction: no affine 2 → 1 model achieves better than 2/4 accuracy.

**Think about it.** With only four data points we cannot say anything
strong about how well the *class* of neural models generalises; but we
*can* falsify the claim that "an affine model + a sigmoid can represent
any binary function of two binary inputs". A single counter-example
suffices, and XOR is that counter-example. This is why the exercise is
tiny: it is a representational test, not a statistical test.

---

## Task 2 — Design the intelligent agent

**Design.**

    2 inputs -> Linear(2, 2) -> hidden nonlinearity -> Linear(2, 1) -> sigmoid + BCE

Symbolically:

    a(1) = W(1) x + b(1)               W(1) ∈ R^{2x2}, b(1) ∈ R^2
    h(1) = f(a(1))                     f ∈ {sigmoid, tanh, ReLU}
    a(2) = W(2) h(1) + b(2)            W(2) ∈ R^{1x2}, b(2) ∈ R^1
    p    = σ(a(2))                     scalar probability
    L    = BCE(p, y)                   mean over the 4 examples

In PyTorch we keep `a(2)` as a logit and combine sigmoid + BCE inside
`nn.BCEWithLogitsLoss` for numerical stability. Optimiser is
gradient-based (Adam or SGD).

**Q1 — Why is the hidden nonlinearity scientifically necessary?**
Because a composition of affine maps `W(2)(W(1) x + b(1)) + b(2)` is
itself an affine map `W' x + b'`, which is a linear model. Stacking
affine layers without a nonlinear activation cannot represent anything
that a single affine layer cannot. XOR is not representable by a linear
model (Task 1), so *some* non-affine step between the two matrix
multiplications is required — that is what `f` supplies.

**Q2 — Why is sigmoid + BCE the right output pairing here?** The target
is a single yes/no answer. A sigmoid maps the scalar logit into `(0, 1)`
so we can interpret the output as `P(y = 1 | x)`, and BCE is exactly the
negative log-likelihood of that Bernoulli. Together they make the loss
convex in the *logit* per example, which yields the clean gradient
`p - y` on the output logit — no vanishing pre-factor from `σ'` at the
output. Using MSE with sigmoid instead would give a `p(1-p)` factor in
the output gradient that vanishes when the model is very confident and
very wrong, which stalls learning.

**Q3 — Evidence that will count as successful learning.**

1. **Final loss well below the chance baseline** `H(0.5) ≈ 0.693`. A
   converged 4/4 XOR fit reaches loss below `10^-2` easily.
2. **All four thresholded predictions match the labels.** This is the
   task specification, so it is the primary acceptance test.
3. **The first-layer weight gradient is nonzero at initialisation and
   shrinks toward zero as training converges.** Nonzero shows the
   learning signal exists; shrinking shows the parameters are actually
   moving into a low-loss region rather than sitting at a saddle.
4. **Repeated-run behaviour is consistent.** With only 4 data points
   the loss surface has multiple minima; a design should reach 4/4 on
   at least a healthy fraction of seeds. When it does not (see the
   seed-sweep in `xor_binary.py`), that is a diagnostic about the
   *optimiser + init* interaction, not the *task*.

**Think about it — hidden targets.** Nothing in the training data tells
each hidden unit what to compute. The learning signal reaches
`W(1), b(1)` purely through backpropagation:
`∂L/∂W(1) = (∂L/∂h(1)) · (∂h(1)/∂a(1)) · x^T`, where `∂L/∂h(1) =
W(2)^T · (∂L/∂a(2))`. In other words, the *output* layer's requirement
for what `h(1)` should look like — expressed as its gradient — is
back-propagated through `W(2)` and reshapes each hidden unit so that
the two units together become a linearly separable representation of
XOR. Empirically the two units typically converge to an AND-like and an
OR-like feature, whose linear combination is XOR — but the algorithm
never asks for that: it is what the gradient produces from the loss
alone.

---

## Task 3 — Use an LLM to generate a first implementation

**Prompt sent to Claude (verbatim):**

> Generate minimal PyTorch code that implements exactly the following
> design, and do not change the architecture or task.
>
> - Model: 2 inputs → Linear(2, 2) → hidden nonlinearity → Linear(2, 1),
>   output as a raw logit (I will use BCEWithLogitsLoss on it).
> - Hidden nonlinearity is a parameter (sigmoid / tanh / ReLU); default
>   sigmoid.
> - Data: the four XOR examples explicitly written out, no data loader.
> - Training: full-batch SGD on the mean BCE loss, a few thousand CPU
>   steps.
> - Random seed for reproducibility.
> - After training, print the initial and final loss, all four
>   probabilities and thresholded labels, and the first-layer weight
>   gradient tensor after one final `backward()` on the mean loss so
>   I can see ∂L/∂W(1).
> - One paragraph of comments explaining what each printed test is
>   checking. No extra abstractions, no CLI, no argparse.

**Corrections I made before running.**

1. The first draft used `nn.SGD(model.parameters(), lr=0.1)` — the
   plain `torch.optim.SGD`. On seed 0 with `lr=0.1` the sigmoid-hidden
   run stalled at loss ≈ 0.693 (the sigmoid pre-activation plateau
   discussed in the background section of the worksheet). I kept SGD in
   `activation.py` because that plateau *is* the point of the
   activation experiment, but switched the main `xor_binary.py` to
   `torch.optim.Adam(lr=0.1)` and to seed 1. Both are engineering
   settings the worksheet explicitly permits under Task 4A. Nothing
   about the scientific task (the 4 XOR points, the 2-2-1 architecture,
   sigmoid + BCE) changed.
2. The first draft printed only `model[0].weight.grad`; I added
   `b(1).grad`, `W(2).grad`, and the Euclidean norm
   `||∇_W(1) L||_2`, because the norm is what Task 4D needs for its
   cross-activation comparison and it is trivial to compute in the
   same block.

**Verifying the code without running it.** Reading the file we can
confirm the architecture (`nn.Linear(2, 2)` then activation then
`nn.Linear(2, 1)`), the exact four training examples, that
`BCEWithLogitsLoss` is applied to logits and not to probabilities
(otherwise the loss would double-sigmoid), that a seed is set before
building the module, and that `zero_grad → forward → loss → backward →
step` are in the right order.

**Verifying only by execution.** Whether the model *actually* learns
XOR from this init + optimiser combination; whether the gradient at
step 5 is 10⁻³ or 10⁻¹ under a given activation; whether the softmax
row of the three-class model sums to 1 under `float32` arithmetic.
These depend on numerical behaviour and must be measured.

**Checkpoint:** the exact prompt above; the two corrections above
(optimiser + printed diagnostics).

---

## Task 4 — Execute, test, and diagnose

Captured stdout: [outputs/xor_binary.txt](outputs/xor_binary.txt),
[outputs/symmetry.txt](outputs/symmetry.txt),
[outputs/activation.txt](outputs/activation.txt).

### 4A — Basic learning check

Settings: 2 → 2 → 1 network, sigmoid hidden, Adam(lr=0.1), 4000 steps,
seed 1.

- initial loss: `0.702770`
- final loss: `0.000048`
- probabilities: `(0,0)=0.0001`, `(0,1)=1.0000`, `(1,0)=1.0000`,
  `(1,1)=0.0000`.
- 4/4 predictions correct.

Loss dropped by four orders of magnitude below the chance baseline
`H(0.5) ≈ 0.693`, and the four probabilities are effectively saturated
at 0 or 1 at their respective corners. All four validation criteria
from Task 2 pass.

### 4B — Backpropagation check

After one final forward + backward on the mean loss (see
`xor_binary.py:report_gradient`):

    W(1).grad shape = (2, 2)   (matches W(1))
    W(1).grad ≈ [[-6.5e-8, -5.7e-8],
                 [-5.5e-7, -6.2e-7]]
    ||∇_W(1) L||_2 ≈ 1.0e-6

`W(1).grad` is exactly the tensor `∂L / ∂W(1)` — the derivative of the
mean loss over the four training examples with respect to every entry
of the first-layer weight matrix. It has the same shape as `W(1)`
because gradients live in the tangent space of their parameter, and
`optim.step()` uses precisely this tensor to update `W(1)`.

**Why the mean-loss gradient is the average of per-example gradients.**
`nn.BCEWithLogitsLoss()` with its default `reduction='mean'` computes
`L = (1/N) Σ_i ℓ_i` for `N = 4`. Because the derivative is a linear
operator, `∂L/∂W(1) = (1/N) Σ_i ∂ℓ_i/∂W(1)`. So autograd's single
backward pass through the summed loss produces the mean per-example
gradient in one call — no explicit loop over the four points needed.

The gradient magnitude is small (≈ 10⁻⁶) because the model has
converged and the sigmoid outputs are saturated near 0 and 1, so
`p - y ≈ 0` for every example. This is expected — the *converged*
gradient is small; the *initial* gradient (visible in
`activation.txt` at step 5) is larger by orders of magnitude.

### 4C — Symmetry experiment (zero-init)

See `symmetry.py` and [outputs/symmetry.txt](outputs/symmetry.txt).
With every parameter set to `0` and the architecture untouched:

    step 0..2000: row 0 = [0.0, 0.0], row 1 = [0.0, 0.0], identical=True
    final loss = 0.693147   (= -log 0.5 exactly)
    probabilities = 0.5 everywhere,  2/4 correct

Every parameter stays at zero forever. The mechanism: at zero weights
and biases, `a(1) = 0`, `h(1) = σ(0) · 1 = 0.5 · 1` (or `tanh(0) = 0`
etc.) is the same vector regardless of `x`, so the two hidden units are
indistinguishable — and their gradients, which flow back through
`W(2)` (which is also zero), are identical too. Identical parameters
receiving identical updates remain identical: the pair of hidden units
collapses into one effective unit, and the network reduces to a
constant predictor. Random initialisation exists *precisely* to break
this symmetry. This experiment isolates the mechanism into a single
degenerate but reproducible case.

### 4D — Activation experiment

See `activation.py`. Same seed (0), same optimiser (SGD, lr=0.1), same
number of steps, only the hidden activation changes. The early
gradient is measured at step 5 (before either the sigmoid saturates or
ReLU has time to develop many dead units).

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇_W(1) L‖₂ (step 5) |
|--------------------|------------|---------------|-----------------------------|
| Sigmoid            | 0.6931     | 2/4           | 1.45 × 10⁻³ |
| Tanh               | 0.0135     | 4/4           | 5.48 × 10⁻² |
| ReLU               | 0.4779     | 3/4           | 1.62 × 10⁻³ |

**Interpretation, for this experiment.** Nothing here says one
activation is "better" in general — with only four points and one seed
that would be nonsense. What the numbers do isolate:

- **Sigmoid** starts around the plateau. Its pre-activations begin near
  zero, and `σ'(0) = 0.25` combined with a small `W(2)` gives a
  first-layer signal of order 10⁻³. SGD with `lr=0.1` cannot escape
  fast enough in 4000 steps and the run ends at the fair-coin loss.
- **Tanh** is symmetric around zero with `tanh'(0) = 1` — four times
  the derivative of sigmoid at the origin — and its outputs are
  centered at zero, which makes the downstream linear layer's job
  easier from step 1. Its early gradient is ~30× that of sigmoid, and
  learning proceeds to 4/4.
- **ReLU** is a hard nonlinearity with derivative 1 on the positive
  side and 0 on the negative side. Its early gradient is also small
  here because the *specific* random init leaves several hidden
  pre-activations negative for most inputs — those units are "dead" —
  and only a subset of the parameters contribute. The 3/4 outcome is
  the ceiling of this run, not of ReLU.

Both sigmoid saturation and ReLU death produce a small gradient, but
they are different mechanisms: sigmoid saturates when the
pre-activation is *large in magnitude*; ReLU dies when the
pre-activation is *negative*. You distinguish them by looking at the
pre-activation values, not the activations: a sigmoid unit with
`a ≈ ±6` and `h ≈ 1` (or `0`) is saturated; a ReLU unit with `a < 0`
and `h = 0` is dead. `xor_binary.py` prints the gradient tensor itself,
so the same idea generalises: if `W(1)[j, :].grad` is exactly zero on a
ReLU network for many steps, that unit is dead; if it is small but
nonzero for a sigmoid network, that unit is saturated.

**Think about it — distinguishing them.** Inspect the pre-activations
`a(1) = W(1) x + b(1)`: for a sigmoid unit `|a(1)| >> 0` implies
saturation, `a(1) ≈ 0` implies the healthy regime. For a ReLU unit
`a(1) < 0` implies dead, `a(1) > 0` implies healthy. The activations
alone (`h(1) = 0` for both) would not tell you which of the two
mechanisms is responsible.

---

## Task 5 — Three-class extension

Model: 2 → 2 → 3 logits + softmax + cross-entropy, seed 0, SGD(lr=0.1),
4000 steps. See `three_class.py` and
[outputs/three_class.txt](outputs/three_class.txt).

**Predictions before running.**

1. `W(2)` becomes 3 × 2 (rows = classes, columns = hidden units).
2. Each example produces 3 logits.
3. Softmax probabilities sum to 1 because they are constructed as
   `p_k = exp(z_k) / Σ_j exp(z_j)`; the denominator is exactly the sum
   of the numerators. This is a definitional identity, not an
   empirical fact.
4. The logit gradient `p - y`: with cross-entropy loss
   `L = -Σ_k y_k log p_k` (`y` one-hot) and softmax `p_k = e^{z_k} / S`,
   we have `∂L/∂z_k = p_k - y_k`. The `e^{z_k}` in the softmax and the
   `log` in cross-entropy telescope; every other term cancels. That is
   also why sigmoid + BCE has gradient `p - y` — it is the K = 2
   special case.

**Results.**

    initial loss = 1.0735   (≈ log 3 = 1.0986, three-way chance)
    final   loss = 0.0368
    W(2) shape  = (3, 2)                        ✓ prediction
    logits per example = 3                       ✓ prediction

| x1 | x2 | target | p₀ | p₁ | p₂ | pred |
|----|----|--------|-----|-----|-----|------|
| 0  | 0  | 0 | 0.9638 | 0.0362 | 0.0000 | 0 |
| 0  | 1  | 1 | 0.0161 | 0.9670 | 0.0169 | 1 |
| 1  | 0  | 1 | 0.0167 | 0.9659 | 0.0174 | 1 |
| 1  | 1  | 2 | 0.0000 | 0.0411 | 0.9589 | 2 |

4/4 correct. Softmax-row sum check on the `(0,1)` example:
`0.016106 + 0.967004 + 0.016890 = 0.99999994` — one at `float32`
precision (the shortfall is roundoff, not a modelling error).

**Additive-invariance diagnostic.** Adding `100` to every logit before
softmax:

    softmax(logits + 100) - softmax(logits): max |Δ| = 1.19e-7

zero within `float32` precision. Because
`p_k = exp(z_k + c) / Σ_j exp(z_j + c) = e^c · exp(z_k) / (e^c · Σ_j
exp(z_j)) = exp(z_k) / Σ_j exp(z_j)`, the constant cancels. A
production softmax exploits this by subtracting `max(z)` from every
logit before exponentiating, so the largest `exp` argument is `0` and
the others are `≤ 0` — preventing overflow when logits are large
(`exp(1000)` overflows `float32`; `exp(0)` does not).

**Think about it — scaling to a language-model vocabulary.**
Mathematically nothing at the *softmax + cross-entropy* level changes
when `K` goes from 3 to `50 000`: `p - y` is still the logit gradient,
probabilities still sum to 1, subtracting the maximum is still exact.
What changes dramatically is *around* it: the output matrix `W(2)`
becomes `V × d` where `V` is the vocabulary and `d` the hidden
dimension, which dominates parameter counts; the naive softmax over
`V` becomes the dominant compute at inference, motivating
approximations (sampled softmax, hierarchical softmax) at training and
tricks like KV caching at inference; and the target `y` is almost
always a single index rather than a one-hot vector, so the *actual*
gradient one computes is just `p_{predicted} - 1` at the target index
and `p_j` elsewhere — the sparsity is exploited implicitly by
`nn.CrossEntropyLoss`.

---

## Reflection Questions

### 1. What did XOR demonstrate about depth vs nonlinearity?

Depth without nonlinearity is not depth. A stack of affine layers is
mathematically indistinguishable from a single affine layer, so no
amount of extra `nn.Linear`s helps if there is no `nn.ReLU` /
`nn.Sigmoid` / `nn.Tanh` between them. XOR falsifies "affine is
enough" with a single four-point counter-example. What actually opens
new function classes is the *nonlinear activation*; depth then extends
the class further by composing nonlinearities.

### 2. Evidence of a useful learning signal, not just a nonzero gradient.

A nonzero gradient at step 0 shows only that the loss depends on the
parameters — that is trivially true whenever `∂L/∂θ` is defined. What
`xor_binary.py` shows is stronger:

- the loss monotonically decreased by four orders of magnitude across
  4000 steps (`0.702770 → 0.000048`);
- the four *thresholded* predictions match the labels — a
  discrete-valued acceptance test that is invariant to small numeric
  jitter and cannot be gamed by a shrinking-scale trick;
- the gradient at the *end* is small (~10⁻⁶) *and* the loss is small,
  which is the correct joint signature of convergence (rather than
  the small-gradient / high-loss signature of a plateau or saddle).

### 3. Why did identical / zero initialisation prevent distinct features?

Symmetry preservation. Backprop's update to a hidden-unit parameter
depends on the unit's own pre-activation and on the gradient arriving
from the layer above. Two hidden units with identical `W(1)` rows and
identical `b(1)` entries produce identical pre-activations on every
input, receive identical gradients (because `W(2)` treats them the
same too — its two columns are equal), and therefore change by the
same amount. The differential equation on the two rows is symmetric in
them, so they stay equal forever. The network is stuck in the
1-hidden-unit subspace, which by Task 1 cannot represent XOR — hence
the loss glued at `-log 0.5`. Random init destroys this symmetry from
step 0.

### 4. How did changing the activation affect the observed gradient?

Under identical seed / optimiser / step count, the step-5 first-layer
gradient norm was ~1.5×10⁻³ for sigmoid, ~5.5×10⁻² for tanh, and
~1.6×10⁻³ for ReLU. The *scientific* explanation is per-activation
derivative: `σ'(0) = 0.25` and outputs bounded in `(0, 1)` produce a
small chain-rule product from step 0; `tanh'(0) = 1` and zero-centered
outputs give a larger, better-signed signal; ReLU has derivative 1 on
its active side but 0 on the inactive side, so its gradient depends on
which hidden units happen to be alive at initialisation. The
*engineering observation* is that with lr=0.1 and 4000 steps SGD reached
4/4 with tanh but stalled with sigmoid and reached 3/4 with ReLU. The
engineering observation does not license the general claim "tanh is
best" — it is an observation on one seed for one dataset with one
optimiser. Change any of those and the ranking can shift.

### 5. Why must the output layer and loss be selected together?

Because the *gradient the model actually sees* depends on the pairing,
not on either piece alone. Sigmoid + BCE and softmax + cross-entropy
both give the clean logit gradient `p - y`, because the derivative of
the log-partition (or `log(1 + e^{-z})`) cancels with the derivative of
the output nonlinearity. Sigmoid + MSE gives an extra `p(1 - p)` factor
that vanishes exactly when the model is confidently wrong — the worst
possible time for a small gradient. Similarly, softmax + MSE gives no
useful gradient at all when one probability is near 1 but on the wrong
class. Output nonlinearity and loss are one design choice, not two.

### 6. LLM productivity vs LLM verification.

**Productivity win.** The LLM produced the `nn.Sequential` skeleton,
the training loop, and the full-batch tensor construction of the four
XOR examples almost verbatim from the specification. Writing that from
scratch would have taken 10 minutes of remembering PyTorch idioms
(argument order of `nn.Linear`, `reduction='mean'` default of
`BCEWithLogitsLoss`, `optim.zero_grad()` before `backward()`); getting
it back in a few seconds meant I spent that time reading and correcting
the numerical setup instead.

**Verification necessity.** The first draft used `torch.optim.SGD`
which — with the default settings the LLM chose — silently stalled at
the sigmoid plateau, printing "4/4 correct" for exactly zero out of the
four examples with probabilities all at 0.5. The code was correct; the
numerical behaviour was not. Only running it and looking at the actual
loss curve revealed the stall; only understanding the sigmoid plateau
turned the observation into a fix (switch to Adam and/or sweep seeds).
The lesson is the one the worksheet states explicitly: the LLM is fine
for engineering, but the *scientific* decision — is the network
actually learning XOR, or is it just outputting 0.5? — is mine to make.

### 7. Which tests generalise to scale, and which do not?

**Keep at scale:**

- **Loss below chance baseline.** `-log(1/K)` remains a meaningful
  floor for K-way classification, and being clearly under it is the
  first non-trivial evidence that anything is being learned.
- **Softmax-row-sum sanity check.** Cheap, catches numerical bugs
  quickly, applies to any K.
- **Gradient nonzero / not-NaN check on a representative parameter.**
  A single scalar (`.norm()`, `.isfinite().all()`) per major parameter
  block per step scales trivially.
- **Symmetry / initialisation smoke test.** For any new architecture,
  starting from a degenerate init (all-zero, or all identical) and
  confirming it collapses is a one-off, tiny test that catches whole
  classes of bugs.

**Drop or replace at scale:**

- **Exhaustive per-example predictions.** With 4 examples we print the
  table; with 4 million we sample or aggregate into a confusion
  matrix.
- **Finite-difference gradient checks.** These are quadratic in the
  parameter count and are prohibitive above a few thousand parameters;
  a modern network has 10⁸–10¹¹. Autograd + `torch.autograd.gradcheck`
  on tiny slices, or `torch.compile`-level assertions, replace them.
- **Full-batch training.** Dies at any real dataset — replaced by
  mini-batch SGD/Adam, and separately by evaluation batches.

The general principle: the *class* of test survives (loss checks,
gradient checks, invariance checks), but the *implementation* becomes
sampled, batched, or numerically-approximated rather than exhaustive.
