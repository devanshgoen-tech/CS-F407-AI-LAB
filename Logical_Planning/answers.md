# Logical Planning Lab — Written Answers

Worksheet: `logic_lab_ex.pdf` — *Logical Reasoning for Planning*.

## Task 0 — Understanding the planning problem

- **Initial state** `I = { At(Robot,A), At(Package,A) }`.
- **Goal** `G = { At(Package,C) }`.
- **Actions**: `Move(X,Y)` for every connected pair `(X,Y)`,
  `PickUp(Package,L)` and `Drop(Package,L)` for every location `L`.
  See the formal signatures in [`planner.py`](planner.py).

| Action | Positive preconditions | Negative preconditions | Positive effects | Negative effects |
|--------|------------------------|------------------------|------------------|------------------|
| `Move(X,Y)` | `At(Robot,X)` | – | `At(Robot,Y)` | `At(Robot,X)` |
| `PickUp(Package,L)` | `At(Robot,L)`, `At(Package,L)` | – | `Holding(Package)` | `At(Package,L)` |
| `Drop(Package,L)` | `At(Robot,L)`, `Holding(Package)` | – | `At(Package,L)` | `Holding(Package)` |

**Is `PickUp(Package,A)` applicable in `I`?** Yes — both of its positive
preconditions, `At(Robot,A)` and `At(Package,A)`, hold in `I`.

**Is `Drop(Package,C)` applicable in `I`?** No — `At(Robot,C)` is false
in `I`, and `Holding(Package)` is false too.

## Task 1 — A plan by hand

```
S0 : { At(Robot,A), At(Package,A) }
   -> PickUp(Package,A)
S1 : { At(Robot,A), Holding(Package) }
   -> Move(A,B)
S2 : { At(Robot,B), Holding(Package) }
   -> Move(B,C)
S3 : { At(Robot,C), Holding(Package) }
   -> Drop(Package,C)
S4 : { At(Robot,C), At(Package,C) }       <-- goal reached
```

The automated BFS planner reproduces exactly this four-action plan (see
`outputs/run.txt`).

## Task 2 — Prompt used

> *Write a Python STRIPS-style planner. Represent a state as a set of
> ground propositions. Each action should carry positive/negative
> preconditions and positive/negative effects. An action is applicable
> when all positive preconditions hold and all negative preconditions
> are absent. Applying an action removes negative effects and adds
> positive effects. Use breadth-first search over states to find a
> sequence of actions reaching a goal expressed as the required
> positive and absent-negative propositions. Report `No plan found` if
> the frontier is exhausted. Also build the warehouse domain
> (three locations A–B–C, a robot and a package) and run it.*

**Where the specification ideas appear in the generated code**
(`planner.py`):

- Preconditions → `Action.applicable(state)`.
- Effects → `Action.apply(state)`.
- Goal → `goal_satisfied(state, goal_pos, goal_neg)` and the early-exit
  check inside `plan_bfs`.
- BFS → the `deque` frontier, `came_from` map, and reconstruction.

## Task 3 — Test results (see `outputs/tests.txt`)

- **Test A — solvable.** Plan of length 4, independently replayed on
  the initial state to confirm every precondition held and that the
  final state contains `At(Package,C)`.
- **Test B — impossible.** With `PickUp` removed, the planner correctly
  reports no plan rather than inventing one.
- **Test C — irrelevant actions.** The robot can still reach `C` on its
  own (plan found when the goal is `At(Robot,C)`), but *the package*
  cannot reach `C` without `PickUp`. The planner never confuses
  `At(Robot,C)` with `At(Package,C)` because the propositions are
  distinct.

| Test | Initial | Goal | Plan found? | Plan length |
|------|---------|------|-------------|-------------|
| A | `{At(Robot,A), At(Package,A)}` | `{At(Package,C)}` | yes | 4 |
| B | same | `{At(Package,C)}`, `PickUp` removed | no | – |
| C-pkg | same | `{At(Package,C)}`, `PickUp` removed | no | – |
| C-rbt | same | `{At(Robot,C)}`, `PickUp` removed | yes | 2 |

## Task 4 — Logic and search

Completed description:

```
Current state
    |
Check action preconditions                 <-- logical reasoning:
    |                                           S |= Preconditions(a)
Preconditions satisfied?
    |
Generate successor state                   <-- state update:
    |                                           S' = (S \ neg_eff) U pos_eff
Search over alternatives                   <-- BFS frontier
    |
Goal?                                      <-- goal_satisfied(S')
```

- **Logical reasoning** decides whether an action can fire at a given
  state and how the state changes. In code: `Action.applicable` and
  `Action.apply`.
- **Search** decides *which* applicable action to try next and keeps
  track of alternatives. In code: BFS in `plan_bfs`.

"Logic determines what is possible; search determines what to try."

## Task 5 — Can the LLM verify its own plan?

The LLM can write an English justification for each action, but it is
not a substitute for executing the plan.  `tests.py::_validate` is the
honest check: it replays the plan on a *fresh* copy of the initial
state, enforcing every precondition, and confirms the goal literally
holds after the final action.  This is the "independent verification"
asked for by Task 5: the Python planner generates, and the replay
(and the optional Prolog file) checks.

Which should you trust more?  The independent replay, every time —
because a textual explanation is reconstructed from the same model that
produced the plan, and it can be fluent and still wrong. The replay has
no access to the LLM at all.

## Section 7 — Optional Prolog extension

See [`planner.pl`](planner.pl).

### Task 6 — Prolog as a plan verifier

```prolog
?- can_move(a, b).   % true
?- can_move(a, c).   % false
```

**Why.** `can_move(X,Y)` is derived from the fact `connected(X,Y)`.
`connected(a,b)` is a fact in the KB, so Prolog answers `true` by
direct lookup. There is no `connected(a,c)` fact and no rule allowing
Prolog to derive it, so by the closed-world assumption the query
`can_move(a,c)` fails.

**Correspondence with logic.** The rule
`can_move(X,Y) :- connected(X,Y).` is exactly the implication
`Connected(X,Y) -> CanMove(X,Y)` under universal quantification over
`X,Y`. Prolog is answering "does this instance of the implication
match a known fact?".

### Task 7 — Checking a proposed plan

```prolog
?- valid_move(a, b).   % true
?- valid_move(b, c).   % true
?- valid_move(a, c).   % false
```

So the sequence `Move(a,b), Move(b,c)` is supported by the KB, but a
single-step `Move(a,c)` is **not** — it would need an edge the KB does
not contain. Prolog is thus acting as a cheap, independent verifier of
movement edges proposed by the Python planner.

The `path/2` rule in `planner.pl` additionally lets Prolog answer the
reachability question `?- path(a, c, P).` with `P = [a, b, c]`, which
confirms that `Move(a,c)` can be *simulated* by two legal moves even
though the single move is itself invalid.

### Task 8 — Logical reasoning in Prolog

```
wet_road.
slippery       :- wet_road.
reduce_speed   :- slippery.

?- reduce_speed.  % true
```

Chain of inference:

`wet_road`  ⇒  (`wet_road → slippery`)  ⇒  `slippery`  ⇒  (`slippery → reduce_speed`)  ⇒  `reduce_speed`.

### Reflection on Prolog

1. **Fact vs rule.** A fact asserts a ground proposition
   (`penguin(polly).`); a rule asserts an implication
   (`bird(X) :- penguin(X).`).
2. **Query as entailment.** `?- Q.` asks whether `KB ⊨ Q`, decided
   by Prolog's resolution procedure (SLD-resolution with the closed
   world assumption).
3. **Why verify a Python plan with Prolog.** Because the verification
   logic is small, declarative, and independent of the code that
   produced the plan — if both agree, that is corroboration from two
   different sources.
4. **Independent verifier + LLM-generated plan.** The LLM may hallucinate
   actions or confuse goals; an independent logical checker refuses any
   step that is not grounded in the knowledge base, which is exactly
   what we want when the plan was generated with help from an LLM.

## Reflection questions (Section 5)

1. **Why specify preconditions and effects first.** Without them the
   LLM has no definition of what "applicable" means; the generated
   code would then invent its own semantics.
2. **Example error without precondition checks.** The planner would
   happily `Drop(Package,C)` while the package is still at `A`,
   magically relocating it.
3. **"Looks reasonable" vs valid.** A plan can be fluent and the wrong
   length, or miss a hidden negative precondition. Only replay against
   the explicit state transitions proves validity.
4. **LLM contribution.** The planner skeleton, the choice of `frozenset`
   for states, and a first-cut BFS. I then added the goal structure
   with both positive and negative literals, path reconstruction, and
   the tests.
5. **Verified independently.** The `_validate` replay in `tests.py`
   and the Prolog edge-checker in `planner.pl`.
6. **Where logical reasoning is used.** Deciding `S ⊨ Preconditions(a)`
   and computing `S' = (S \ neg_eff) ∪ pos_eff`.
7. **Planning vs search.** Planning *is* search over states, where the
   transitions are produced by applying logical actions. The frontier,
   closed set, and goal test are the same mechanisms seen in Search Lab,
   now operating on sets of propositions rather than grid cells.

## Takeaway

```
Understand -> Specify -> Generate -> Execute -> Verify
```

The planner was *generated* with the LLM's help; it is *verified* by
`tests.py::_validate` and (optionally) by the Prolog rules.
