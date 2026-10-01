# Search Lab — Written Answers

Worksheet: `search_lab_ex.pdf` — *Search and A\*: using an LLM as an
engineering assistant*.

## Task 0 — Formulating the search problem

| Component | Specification |
|-----------|---------------|
| State `S` | `(row, col)` position of the robot in the grid. |
| Actions `A` | `{Up, Down, Left, Right}`. |
| Transition `T` | `T(s, a) = s + delta(a)` if the target cell is inside the grid and not `#`; otherwise `a` is inapplicable at `s`. |
| Initial state `s0` | Position of the cell marked `S`. |
| Goal set `G` | `{ position of the cell marked G }`. |
| Cost `c` | `c(s, a, s') = 1` for every move. |

Short answers:

- **(a) Information needed to specify a state.** Just the robot's
  `(row, col)` — the warehouse map itself is static, so no extra state
  is required.
- **(b) What makes an action invalid?** It would leave the grid or step
  onto a `#` cell.
- **(c) Deterministic?** Yes. For a given `(s, a)` the successor is
  fixed; there is no randomness.
- **(d) What constitutes a solution?** A finite sequence of actions
  `a1, …, ak` whose cumulative transitions take `s0` to a state in `G`,
  avoiding obstacles at every intermediate step.

## Task 1 — Agent plan

1. States are Python `tuple[int, int]`.
2. The warehouse is a `list[str]`; `grid[r][c]` is `'#'`, `.`, `S`, or `G`.
3. Valid actions: iterate over the four offsets; a move is valid iff
   the target cell is inside the grid and not `#`.
4. Goal recognition: `current == goal_position`.
5. Frontier for A*: a `heapq` of `(f, tie, position)`. Also required:
   `g_score[position]` and a `came_from[position] = (parent, action)`
   dictionary.
6. Path reconstruction: follow `came_from` from the goal back to the
   start, reversing the collected list of positions and actions.

Reported at termination: whether a solution was found, the path, the
path length, and the number of states expanded.

## Task 2 — Prompt used

> *Implement A\* search in Python for a grid-world warehouse robot. The
> grid is an ASCII map with `#` obstacles, `.` free cells, and a single
> `S` and `G`. Actions are Up/Down/Left/Right, each with cost 1. Use
> Manhattan distance as the heuristic. The function should return the
> action sequence, the path length, and the number of states expanded,
> and should use `heapq` for the open set and a closed set to avoid
> re-expanding states.*

I made two targeted follow-ups: (i) store `(parent, action)` in
`came_from` so action names could be printed, and (ii) tie-break on an
increasing counter so that heap pops are deterministic for ties.

## Task 3 — Test results (see `outputs/tests.txt`)

| Test | Outcome |
|------|---------|
| Original warehouse | path found, length 40, 64 states expanded |
| Trivial adjacent (`#SG##`) | single `Right` action, as expected |
| Sealed-off goal | failure reported, no infinite loop |
| Alternative-paths map | shortest path of length 4; BFS agrees |

## Task 4 — Where each concept appears

| Concept | Where in the code |
|---------|-------------------|
| State | `Position = tuple[int, int]` used throughout `search_agent.py` |
| Action | `ACTIONS` list in `search_agent.py` |
| Transition | `neighbours()` in `search_agent.py` |
| Goal test | `if current == goal:` inside `a_star` / `bfs` |
| g(n) | `g_score[current] + 1` in `a_star` |
| h(n) | `manhattan`, `euclidean`, `zero`, `manhattan_times_two` |
| f(n) | `f = tentative_g + h(nxt, goal)` in `a_star` |
| Frontier | `open_heap` (min-heap) in `a_star`; `deque` in `bfs` |
| Visited | `closed` set in `a_star`; the `came_from` dict in `bfs` |
| Path reconstruction | `_reconstruct()` |

Answers:

- **(a) Frontier data structure.** `heapq` priority queue keyed by
  `f(n)`, with an increasing tie-breaker.
- **(b) Next state to expand.** The one with the smallest `f`.
- **(c) Heuristic calculation.** In the `h` function (passed in as a
  parameter) called every time a successor's `f` is computed.
- **(d) Is f = g + h explicit?** Yes:
  `f = tentative_g + h(nxt, goal)`.
- **(e) Avoiding repeated exploration.** The `closed` set skips any
  position popped from the heap twice, and the g-score check prevents
  pushing a worse path to an already-reached cell.

## Task 5 — A\* vs BFS

Running `search_agent.py` prints:

| Measure | BFS | A\* (Manhattan) |
|---------|------|-----------------|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

Answers:

- (a) Both find a solution.
- (b) Both paths have length 40 (A\* with an admissible heuristic is
  optimal under unit costs).
- (c) **On *this* map A\* does not expand fewer states**, because the
  maze has essentially one corridor from `S` to `G`: the entire
  reachable region is 64 cells and every algorithm has to touch all of
  them. On the smaller alternative-paths map from Task 4, A\* and BFS
  also agree on length, but on an open map with many detours A\* would
  expand strictly fewer.
- (d) A\* can expand fewer states because `f = g + h` biases expansion
  toward the goal: states that would clearly take longer than the
  current best candidate are deferred.

## Task 6 — Heuristic experiments

| Heuristic | Path found | Path length | States expanded |
|-----------|-----------|-------------|-----------------|
| `h = 0` (uniform cost) | yes | 40 | 64 |
| Manhattan (admissible) | yes | 40 | 64 |
| Euclidean (admissible, weaker) | yes | 40 | 64 |
| `2 × Manhattan` (inadmissible) | yes | 40 | 64 |

Observations:

- The maze is so constrained that the heuristic cannot prune anything:
  the only route is forced. In that regime, A\* degenerates to the
  behaviour of uniform-cost search no matter what heuristic is used.
- With `h = 0`, A\* is exactly uniform-cost search; path length stays
  optimal.
- With Euclidean (admissible but a weaker lower bound than Manhattan on
  a 4-connected grid), optimality is preserved and expansion is
  identical here.
- With `2 × Manhattan`, the heuristic is **inadmissible**: on a map
  with alternative routes, this can lead A\* to return a non-optimal
  path. On the single-corridor warehouse there are no alternative
  routes, so the result is incidentally still length 40 — this is why
  one must test such claims on a map with real choice (see
  `tests.py::test4_shortest_path`).

### Think About It — admissibility

If `h` is admissible (`h(n) ≤ h*(n)`), A\* is guaranteed to return an
optimal path. If `h` is inflated past `h*`, the search becomes more
greedy: it reaches *a* goal quickly but may miss a shorter path. The
single-corridor map hides this; a map with branches exposes it.

## Task 7 — Reflection on the LLM-generated agent

- *Correct immediately:* the overall A\* skeleton, Manhattan distance,
  the use of `heapq` for the open set.
- *Problems found:* initial version stored only the parent in
  `came_from`, so I could not print the action names; it also had no
  tie-breaker, which made heap behaviour non-deterministic on equal
  `f`-values.
- *How discovered:* by running the trivial adjacent map and the
  sealed-off map, and by inspecting the printed output.
- *Terminology that needed thought:* the LLM's use of the term "closed
  set" was fine, but the explanation of why Manhattan is admissible on
  a 4-connected grid needed me to re-derive it from the move set.
- *My changes:* `(parent, action)` tuples in `came_from`; explicit
  g-score comparison for re-opening; the `heuristic` argument to
  `a_star`; tests; the four heuristics for Task 6.
- *Most useful tests:* the sealed-off map (ruled out infinite loops),
  the alternative-paths map (proved optimality end-to-end), and
  comparing BFS and A\* on the same warehouse.
- *Trust without testing?* No. The map structure determines whether A\*
  will expand fewer states than BFS, and the Task 6 experiments only
  make sense when run on a map that actually offers choices.

## Final reflection

1. **Formulating before coding.** Without `(S, A, T, s0, G, c)` the
   code is not actually solving any particular problem — testing is
   just hoping. The formulation is also what tells us why BFS and A\*
   are even applicable.
2. **A\* as informed search.** A\* uses `h(n)`, an estimate of the
   remaining cost, in addition to `g(n)`. "Informed" means that domain
   knowledge (positions on a grid, the fact that a step changes one
   coordinate by one) is injected into the search strategy.
3. **Choice of heuristic.** It controls both optimality and efficiency:
   admissibility preserves optimality, dominance (`h1 ≥ h2 ≥ 0` both
   admissible) decides which admissible heuristic expands fewer nodes.
4. **LLM contribution.** It accelerated the implementation (open set,
   g-scoring, closed set), but the engineering judgments —
   admissibility, tests, visualising the path — were mine.
5. **Risks of trusting unchecked LLM code.** Non-deterministic heap
   behaviour, silent off-by-one errors in bounds checks, and the
   seductive case of a map where every heuristic looks equally good.
