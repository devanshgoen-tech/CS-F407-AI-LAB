"""A tiny STRIPS-style planner.

Worksheet: logic_lab_ex.pdf.

A state is a frozen set of ground logical propositions represented as
strings (e.g. "At(Robot,A)").  An action has:

* a name,
* positive preconditions (propositions that must be in the state),
* negative preconditions (propositions that must NOT be in the state),
* positive effects (added to the state),
* negative effects (removed from the state).

An action is applicable in state S iff all its positive preconditions are
in S and none of its negative preconditions are in S.  Applying it
produces S' = (S \\ negative_effects) union positive_effects.

Planning is done by breadth-first search over states.  BFS guarantees
the shortest plan when every action has unit cost, and ``no plan found''
is reported when the frontier is exhausted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


State = frozenset[str]


@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: frozenset[str] = field(default_factory=frozenset)
    neg_pre: frozenset[str] = field(default_factory=frozenset)
    pos_eff: frozenset[str] = field(default_factory=frozenset)
    neg_eff: frozenset[str] = field(default_factory=frozenset)

    def applicable(self, state: State) -> bool:
        return self.pos_pre.issubset(state) and self.neg_pre.isdisjoint(state)

    def apply(self, state: State) -> State:
        return (state - self.neg_eff) | self.pos_eff


@dataclass
class PlanResult:
    found: bool
    plan: list[Action]
    trace: list[State]        # trace[i] = state after plan[:i]  (so trace[0] = initial)
    expanded: int

    def pretty(self) -> str:
        if not self.found:
            return "No plan found"
        lines = [f"Plan has {len(self.plan)} action(s); explored {self.expanded} state(s)."]
        for i, s in enumerate(self.trace):
            label = "S0" if i == 0 else f"S{i}"
            lines.append(f"  {label}: {sorted(s)}")
            if i < len(self.plan):
                lines.append(f"    -> {self.plan[i].name}")
        return "\n".join(lines)


def goal_satisfied(state: State, goal_pos: frozenset[str], goal_neg: frozenset[str] = frozenset()) -> bool:
    return goal_pos.issubset(state) and goal_neg.isdisjoint(state)


def plan_bfs(
    initial: State,
    actions: list[Action],
    goal_pos: frozenset[str],
    goal_neg: frozenset[str] = frozenset(),
    max_expansions: int = 10_000,
) -> PlanResult:
    """BFS over states; returns the first (shortest) plan reaching the goal."""
    if goal_satisfied(initial, goal_pos, goal_neg):
        return PlanResult(True, [], [initial], 0)

    frontier: deque[State] = deque([initial])
    came_from: dict[State, tuple[State, Action] | None] = {initial: None}
    expanded = 0

    while frontier:
        if expanded >= max_expansions:
            break
        current = frontier.popleft()
        expanded += 1

        for a in actions:
            if not a.applicable(current):
                continue
            successor = a.apply(current)
            if successor in came_from:
                continue
            came_from[successor] = (current, a)
            if goal_satisfied(successor, goal_pos, goal_neg):
                # Reconstruct.
                plan: list[Action] = []
                trace: list[State] = [successor]
                node: State = successor
                while True:
                    link = came_from[node]
                    if link is None:
                        break
                    prev, act = link
                    plan.append(act)
                    trace.append(prev)
                    node = prev
                plan.reverse()
                trace.reverse()
                return PlanResult(True, plan, trace, expanded)
            frontier.append(successor)

    return PlanResult(False, [], [], expanded)


# ---------------------------------------------------------------------------
# Warehouse domain
# ---------------------------------------------------------------------------

LOCATIONS = ("A", "B", "C")
CONNECTED = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def move(x: str, y: str) -> Action:
    return Action(
        name=f"Move({x},{y})",
        pos_pre=frozenset({f"At(Robot,{x})"}),
        pos_eff=frozenset({f"At(Robot,{y})"}),
        neg_eff=frozenset({f"At(Robot,{x})"}),
    )


def pickup(package: str, loc: str) -> Action:
    return Action(
        name=f"PickUp({package},{loc})",
        pos_pre=frozenset({f"At(Robot,{loc})", f"At({package},{loc})"}),
        pos_eff=frozenset({f"Holding({package})"}),
        neg_eff=frozenset({f"At({package},{loc})"}),
    )


def drop(package: str, loc: str) -> Action:
    return Action(
        name=f"Drop({package},{loc})",
        pos_pre=frozenset({f"At(Robot,{loc})", f"Holding({package})"}),
        pos_eff=frozenset({f"At({package},{loc})"}),
        neg_eff=frozenset({f"Holding({package})"}),
    )


def warehouse_actions(include_pickup: bool = True) -> list[Action]:
    actions: list[Action] = []
    for x, y in CONNECTED:
        actions.append(move(x, y))
    if include_pickup:
        for loc in LOCATIONS:
            actions.append(pickup("Package", loc))
    for loc in LOCATIONS:
        actions.append(drop("Package", loc))
    return actions


WAREHOUSE_INITIAL: State = frozenset({"At(Robot,A)", "At(Package,A)"})
WAREHOUSE_GOAL = frozenset({"At(Package,C)"})


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Task 1 - Manual plan (expected sequence) ===")
    manual = ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    print("  " + "\n  ".join(manual))

    print("\n=== Task 3 Test A - Solvable warehouse ===")
    result = plan_bfs(
        WAREHOUSE_INITIAL,
        warehouse_actions(include_pickup=True),
        WAREHOUSE_GOAL,
    )
    print(result.pretty())

    print("\n=== Task 3 Test B - Impossible (PickUp removed) ===")
    result_b = plan_bfs(
        WAREHOUSE_INITIAL,
        warehouse_actions(include_pickup=False),
        WAREHOUSE_GOAL,
    )
    print(result_b.pretty())

    print("\n=== Task 3 Test C - Irrelevant actions do not fool the goal ===")
    # Robot wanders to C without the package; goal is Package at C.
    irrelevant = plan_bfs(
        WAREHOUSE_INITIAL,
        warehouse_actions(include_pickup=False),  # no PickUp available
        WAREHOUSE_GOAL,
    )
    # Also show that the robot could reach C without carrying the package
    # if its position were the goal:
    robot_only = plan_bfs(
        WAREHOUSE_INITIAL,
        warehouse_actions(include_pickup=False),
        goal_pos=frozenset({"At(Robot,C)"}),
    )
    print("Package goal, no PickUp:")
    print(irrelevant.pretty())
    print("\nRobot-only goal (sanity: BFS still works):")
    print(robot_only.pretty())


if __name__ == "__main__":
    main()
