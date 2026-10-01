"""Tests for the warehouse planner (Task 3)."""

from planner import (
    Action,
    WAREHOUSE_INITIAL,
    WAREHOUSE_GOAL,
    plan_bfs,
    warehouse_actions,
)


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def _validate(plan, initial, goal_pos):
    """Independently replay the plan on the initial state."""
    state = set(initial)
    for a in plan:
        if not a.pos_pre.issubset(state) or not a.neg_pre.isdisjoint(state):
            return False, state
        state -= a.neg_eff
        state |= a.pos_eff
    return goal_pos.issubset(state), state


def test_A_solvable() -> None:
    print("Test A - solvable warehouse")
    r = plan_bfs(WAREHOUSE_INITIAL, warehouse_actions(True), WAREHOUSE_GOAL)
    check("plan found", r.found)
    valid, final = _validate(r.plan, WAREHOUSE_INITIAL, WAREHOUSE_GOAL)
    check("independent replay confirms the plan", valid)
    check("package delivered to C", "At(Package,C)" in final)
    check("BFS gave shortest plan of length 4", len(r.plan) == 4)


def test_B_impossible() -> None:
    print("Test B - PickUp removed, package stays at A")
    r = plan_bfs(WAREHOUSE_INITIAL, warehouse_actions(False), WAREHOUSE_GOAL)
    check("planner reports failure", not r.found and r.plan == [])


def test_C_irrelevant_actions() -> None:
    print("Test C - robot reaching C is not package reaching C")
    acts = warehouse_actions(False)  # no PickUp
    # Robot can reach C; package cannot.
    robot_goal = plan_bfs(WAREHOUSE_INITIAL, acts, frozenset({"At(Robot,C)"}))
    pkg_goal = plan_bfs(WAREHOUSE_INITIAL, acts, frozenset({"At(Package,C)"}))
    check("robot can reach C by itself", robot_goal.found)
    check("package cannot reach C without PickUp", not pkg_goal.found)


def test_preconditions_enforced() -> None:
    print("Test D - preconditions actually block ungrounded Drops")
    acts = warehouse_actions(True)
    # Try to achieve Holding(Package) in C immediately -- must require PickUp first.
    r = plan_bfs(WAREHOUSE_INITIAL, acts, frozenset({"Holding(Package)"}))
    check("plan found to pick package up", r.found)
    valid, _ = _validate(r.plan, WAREHOUSE_INITIAL, frozenset({"Holding(Package)"}))
    check("replay confirms every precondition held", valid)


def test_initial_already_goal() -> None:
    print("Test E - trivial case: initial already satisfies the goal")
    r = plan_bfs(WAREHOUSE_INITIAL, warehouse_actions(True), frozenset({"At(Robot,A)"}))
    check("empty plan", r.found and r.plan == [])


if __name__ == "__main__":
    for t in (
        test_A_solvable,
        test_B_impossible,
        test_C_irrelevant_actions,
        test_preconditions_enforced,
        test_initial_already_goal,
    ):
        t()
    print("\nAll tests passed.")
