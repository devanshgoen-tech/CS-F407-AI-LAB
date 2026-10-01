"""Tests for the goal-based warehouse agent."""

from warehouse_agent import goal_based_search, WAREHOUSE_MAP, parse_map


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_original_map() -> None:
    print("test_original_map")
    result = goal_based_search(WAREHOUSE_MAP)
    check("path found", result.found)
    _, start, goal = parse_map(WAREHOUSE_MAP)
    check("path starts at S", result.path[0] == start)
    check("path ends at G", result.path[-1] == goal)
    check("path length matches action count", len(result.actions) == len(result.path) - 1)


def test_trivial_adjacent() -> None:
    print("test_trivial_adjacent")
    tiny = "\n".join([
        "#####",
        "#SG##",
        "#####",
    ])
    result = goal_based_search(tiny)
    check("one-step path found", result.found and len(result.actions) == 1)
    check("single Right action", result.actions == ["Right"])


def test_no_solution() -> None:
    print("test_no_solution")
    sealed = "\n".join([
        "#######",
        "#S....#",
        "###.###",
        "#...#G#",
        "#######",
    ])
    result = goal_based_search(sealed)
    check("no path reported", not result.found)
    check("search terminates (finite expansions)", result.expanded >= 1)


def test_shortest_path() -> None:
    print("test_shortest_path")
    # Two alternative routes: a direct one (length 4) and a longer detour.
    alt = "\n".join([
        "#######",
        "#S...G#",
        "#.###.#",
        "#.....#",
        "#######",
    ])
    result = goal_based_search(alt)
    check("alternative map solved", result.found)
    check("BFS finds length-4 shortest path", len(result.actions) == 4)


if __name__ == "__main__":
    for t in (test_original_map, test_trivial_adjacent, test_no_solution, test_shortest_path):
        t()
    print("\nAll tests passed.")
