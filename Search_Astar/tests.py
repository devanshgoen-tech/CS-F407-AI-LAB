"""Tests for the A* / BFS implementations (Task 3)."""

from search_agent import (
    WAREHOUSE_MAP,
    a_star,
    bfs,
    manhattan,
    euclidean,
    manhattan_times_two,
    zero,
    parse_map,
)


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test1_original_warehouse() -> None:
    print("Test 1 - original warehouse")
    r = a_star(WAREHOUSE_MAP, manhattan)
    _, start, goal = parse_map(WAREHOUSE_MAP)
    check("path found", r.found)
    check("starts at S", r.path[0] == start)
    check("ends at G", r.path[-1] == goal)


def test2_trivial_case() -> None:
    print("Test 2 - trivial adjacent goal")
    tiny = "\n".join([
        "#####",
        "#SG##",
        "#####",
    ])
    r = a_star(tiny, manhattan)
    check("one-step solution", r.found and r.path_length == 1)
    check("action is Right", r.actions == ["Right"])


def test3_no_solution() -> None:
    print("Test 3 - unreachable goal")
    sealed = "\n".join([
        "#######",
        "#S....#",
        "###.###",
        "#...#G#",
        "#######",
    ])
    r = a_star(sealed, manhattan)
    check("failure reported, no path", not r.found and r.path == [])


def test4_shortest_path() -> None:
    print("Test 4 - map with alternative paths")
    alt = "\n".join([
        "#######",
        "#S...G#",
        "#.###.#",
        "#.....#",
        "#######",
    ])
    r_astar = a_star(alt, manhattan)
    r_bfs = bfs(alt)
    check("A* returns a shortest path", r_astar.found and r_astar.path_length == 4)
    check("BFS and A* agree on length", r_bfs.path_length == r_astar.path_length)


def test5_astar_optimal_matches_bfs() -> None:
    print("Test 5 - A* with admissible h matches BFS optimality")
    r_astar = a_star(WAREHOUSE_MAP, manhattan)
    r_bfs = bfs(WAREHOUSE_MAP)
    check("both find a path", r_astar.found and r_bfs.found)
    check("same shortest path length", r_astar.path_length == r_bfs.path_length)
    check("A* expands no more than BFS", r_astar.expanded <= r_bfs.expanded)


def test6_heuristic_variants() -> None:
    print("Test 6 - heuristic variants still terminate")
    for name, h in [("zero", zero), ("euclidean", euclidean), ("2x manhattan", manhattan_times_two)]:
        r = a_star(WAREHOUSE_MAP, h)
        check(f"{name}: path found", r.found)


if __name__ == "__main__":
    for t in (
        test1_original_warehouse,
        test2_trivial_case,
        test3_no_solution,
        test4_shortest_path,
        test5_astar_optimal_matches_bfs,
        test6_heuristic_variants,
    ):
        t()
    print("\nAll tests passed.")
