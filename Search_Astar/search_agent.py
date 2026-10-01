"""A* and BFS search for the warehouse robot navigation problem.

Worksheet: search_lab_ex.pdf.

The search problem (I, A, T, s0, G, c):

* S  — the set of (row, col) grid positions that are not walls.
* A  — the four moves Up / Down / Left / Right.
* T  — T(s, a) = s + delta(a) if that cell is free, else undefined.
* s0 — the position marked `S`.
* G  — the singleton containing the position marked `G`.
* c  — c(s, a, s') = 1 for every move.

A* uses f(n) = g(n) + h(n) with h = Manhattan distance by default.
Manhattan distance is admissible on a 4-connected grid because any step
changes exactly one coordinate by one, so the number of remaining steps
is at least |drow| + |dcol|.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


WAREHOUSE_MAP = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""


ACTIONS: list[tuple[str, int, int]] = [
    ("Up", -1, 0),
    ("Down", 1, 0),
    ("Left", 0, -1),
    ("Right", 0, 1),
]


Position = tuple[int, int]
Heuristic = Callable[[Position, Position], float]


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    path: list[Position]
    actions: list[str]
    expanded: int
    path_length: int = field(init=False)

    def __post_init__(self) -> None:
        self.path_length = len(self.actions)


def parse_map(ascii_map: str) -> tuple[list[str], Position, Position]:
    grid = ascii_map.splitlines()
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Map must contain 'S' and 'G'.")
    return grid, start, goal


def passable(grid: list[str], pos: Position) -> bool:
    r, c = pos
    if r < 0 or r >= len(grid):
        return False
    if c < 0 or c >= len(grid[r]):
        return False
    return grid[r][c] != "#"


def neighbours(grid: list[str], pos: Position):
    for name, dr, dc in ACTIONS:
        nxt = (pos[0] + dr, pos[1] + dc)
        if passable(grid, nxt):
            yield name, nxt


def manhattan(a: Position, b: Position) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero(_: Position, __: Position) -> float:
    return 0.0


def euclidean(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def manhattan_times_two(a: Position, b: Position) -> float:
    return 2 * manhattan(a, b)


def _reconstruct(came_from, end: Position) -> tuple[list[Position], list[str]]:
    path: list[Position] = []
    actions: list[str] = []
    node = end
    while node is not None:
        path.append(node)
        link = came_from[node]
        if link is None:
            break
        prev, action = link
        actions.append(action)
        node = prev
    path.reverse()
    actions.reverse()
    return path, actions


def bfs(ascii_map: str) -> SearchResult:
    grid, start, goal = parse_map(ascii_map)
    frontier: deque[Position] = deque([start])
    came_from = {start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == goal:
            path, actions = _reconstruct(came_from, current)
            return SearchResult("BFS", True, path, actions, expanded)
        for action, nxt in neighbours(grid, current):
            if nxt in came_from:
                continue
            came_from[nxt] = (current, action)
            frontier.append(nxt)

    return SearchResult("BFS", False, [], [], expanded)


def a_star(ascii_map: str, h: Heuristic = manhattan, label: str = "A*") -> SearchResult:
    grid, start, goal = parse_map(ascii_map)

    tie = 0
    open_heap: list[tuple[float, int, Position]] = []
    heapq.heappush(open_heap, (h(start, goal), tie, start))

    g_score: dict[Position, float] = {start: 0.0}
    came_from: dict[Position, tuple[Position, str] | None] = {start: None}
    closed: set[Position] = set()
    expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            path, actions = _reconstruct(came_from, current)
            return SearchResult(label, True, path, actions, expanded)

        for action, nxt in neighbours(grid, current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative_g
                came_from[nxt] = (current, action)
                tie += 1
                f = tentative_g + h(nxt, goal)
                heapq.heappush(open_heap, (f, tie, nxt))

    return SearchResult(label, False, [], [], expanded)


def render_path(ascii_map: str, path: list[Position]) -> str:
    grid = [list(row) for row in ascii_map.splitlines()]
    for r, c in path:
        if grid[r][c] not in ("S", "G"):
            grid[r][c] = "*"
    return "\n".join("".join(row) for row in grid)


def _print_result(result: SearchResult) -> None:
    print(f"[{result.algorithm}]")
    if not result.found:
        print("  no path found")
    else:
        print(f"  path length (moves): {result.path_length}")
        print(f"  states expanded:     {result.expanded}")
        print(f"  actions:             {', '.join(result.actions)}")


def main() -> None:
    print("Warehouse map:")
    print(WAREHOUSE_MAP)
    print()

    print("=== Task 3 - Original warehouse (A*, Manhattan) ===")
    r = a_star(WAREHOUSE_MAP)
    _print_result(r)
    print("  path drawn:")
    for line in render_path(WAREHOUSE_MAP, r.path).splitlines():
        print("    " + line)

    print("\n=== Task 5 - A* vs BFS ===")
    r_bfs = bfs(WAREHOUSE_MAP)
    r_astar = a_star(WAREHOUSE_MAP)
    print(f"  {'measure':<22}{'BFS':<10}{'A*':<10}")
    print(f"  {'solution found':<22}{str(r_bfs.found):<10}{str(r_astar.found):<10}")
    print(f"  {'path length':<22}{r_bfs.path_length:<10}{r_astar.path_length:<10}")
    print(f"  {'states expanded':<22}{r_bfs.expanded:<10}{r_astar.expanded:<10}")

    print("\n=== Task 6 - Heuristic experiments ===")
    for name, h in [
        ("h = 0 (uniform cost)", zero),
        ("Manhattan (admissible)", manhattan),
        ("Euclidean (admissible, weaker)", euclidean),
        ("2 x Manhattan (inadmissible)", manhattan_times_two),
    ]:
        res = a_star(WAREHOUSE_MAP, h, label=f"A* / {name}")
        _print_result(res)


if __name__ == "__main__":
    main()
