"""Goal-based agent for the warehouse navigation problem.

Worksheet: agents_lab.pdf (Task 2 and Task 3).

The warehouse is represented by an ASCII grid.  The vehicle starts at `S`
and must reach `G` while avoiding `#` cells.  Allowed moves are Up, Down,
Left, Right.

This is a goal-based agent (not a reflex agent): it maintains an explicit
representation of the world, the goal, and the frontier of states it has
yet to explore.  It chooses actions based on where those actions are
expected to lead, not purely on the current percept.

The search strategy is Breadth-First Search (BFS).  BFS is appropriate
here because every move has unit cost: the first time BFS reaches the
goal it has found a shortest path (optimal under uniform cost), and the
state space is finite and small enough that completeness is not an issue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


WAREHOUSE_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""


# Actions as (name, drow, dcol).
ACTIONS: list[tuple[str, int, int]] = [
    ("Up", -1, 0),
    ("Down", 1, 0),
    ("Left", 0, -1),
    ("Right", 0, 1),
]


@dataclass
class SearchResult:
    path: list[tuple[int, int]] | None
    actions: list[str]
    expanded: int

    @property
    def found(self) -> bool:
        return self.path is not None


def parse_map(ascii_map: str) -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    """Return (grid, start, goal) from an ASCII map."""
    grid = ascii_map.splitlines()
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Map must contain exactly one 'S' and one 'G'.")
    return grid, start, goal


def passable(grid: list[str], pos: tuple[int, int]) -> bool:
    r, c = pos
    if r < 0 or r >= len(grid):
        return False
    if c < 0 or c >= len(grid[r]):
        return False
    return grid[r][c] != "#"


def goal_based_search(ascii_map: str) -> SearchResult:
    """Breadth-first search from S to G."""
    grid, start, goal = parse_map(ascii_map)

    frontier: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1

        if current == goal:
            # Reconstruct path.
            path: list[tuple[int, int]] = []
            actions: list[str] = []
            node = current
            while node is not None:
                path.append(node)
                parent = came_from[node]
                if parent is None:
                    break
                prev, action = parent
                actions.append(action)
                node = prev
            path.reverse()
            actions.reverse()
            return SearchResult(path=path, actions=actions, expanded=expanded)

        for name, dr, dc in ACTIONS:
            nxt = (current[0] + dr, current[1] + dc)
            if nxt in came_from:
                continue
            if not passable(grid, nxt):
                continue
            came_from[nxt] = (current, name)
            frontier.append(nxt)

    return SearchResult(path=None, actions=[], expanded=expanded)


def render_path(ascii_map: str, path: list[tuple[int, int]]) -> str:
    """Draw the path on top of the map using '*' (keeping S and G)."""
    grid = [list(row) for row in ascii_map.splitlines()]
    for r, c in path:
        if grid[r][c] not in ("S", "G"):
            grid[r][c] = "*"
    return "\n".join("".join(row) for row in grid)


def main() -> None:
    print("Warehouse map:")
    print(WAREHOUSE_MAP)
    print()

    result = goal_based_search(WAREHOUSE_MAP)
    if not result.found:
        print("No path from S to G.")
        return

    print(f"Path length (moves): {len(result.actions)}")
    print(f"States expanded:     {result.expanded}")
    print(f"Action sequence:     {', '.join(result.actions)}")
    print()
    print("Path visualised:")
    print(render_path(WAREHOUSE_MAP, result.path))


if __name__ == "__main__":
    main()
