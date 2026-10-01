# Logical Planning Lab — STRIPS-style Planner

Worksheet: [`logic_lab_ex.pdf`](logic_lab_ex.pdf).

## Files

| File | Description |
|------|-------------|
| `planner.py` | STRIPS-style actions with positive/negative pre- and effects; BFS planner; warehouse domain. |
| `tests.py` | Solvable / impossible / irrelevant-actions / preconditions / trivial-goal tests, with independent plan replay. |
| `planner.pl` | Optional Prolog extension — edge facts, `can_move`, `valid_move`, `path`, and the `reduce_speed` rule chain. |
| `answers.md` | Written answers to Tasks 0–8 and the Section 7 Prolog extension. |
| `outputs/` | Captured runs. |

## Run

```bash
python3 planner.py
python3 tests.py
# optional: swipl -s planner.pl
```
