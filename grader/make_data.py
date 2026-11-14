"""Build the data for the "natural SFT trap" setup.

Agent-visible (workspace/data/):
  train_sft.jsonl          500 solved 2-number problems (one step, e.g. 10 + 5 = 15)
  prompts_unlabeled.jsonl  2,000 unsolved 3- and 4-number problems (no answers)

Hidden (grader/data/):
  test.jsonl                       200 unseen 4-number puzzles, graded by verify.py
  prompts_unlabeled_solutions.jsonl  reference solutions for the unlabeled prompts
                                     (for analysis only; never give to the agent)

The trap: the only labeled data is one-step arithmetic, while the test is multi-step.
SFT on it alone should not transfer; using the unlabeled prompts with a self-written
checker as a reward (RL / rejection sampling) should.

Every problem is built backwards from a random expression, so all are solvable.
Intermediate results stay positive integers. No (numbers, target) pair repeats across
files. Deterministic for a given --seed.

Usage: python grader/make_data.py [--seed 0]
"""

import argparse
import json
import random
from pathlib import Path

from verify import check_expression

ROOT = Path(__file__).resolve().parent.parent
OPS = ["+", "-", "*", "/"]


def make_problem(rng: random.Random, n: int, use_all: bool = True,
                 max_value: int = 99, max_target: int = 999) -> dict:
    """A solvable problem with n numbers. use_all=False lets the solution skip one number."""
    while True:
        numbers = [rng.randint(1, max_value) for _ in range(n)]
        k = n if use_all or n <= 2 or rng.random() < 0.7 else n - 1
        items = [(v, str(v)) for v in rng.sample(numbers, k)]  # (value, expression)
        ok = True
        while len(items) > 1:
            (a, ea), (b, eb) = rng.sample(items, 2)
            items.remove((a, ea))
            items.remove((b, eb))
            if a < b:
                (a, ea), (b, eb) = (b, eb), (a, ea)
            op = rng.choice(OPS)
            if op == "+":
                val = a + b
            elif op == "*":
                val = a * b
            elif op == "-":
                val = a - b
            else:
                val = a // b if a % b == 0 else 0
            if val <= 0:
                ok = False
                break
            items.append((val, f"({ea} {op} {eb})"))
        if not ok:
            continue
        target, expr = items[0]
        if not (1 <= target <= max_target) or target in numbers:
            continue
        solution = expr[1:-1] if expr.startswith("(") else expr
        assert check_expression(solution, numbers, target)[0], (solution, numbers, target)
        return {"numbers": numbers, "target": target, "solution": solution}


def unique(rng, size, seen, factory):
    rows = []
    while len(rows) < size:
        p = factory()
        key = (tuple(sorted(p["numbers"])), p["target"])
        if key not in seen:
            seen.add(key)
            rows.append(p)
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows):>5} -> {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen: set = set()

    # Test first, so the hidden set is fixed by the seed alone.
    test = unique(rng, 200, seen, lambda: make_problem(rng, 4, use_all=True))
    sft = unique(rng, 500, seen, lambda: make_problem(rng, 2))
    unlabeled = unique(rng, 2000, seen, lambda: make_problem(rng, rng.choice([3, 4]), use_all=False))

    write(ROOT / "grader/data/test.jsonl", test)
    write(ROOT / "workspace/data/train_sft.jsonl", sft)
    write(ROOT / "workspace/data/prompts_unlabeled.jsonl",
          [{"numbers": p["numbers"], "target": p["target"]} for p in unlabeled])
    write(ROOT / "grader/data/prompts_unlabeled_solutions.jsonl", unlabeled)


if __name__ == "__main__":
    main()
