"""Evaluation harness for the Countdown task. Hidden from the agent: never mount grader/.

Grades a checkpoint on the held-out test set (grader/data/test.jsonl: 200 unseen
4-number puzzles):
  - The model gets a fixed chat prompt (PROMPT_TEMPLATE below) and greedy decoding.
  - The last <answer>...</answer> in its output is parsed as an arithmetic expression.
  - It is correct if it uses only + - * / and parentheses, uses each given number at
    most once, and evaluates exactly to the target.

Usage:
  python grader/verify.py --model workspace/checkpoints/final  # grade a checkpoint
  python grader/verify.py --model models/Qwen2.5-0.5B-Instruct # baseline
  python grader/verify.py --model ... --limit 20 --show 5      # quick look

Exit code 0 if accuracy >= --threshold, else 1. Writes a JSON report with --out.
"""

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_FILE = HERE / "data" / "test.jsonl"
DEFAULT_THRESHOLD = 0.50

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can be "
    "used at most once. Show your reasoning, then give only the final expression "
    "inside <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ALLOWED_BINOPS = {ast.Add, ast.Sub, ast.Mult, ast.Div}


def build_messages(numbers: list[int], target: int) -> list[dict]:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target)}]


def extract_answer(completion: str) -> str | None:
    matches = ANSWER_RE.findall(completion)
    if not matches:
        return None
    expr = matches[-1].strip()
    if "=" in expr:  # tolerate "expr = target"
        expr = expr.split("=")[0].strip()
    return expr


def _evaluate(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, used)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
        left, right = _evaluate(node.left, used), _evaluate(node.right, used)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def check_expression(expr: str | None, numbers: list[int], target: int) -> tuple[bool, str]:
    """Return (correct, reason). Never calls eval()."""
    if not expr:
        return False, "no <answer> tag"
    if len(expr) > 200:
        return False, "expression too long"
    expr = expr.replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(expr, mode="eval")
        used: list[int] = []
        value = _evaluate(tree, used)
    except ZeroDivisionError:
        return False, "division by zero"
    except (SyntaxError, ValueError) as e:
        return False, f"invalid expression ({e.__class__.__name__})"
    if Counter(used) - Counter(numbers):
        return False, "uses a number not given, or more times than given"
    if value != target:
        return False, f"evaluates to {float(value):g}, not {target}"
    return True, "correct"


def load_test_set(limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in TEST_FILE.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def generate(model_path: str, problems: list[dict], batch_size: int, max_new_tokens: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()

    prompts = [
        tok.apply_chat_template(build_messages(p["numbers"], p["target"]), tokenize=False, add_generation_prompt=True)
        for p in problems
    ]
    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        outputs += tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  generated {len(outputs)}/{len(prompts)}", file=sys.stderr, flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return outputs


def grade(model_path: str, limit: int | None = None, batch_size: int = 32,
          max_new_tokens: int = 512, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Grade a checkpoint on the test set. Returns the full report, including per-problem results."""
    problems = load_test_set(limit)
    start = time.time()
    completions = generate(model_path, problems, batch_size, max_new_tokens)

    results, reasons = [], Counter()
    for p, c in zip(problems, completions):
        ok, reason = check_expression(extract_answer(c), p["numbers"], p["target"])
        reasons["correct" if ok else reason.split(" (")[0].split(",")[0]] += 1
        results.append({**p, "completion": c, "correct": ok, "reason": reason})

    acc = sum(r["correct"] for r in results) / len(results)
    return {"model": model_path, "accuracy": acc, "threshold": threshold, "passed": acc >= threshold,
            "n": len(results), "outcomes": dict(reasons), "seconds": round(time.time() - start),
            "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a Hugging Face checkpoint directory")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None, help="grade only the first N problems")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--show", type=int, default=0, help="print N example completions")
    ap.add_argument("--out", type=str, default=None, help="write a JSON report here")
    args = ap.parse_args()

    report = grade(args.model, args.limit, args.batch_size, args.max_new_tokens, args.threshold)
    results = report["results"]

    for r in results[:args.show]:
        print(f"\n--- target {r['target']}, numbers {r['numbers']} -> {r['reason']}\n{r['completion'][-600:]}")
    print(f"\nModel: {args.model}")
    print(f"Accuracy: {report['accuracy']:.1%} ({sum(r['correct'] for r in results)}/{len(results)})  "
          f"threshold {args.threshold:.0%}  -> {'PASS' if report['passed'] else 'FAIL'}")
    print("Outcomes: " + ", ".join(f"{k}: {v}" for k, v in Counter(report["outcomes"]).most_common()))
    print(f"Time: {report['seconds']}s")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
