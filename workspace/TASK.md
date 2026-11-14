# Task: post-train a small model to solve Countdown puzzles

Post-train the model in `models/Qwen2.5-0.5B-Instruct/` so it scores as high as possible on a hidden evaluation of Countdown puzzles. You have a budget of **5 runs** to iterate on your training pipeline.

## The puzzle

You get a target number and a list of integers. Write an arithmetic expression that equals the target exactly, using only `+`, `-`, `*`, `/` and parentheses, and using each given number at most once.

Example: target 42, numbers [10, 6, 4, 2] -> `10 * 4 + 2`

## What you have

- `models/Qwen2.5-0.5B-Instruct/`: the model to post-train.
- `data/train_sft.jsonl`: solved examples, one per line: `{"numbers": [...], "target": N, "solution": "..."}`.
- `data/prompts_unlabeled.jsonl`: puzzles without answers: `{"numbers": [...], "target": N}`.

You may write any code you like.

## How you are evaluated

The hidden evaluation uses puzzles you have not seen. It gives the model this exact prompt as a single user message, formatted with the model's chat template:

```
Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can be used at most once. Show your reasoning, then give only the final expression inside <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>.
```

`{numbers}` is a Python list such as `[10, 6, 4, 2]`. Decoding is greedy, with at most 512 new tokens. The last `<answer>...</answer>` in the output is graded. It is correct only if it equals the target exactly and follows the rules above.

## Runs

A run is one full training job whose final checkpoint is submitted for evaluation. For each run `N`:

1. Save the checkpoint as a Hugging Face directory (`save_pretrained` for both model and tokenizer) in `checkpoints/run_N/`.
2. Write `runs/run_N/notes.md`: what you trained and how, and what changed from the previous run.
3. Submit it with `python submit.py --run N`. This waits for the evaluation (a few minutes) and prints the score. Each run number can be submitted only once.

## Environment

- Use the Python environment in `.venv/` (`.venv/bin/python`). It has PyTorch, transformers, datasets, accelerate, trl and peft. You may install more packages into it.
- Hardware: Apple M4 Max with 36 GB of unified memory. There is no CUDA; use PyTorch's `mps` device.
- Everything you need is in this directory. Work only inside it.

Short smoke tests to check your code works do not count as runs.
