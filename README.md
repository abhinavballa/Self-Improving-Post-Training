# Self-Improving Post-Training

Do AI agents that post-train language models change their approach when it stops working?

This repo is an experiment on **strategy lock-in**. A frontier coding agent (Claude Code) is given a small model and asked to post-train it on a math puzzle, with 5 training runs. The task is designed so the obvious approach plateaus. We measure whether the agent keeps tuning that approach or switches to a better one, and then test an automatic trigger that pushes it to switch.

It builds on [*What is Missing from AI Post-Training AI: An Empirical Analysis*](https://arxiv.org/abs/2608.19072) (Lim et al., 2026). That paper found that across 1,338 trajectories, post-training agents changed strategy in only 2.1% of run-to-run steps, and that one mid-run instruction to switch strategy gained up to 17.44 points.

**Status:** setup complete and baseline measured. No agent runs yet. The full record is in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Roles

- **The agent (Claude Code)** chooses the training strategy, writes the training code, runs it, and decides what to try next. Strategy lock-in happens here.
- **The target model (Qwen2.5-0.5B-Instruct)** is only trained and graded. It never sees the task, the scores or the code.

This matches the paper: frontier agents post-train small models, and no model updates its own weights.

## The task: Countdown

Given a target and a list of integers, write an expression using `+ - * /` and parentheses, with each number used at most once, that equals the target exactly.

> Target 42, numbers [10, 6, 4, 2] → `10 * 4 + 2`

## The trap

| Data | What it is | Agent sees it? |
|---|---|---|
| `workspace/data/train_sft.jsonl` | 500 **solved** puzzles, all with 2 numbers (one step) | Yes |
| `workspace/data/prompts_unlabeled.jsonl` | 2,000 **unsolved** puzzles with 3 or 4 numbers (multi-step) | Yes |
| `grader/data/test.jsonl` | 200 four-number puzzles | No, hidden |

The only labeled data is one-step arithmetic, but the test is multi-step. Fine-tuning on the solved examples (SFT), which Claude Code chose first in 71.9% of the paper's trajectories, should give a small gain that then levels off. Some test puzzles can be solved with fewer numbers, which caps SFT's payoff instead of zeroing it; this is a deliberate choice (Entry 2 in the log). Breaking through should take a different strategy, such as having the model attempt the unsolved puzzles and rewarding attempts that pass a self-written checker (RL or rejection sampling). The agent is told none of this.

## Experiment plan

1. **Reproduce lock-in.** Let the agent use its 5 runs unaided and record what it does.
2. **Measure strategy, not just score.** An LLM judge reads each run's code diff and notes and labels the step a *tweak* (same strategy) or a *switch*. It uses the paper's definition: a strategy is (training algorithm, data source, training stages).
3. **Automate the switch.** Detect when scores plateau with no strategy change, then give the agent an instruction naming a specific alternative strategy. In the paper, an instruction that only said "reconsider" never worked.

## How a run is observed

For each run the harness records:
- test accuracy and why wrong answers failed (no answer, invalid expression, wrong numbers, wrong value)
- a git snapshot of the agent's folder, giving an exact diff from the previous run
- the agent's own notes on what it trained and what it changed (it is not asked whether it changed strategy)
- every model output on the test set
- the judge's tweak/switch label (planned)

## Results so far

| Measurement | Test accuracy |
|---|---|
| Untrained Qwen2.5-0.5B-Instruct (baseline) | **5.5%** (11 / 200) |

The untrained model usually reasons in circles until it hits the 512-token limit (117 of 200 outputs gave no answer). See Entry 1 in the log.

## Repository layout

```
grader/                  hidden from the agent
  verify.py              deterministic grader: fixed prompt, greedy decoding, safe expression parser
  serve.py               local grading server: snapshots the agent's folder, grades, returns only the score
  make_data.py           builds all data files (seed 0); every puzzle is solvable by construction
  data/                  hidden test set + reference answers for the unlabeled puzzles
workspace/               template for the agent's folder
  TASK.md                the agent's only instructions
  submit.py              submits a checkpoint and prints its score
  data/                  the agent's two data files
scripts/new_agent_run.sh creates an isolated agent folder outside this repo
EXPERIMENT_LOG.md        dated record of every measurement, run and decision
CLAUDE.md                notes for Claude Code sessions working on this repo
```

## Reproducing

Requires an Apple Silicon Mac (training uses PyTorch's `mps` backend), [uv](https://docs.astral.sh/uv/), and [Claude Code](https://claude.com/claude-code).

```bash
# 1. Environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Model (if `hf download` stalls on the weights, fetch model.safetensors with curl from
#    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors)
.venv/bin/hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# 3. Data (already committed; this regenerates it identically)
.venv/bin/python grader/make_data.py

# 4. Baseline
.venv/bin/python grader/verify.py --model models/Qwen2.5-0.5B-Instruct

# 5. An agent experiment
scripts/new_agent_run.sh exp1                                # agent folder in ~/countdown-agent-runs/exp1
.venv/bin/python grader/serve.py --experiment exp1           # terminal 1: grading server
cd ~/countdown-agent-runs/exp1 && claude                     # terminal 2: "Read TASK.md and complete the task."
```

## Isolation and its limits

The agent works in a folder outside this repo, with Claude Code permission rules blocking access to it, and can reach the grader only through `submit.py`, which returns a single score. This boundary is based on permissions, not enforced by the operating system, so transcripts should be checked for any access to the repo. A stronger design, with the agent in a Docker container and GPU training jobs run by a job runner on the host, is described in CLAUDE.md but not built.

Because the agent can browse the web and `make_data.py` with seed 0 recreates the test set, this repository stays private while agent experiments are running.
