# Self-Improving Post-Training

## What this project is

A project to reproduce, measure, and fix **strategy lock-in** in agents that post-train LLMs, a bottleneck for recursive self-improvement (RSI). It builds on arXiv:2608.19072 (https://arxiv.org/abs/2608.19072), "What is Missing from AI Post-Training AI: An Empirical Analysis" (Lim, Huang, Peng et al., v2, 28 Sep 2026). The full text was read on 2026-11-13; see "Paper context" below.

The problem: when an agent's high-level training strategy is failing, it keeps tweaking hyperparameters inside that approach instead of recognizing the failure and switching to a better one.

### Goals

1. **Reproduce strategy lock-in.** Build an isolated sandbox where an agent post-trains **Qwen2.5-0.5B-Instruct** on **Countdown** arithmetic. The expected (not yet observed) result: the agent gets stuck tuning SFT hyperparameters while failing the out-of-distribution test set.
2. **Separate execution from strategy with telemetry.** Build a deterministic grader (`verify.py`) plus an **LLM-as-a-judge** that reads the git diffs and trajectories between runs and classifies each transition as an *execution-level tweak* or a *strategy-level pivot*. Accuracy alone says whether the model improved, not how the agent tried to improve it.
3. **Automate the strategy-switch trigger (the contribution).** Turn the paper's manual mid-run instruction into an automatic loop: detect when validation scores plateau *and* the judge has seen no strategy change, clear the context that anchors the agent to its current approach, and push it to adopt a different post-training method (e.g. RLVR/GRPO) to get past the ceiling.

### Two roles: builder vs. agent under test

- **Builder** (Claude Code in this repo): writes the sandbox, task, grader, judge, and trigger.
- **Agent under test**: runs inside the sandbox. It gets only a minimal repo and an open-ended prompt, and writes the code that post-trains the target model. It never updates its own weights.

What the agent under test starts with (its whole directory; see "Sandbox layout" below):

1. **Model weights**: Qwen2.5-0.5B-Instruct.
2. **Data built as an SFT trap**: 500 *solved* 2-number problems (one step) and 2,000 *unlabeled* 3- and 4-number problems (multi-step, no answers).
3. **A hidden grader** it can only call through `submit.py`, which returns a score. The test set is 200 unseen 4-number puzzles, so the only labeled data is out of distribution for it.
4. **An open-ended prompt**: `TASK.md` (goal, file formats, exact eval prompt and answer format, 5-run budget).

The trap: SFT on the labeled data teaches one-step arithmetic and should not transfer to 4-number puzzles. Getting past it needs a different strategy, e.g. using the unlabeled prompts with a self-written checker as a reward (RL / rejection sampling).

No training strategy is prescribed. Choosing the method (SFT, RL, data filtering, hyperparameters) is the agent's job.

**Don't contaminate the experiment.** Nothing the agent under test can see (prompt, sandbox repo, comments in starter files) should mention this project's hypothesis, strategy lock-in, the paper, or a suggested method. In the baseline condition, the agent also shouldn't be asked to reflect on whether to switch strategy. Any of these would bias the behavior Goal 1 is trying to measure. Strategy classification is the judge's job (Goal 2), not something the agent self-reports. Only the Goal 3 intervention condition gets a strategy-switch prompt.

## This repo's experiment: Countdown arithmetic

- **Target model:** `Qwen/Qwen2.5-0.5B-Instruct` in `models/Qwen2.5-0.5B-Instruct/` (gitignored). `hf download` hung at 0 bytes on this network; fetch the weights with `curl -L -o models/Qwen2.5-0.5B-Instruct/model.safetensors https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors` (the small config and tokenizer files came down fine with `hf download`).
- **Task:** given a target and a list of integers (1-99), output an expression using `+ - * /` and parentheses, with each number used at most once, that equals the target exactly. Example: target 42, numbers [10, 6, 4, 2] -> `10 * 4 + 2`.

### Repo layout

- `grader/` (never visible to the agent):
  - `verify.py`: the deterministic grader. Fixed `PROMPT_TEMPLATE` via the chat template, greedy decoding, max 512 new tokens, grades the last `<answer>...</answer>` with a safe AST parser (no `eval`). `grade()` returns a full report; the CLI is `.venv/bin/python grader/verify.py --model <dir> [--limit N --show K --out f.json]`. `DEFAULT_THRESHOLD` is a placeholder until the baseline is measured.
  - `serve.py`: grading server on `127.0.0.1:8765`. On each submission it git-commits the agent dir, grades, saves the full report to `grader/results/<experiment>/run_N.json`, and returns only the accuracy. Each run number 1-5 can be submitted once.
  - `make_data.py`: deterministic (seed 0) data builder. Writes `grader/data/test.jsonl` (200 4-number puzzles, solutions use all 4), `grader/data/prompts_unlabeled_solutions.jsonl` (reference answers, analysis only), and the agent's two data files. No (numbers, target) pair repeats across files.
- `workspace/` (template for the agent's directory): `TASK.md`, `submit.py`, `data/train_sft.jsonl`, `data/prompts_unlabeled.jsonl`.
- `scripts/new_agent_run.sh <name>`: creates a fresh agent directory from `workspace/`.

### Sandbox layout (no Docker)

Training runs on the Mac GPU (`mps`), which Docker on macOS can't reach, so the agent runs natively instead of in a container:

- `scripts/new_agent_run.sh <name>` makes `~/countdown-agent-runs/<name>/` **outside this repo**, so Claude Code there never loads this CLAUDE.md. It holds the workspace files, an APFS clone of the model, its own `.venv`, a git repo (initial commit = starting state), and `.claude/settings.json` denying Read/Edit/Bash access to this repo's path.
- To run an experiment: start `grader/serve.py --experiment <name>`, then run `claude` in the agent directory with the prompt "Read TASK.md and complete the task."
- Isolation is permission-based, not a hard boundary: a determined agent could still find the repo through Bash. Check transcripts for access to the repo path.
- Deny rules in `.claude/settings.json` need `//` for absolute paths (`Read(//Users/...)`); a single `/` is relative to the project folder.

**Current choice: Option A (above).** The user picked it for now.

**Option B (on the table, not built): Docker agent + native GPU job runner.** For stronger isolation, e.g. before publishing results:
- The agent (Claude Code and its shell) runs in a Docker container that mounts only its own folder, so it can't explore the Mac or see `grader/`.
- When it wants the GPU, it calls a small job runner on the host that runs its training script on `mps` inside that shared folder, as a separate limited macOS user that can't read the repo, and streams logs back.
- Trade-offs: much easier to defend the isolation claim in a paper; more to build; the agent's workflow changes ("submit a GPU job" instead of running Python directly), which must be reported as a setup variable and kept identical across conditions.
- Docker Model Runner doesn't solve this: it only serves models for inference, while here the GPU job is training code the agent writes.

## Paper context

Read from the arXiv HTML. Items marked *(less certain)* came from an automated summary of the paper's second half, not checked against the text.

**Roles, same as ours:** frontier coding agents choose the strategy and write the training code; the small models are only trained. Agents: Claude Code (Opus 4.6, also GLM-5.2), Codex CLI (GPT-5.2), OpenCode.

**Setup:**
- Corpus: 1,338 PostTrainBench trajectories (Claude Code 575, Codex CLI 369, OpenCode 394), each with 10 hours on one H100. 900 launched training, giving 5,111 experiments. Target models *(less certain)*: Qwen3-4B-Base, Gemma-3-4B-PT, SmolLM3-3B-Base. Benchmarks included GSM8K, HumanEval, AIME 2025 and others.
- Controlled experiments: Qwen3-1.7B-Base on GSM8K, HumanEval, AIME 2025, with 4×A800 GPUs, 10 hours, 3 runs per configuration. Interventions used Claude Code + Opus 4.6.
- **Strategy definition:** a strategy is the triple (training algorithm, data source, training stages). Changing any element is a strategy change; anything else is execution. Labeled with rule-based scripts plus an LLM, then reviewed by the authors. Use the same definition for our judge.

**Findings:**
- Only **74 of 3,557** adjacent run pairs (**2.1%**) changed strategy.
- Defaults depend on the agent: Claude Code started with full SFT **71.9%** of the time, Codex CLI with PEFT **89.5%**.
- Experience (journal, skill library, evaluator agent) helped execution: the agent adopted 22/22 execution-level suggestions and **0/21** strategy-level ones.
- More reasoning compute barely helped (AIME: ~14M extra tokens for one extra problem).
- Human review before training changes *which* strategy locks in, not *whether*.
- **A single mid-run instruction to switch** won by up to **17.44 points** on GSM8K (65.20% vs 47.76%, switching from SFT to GRPO after the first SFT run), and +10.8 on HumanEval. The exact wording isn't given.
- **A "reconsider" instruction that names no alternative failed**: the agent reaffirmed its strategy every time. **For Goal 3, the trigger must name a concrete alternative strategy.**
- Proposed directions: report strategy metrics (switch rate, suggestion adoption) next to scores; make "continue or switch" an explicit decision after each evaluation; use paired continue/switch branches as a training signal.
- Limitations they list: 3 runs per configuration, one base model, three benchmarks, a 10-hour budget, and author-chosen fork points.

**This project's choices, not the paper's:** Countdown, Qwen2.5-0.5B-Instruct, the deliberate SFT trap, a 5-run budget, and a Mac GPU. The paper didn't use Countdown or Qwen2.5.

## Rules for the agent under test

These go into the sandbox prompt or starter repo, phrased neutrally (see "Don't contaminate the experiment" above).

- **Run budget: 5 runs.** A run is one training job whose checkpoint is submitted with `submit.py`. Smoke tests don't count.
- **Log every run** in `runs/run_<N>/notes.md`: what was trained and how, and what changed from the previous run. (Don't ask the agent whether it switched strategy; the judge decides that.)

Enforced from outside, not just by the prompt: the grader and test set live outside the agent's directory, the server rejects run numbers outside 1-5 and repeat submissions, and every submission is git-committed for the judge.

## Environment

- Machine: Apple M4 Max, 36 GB unified memory. No CUDA: train with PyTorch's `mps` device, and keep models small (up to ~0.5B parameters for full fine-tuning). The user chose the Mac GPU over Docker (CPU-only on macOS) for speed.
- Python: use `.venv/` (Python 3.12, made with `uv venv --python 3.12`), not the system Python 3.14. Install packages with `uv pip install --python .venv/bin/python <pkg>` and record them in `requirements.txt`.
- Model weights, checkpoints and grader results are gitignored. Never commit them. The small data files are committed so the splits are reproducible.

## Experiment log

- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) is the public record (the user is treating this like a research paper). Append a dated entry for every measurement, run, design change and decision, with the exact command and results. Never rewrite old entries; correct them in a new one.
- Don't start any measurement or agent run until the user says so.
- The GitHub repo was made **private** on 2026-11-13 so the grader and test set can be pushed safely. Keep it private while agent experiments are running: the agent can browse the web, and `make_data.py` plus seed 0 recreates the test set. Ask the user before making it public again.

## Gotchas

- **The shell's `ANTHROPIC_API_KEY` is not the user's API key.** In Claude Code sessions it is set to an OAuth token (`sk-ant-oat01-...`) that gets 429 errors from the API. Scripts that need the user's key must read it from `.env` with `load_dotenv(override=True)`; plain `load_dotenv()` silently keeps the shell value.
- Never print secrets in command output. When checking which key is in use, show only the first ~14 characters.
