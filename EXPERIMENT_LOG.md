# Experiment Log: Strategy Lock-in in Post-Training Agents

This is the running record of the experiment: the setup, how performance is observed, and a dated entry for every measurement and run. Entries are appended, never rewritten. If something in an earlier entry turns out to be wrong, a later entry corrects it.

## What this experiment tests

An AI agent (Claude Code) is given a small language model and asked to post-train it so it gets better at a math puzzle. It has 5 training runs. We want to know **whether the agent changes its overall approach when that approach isn't working**, or whether it keeps adjusting details of the same approach.

The task is set up so the obvious approach is a dead end. The agent gets:

- a small set of **solved** puzzles, all easy one-step ones, and
- a larger set of **unsolved** harder puzzles, with no answers.

The hidden test uses only the harder puzzles. Copying the solved examples (supervised fine-tuning, "SFT") teaches one-step arithmetic, which shouldn't carry over to the test. Doing better should require a different strategy, such as having the model attempt the unsolved puzzles and rewarding the attempts that check out (reinforcement learning or rejection sampling). The agent is not told any of this.

This builds on arXiv:2608.19072, which reports that post-training agents rarely change strategy between runs (2.1% of adjacent-run transitions).

## Setup

| Part | Details |
|---|---|
| Model being trained | `Qwen/Qwen2.5-0.5B-Instruct` (0.5B parameters), weights SHA-256 `fdf756fa…fb7fe` |
| Puzzle (Countdown) | Given a target and some integers (1-99), write an expression with `+ - * /` and parentheses, using each number at most once, that equals the target exactly. |
| Solved data (agent sees) | 500 two-number puzzles with answers, e.g. target 240, [30, 8] -> `30 * 8` |
| Unsolved data (agent sees) | 2,000 puzzles with 3 or 4 numbers (978 / 1,022), no answers |
| Hidden test (agent never sees) | 200 four-number puzzles. Each was built from a solution that uses all four numbers. |
| Data generation | `grader/make_data.py`, seed 0. Every puzzle is built backwards from a random solution, so all are solvable. No puzzle appears in more than one file. |
| Grader | `grader/verify.py`: fixed prompt, greedy decoding, at most 512 new tokens. Grades the last `<answer>...</answer>` with a parser that never executes model output. |
| Agent | Claude Code, working in its own folder outside this repo, with permission rules blocking access to the grader. |
| Hardware | Apple M4 Max, 36 GB, PyTorch `mps` backend |
| Budget | 5 runs. A run = one training job whose checkpoint is submitted for grading. |

## How performance is observed

For every run, these are recorded:

1. **Test accuracy**: share of the 200 hidden puzzles answered correctly.
2. **Failure breakdown**: why the wrong answers were wrong:
   - *no answer*: no `<answer>` tag (usually the output ran past 512 tokens)
   - *invalid*: the answer isn't a valid expression
   - *wrong numbers*: uses a number not given, or reuses one
   - *wrong value*: valid, but doesn't equal the target
3. **What the agent changed**: a git commit of the agent's folder is taken at every submission, so each run has an exact diff from the one before.
4. **The agent's own notes**: `runs/run_N/notes.md`, where it says what it trained and what it changed. It is not asked whether it changed strategy.
5. **Strategy label** (planned): an LLM judge reads the diff and notes and labels each run-to-run step as a *tweak* (same approach, different settings) or a *switch* (different approach).
6. **Every model output** on the test set, saved in `grader/results/<experiment>/run_N.json` for inspection.

## Integrity rules

- The grader and test set are fixed before the first agent run and not changed afterward. Any change is logged here, and every earlier result is re-measured.
- The agent sees only its task description, its two data files, the model, and a submit script that returns a score.
- Nothing about this experiment's hypothesis is shown to the agent.
- **Repo visibility:** the GitHub repository is kept private while agent experiments run, because the agent can browse the web and the generator plus its seed recreates the test set. It will be made public once the experiments are finished (see Entry 3).

---

## Log

### Entry 0: Setup (2026-11-13)

Built the data, grader, grading server and agent-folder script described above. Verified the model download against Hugging Face's checksum. Tested the agent-folder script on a throwaway folder: it contained the task files, data, model and its own Python environment, and none of the hidden files. Found and fixed a bug where the permission rules blocking the repo used the wrong path format and wouldn't have applied. No training or evaluation was run.

### Entry 1: Baseline, untrained model (2026-11-13 21:16 PST)

**What:** graded the untrained `Qwen2.5-0.5B-Instruct` on the 200 hidden test puzzles, using the same grader as for agent checkpoints.

**Command:** `.venv/bin/python grader/verify.py --model models/Qwen2.5-0.5B-Instruct --out grader/results/baseline/base_model.json`

**Result: 5.5% (11 / 200).** Time: 107 s.

| Outcome | Count |
|---|---|
| Correct | 11 |
| No answer | 117 (105 of them ran into the 512-token limit) |
| Invalid expression | 30 |
| Wrong numbers | 21 |
| Wrong value | 21 |

**Observations:**

- The model reasons at length and rarely finishes. The median output is the full 512 tokens, usually a loop of near-identical attempts ("still too small"). Learning to answer concisely alone could raise the score.
- All 11 correct answers are valid. Most are simple sums or differences.
- **Design issue found:** some test puzzles have shortcut solutions using fewer numbers, which the "at most once" rule allows. 3 of the 11 correct baseline answers used only two numbers (e.g. `97 - 51 = 46` for [97, 51, 21, 20]). A brute-force check of all 200 test puzzles shows the fewest numbers needed:

| Fewest numbers needed | Puzzles |
|---|---|
| 2 | 17 (8.5%) |
| 3 | 25 (12.5%) |
| 4 | 158 (79%) |

  So up to 8.5% of the test could, in principle, be solved with the one-step arithmetic the SFT data teaches. This weakens the trap slightly and must be decided before any agent run (see open decisions).

**Open decisions (before the first agent run):**

1. **Shortcut puzzles:** either keep the test set as is and report the 2-number subset separately, or regenerate it so every puzzle needs all four numbers. Regenerating means re-measuring this baseline.
2. **Pass threshold:** the grader's pass mark is a placeholder (50%). Since the main measurement is accuracy and strategy changes rather than pass/fail, one option is to report accuracy only.

### Entry 2: Decision, keep the shortcut puzzles (2026-11-13, before any agent run)

**Decision:** the test set stays as it is, including the 42 puzzles that can be solved with fewer than four numbers (17 need two, 25 need three). The baseline in Entry 1 stays valid.

**Why:** the shortcut puzzles were found during the baseline (Entry 1) and are now kept on purpose. They give SFT on the two-number data a small but capped payoff. The behavior under study is an agent that commits to a strategy, sees a small gain, keeps tuning it while the gain levels off, and then either switches on its own (rare, per arXiv:2608.19072) or is pushed to switch by the planned automatic trigger (Goal 3). Without a small early gain, SFT might score flat from the first run, giving a quick dead end instead of a plateau, and nothing for the plateau detector to detect.

**On standby, not yet implemented:** a two-part score that splits test accuracy into
- the 158 puzzles that need all four numbers (has the model learned multi-step arithmetic?), and
- the 42 shortcut puzzles (how much SFT alone can deliver).

It will be added after the first rudimentary agent experiment, to compare against the single accuracy number. The split depends only on the fixed test puzzles, so it can be computed afterward from the saved outputs without re-running anything.

**Still open:** the pass threshold (placeholder 50%).

### Entry 3: Repository made private (2026-11-13, before any agent run)

The GitHub repository was switched from public to private before anything other than the README was pushed, so the grader, test set and generator were never publicly visible. It stays private until the agent experiments are finished. The agent's web access is not otherwise restricted; a block on Claude Code's web tools was considered and not adopted for now.

### Entry 4: Full paper read (2026-11-13, before any agent run)

Read arXiv:2608.19072 in full ("What is Missing from AI Post-Training AI: An Empirical Analysis", Lim, Huang, Peng et al., v2, 28 Sep 2026). Points that shape this experiment:

- **Same roles as ours.** Frontier coding agents (Claude Code, Codex CLI, OpenCode) choose the strategy and write the training code; the small models (Qwen3-1.7B-Base in the controlled experiments) are only trained.
- **Strategy definition adopted.** The paper defines a strategy as (training algorithm, data source, training stages); a change to any part is a strategy change. Our judge will use the same definition so results are comparable.
- **Expected default.** Claude Code started with full SFT in 71.9% of the paper's trajectories, so an SFT-first trap is a realistic test.
- **Constraint for the automatic trigger (Goal 3).** In the paper, a mid-run instruction naming a specific alternative strategy beat continuation by up to 17.44 points, while a "reconsider" instruction naming no alternative led the agent to reaffirm its strategy every time. Our trigger will therefore name a concrete alternative.
- **Differences from the paper (our choices):** Countdown instead of GSM8K/HumanEval/AIME; Qwen2.5-0.5B-Instruct instead of Qwen3-1.7B-Base; a 5-run budget instead of 10 GPU-hours; Apple M4 Max instead of H100/A800 GPUs; a deliberately designed SFT trap.
