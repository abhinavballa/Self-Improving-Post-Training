#!/usr/bin/env bash
# Create a fresh, isolated agent directory for one experiment, outside this repo.
#
#   scripts/new_agent_run.sh exp1        -> ~/countdown-agent-runs/exp1/
#
# The agent dir gets only: TASK.md, submit.py, data/ (train_sft + prompts_unlabeled),
# a copy of the model, its own .venv, and a .claude/settings.json that denies access
# to this repo (so it can't read the grader, the test set, or our CLAUDE.md).
set -euo pipefail
NAME="${1:?usage: new_agent_run.sh <experiment-name>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${AGENT_RUNS_DIR:-$HOME/countdown-agent-runs}/$NAME"
[[ -e "$DEST" ]] && { echo "$DEST already exists"; exit 1; }

mkdir -p "$DEST/data" "$DEST/models" "$DEST/checkpoints" "$DEST/runs" "$DEST/.claude"
cp "$REPO/workspace/TASK.md" "$REPO/workspace/submit.py" "$DEST/"
cp "$REPO/workspace/data/train_sft.jsonl" "$REPO/workspace/data/prompts_unlabeled.jsonl" "$DEST/data/"
cp -cR "$REPO/models/Qwen2.5-0.5B-Instruct" "$DEST/models/"   # APFS clone: no extra disk space
rm -rf "$DEST/models/Qwen2.5-0.5B-Instruct/.cache"

cat > "$DEST/.claude/settings.json" <<JSON
{
  "permissions": {
    "deny": [
      "Read(/$REPO/**)",
      "Edit(/$REPO/**)",
      "Bash(*$REPO*)",
      "Bash(*Self-Improving-Post-Training*)"
    ]
  }
}
JSON

printf '.venv/\nmodels/\ncheckpoints/\n__pycache__/\n' > "$DEST/.gitignore"
git -C "$DEST" init -q
git -C "$DEST" -c user.name=harness -c user.email=harness@localhost add -A
git -C "$DEST" -c user.name=harness -c user.email=harness@localhost commit -q -m "starting state"

uv venv --quiet --python 3.12 "$DEST/.venv"
uv pip install --quiet --python "$DEST/.venv/bin/python" -r "$REPO/requirements.txt" trl peft

echo "Agent directory ready: $DEST"
echo "Next:"
echo "  1. In one terminal:  $REPO/.venv/bin/python $REPO/grader/serve.py --experiment $NAME"
echo "  2. In another:       cd $DEST && claude"
echo "     Prompt it:        Read TASK.md and complete the task."
