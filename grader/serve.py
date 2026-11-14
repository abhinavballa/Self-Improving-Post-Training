"""Local grading server: the agent submits checkpoints here and gets back only a score.

The agent never sees grader/ or the test set. Its submit.py POSTs
{"run": N, "checkpoint": "<absolute path>"} to http://127.0.0.1:<port>/submit.
On each submission the server:
  1. git-commits the agent directory (the checkpoint's grandparent), so the judge
     gets a diff per run (checkpoints, models and .venv are gitignored there),
  2. grades with verify.grade(),
  3. saves the full report (every completion + the commit hash) to
     grader/results/<experiment>/run_N.json, and replies with the accuracy only.

It enforces the run budget: each run number 1..--max-runs can be submitted once.

Usage: .venv/bin/python grader/serve.py --experiment exp1 [--port 8765 --max-runs 5]
"""

import argparse
import json
import subprocess
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import verify

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()  # one grading job at a time


def snapshot(agent_dir: Path, run: int) -> str | None:
    """Commit the agent directory's current state; return the commit hash (None if not a git repo)."""
    if not (agent_dir / ".git").exists():
        return None
    git = ["git", "-C", str(agent_dir), "-c", "user.name=harness", "-c", "user.email=harness@localhost"]
    subprocess.run(git + ["add", "-A"], check=True, capture_output=True)
    subprocess.run(git + ["commit", "--allow-empty", "-q", "-m", f"run {run} submitted"],
                   check=True, capture_output=True)
    return subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def make_handler(results_dir: Path, max_runs: int):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/submit":
                return self._reply(404, {"error": "unknown endpoint"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                run, ckpt = int(req["run"]), Path(req["checkpoint"])
            except (ValueError, KeyError, TypeError):
                return self._reply(400, {"error": 'expected JSON {"run": N, "checkpoint": "/abs/path"}'})
            if not 1 <= run <= max_runs:
                return self._reply(400, {"error": f"run must be between 1 and {max_runs}"})
            if not (ckpt.is_absolute() and (ckpt / "config.json").exists()):
                return self._reply(400, {"error": f"no Hugging Face checkpoint at {ckpt}"})

            with LOCK:
                out = results_dir / f"run_{run}.json"
                if out.exists():
                    return self._reply(409, {"error": f"run {run} was already submitted"})
                print(f"[{datetime.now():%H:%M:%S}] grading run {run}: {ckpt}", flush=True)
                commit = snapshot(ckpt.parent.parent, run)
                try:
                    report = verify.grade(str(ckpt))
                except Exception as e:  # report load/generation failures to the agent
                    return self._reply(500, {"error": f"grading failed: {e.__class__.__name__}: {e}"})
                report.update(run=run, commit=commit, checkpoint=str(ckpt),
                              submitted_at=datetime.now().isoformat(timespec="seconds"))
                out.write_text(json.dumps(report, indent=2))
                used = len(list(results_dir.glob("run_*.json")))
                print(f"  run {run}: accuracy {report['accuracy']:.1%} ({report['seconds']}s)", flush=True)
                return self._reply(200, {"run": run, "accuracy": report["accuracy"],
                                         "runs_left": max_runs - used})

        def log_message(self, *args) -> None:  # keep the console to our own lines
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="name for this agent trajectory")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-runs", type=int, default=5)
    args = ap.parse_args()

    results_dir = HERE / "results" / args.experiment
    results_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(results_dir, args.max_runs))
    print(f"Grading server for '{args.experiment}' on http://127.0.0.1:{args.port}/submit "
          f"(max {args.max_runs} runs; reports in {results_dir.relative_to(HERE.parent)})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
