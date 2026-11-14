"""Submit a checkpoint for evaluation and print its score.

Usage: python submit.py --run N [--checkpoint checkpoints/run_N]

Each run number can be submitted once. Grading takes a few minutes; this waits for it.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8765/submit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--checkpoint", default=None, help="defaults to checkpoints/run_<N>")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint or f"checkpoints/run_{args.run}").resolve()
    body = json.dumps({"run": args.run, "checkpoint": str(ckpt)}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    print(f"Submitting run {args.run} ({ckpt}) for evaluation...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print("Error:", json.loads(e.read()).get("error", e.reason))
        return 1
    except urllib.error.URLError as e:
        print(f"Error: evaluation server not reachable ({e.reason})")
        return 1

    print(f"Run {result['run']}: accuracy {result['accuracy']:.1%}  ({result['runs_left']} runs left)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
