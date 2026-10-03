"""
run_pipeline.py

Runs the whole data pipeline in order with one command, from the repo root:

    python run_pipeline.py                  # everything
    python run_pipeline.py --from features  # start at a later step (reuses earlier outputs)
    python run_pipeline.py --only eda       # just one step

Steps: data -> clean_data -> features -> preprocess_data -> eda
Needs the raw CSVs in data/train and data/test.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# (name, script) in the order they have to run
STEPS = [
    ("data", "data.py"),                      # raw tables -> data/aggregated
    ("clean_data", "clean_data.py"),          # -> data/cleaned
    ("features", "features.py"),              # -> data/features
    ("preprocess_data", "preprocess_data.py"),  # -> data/preprocessed
    ("eda", "eda.py"),                        # -> reports/eda/figures
]


def main() -> None:
    names = [name for name, _ in STEPS]
    parser = argparse.ArgumentParser(description="Run the full data pipeline.")
    parser.add_argument("--from", dest="start", choices=names, help="start at this step")
    parser.add_argument("--only", choices=names, help="run just this step")
    args = parser.parse_args()

    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        steps = STEPS[names.index(args.start):]
    else:
        steps = STEPS

    total_start = time.time()
    for name, script in steps:
        print(f"\n===== {name} ({script}) =====", flush=True)
        step_start = time.time()
        # The scripts use paths like "data/train/..." so they must run from the repo root.
        result = subprocess.run([sys.executable, str(NOTEBOOKS_DIR / script)], cwd=ROOT_DIR)
        if result.returncode != 0:
            sys.exit(f"\nPipeline stopped: '{name}' failed (exit code {result.returncode}).")
        print(f"===== {name} done in {time.time() - step_start:.0f}s =====", flush=True)

    print(f"\nAll done in {time.time() - total_start:.0f}s.")


if __name__ == "__main__":
    main()
