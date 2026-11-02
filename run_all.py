"""Run the whole pipeline in the right order.

    python run_all.py           # ~15 minutes
    python run_all.py --quick   # fewer seeds, ~5 minutes (for testing)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
quick = "--quick" in sys.argv

steps = [
    ("1/4  sanity checks + baseline table", ["run_checks.py"]),
    ("2/4  datasets -> data/", ["make_datasets.py"] + (["--seeds", "2"] if quick else [])),
    ("3/4  figures -> figures/, results/", ["make_figures.py"] + (["--quick"] if quick else [])),
    ("4/4  model comparison: logistic regression, XGBoost, surrogate-fair", ["compare_models.py"] + (["--seeds", "1"] if quick else [])),
]
for title, cmd in steps:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, str(ROOT / cmd[0]), *cmd[1:]], check=True, cwd=ROOT)
print("\nDone. See figures/, results/ and data/.")
