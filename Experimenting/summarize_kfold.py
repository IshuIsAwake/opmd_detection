"""
summarize_kfold.py — CLI wrapper around common.summarize.write_summary.

Aggregates whatever fold_*/metrics.json files are on disk under
Experimenting/results/<experiment>/ and writes summary.{json,txt}. Safe to
run repeatedly (idempotent) and against a partial set of folds — useful when
you want to look at the running mean ± std between folds without waiting for
all 10 to finish.

    python Experimenting/summarize_kfold.py kfold10_binary
    python Experimenting/summarize_kfold.py kfold10_5class
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings
from common.summarize import write_summary


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python Experimenting/summarize_kfold.py "
                 "<experiment-name e.g. kfold10_binary>")
    name = sys.argv[1]
    out_dir = settings.RESULTS_ROOT / name
    if not out_dir.is_dir():
        sys.exit(f"no such results dir: {out_dir}")
    report = write_summary(out_dir)
    print(f"[{name}] {report['n_folds']} fold(s) summarised → "
          f"{out_dir}/summary.txt\n")
    print((out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
