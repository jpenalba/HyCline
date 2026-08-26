#!/usr/bin/env python
"""
Run HyCline directly from a cloned repository, no installation step required
beyond `pip install -r requirements.txt` (or the equivalent `pip install
-e .[bayes]`).

Examples
--------
    python run_hycline.py fit --input mydata.csv --method classic \\
        --scaling fixed --output-dir results/

    python run_hycline.py fit --input mydata.csv --method bayesian \\
        --num-chains 4 --output-dir results/

    python run_hycline.py compare-tails --input mydata.csv \\
        --candidates none,mirror,both --output-dir results/

Run with --help (or a subcommand plus --help) to see the full flag list:

    python run_hycline.py --help
    python run_hycline.py fit --help
    python run_hycline.py compare-tails --help

This is a thin wrapper around hycline.cli.main() -- see README.md's
"Command-line usage" section for the full walkthrough, and
BACKEND_PHASE1.md / BACKEND_PHASE2.md for what's actually happening under
the hood of each fitting method.
"""

import sys
from pathlib import Path

# Make the cloned repo importable even without `pip install -e .` -- this
# is what lets someone `git clone` and immediately run this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hycline.cli import main

if __name__ == "__main__":
    sys.exit(main())
