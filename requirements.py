"""
Project dependency list for ClinsightAI.

Usage:
1) Install core:
   pip install -r <(python requirements.py --format txt)

2) Install core + optional model stack:
   pip install -r <(python requirements.py --format txt --with-optional)
"""

from __future__ import annotations

import argparse


CORE_REQUIREMENTS = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "streamlit>=1.30",
    "altair>=5.0",
]

# Optional but recommended for embedding + transformer-based severity scoring.
OPTIONAL_REQUIREMENTS = [
    "sentence-transformers>=2.7",
    "transformers>=4.40",
    "torch>=2.2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print project requirements")
    parser.add_argument("--format", choices=["txt"], default="txt")
    parser.add_argument("--with-optional", action="store_true")
    args = parser.parse_args()

    reqs = list(CORE_REQUIREMENTS)
    if args.with_optional:
        reqs.extend(OPTIONAL_REQUIREMENTS)

    for dep in reqs:
        print(dep)


if __name__ == "__main__":
    main()
