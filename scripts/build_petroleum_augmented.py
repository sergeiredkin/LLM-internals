#!/usr/bin/env python3
"""Build an optional petroleum retrieval corpus with conservative table facts."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--table-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for path in (args.base, args.table_facts):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        output.write(line)
                        count += 1
    print(f"Augmented chunks: {count}")
    print(f"Output:           {args.output}")


if __name__ == "__main__":
    main()
