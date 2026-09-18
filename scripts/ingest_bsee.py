#!/usr/bin/env python3
"""Import an explicitly mapped BSEE production CSV into the local truth store."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.bsee import connect, insert, read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("data/petroleum/truth.sqlite"))
    args = parser.parse_args()
    rows = read_csv(args.input)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    count = insert(connect(str(args.database)), rows)
    print(f"Imported BSEE rows: {count}")
    print(f"Database: {args.database}")


if __name__ == "__main__":
    main()
