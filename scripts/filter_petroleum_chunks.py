#!/usr/bin/env python3
"""Filter obvious extraction failures while preserving rejected chunks for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.petroleum_quality import quality_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--minimum-characters", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_characters <= 0:
        raise SystemExit("--minimum-characters must be positive")
    args.clean.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)
    clean_count = 0
    rejected_count = 0
    reasons: dict[str, int] = {}
    with args.input.open(encoding="utf-8") as source, \
            args.clean.open("w", encoding="utf-8") as clean, \
            args.rejected.open("w", encoding="utf-8") as rejected:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            flags = quality_flags(record.get("text", ""), minimum_characters=args.minimum_characters)
            if flags:
                record["quality_flags"] = flags
                record["input_line"] = line_number
                rejected.write(json.dumps(record, ensure_ascii=False) + "\n")
                rejected_count += 1
                for flag in flags:
                    reasons[flag] = reasons.get(flag, 0) + 1
            else:
                record["quality_flags"] = []
                clean.write(json.dumps(record, ensure_ascii=False) + "\n")
                clean_count += 1
    print(f"Clean chunks:     {clean_count}")
    print(f"Rejected chunks:  {rejected_count}")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")
    print(f"Clean output:     {args.clean}")
    print(f"Rejected output:  {args.rejected}")


if __name__ == "__main__":
    main()
