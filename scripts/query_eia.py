#!/usr/bin/env python3
"""Run read-only EIA series queries and calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.eia import connect, latest_value, series_values, year_over_year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/petroleum/eia.sqlite"))
    parser.add_argument("--series", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--yoy-period", help="calculate YoY for YYYY-MM or YYYY-MM-DD")
    parser.add_argument("--latest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = connect(str(args.database))
    if args.yoy_period:
        result = year_over_year(connection, args.series, args.yoy_period)
        print(json.dumps(result, indent=2))
        return
    if args.latest:
        row = latest_value(connection, args.series)
        print(json.dumps(dict(row) if row else None, indent=2))
        return
    rows = series_values(connection, args.series, start=args.start, end=args.end)
    print(json.dumps([dict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
