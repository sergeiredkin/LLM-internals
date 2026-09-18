#!/usr/bin/env python3
"""Pull selected EIA petroleum datasets into a normalized local SQLite store."""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from llm.eia import connect, insert_observations, observation_from_record

DATASETS = {
    "crude-production": ("petroleum/crd/crpdn", "monthly"),
    "inventories": ("petroleum/stoc/wstk", "weekly"),
    "refinery-inputs": ("petroleum/pnp/wiup", "weekly"),
    "product-supplied": ("petroleum/cons/wpsup", "weekly"),
    "spot-prices": ("petroleum/pri/spt", "weekly"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/petroleum/eia.sqlite"))
    parser.add_argument("--api-key", default=os.environ.get("EIA_API_KEY", "DEMO_KEY"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record failed datasets and continue the pilot pull",
    )
    return parser.parse_args()


def fetch(path: str, frequency: str, args: argparse.Namespace) -> tuple[str, list[dict[str, object]]]:
    endpoint = f"https://api.eia.gov/v2/{path}/data/"
    parameters = {
        "api_key": args.api_key,
        "frequency": frequency,
        "start": args.start,
        "end": args.end,
        "length": str(args.length),
        "data[0]": "value",
    }
    url = endpoint + "?" + urlencode(parameters)
    request = Request(url, headers={"User-Agent": "learnGPT-eia-pilot/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return url, payload.get("response", {}).get("data", [])


def main() -> None:
    args = parse_args()
    if args.length <= 0:
        raise SystemExit("--length must be positive")
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; choose from {sorted(DATASETS)}")
    args.database.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(str(args.database))
    fetched_at = datetime.now(timezone.utc).isoformat()
    total = 0
    failures: list[str] = []
    for name in names:
        path, frequency = DATASETS[name]
        try:
            url, records = fetch(path, frequency, args)
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            if not args.continue_on_error:
                raise
            failures.append(f"{name}: {type(error).__name__}: {error}")
            print(f"WARN {failures[-1]}")
            continue
        observations = [
            observation
            for record in records
            if (observation := observation_from_record(name, record, url)) is not None
        ]
        inserted = insert_observations(connection, observations, fetched_at)
        total += inserted
        print(f"{name}: API rows={len(records)} numeric rows={inserted}")
    print(f"Database: {args.database}")
    print(f"Inserted: {total}")
    if failures:
        print("Failed datasets:")
        print("\n".join(f"  {failure}" for failure in failures))


if __name__ == "__main__":
    main()
