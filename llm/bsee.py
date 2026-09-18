"""Strict, provenance-preserving storage for BSEE production observations."""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BSEEProduction:
    source_id: str
    period: str
    area: str
    operator: str
    oil_bbl: float | None
    gas_mcf: float | None
    water_bbl: float | None
    source_url: str
    source_checksum: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS bsee_production (
    source_id TEXT NOT NULL,
    period TEXT NOT NULL,
    area TEXT NOT NULL,
    operator TEXT NOT NULL,
    oil_bbl REAL,
    gas_mcf REAL,
    water_bbl REAL,
    source_url TEXT NOT NULL,
    source_checksum TEXT NOT NULL,
    PRIMARY KEY (source_id, period, area, operator)
);
CREATE INDEX IF NOT EXISTS idx_bsee_period_area
    ON bsee_production(period, area);
"""

REQUIRED_COLUMNS = {
    "source_id", "period", "area", "operator", "oil_bbl", "gas_mcf", "water_bbl",
    "source_url", "source_checksum",
}


def connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: str) -> float | None:
    value = value.strip()
    return None if not value or value.lower() in {"na", "n/a", "null", "-"} else float(value)


def read_csv(path: Path) -> list[BSEEProduction]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"BSEE CSV missing required columns: {sorted(missing)}")
        rows = []
        for line_number, row in enumerate(reader, 2):
            try:
                rows.append(BSEEProduction(
                    source_id=row["source_id"].strip(), period=row["period"].strip(),
                    area=row["area"].strip(), operator=row["operator"].strip(),
                    oil_bbl=_number(row["oil_bbl"]), gas_mcf=_number(row["gas_mcf"]),
                    water_bbl=_number(row["water_bbl"]), source_url=row["source_url"].strip(),
                    source_checksum=row["source_checksum"].strip(),
                ))
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"invalid BSEE row {line_number}: {error}") from error
    return rows


def insert(connection: sqlite3.Connection, rows: list[BSEEProduction]) -> int:
    connection.executemany(
        """INSERT OR REPLACE INTO bsee_production
        (source_id, period, area, operator, oil_bbl, gas_mcf, water_bbl, source_url, source_checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(r.source_id, r.period, r.area, r.operator, r.oil_bbl, r.gas_mcf, r.water_bbl,
          r.source_url, r.source_checksum) for r in rows],
    )
    connection.commit()
    return len(rows)
