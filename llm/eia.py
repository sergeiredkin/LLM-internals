"""Normalized local storage and calculations for EIA petroleum observations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EIAObservation:
    dataset: str
    series_id: str
    period: str
    value: float
    unit: str
    geography: str
    product: str
    description: str
    source_url: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS eia_observations (
    dataset TEXT NOT NULL,
    series_id TEXT NOT NULL,
    period TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL DEFAULT '',
    geography TEXT NOT NULL DEFAULT '',
    product TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (dataset, series_id, period)
);
CREATE INDEX IF NOT EXISTS idx_eia_series_period
    ON eia_observations(series_id, period);
CREATE INDEX IF NOT EXISTS idx_eia_dataset_period
    ON eia_observations(dataset, period);
"""


def connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def observation_from_record(
    dataset: str,
    record: dict[str, Any],
    source_url: str,
) -> EIAObservation | None:
    raw_value = record.get("value")
    if raw_value in (None, "", "null", "NA"):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return EIAObservation(
        dataset=dataset,
        series_id=str(record.get("series") or record.get("seriesId") or ""),
        period=str(record["period"]),
        value=value,
        unit=str(record.get("unit") or record.get("units") or ""),
        geography=str(record.get("area-name") or record.get("duoarea") or ""),
        product=str(record.get("product-name") or record.get("product") or ""),
        description=str(record.get("series-description") or ""),
        source_url=source_url,
    )


def insert_observations(
    connection: sqlite3.Connection,
    observations: list[EIAObservation],
    fetched_at: str,
) -> int:
    rows = [
        (
            item.dataset,
            item.series_id,
            item.period,
            item.value,
            item.unit,
            item.geography,
            item.product,
            item.description,
            item.source_url,
            fetched_at,
        )
        for item in observations
        if item.series_id
    ]
    connection.executemany(
        """INSERT OR REPLACE INTO eia_observations
        (dataset, series_id, period, value, unit, geography, product, description, source_url, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    connection.commit()
    return len(rows)


def series_values(
    connection: sqlite3.Connection,
    series_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[sqlite3.Row]:
    query = "SELECT * FROM eia_observations WHERE series_id = ?"
    parameters: list[str] = [series_id]
    if start is not None:
        query += " AND period >= ?"
        parameters.append(start)
    if end is not None:
        query += " AND period <= ?"
        parameters.append(end)
    query += " ORDER BY period"
    return list(connection.execute(query, parameters))


def latest_value(connection: sqlite3.Connection, series_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM eia_observations WHERE series_id = ? ORDER BY period DESC LIMIT 1",
        (series_id,),
    ).fetchone()


def change(first: float, last: float) -> float:
    return last - first


def percent_change(first: float, last: float) -> float | None:
    return None if first == 0 else (last - first) / abs(first) * 100.0


def year_over_year(
    connection: sqlite3.Connection,
    series_id: str,
    period: str,
) -> dict[str, float | str | None] | None:
    current = connection.execute(
        "SELECT * FROM eia_observations WHERE series_id = ? AND period = ?",
        (series_id, period),
    ).fetchone()
    if current is None:
        return None
    prior_period = f"{int(period[:4]) - 1}{period[4:]}"
    prior = connection.execute(
        "SELECT * FROM eia_observations WHERE series_id = ? AND period = ?",
        (series_id, prior_period),
    ).fetchone()
    if prior is None:
        return {
            "series_id": series_id,
            "period": period,
            "value": current["value"],
            "prior_period": prior_period,
            "prior_value": None,
            "change": None,
            "percent_change": None,
        }
    return {
        "series_id": series_id,
        "period": period,
        "value": current["value"],
        "prior_period": prior_period,
        "prior_value": prior["value"],
        "change": change(prior["value"], current["value"]),
        "percent_change": percent_change(prior["value"], current["value"]),
    }
