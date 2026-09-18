# EIA Week 2 Petroleum Data Pilot

**Date:** 2026-09-18

## Implementation

The local EIA layer uses a normalized SQLite database because PostgreSQL is not installed on the
current workstation. The schema is relational and portable to PostgreSQL:

`data/petroleum/eia.sqlite`

Tables and indexes are created by `llm.eia.connect()`.

The ingestion CLI supports:

- crude production
- inventories
- refinery inputs
- product supplied
- spot prices

Every observation stores dataset, series ID, period, value, unit, geography, product, description,
source API URL, and fetch timestamp.

## Live pilot

The EIA API `DEMO_KEY` was sufficient for the pilot, but it is rate-limited. The pull used a two-year
window and a 500-row page size. The local database currently contains:

- crude production: 500 observations
- inventories: 500 observations
- refinery inputs: 500 observations
- spot prices: 500 observations
- product supplied: temporarily blocked by EIA HTTP 403/rate limiting

The ingestion command supports `--continue-on-error` so one unavailable dataset does not destroy the
successful portion of a scheduled pull.

## Verified calculation

For EIA series `MCRFPUS1` (U.S. field production of crude oil), the local store calculated:

```text
2024-01: 389,080 thousand barrels
2025-01: 408,165 thousand barrels
absolute change: +19,085 thousand barrels
percent change: +4.9052%
```

The calculation preserved the EIA series ID, unit, period, description, and source URL.

## Commands

```bash
python -m scripts.ingest_eia \
  --database data/petroleum/eia.sqlite \
  --start 2024-01-01 --end 2025-12-31 \
  --length 500 --continue-on-error
```

```bash
python -m scripts.query_eia \
  --database data/petroleum/eia.sqlite \
  --series MCRFPUS1 \
  --yoy-period 2025-01
```

For production, register an EIA API key and set it outside Git:

```bash
export EIA_API_KEY='your-key'
```

## Decision

Week 2 engineering is complete for the local pilot: normalized storage, ingestion, read-only series
queries, and calculations work. Before claiming complete coverage, add pagination, scheduled pulls,
PostgreSQL deployment, and retry/backoff for EIA rate limits. Product-supplied ingestion should be
rerun with a registered key after the temporary API restriction clears.
