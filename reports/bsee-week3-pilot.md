# BSEE Week 3 Pilot

The BSEE layer is intentionally constrained to Gulf of Mexico production observations. It is kept
separate from EIA and from explanatory-document retrieval.

## Safety decision

BSEE exposes official downloadable production reports, including ranked operator files. The ranked
files currently expose aggregate fixed-width values whose column meanings and units are not explicit
enough for safe automatic interpretation. They are therefore not ingested as fabricated monthly
well-level facts.

Instead, `scripts/ingest_bsee.py` requires an explicitly mapped CSV with these required fields:

```text
source_id, period, area, operator, oil_bbl, gas_mcf, water_bbl,
source_url, source_checksum
```

Rows fail closed when columns are missing or numeric values are malformed. Every row preserves its
source URL and checksum.

## Import command

```bash
python -m scripts.ingest_bsee \
  --input data/petroleum/bsee-production-mapped.csv \
  --database data/petroleum/truth.sqlite
```

The database is ignored by Git. No BSEE records are committed yet because the official export schema
still needs to be manually verified and mapped.

## Next action

Download one official BSEE production export, document its field definitions and units, create the
explicit mapping CSV, inspect sample totals against the BSEE page, and only then load it into the
truth store.
