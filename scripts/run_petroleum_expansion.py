#!/usr/bin/env python3
"""One-command petroleum corpus expansion with quality gates.

Collapses the manual per-source workflow (handoff Phase 3) into a single
automated loop:

    discover -> download -> checksum -> manifest entry -> validate ->
    ingest pages -> chunks -> filter -> table facts -> augmented corpus ->
    50-question production benchmark -> regression gate -> test suite ->
    expansion report -> commit/push

Usage:
    # Verify the harness reproduces the frozen baseline on the current corpus:
    conda run -n gpu-test python -m scripts.run_petroleum_expansion --selfcheck

    # Automatically discover and add N verified USGS/OSTI sources:
    conda run -n gpu-test python -m scripts.run_petroleum_expansion --auto 1

    # Add one explicit source (all provenance required):
    conda run -n gpu-test python -m scripts.run_petroleum_expansion \
        --url https://www.osti.gov/servlets/purl/1234567 \
        --title "Report title" --year 1987 --source-id usgs-osti-1234567

    # Plan without touching anything:
    ... --dry-run

Acceptance rule (handoff): preserve provenance, pass tests, and do not
regress the production benchmark. A failed candidate is rolled back
automatically (manifest entry and raw PDF removed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from llm.rag import BM25Retriever, load_jsonl_chunks, retrieval_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "petroleum"
RESULTS = REPO_ROOT / "reports" / "results"

MANIFEST = DATA / "manifest.jsonl"
RAW_DIR = DATA / "raw"
PAGES = DATA / "pages-expanded.jsonl"
CHUNKS = DATA / "chunks-expanded.jsonl"
CHUNKS_REVIEW = DATA / "review-auto.jsonl"
CHUNKS_CLEAN = DATA / "chunks-expanded-clean.jsonl"
CHUNKS_REJECTED = DATA / "chunks-expanded-rejected.jsonl"
TABLE_CANDIDATES = DATA / "table-candidates.jsonl"
TABLE_ROWS = DATA / "table-row-candidates.jsonl"
TABLE_FACTS = DATA / "table-fact-candidates.jsonl"
AUGMENTED = DATA / "chunks-augmented.jsonl"
QUERIES = DATA / "queries-50.jsonl"
BASELINE_FILE = RESULTS / "petroleum-production-baseline.json"
REJECTED_FILE = DATA / "auto-expansion-rejected.json"
USER_AGENT = "learnGPT-petroleum-research/1.0"

# Frozen production baseline (reports/results/petroleum-expansion-26-report.md).
FROZEN_BASELINE = {"hit_at_5": 0.980, "recall_at_5": 0.981, "mrr_at_5": 0.761}
HIT_RECALL_TOLERANCE = 0.005
MRR_TOLERANCE = 0.015

PETROLEUM_TERMS = re.compile(
    r"\b(petroleum|oil|gas|hydrocarbon|crude|source rock|basin|drilling)\b", re.IGNORECASE
)
USGS_MARKERS = re.compile(r"Geological Survey|USGS", re.IGNORECASE)
# Provenance gate: the PUBLISHER field itself must be USGS/Geological Survey.
# Journal articles (ACS, Elsevier, ...) keep publisher copyright even when
# USGS/DOE-affiliated authors wrote them, so they are never acceptable.
USGS_PUBLISHER = re.compile(r"(U\.?S\.? )?Geological Survey|USGS", re.IGNORECASE)
OSTI_API = "https://www.osti.gov/api/v1/records"
PUBS_API = "https://pubs.usgs.gov/pubs-services/publication"
PUBS_QUERIES = [
    "assessment of undiscovered oil and gas resources",
    "geologic assessment petroleum province",
    "petroleum geology and assessment basin",
    "total petroleum system assessment",
]
# Series accepted as substantial domain reports (fact sheets are page-scale
# summaries that dilute retrieval; data releases often lack a prose PDF).
ACCEPTED_SERIES = {"SIR", "OFR", "PP", "PROF", "CIR", "B", "DS", "TM"}
USGS_DOI_PREFIX = "10.3133/"
OSTI_QUERIES = [
    "USGS petroleum resource assessment oil gas",
    "geologic assessment undiscovered oil gas resources USGS",
    "USGS petroleum geology basin report",
    "USGS oil and gas exploration report",
]


def log(message: str) -> None:
    print(f"[expansion] {message}", flush=True)


def run(step: str, argv: list[str], timeout: int = 1800) -> dict:
    log(f"{step}: {' '.join(str(a) for a in argv[:2])} ...")
    result = subprocess.run(
        [sys.executable, "-m", f"scripts.{argv[0]}", *argv[1:]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + "\n" + result.stderr[-4000:] + "\n")
        raise SystemExit(f"step failed: {step}")
    return {"stdout": result.stdout}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.open(encoding="utf-8") if line.strip()]


def load_rejected() -> list[dict]:
    if not REJECTED_FILE.exists():
        return []
    return json.loads(REJECTED_FILE.read_text(encoding="utf-8"))


def record_rejection(record: dict, reason: str) -> None:
    """Remember a gate-failed source so discovery never re-proposes it."""
    rejected = load_rejected()
    rejected.append({
        "source_id": record["source_id"],
        "url": record["url"],
        "title": record["title"],
        "reason": reason,
        "date": date.today().isoformat(),
    })
    REJECTED_FILE.write_text(json.dumps(rejected, indent=2) + "\n", encoding="utf-8")
    log(f"recorded rejection: {record['source_id']} ({reason})")


def fetch_jsonl_counts(path: Path) -> dict:
    counts = {"lines": 0}
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                counts["lines"] += 1
    return counts


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def pubs_candidates(limit: int) -> list[dict]:
    """Discover USGS Publications Warehouse petroleum reports with full provenance."""
    manifest = load_manifest()
    known_urls = {record["url"] for record in manifest}
    known_ids = {record["source_id"] for record in manifest}
    rejected_ids = {entry["source_id"] for entry in load_rejected()}
    rejected_urls = {entry["url"] for entry in load_rejected()}
    found: list[dict] = []
    seen: set[str] = set()
    for query in PUBS_QUERIES:
        url = f"{PUBS_API}?title={urllib.parse.quote(query)}&pageSize=50"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                records = json.load(response).get("records", [])
        except Exception as error:  # noqa: BLE001 - discovery is best-effort
            log(f"pubs discovery query failed ({query!r}): {error}")
            continue
        for hit in records:
            index_id = str(hit.get("indexId") or hit.get("id") or "").strip()
            if not index_id or index_id in seen:
                continue
            year = str(hit.get("text") or "").split(" - ")[1] if " - " in str(hit.get("text") or "") else None
            seen.add(index_id)
            detail = pubs_detail(index_id)
            if not detail:
                continue
            candidate = pubs_record_to_candidate(detail, year)
            if not candidate:
                continue
            if candidate["url"] in known_urls or candidate["source_id"] in known_ids:
                continue
            if candidate["source_id"] in rejected_ids or candidate["url"] in rejected_urls:
                continue
            found.append(candidate)
            if len(found) >= limit:
                return found
    return found


def pubs_detail(index_id: str) -> dict | None:
    request = urllib.request.Request(
        f"{PUBS_API}/{urllib.parse.quote(index_id)}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)
    except Exception as error:  # noqa: BLE001
        log(f"pubs detail failed ({index_id}): {error}")
        return None


def pubs_record_to_candidate(detail: dict, fallback_year: str | None) -> dict | None:
    """Convert a Pubs Warehouse record into a manifest candidate, or None.

    Provenance gates: USGS DOI prefix, accepted numbered series, official
    pubs.usgs.gov PDF link.
    """
    title = str(detail.get("title") or detail.get("displayTitle") or "").strip()
    doi = str(detail.get("doi") or "")
    series = (detail.get("seriesTitle") or {}).get("code", "")
    series_number = str(detail.get("seriesNumber") or "")
    if not title or not doi.startswith(USGS_DOI_PREFIX):
        return None
    if series not in ACCEPTED_SERIES or not series_number:
        return None
    pdf_url = None
    for link in detail.get("links") or []:
        if str((link.get("linkFileType") or {}).get("text") or "").lower() == "pdf" and str(
            link.get("url", "")
        ).startswith("https://pubs.usgs.gov/"):
            pdf_url = link["url"]
            break
    if not pdf_url:
        return None
    year = str(detail.get("publicationYear") or fallback_year or "unknown")
    series_slug = series.lower()
    number_slug = re.sub(r"[\u2010-\u2015/]+", "-", series_number.lower()).strip("-")
    pdf_stem = pdf_url.rsplit("/", 1)[-1].lower().removesuffix(".pdf")
    source_id = f"usgs-{series_slug}-{number_slug}"
    # Numbered-series chapters (e.g. PP 1824 R/I/...) share one seriesNumber;
    # append the PDF stem when it disambiguates so source IDs stay unique.
    if pdf_stem.replace("-", "") not in source_id.replace("-", ""):
        source_id = f"{source_id}-{pdf_stem}"
    return {
        "source_id": source_id,
        "url": pdf_url,
        "title": re.sub(r"\s+", " ", title),
        "publisher": "USGS",
        "publication_date": year,
        "license": "USGS-public-domain-verify-source",
        "format": "pdf",
        "source_role": "domain-report",
        "doi": doi,
    }


def osti_candidates(limit: int) -> list[dict]:
    """Query the public OSTI API for USGS petroleum reports not yet in the manifest."""
    manifest = load_manifest()
    known_urls = {record["url"] for record in manifest}
    known_ids = {record["source_id"] for record in manifest}
    rejected_ids = {entry["source_id"] for entry in load_rejected()}
    rejected_urls = {entry["url"] for entry in load_rejected()}
    found: list[dict] = []
    seen: set[str] = set()
    for query in OSTI_QUERIES:
        url = f"{OSTI_API}?q={urllib.parse.quote(query)}&rows=50"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                records = json.load(response)
        except Exception as error:  # noqa: BLE001 - discovery is best-effort
            log(f"discovery query failed ({query!r}): {error}")
            continue
        for record in records:
            osti_id = str(record.get("osti_id") or "").strip()
            title = str(record.get("title") or "").strip()
            publisher = str(record.get("publisher") or "").strip()
            if not osti_id or not title:
                continue
            pdf_url = f"https://www.osti.gov/servlets/purl/{osti_id}"
            if pdf_url in known_urls or osti_id in seen:
                continue
            if f"usgs-osti-{osti_id}" in known_ids:
                continue
            if f"usgs-osti-{osti_id}" in rejected_ids or pdf_url in rejected_urls:
                continue
            haystack = " ".join(
                [
                    title,
                    publisher,
                    str(record.get("description") or "")[:600],
                    str(record.get("authors") or ""),
                ]
            )
            if not PETROLEUM_TERMS.search(haystack):
                continue
            if not USGS_MARKERS.search(haystack):
                continue
            if record.get("journal_name"):
                continue  # journal article: publisher copyright, not public domain
            if not USGS_PUBLISHER.search(publisher) and not USGS_PUBLISHER.search(
                str(record.get("office") or "") + " " + str(record.get("sponsor_org") or "")
            ):
                continue
            if re.search(r"\b(coal|uranium|wind|solar|geothermal|nuclear|carbon storage)\b", title, re.IGNORECASE):
                continue
            year = str(record.get("publication_date") or "")[:4] or "unknown"
            seen.add(osti_id)
            found.append(
                {
                    "source_id": f"usgs-osti-{osti_id}",
                    "url": pdf_url,
                    "title": re.sub(r"\s+", " ", title),
                    "publisher": publisher or "USGS/OSTI",
                    "publication_date": year,
                    "license": "USGS-public-domain-verify-source",
                    "format": "pdf",
                    "source_role": "domain-report",
                }
            )
            if len(found) >= limit:
                return found
    return found


def looks_like_pdf(url: str) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            if "pdf" in content_type.lower():
                return True
            if int(response.headers.get("Content-Length") or 0) < 20_000:
                return False
    except Exception:  # noqa: BLE001
        return False
    # Ambiguous content type: probe the magic bytes.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(5) == b"%PDF-"
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# Source lifecycle
# --------------------------------------------------------------------------

def download_pdf(record: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / f"{record['source_id']}.pdf"
    temporary = destination.with_suffix(".pdf.tmp")
    request = urllib.request.Request(record["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if temporary.read_bytes()[:5] != b"%PDF-":
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"downloaded file is not a PDF: {record['url']}")
    temporary.replace(destination)
    return destination


def append_manifest(record: dict, digest: str) -> None:
    entry = {
        "source_id": record["source_id"],
        "url": record["url"],
        "title": record["title"],
        "publisher": record["publisher"],
        "publication_date": record["publication_date"],
        "license": record["license"],
        "format": "pdf",
        "sha256": digest,
        "source_role": record["source_role"],
    }
    with MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def rollback(record: dict) -> None:
    """Remove a failed candidate from the manifest and raw dir."""
    entries = load_manifest()
    kept = [e for e in entries if e["source_id"] != record["source_id"]]
    if len(kept) != len(entries):
        with MANIFEST.open("w", encoding="utf-8") as handle:
            for entry in kept:
                handle.write(json.dumps(entry) + "\n")
    (RAW_DIR / f"{record['source_id']}.pdf").unlink(missing_ok=True)


def rebuild_corpus() -> dict:
    # Ingest first: it downloads any raw PDFs missing locally (e.g. sources added
    # on another machine) with retry/backoff, and checksum-verifies every file.
    run("ingest pages", ["ingest_petroleum_pdfs", "--manifest", MANIFEST, "--raw-dir", RAW_DIR, "--output", PAGES])
    run("validate manifest", ["validate_petroleum_manifest", "--manifest", MANIFEST, "--raw-dir", RAW_DIR])
    run("prepare chunks", ["prepare_petroleum_chunks", "--pages", PAGES, "--chunks", CHUNKS, "--review", CHUNKS_REVIEW])
    run("filter chunks", ["filter_petroleum_chunks", "--input", CHUNKS, "--clean", CHUNKS_CLEAN, "--rejected", CHUNKS_REJECTED])
    run("extract tables", ["extract_petroleum_tables", "--pages", PAGES, "--output", TABLE_CANDIDATES, "--merge-continuations"])
    run("reconstruct table rows", ["reconstruct_petroleum_tables", "--pages", PAGES, "--output", TABLE_ROWS])
    run("construct table facts", ["construct_petroleum_table_facts", "--rows", TABLE_ROWS, "--output", TABLE_FACTS])
    run("build augmented corpus", ["build_petroleum_augmented", "--base", CHUNKS_CLEAN, "--table-facts", TABLE_FACTS, "--output", AUGMENTED])
    return {
        "pages": fetch_jsonl_counts(PAGES)["lines"],
        "chunks": fetch_jsonl_counts(CHUNKS)["lines"],
        "clean": fetch_jsonl_counts(CHUNKS_CLEAN)["lines"],
        "rejected": fetch_jsonl_counts(CHUNKS_REJECTED)["lines"],
        "table_facts": fetch_jsonl_counts(TABLE_FACTS)["lines"],
        "augmented": fetch_jsonl_counts(AUGMENTED)["lines"],
    }


def baseline() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return dict(FROZEN_BASELINE)


def run_benchmark(tag: str) -> dict:
    output = RESULTS / f"petroleum-benchmark-{tag}.json"
    run("production benchmark", [
        "run_petroleum_production_benchmark",
        "--documents", AUGMENTED,
        "--queries", QUERIES,
        "--output", output,
    ])
    return json.loads(output.read_text(encoding="utf-8"))


def metrics_ok(metrics: dict, reference: dict) -> tuple[bool, str]:
    hit = metrics.get("hit_rate_at_k", 0.0)
    recall = metrics.get("recall_at_k", 0.0)
    mrr = metrics.get("mrr_at_k", 0.0)
    problems = []
    if hit < reference["hit_at_5"] - HIT_RECALL_TOLERANCE:
        problems.append(f"Hit@5 {hit:.3f} < {reference['hit_at_5']:.3f}")
    if recall < reference["recall_at_5"] - HIT_RECALL_TOLERANCE:
        problems.append(f"Recall@5 {recall:.3f} < {reference['recall_at_5']:.3f}")
    if mrr < reference["mrr_at_5"] - MRR_TOLERANCE:
        problems.append(f"MRR@5 {mrr:.3f} < {reference['mrr_at_5']:.3f}")
    return (not problems), "; ".join(problems) or "pass"


def run_tests() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
    return result.returncode == 0, tail[0]


def diagnose_regression(documents: Path, reference_metrics: dict) -> list[str]:
    """Identify the individual benchmark questions that regressed and why.

    Serves the handoff Phase-2 requirement to track per-question regressions;
    the output is embedded in rejection records for later analysis.
    """
    chunks = load_jsonl_chunks(documents)
    retriever = BM25Retriever(chunks)
    queries = []
    for line in QUERIES.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            queries.append((record["query"], set(record["relevant_document_ids"])))
    baseline_hits = int(round(reference_metrics["hit_at_5"] * len(queries)))
    regressed = []
    for query, relevant in queries:
        single = retrieval_metrics(
            retriever, [(query, relevant)], top_k=5, expand_query=True,
            prefer_primary_evidence=True, section_aware=True, summary_aware=True,
            semantic_rerank=True, source_aware=True,
        )
        if single["hit_rate_at_k"] < 1.0:
            regressed.append(query)
    log(
        f"diagnostic: {len(regressed)} query/queries miss in top-5 "
        f"(baseline expected ~{len(queries) - baseline_hits} miss/es)"
    )
    return regressed[:5]


def write_report(tag: str, record: dict, digest: str, ingestion: dict,
                 benchmark: dict, tests_ok: bool, tests_line: str) -> Path:
    metrics = benchmark["metrics"]
    hit = metrics["hit_rate_at_k"]
    recall = metrics["recall_at_k"]
    mrr = metrics["mrr_at_k"]
    report = RESULTS / f"petroleum-expansion-{tag}-report.md"
    body = f"""# Petroleum Corpus Expansion: {tag}

Added and checksum-verified (automated expansion):

- `{record['source_id']}`
- {record['title']}
- {record['publisher']}, {record['publication_date']}
- {record['url']}
- SHA-256: `{digest}`

## Ingestion

```text
Manifest sources: {len(load_manifest())}
Recovered pages:  {ingestion['pages']}
Original chunks:  {ingestion['chunks']}
Clean chunks:     {ingestion['clean']}
Rejected chunks:  {ingestion['rejected']}
Table facts:      {ingestion['table_facts']}
Augmented corpus: {ingestion['augmented']}
```

## Regression

Production configuration on the frozen 50-question benchmark:

```text
Hit@5:    {hit:.3f}
Recall@5: {recall:.3f}
MRR@5:    {mrr:.3f}
```

Baseline: Hit@5 {baseline()['hit_at_5']:.3f}, Recall@5 {baseline()['recall_at_5']:.3f}, MRR@5 {baseline()['mrr_at_5']:.3f}.

## Tests

```text
{'PASS: ' + tests_line if tests_ok else 'FAIL: ' + tests_line}
```

Generated by `scripts/run_petroleum_expansion.py` on {date.today().isoformat()}.
"""
    report.write_text(body, encoding="utf-8")
    return report


def commit_and_push(report: Path, record: dict) -> None:
    subprocess.run(["git", "add", str(MANIFEST), str(report)], cwd=REPO_ROOT, check=True)
    short_title = record["title"][:60].replace('"', "'")
    subprocess.run(
        ["git", "commit", "-m", f"Expand petroleum corpus with {short_title}"],
        cwd=REPO_ROOT,
        check=True,
    )
    # The remote may have moved (other machines push corpus work); rebase and retry.
    rebase = subprocess.run(["git", "pull", "--rebase"], cwd=REPO_ROOT, capture_output=True, text=True)
    if rebase.returncode != 0:
        raise SystemExit(f"git pull --rebase failed (resolve manually): {rebase.stderr[-2000:]}")
    push = subprocess.run(["git", "push"], cwd=REPO_ROOT, capture_output=True, text=True)
    if push.returncode != 0:
        raise SystemExit(f"git push failed: {push.stderr[-2000:]}")
    log("committed and pushed")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def add_source(record: dict, tag: str, do_commit: bool, skip_tests: bool) -> bool:
    """Run the full gated loop for one candidate. Returns True if kept."""
    log(f"adding source: {record['source_id']} — {record['title'][:70]}")
    if not looks_like_pdf(record["url"]):
        rollback(record)
        raise SystemExit(f"not a verifiable PDF: {record['url']}")
    pdf = download_pdf(record)
    digest = sha256(pdf)
    log(f"downloaded {pdf.stat().st_size / 1e6:.1f} MB, sha256={digest[:16]}...")
    append_manifest(record, digest)
    try:
        ingestion = rebuild_corpus()
        benchmark = run_benchmark(tag)
        ok, gate = metrics_ok(benchmark["metrics"], baseline())
        if not ok:
            regressed = diagnose_regression(AUGMENTED, baseline())
            detail = "; ".join(f"miss: {q}" for q in regressed) or "see benchmark json"
            raise SystemExit(f"benchmark regression gate failed: {gate} | {detail}")
        log(f"benchmark gate: {gate}")
        if not skip_tests:
            tests_ok, tests_line = run_tests()
            if not tests_ok:
                raise SystemExit(f"test suite failed: {tests_line}")
            log(f"tests: {tests_line}")
        else:
            tests_ok, tests_line = True, "skipped"
        report = write_report(tag, record, digest, ingestion, benchmark, tests_ok, tests_line)
        log(f"report: {report.relative_to(REPO_ROOT)}")
        if do_commit:
            commit_and_push(report, record)
    except SystemExit:
        rollback(record)
        log(f"ROLLED BACK: {record['source_id']}")
        raise
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", type=int, metavar="N", help="discover and add N verified sources automatically")
    parser.add_argument("--url", help="official PDF URL for an explicit source")
    parser.add_argument("--title")
    parser.add_argument("--year", dest="publication_date")
    parser.add_argument("--publisher", default="USGS")
    parser.add_argument("--source-id")
    parser.add_argument("--source-role", default="domain-report", choices=["domain-report", "methodology", "overview"])
    parser.add_argument("--selfcheck", action="store_true", help="rebuild current corpus, run benchmark + tests, verify baseline")
    parser.add_argument("--commit", action="store_true", help="commit and push successful additions")
    parser.add_argument("--skip-tests", action="store_true", help="skip the unittest suite (not recommended)")
    parser.add_argument("--dry-run", action="store_true", help="show the plan without changing anything")
    args = parser.parse_args()

    if args.selfcheck:
        log("selfcheck: rebuilding current corpus from manifest")
        if args.dry_run:
            log(f"dry-run: would rebuild {len(load_manifest())} sources and run the 50q benchmark")
            return
        ingestion = rebuild_corpus()
        log(f"ingestion: {ingestion}")
        benchmark = run_benchmark("selfcheck")
        ok, gate = metrics_ok(benchmark["metrics"], baseline())
        tests_ok, tests_line = run_tests()
        log(f"baseline gate: {gate}")
        log(f"tests: {tests_line}")
        if not (ok and tests_ok):
            raise SystemExit("selfcheck FAILED")
        measured = {
            "hit_at_5": round(benchmark["metrics"]["hit_rate_at_k"], 4),
            "recall_at_5": round(benchmark["metrics"]["recall_at_k"], 4),
            "mrr_at_5": round(benchmark["metrics"]["mrr_at_k"], 4),
        }
        if measured != baseline():
            BASELINE_FILE.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
            log(f"baseline re-frozen from measured corpus: {measured} -> {BASELINE_FILE.name}")
        log("selfcheck PASS")
        return

    candidates: list[dict] = []
    if args.url:
        if not (args.title and args.source_id):
            raise SystemExit("--url requires --title and --source-id")
        candidates.append({
            "source_id": args.source_id,
            "url": args.url,
            "title": args.title,
            "publisher": args.publisher,
            "publication_date": args.publication_date or "unknown",
            "license": "USGS-public-domain-verify-source",
            "source_role": args.source_role,
        })
    elif args.auto:
        log(f"discovering up to {args.auto} candidates (USGS Pubs Warehouse, then OSTI)")
        candidates = pubs_candidates(args.auto)
        if len(candidates) < args.auto:
            candidates += osti_candidates(args.auto - len(candidates))
        log(f"discovered {len(candidates)} candidates")
    else:
        parser.print_help()
        raise SystemExit("choose --selfcheck, --auto N, or an explicit --url source")

    if args.dry_run:
        for record in candidates:
            log(f"dry-run: would add {record['source_id']} | {record['title'][:70]} | {record['url']}")
        return

    added = 0
    for record in candidates:
        manifest_before = len(load_manifest())
        tag = str(manifest_before + 1)
        try:
            if add_source(record, tag, args.commit, args.skip_tests):
                added += 1
        except SystemExit as error:
            rollback(record)
            record_rejection(record, str(error))
            log(f"candidate failed: {error}")
            continue
    log(f"done: {added}/{len(candidates)} sources added")


if __name__ == "__main__":
    main()
