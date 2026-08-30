# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Numbers and provenance for the README performance charts.

Two measurement windows feed these charts and they are *not* directly
comparable — different commits, different hardware:

  1M  — CI Benchmark (Azure Blob, 4 cores pinned), run 33245831329 on
        commit 3aaffb64, 2026-08-29.
  10M — the default supertable scale, measured separately (commit 339e621
        window). See benches/README.md.

Every chart therefore labels its scale groups and carries both provenances in
the footnote. Footnote text is drawn straight into SVG, so it must stay plain:
no markdown, no angle brackets.
"""

from __future__ import annotations

from dataclasses import dataclass

# Milliseconds is the single internal unit; the renderer picks µs/ms/s per value.
US = 1 / 1000


@dataclass(frozen=True)
class Bar:
    """One horizontal bar. `group` heads a run of consecutive bars."""

    group: str
    label: str
    ms: float
    cold: bool = False


@dataclass(frozen=True)
class CompareRow:
    name: str
    value: float
    label: str
    config: str = ""
    self_row: bool = False


CI_RUN = "33245831329"
CI_RUN_URL = f"https://github.com/infino-ai/infino/actions/runs/{CI_RUN}"
CI_COMMIT = "3aaffb64"

# Both windows in one line, plain text, for the chart footnotes.
PROVENANCE = (
    f"1M: Azure Blob, 4 cores pinned, CI run {CI_RUN} ({CI_COMMIT}) "
    "· 10M: supertable default scale, separate run"
)

# ── Internal latency ────────────────────────────────────────────────────────

VECTOR = {
    "title": "Vector search",
    "subtitle": "1024-d cosine · top-10 · post-drain · recall@10 0.992",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "warm p50", 591 * US),
        Bar("1M", "warm p99", 687 * US),
        Bar("1M", "cold p50", 114, cold=True),
        Bar("10M", "warm p50", 5),
        Bar("10M", "warm p99", 12),
        Bar("10M", "cold p50", 314, cold=True),
    ],
}

FTS = {
    "title": "Full-text search (BM25)",
    "subtitle": "top-10 including row fetch · 1M: single_rare · 10M: median shape",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "warm p50", 125 * US),
        Bar("1M", "warm p99", 143 * US),
        Bar("1M", "cold p50", 16.4, cold=True),
        Bar("10M", "warm p50", 2),
        Bar("10M", "warm p99", 7),
        Bar("10M", "cold p50", 275, cold=True),
    ],
}

SQL = {
    "title": "SQL query shapes",
    "subtitle": "warm p50 · supertable on object storage",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "metadata", 186 * US),
        Bar("1M", "lookup", 4.41),
        Bar("1M", "scan", 5.16),
        Bar("1M", "crosstab", 7.63),
        Bar("10M", "metadata", 260 * US),
        Bar("10M", "lookup", 2.74),
        Bar("10M", "scan", 41.14),
        Bar("10M", "crosstab", 75.14),
    ],
}

# ── External comparisons ────────────────────────────────────────────────────

VDB_ROWS: list[CompareRow] = [
    CompareRow("Infino", 1.1, "1.1 ms", "16c64g", self_row=True),
    CompareRow("Zilliz Cloud", 2.0, "2.0 ms", "8cu-perf"),
    CompareRow("Qdrant Cloud", 6.4, "6.4 ms", "16c64g"),
    CompareRow("OpenSearch", 7.2, "7.2 ms", "16c128g force-merge"),
    CompareRow("Elastic Cloud", 9.5, "9.5 ms", "8c60g force-merge"),
    CompareRow("Pinecone", 13.7, "13.7 ms", "p2.x8 1node"),
]

VDB_META = {
    "title": "Vector search vs vector databases",
    "subtitle": "VectorDBBench · Cohere 1M · 768-d · top-100 · serial p99 · lower is faster",
    "url": "https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch",
}

# (name, version, search ratio, count ratio, is_infino)
SBG_ROWS: list[tuple[str, str, float, float, bool]] = [
    ("Infino", "0.1", 1.19, 0.74, True),
    ("Lucene", "10.5.0", 1.0, 1.0, False),
    ("Tantivy", "0.26", 1.15, 0.80, False),
]

SBG_META = {
    "title": "Full-text vs search libraries",
    "subtitle": "Search Benchmark, the Game · latency vs Lucene = 1.00 · lower is faster",
    "url": "https://tantivy-search.github.io/bench/",
}

SQL_EXT_ROWS: list[CompareRow] = [
    CompareRow("ClickHouse", 6.8, "6.8", "18.4 s suite"),
    CompareRow("DuckDB", 9.8, "9.8", "26.3 s suite"),
    CompareRow("Infino", 12.8, "12.8", "34.0 s suite", self_row=True),
    CompareRow("DataFusion", 17.0, "17.0", "45.6 s suite"),
    CompareRow("Spark", 123.7, "123.7", "332.4 s suite"),
    CompareRow("Postgres", 1519.0, "1519", "4085.5 s suite"),
]

SQL_EXT_META = {
    "title": "SQL on Parquet vs analytic engines",
    "subtitle": "ClickBench 100M rows · vCPU-sec per query · hot · c6a.4xlarge · lower is faster",
    "url": (
        "https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino"
        "%7CDataFusion%20%28Parquet%2C%20single%29%7CSpark%7CPostgreSQL%20%28with%20indexes%29"
        "&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot"
    ),
}
