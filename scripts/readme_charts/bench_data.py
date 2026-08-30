# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Numbers and reproduction metadata for README performance charts.

Internal 1M rows: CI Benchmark (Azure) run 33245831329 on commit 3aaffb64
(2026-08-29). Internal 10M rows: measured supertable defaults (see benches/
README.md); latencies match infino.ai as of the same measurement window.

External comparison rows mirror the published harnesses cited on infino.ai.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyRow:
    label: str
    warm_us: float | None = None
    cold_us: float | None = None
    warm_ms: float | None = None
    cold_ms: float | None = None
    note: str = ""


@dataclass(frozen=True)
class CompareRow:
    name: str
    value: float
    label: str
    config: str = ""
    self_row: bool = False


CONFIG_PATH = "src/config/config.yaml"
PROJECT_CONFIG = "infino.yaml"

# ── Internal: 1M supertable on Azure (CI) ───────────────────────────────────

CI_RUN_URL = "https://github.com/infino-ai/infino/actions/runs/33245831329"
CI_COMMIT = "3aaffb649f6afb119162c3050cfbbe2983d8eae6"

VECTOR_1M = {
    "title": "Vector search · 1M docs",
    "subtitle": "1024-d cosine · top-10 · post-drain · recall@10 0.992",
    "footnote": f"Azure Blob · 4 cores pinned · [{CI_COMMIT[:7]}]({CI_RUN_URL})",
    "rows": [
        LatencyRow("p50", warm_us=591, cold_ms=114),
        LatencyRow("p99", warm_us=687, cold_ms=114),
    ],
    "repro": """\
```sh
# Start from shipped defaults (optional project override):
cp {config} {project}

# Match CI scale + backend (local alternative: omit INFINO_BENCH_STORE=azure):
INFINO_BENCH_SUPERTABLE_DOCS=1000000 \\
INFINO_BENCH_STORE=azure \\
INFINO_REAL_AZURE_CONTAINER=$CONTAINER \\
AZURE_STORAGE_ACCOUNT_NAME=$ACCOUNT \\
AZURE_STORAGE_ACCOUNT_KEY=$KEY \\
  cargo bench -- supertable vector warm cold
```

Post-drain `default` row in the log. Vector tuning knobs live under `vector:` in
`{project}` (rerank codec, cell counts, fine-probe floor); defaults reproduce CI.""",
}

FTS_1M = {
    "title": "Full-text search · 1M docs",
    "subtitle": "BM25 · single_rare · top-10 · including row fetch",
    "footnote": f"Azure Blob · 4 cores pinned · [{CI_COMMIT[:7]}]({CI_RUN_URL})",
    "rows": [
        LatencyRow("p50", warm_us=125, cold_ms=16.4),
        LatencyRow("p99", warm_us=143, cold_ms=16.4),
    ],
    "repro": """\
```sh
cp {config} {project}

INFINO_BENCH_SUPERTABLE_DOCS=1000000 \\
INFINO_BENCH_STORE=azure \\
INFINO_REAL_AZURE_CONTAINER=$CONTAINER \\
AZURE_STORAGE_ACCOUNT_NAME=$ACCOUNT \\
AZURE_STORAGE_ACCOUNT_KEY=$KEY \\
  cargo bench -- supertable fts warm cold
```

Look for `single_rare` in **Supertable FTS — queries + cost**. FTS tokenizer
settings are compile-time; corpus shape is fixed Zipfian 200 tokens/doc in the
harness (`benches/utils/corpus.rs`).""",
}

SQL_1M = {
    "title": "SQL · 1M rows",
    "subtitle": "Warm p50 · supertable on object storage",
    "footnote": f"Azure Blob · [{CI_COMMIT[:7]}]({CI_RUN_URL})",
    "rows": [
        LatencyRow("metadata aggregate", warm_us=186, note="agg_max_title"),
        LatencyRow("lookup aggregate", warm_ms=4.41, note="WHERE key = ?"),
        LatencyRow("scan aggregate", warm_ms=5.16, note="AVG GROUP BY category"),
        LatencyRow("crosstab aggregate", warm_ms=7.63, note="GROUP BY bucket, category"),
    ],
    "repro": """\
```sh
cp {config} {project}

INFINO_BENCH_SUPERTABLE_DOCS=1000000 \\
INFINO_BENCH_STORE=azure \\
INFINO_REAL_AZURE_CONTAINER=$CONTAINER \\
AZURE_STORAGE_ACCOUNT_NAME=$ACCOUNT \\
AZURE_STORAGE_ACCOUNT_KEY=$KEY \\
  cargo bench -- supertable sql warm
```

Query names match the **Supertable SQL — queries + cost** table in the log.""",
}

# ── Internal: 10M supertable (default scale) ───────────────────────────────

VECTOR_10M = {
    "title": "Vector search · 10M docs",
    "subtitle": "1024-d cosine · top-10 · supertable default scale",
    "footnote": "Measured supertable path · see [benches/README.md](benches/README.md)",
    "rows": [
        LatencyRow("p50", warm_ms=5, cold_ms=314),
        LatencyRow("p99", warm_ms=12, cold_ms=850),
    ],
    "repro": """\
```sh
cp {config} {project}

# Default supertable scale is 10M docs (1024-d synthetic vectors):
cargo bench -- supertable vector warm cold
```

Requires ~32 GiB+ RAM for the mmap corpus. Pin cores locally with
`taskset`/`cpuset` if comparing across runs. Results land in
`target/infino-bench/supertable_vector.json`.""",
}

FTS_10M = {
    "title": "Full-text search · 10M docs",
    "subtitle": "BM25 · median query shape · top-10",
    "footnote": "Measured supertable path · see [benches/README.md](benches/README.md)",
    "rows": [
        LatencyRow("p50", warm_ms=2, cold_ms=275),
        LatencyRow("p99", warm_ms=7, cold_ms=720),
    ],
    "repro": """\
```sh
cp {config} {project}

cargo bench -- supertable fts warm cold
```""",
}

SQL_10M = {
    "title": "SQL · 10M rows",
    "subtitle": "Warm p50 · bounded-result query shapes",
    "footnote": "Measured supertable path (Azure, commit 339e621 window)",
    "rows": [
        LatencyRow("metadata aggregate", warm_ms=0.26),
        LatencyRow("lookup aggregate", warm_ms=2.74),
        LatencyRow("scan aggregate", warm_ms=41.14),
        LatencyRow("crosstab aggregate", warm_ms=75.14),
    ],
    "repro": """\
```sh
cp {config} {project}

cargo bench -- supertable sql warm
```""",
}

# ── External comparisons ────────────────────────────────────────────────────

VDB_ROWS: list[CompareRow] = [
    CompareRow("Infino", 1.1, "1.1ms", "16c64g", self_row=True),
    CompareRow("Zilliz Cloud", 2.0, "2.0ms", "8cu-perf"),
    CompareRow("Qdrant Cloud", 6.4, "6.4ms", "16c64g"),
    CompareRow("OpenSearch", 7.2, "7.2ms", "16c128g force-merge"),
    CompareRow("Elastic Cloud", 9.5, "9.5ms", "8c60g force-merge"),
    CompareRow("Pinecone", 13.7, "13.7ms", "p2.x8 1node"),
]

VDB_META = {
    "title": "Vector search vs vector databases",
    "subtitle": "VectorDBBench · Cohere 1M · 768-d · top-100 · serial p99",
    "url": "https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch",
    "repro": """\
Infino ships a VectorDBBench client in
[`infino-ai/VectorDBBench`](https://github.com/infino-ai/VectorDBBench/tree/main/vectordb_bench/backend/clients/infino).
Follow the upstream harness, then compare against the published leaderboard row above.""",
}

SBG_ROWS: list[tuple[str, str, float, float, bool]] = [
    ("Infino", "0.1", 1.19, 0.74, True),
    ("Lucene", "10.5.0", 1.0, 1.0, False),
    ("Tantivy", "0.26", 1.15, 0.8, False),
]

SBG_META = {
    "title": "Full-text vs search libraries",
    "subtitle": "Search Benchmark, the Game · latency relative to Lucene = 1.00",
    "url": "https://tantivy-search.github.io/bench/",
    "repro": """\
Submit/build via the [search-benchmark-game](https://github.com/quickwit-oss/search-benchmark-game)
harness. Infino rows are pending publication on the public board; numbers above
are from our submitted run.""",
}

SQL_EXT_ROWS: list[CompareRow] = [
    CompareRow("ClickHouse", 6.8, "6.8", "18.4s suite"),
    CompareRow("DuckDB", 9.8, "9.8", "26.3s suite"),
    CompareRow("Infino", 12.8, "12.8", "34.0s suite", self_row=True),
    CompareRow("DataFusion", 17.0, "17.0", "45.6s suite"),
    CompareRow("Spark", 123.7, "123.7", "332.4s suite"),
    CompareRow("Postgres", 1519.0, "1519", "4085.5s suite"),
]

SQL_EXT_META = {
    "title": "SQL on Parquet vs analytic engines",
    "subtitle": "ClickBench 100M rows · vCPU-seconds per query · hot runs · c6a.4xlarge",
    "url": "https://benchmark.clickhouse.com/#system=+ClickHouse|DuckDB|Infino|DataFusion%20(Parquet,%20single)|Spark|PostgreSQL%20(with%20indexes)&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot",
    "repro": """\
Infino's ClickBench port lives in
[`infino-ai/clickbench`](https://github.com/infino-ai/clickbench/tree/add-infino/infino).
Run the 43-query suite on c6a.4xlarge at 100M rows, hot runs, Parquet single-file.""",
}
