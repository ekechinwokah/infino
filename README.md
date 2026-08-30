# Infino

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/infino-ai/infino)
[![Crates.io](https://img.shields.io/crates/v/infino.svg)](https://crates.io/crates/infino)
[![docs.rs](https://img.shields.io/docsrs/infino)](https://docs.rs/infino)
[![CI](https://github.com/infino-ai/infino/actions/workflows/ci.yml/badge.svg)](https://github.com/infino-ai/infino/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**SQL, full-text, and vector search over one copy of your data on object storage.**

Infino is an embedded retrieval engine. A "superfile" is a valid Apache Parquet file with
BM25 and vector indexes spliced into it, so one copy serves all three query types — no
daemon, no separate search cluster, no second copy to keep in sync.

![BM25, vector, hybrid, and SQL query one Parquet copy through Infino](docs/assets/readme/one-parquet-copy.svg)

```sh
pip install infino              # Python
npm install @infino-ai/infino   # Node.js
cargo add infino                # Rust
```

## Performance

Warm p50, supertable on object storage:

| | 1M docs | 10M docs |
|---|---|---|
| **Vector** top-10, recall@10 0.992 | **591 µs** | **5 ms** |
| **BM25** top-10, incl. row fetch | **125 µs** | **2 ms** |
| **SQL** point lookup → crosstab | **186 µs – 7.6 ms** | **260 µs – 75 ms** |

Cold — the first query against an idle table, while handles open and the cache fills — is
114 ms / 314 ms for vector and 16 ms / 275 ms for BM25. Charts are log-scale, since warm
and cold are ~200× apart.

The two scales come from **separate measurement windows** and are not directly comparable
to each other: 1M is Azure CI ([run 33245831329](https://github.com/infino-ai/infino/actions/runs/33245831329),
4 cores pinned), 10M is the default supertable scale measured on its own run.

![Vector search latency, log scale, 1M and 10M documents](docs/assets/readme/vector.svg)

![BM25 full-text search latency, log scale, 1M and 10M documents](docs/assets/readme/fts.svg)

![SQL query shape latency, log scale, 1M and 10M rows](docs/assets/readme/sql.svg)

<details>
<summary><b>Reproduce these numbers</b> — config and exact commands</summary>

Engine behavior is configured in YAML only; environment variables never override it. Start
from the shipped defaults, which are what the charts above measure:

```sh
cp src/config/config.yaml infino.yaml    # or $XDG_CONFIG_HOME/infino/config.yaml
```

Edit the `vector:` block to change probe depth, rerank codec, or cell counts; `supertable:`
for commit and cache behavior. Leave both alone to reproduce the charts as published.

Corpus size is the one bench knob that *is* an environment variable, and it takes a plain
integer (`1000000`, not `1M`). Supertable defaults to 10M:

| Chart | Command |
|---|---|
| Vector, 10M | `cargo bench -- supertable vector warm cold` |
| BM25, 10M | `cargo bench -- supertable fts warm cold` |
| SQL, 10M | `cargo bench -- supertable sql warm` |
| Any chart, 1M | prefix with `INFINO_BENCH_SUPERTABLE_DOCS=1000000` |

That runs against a local RustFS daemon (an HTTPS S3 stand-in) by default. To match CI
exactly, add the Azure backend:

```sh
INFINO_BENCH_SUPERTABLE_DOCS=1000000 \
INFINO_BENCH_STORE=azure \
INFINO_REAL_AZURE_CONTAINER=$CONTAINER \
AZURE_STORAGE_ACCOUNT_NAME=$ACCOUNT \
AZURE_STORAGE_ACCOUNT_KEY=$KEY \
  cargo bench -- supertable vector warm cold
```

Reading the output: vector numbers are the **post-drain `default`** row; BM25 is
`single_rare` under **Supertable FTS — queries + cost**; SQL query names are
`agg_max_title` (metadata), `WHERE key = ?` (lookup), `AVG(rating) GROUP BY category`
(scan), and `COUNT(*) GROUP BY bucket, category` (crosstab). Structured results land in
`target/infino-bench/*.json`. Full methodology: [benches/README.md](benches/README.md).

</details>

### How Infino compares

Infino is a retrieval engine that also runs analytics, so it is measured against
specialists in each category on their own published harnesses.

![Vector search versus vector databases on VectorDBBench](docs/assets/readme/compare-vdb.svg)

![Full-text search versus Lucene and Tantivy on Search Benchmark, the Game](docs/assets/readme/compare-fts.svg)

![SQL on Parquet versus analytic engines on ClickBench](docs/assets/readme/compare-sql.svg)

Read honestly: Infino leads on vector, trades with Lucene on full-text (19% slower on
search, 26% faster on count), and lands mid-pack on ClickBench behind ClickHouse and
DuckDB. The point is that one engine covers all three from a single Parquet copy while
staying within range of each specialist.

Harnesses: [VectorDBBench](https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch)
(client in [infino-ai/VectorDBBench](https://github.com/infino-ai/VectorDBBench/tree/main/vectordb_bench/backend/clients/infino)) ·
[Search Benchmark, the Game](https://tantivy-search.github.io/bench/)
([harness](https://github.com/quickwit-oss/search-benchmark-game); Infino rows pending on the public board) ·
[ClickBench](https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino%7CDataFusion%20%28Parquet%2C%20single%29%7CSpark%7CPostgreSQL%20%28with%20indexes%29&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot)
([port](https://github.com/infino-ai/clickbench/tree/add-infino/infino)).

## Hybrid search is SQL

`bm25_search`, `vector_search`, `hybrid_search`, `token_match`, and `exact_match` are SQL
table-valued functions, so a ranked result set is an ordinary relation. Retrieval, filters,
joins, and aggregation compose in one statement against one pinned snapshot — no
client-side stitching between a search engine and a database.

```sql
SELECT _id, title, score
FROM hybrid_search('logs', 'body', 'disk full', 'embedding', :q, 50)
WHERE level = 'error'
  AND ts > now() - interval '24 hours'
ORDER BY score DESC
LIMIT 10;
```

## Quickstart

```python
import infino
import pyarrow as pa

db = infino.connect("memory://")   # or "./data", "s3://bucket/prefix"

schema = pa.schema([
    pa.field("body", pa.large_utf8(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32(), 16), nullable=False),
])
docs = db.create_table(
    "docs", schema,
    infino.IndexSpec().fts("body").vector("embedding", 16, "cosine"),
)

billing, appearance = [1.0] + [0.0] * 15, [0.0, 1.0] + [0.0] * 14
docs.append([
    {"body": "To cancel a subscription, open Settings then Billing.", "embedding": billing},
    {"body": "Enable dark mode under Settings then Appearance.",      "embedding": appearance},
])

hits = docs.hybrid_search("body", "cancel subscription", "embedding", billing, 5)
```

| Language | Quickstart | Examples |
|----------|------------|----------|
| Python | [infino-python/](infino-python/) | [examples/](infino-python/examples/) |
| Node.js | [infino-node/](infino-node/) | [examples/](infino-node/examples/) |
| Rust | [docs.rs/infino](https://docs.rs/infino) | [examples/](examples/) |

Because a superfile is valid Parquet, DuckDB, pyarrow, and DataFusion read the columns with
no Infino in the read path — see
[`parquet_interop.py`](infino-python/examples/parquet_interop.py).

## Architecture

A supertable manifest composes many immutable superfiles on object storage, giving
snapshot-isolated reads, append-only writes, and atomic commits.

- **[Overview →](docs/architecture/overview.md)** — mental model and comparisons
- **[Superfile format →](docs/architecture/superfile.md)** — on-disk layout
- **[Supertable layer →](docs/architecture/supertable.md)** — manifest, commit, query fan-out

Concepts and guides: **[infino.ai/docs](https://infino.ai/docs)**.

## Development

```sh
git clone git@github.com:infino-ai/infino.git && cd infino
cargo build
cargo run --example demo
make ci                # gates before a PR
make readme-charts     # regenerate the charts above
```

The public API is pinned by `public-api.txt` (`make public-api`). The crate is 0.x; MSRV
**1.95**. Python and Node version on their own SemVer lines — see
[docs/versioning.md](docs/versioning.md).

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed [Apache-2.0](LICENSE).
