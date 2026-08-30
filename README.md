# Infino

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/infino-ai/infino)
[![Crates.io](https://img.shields.io/crates/v/infino.svg)](https://crates.io/crates/infino)
[![docs.rs](https://img.shields.io/docsrs/infino)](https://docs.rs/infino)
[![CI](https://github.com/infino-ai/infino/actions/workflows/ci.yml/badge.svg)](https://github.com/infino-ai/infino/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**The search index lives inside the Parquet file.**

An Infino superfile is a valid Apache Parquet file with BM25 and vector indexes spliced
into it. DuckDB, pyarrow, and DataFusion open it as ordinary Parquet and never see the
indexes. Infino reads the same bytes and returns top-10 in **591 µs** warm — data resident on Azure
Blob, no daemon running, no index loaded into a cluster anywhere.

That is the whole idea. One copy of your data, on object storage, answering full-text,
vector, hybrid, and SQL queries. Nothing to deploy, nothing to keep in sync.

![BM25, vector, hybrid, and SQL query one Parquet copy through Infino](docs/assets/readme/one-parquet-copy.svg)

```sh
pip install infino              # Python
npm install @infino-ai/infino   # Node.js
cargo add infino                # Rust
```

```python
import infino

db = infino.connect("s3://my-bucket/prefix")     # or "./data", "memory://"
docs = db.open_table("docs")

# BM25, vector, and hybrid all return Arrow. `query_vec` is your embedding.
hits = docs.hybrid_search("body", "disk full", "embedding", query_vec, k=10)
```

## Why it's fast

Object storage is 20–100 ms away, so serving sub-millisecond search off S3 sounds like a
category error. Three things make it work.

**The index is byte-addressable.** A superfile's posting lists and IVF cells are laid out so
a top-k query resolves to a bounded set of byte ranges rather than the whole file. Cold,
those are exact range GETs; when the manifest carries the open batch, opening costs *zero*
GETs against the superfile itself. There is no "load the index" step, because the index is
never loaded — it's read in place.

**Warm queries never leave the machine.** Fetched ranges land in an NVMe-backed disk cache
and are memory-mapped, so a repeat query is page-cache reads and SIMD scoring. That is where
591 µs comes from. The cache is reclaimable: it shrinks under memory pressure, down to zero.

**Commits are a pointer swap.** A supertable manifest lists immutable superfiles. Writers
append new files and atomically swap the manifest; readers pin a snapshot and are never
blocked. No coordinator, no leader election, no daemon to run — which is why this is a
library you link, not a service you operate.

## Performance

Warm p50, supertable on object storage:

| | 1M docs | 10M docs |
|---|---|---|
| **Vector** top-10, recall@10 0.992 | **591 µs** | **5 ms** |
| **BM25** top-10, incl. row fetch | **125 µs** | **2 ms** |
| **SQL** point lookup → crosstab | **186 µs – 7.6 ms** | **260 µs – 75 ms** |

Cold — first query on an idle table, while handles open and the cache fills — is 114 ms and
314 ms for vector, 16 ms and 275 ms for BM25. Charts are log-scale because warm and cold sit
~200× apart.

![Vector search latency, log scale, 1M and 10M documents](docs/assets/readme/vector.svg)

![BM25 full-text search latency, log scale, 1M and 10M documents](docs/assets/readme/fts.svg)

![SQL query shape latency, log scale, 1M and 10M rows](docs/assets/readme/sql.svg)

Measured on Azure Blob, 4 cores pinned ([CI run 33245831329](https://github.com/infino-ai/infino/actions/runs/33245831329)) for 1M,
and on a separate run at the default 10M scale. The two scales are not comparable to each
other — different commits, different hardware.

<details>
<summary><b>Reproduce every chart</b> — config and exact commands</summary>

Engine behavior is configured in YAML only; environment variables never override it. Start
from the shipped defaults, which are exactly what the charts measure:

```sh
cp src/config/config.yaml infino.yaml    # or $XDG_CONFIG_HOME/infino/config.yaml
```

The `vector:` block holds probe depth, rerank codec, and cell counts; `supertable:` holds
commit and cache behavior. Leave both alone to reproduce the published charts.

Corpus size is the one bench knob that *is* an environment variable, and it takes a plain
integer (`1000000`, not `1M`). Supertable defaults to 10M:

| Chart | Command |
|---|---|
| Vector, 10M | `cargo bench -- supertable vector warm cold` |
| BM25, 10M | `cargo bench -- supertable fts warm cold` |
| SQL, 10M | `cargo bench -- supertable sql warm` |
| Any chart, 1M | prefix with `INFINO_BENCH_SUPERTABLE_DOCS=1000000` |

That runs against a local RustFS daemon (an HTTPS S3 stand-in) by default. To match CI:

```sh
INFINO_BENCH_SUPERTABLE_DOCS=1000000 \
INFINO_BENCH_STORE=azure \
INFINO_REAL_AZURE_CONTAINER=$CONTAINER \
AZURE_STORAGE_ACCOUNT_NAME=$ACCOUNT \
AZURE_STORAGE_ACCOUNT_KEY=$KEY \
  cargo bench -- supertable vector warm cold
```

Reading the output: vector is the **post-drain `default`** row; BM25 is `single_rare` under
**Supertable FTS — queries + cost**; SQL shapes are `agg_max_title` (metadata),
`WHERE key = ?` (lookup), `AVG(rating) GROUP BY category` (scan), and
`COUNT(*) GROUP BY bucket, category` (crosstab). Structured results land in
`target/infino-bench/*.json`. Methodology: [benches/README.md](benches/README.md).

</details>

### Against the specialists

![Vector search versus vector databases on VectorDBBench](docs/assets/readme/compare-vdb.svg)

![Full-text search versus Lucene and Tantivy on Search Benchmark, the Game](docs/assets/readme/compare-fts.svg)

![SQL on Parquet versus analytic engines on ClickBench](docs/assets/readme/compare-sql.svg)

Infino is fastest on VectorDBBench, trades with Lucene on full-text (19% slower on search,
26% faster on count), and sits behind ClickHouse and DuckDB on ClickBench. Every one of
those systems does one of the three things Infino does, and none of them read a file the
others can also read.

Harnesses: [VectorDBBench](https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch)
([client](https://github.com/infino-ai/VectorDBBench/tree/main/vectordb_bench/backend/clients/infino)) ·
[Search Benchmark, the Game](https://tantivy-search.github.io/bench/)
([harness](https://github.com/quickwit-oss/search-benchmark-game); Infino rows pending on the public board) ·
[ClickBench](https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino%7CDataFusion%20%28Parquet%2C%20single%29%7CSpark%7CPostgreSQL%20%28with%20indexes%29&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot)
([port](https://github.com/infino-ai/clickbench/tree/add-infino/infino)).

## Retrieval is a relation

`bm25_search`, `vector_search`, `hybrid_search`, `token_match`, and `exact_match` are SQL
table-valued functions. A ranked result set is an ordinary relation, so retrieval, filters,
joins, and aggregation compose in one statement against one pinned snapshot — instead of
ranking in a search engine and re-filtering the results in your application.

```sql
SELECT _id, title, score
FROM hybrid_search('logs', 'body', 'disk full', 'embedding', :q, 50)
WHERE level = 'error'
  AND ts > now() - interval '24 hours'
ORDER BY score DESC
LIMIT 10;
```

The same table is still Parquet, so anything in the Arrow ecosystem can read the columns
with Infino nowhere in the read path — see
[`parquet_interop.py`](infino-python/examples/parquet_interop.py).

## Indexes calibrate themselves

You declare a recall target; the engine decides how to hit it. On `optimize()`, Infino
measures the corpus and picks the vector index path — an IVF cell scan over 1-bit RaBitQ
codes, or a resident HNSW graph when the table fits RAM and the distribution suits a graph.
If a calibrated graph can't reach the target it is discarded and the scan path serves. It
also re-measures per-cell probe width and depth whenever compaction or a cell split changes
the geometry those laws were fitted against.

```yaml
# infino.yaml
vector:
  target_recall: 0.99      # declare the goal
  search_mode: hnsw_ivf    # default: ivf. hnsw_ivf falls back to ivf automatically
```

```python
table.optimize()    # drain, compact, recalibrate, sweep
```

Every knob, with the measurements behind each default, is documented inline in
[`src/config/config.yaml`](src/config/config.yaml) — including graph build parameters, cell
split triggers, and the maintenance thread budget.

## When not to use Infino

- **You need per-row durability.** Commit is the durability boundary. Rows are durable when
  a commit lands, not when `append()` returns.
- **You need OLTP.** Tables are append-only and time-ordered; updates are delete + insert
  via tombstones. There are no transactions across tables.
- **You need many concurrent writers per table.** Writes go through a single writer slot.
  Readers are unbounded and never blocked.
- **You want a server.** Infino is a library. If you want something to point a cluster at,
  this is the wrong shape.

The crate is 0.x and the API can move; the public surface is pinned by `public-api.txt`.

## Documentation

- **[Overview →](docs/architecture/overview.md)** — the mental model, and how this compares
- **[Superfile format →](docs/architecture/superfile.md)** — how indexes fit inside Parquet
- **[Supertable layer →](docs/architecture/supertable.md)** — manifest, commit, query fan-out
- **[infino.ai/docs](https://infino.ai/docs)** — concepts and guides

| Language | Package | Examples |
|----------|---------|----------|
| Python | [infino-python/](infino-python/) | [examples/](infino-python/examples/) |
| Node.js | [infino-node/](infino-node/) | [examples/](infino-node/examples/) |
| Rust | [docs.rs/infino](https://docs.rs/infino) | [examples/](examples/) |

## Development

```sh
git clone git@github.com:infino-ai/infino.git && cd infino
cargo build
cargo run --example demo
make ci                # gates before a PR
make readme-charts     # regenerate the charts above
```

MSRV **1.95**. Python and Node version on their own SemVer lines
([docs/versioning.md](docs/versioning.md)). See [CONTRIBUTING.md](CONTRIBUTING.md).
Licensed [Apache-2.0](LICENSE).
