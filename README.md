# Infino

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/infino-ai/infino)
[![Crates.io](https://img.shields.io/crates/v/infino.svg)](https://crates.io/crates/infino)
[![docs.rs](https://img.shields.io/docsrs/infino)](https://docs.rs/infino)
[![CI](https://github.com/infino-ai/infino/actions/workflows/ci.yml/badge.svg)](https://github.com/infino-ai/infino/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

# Fast, embedded search and analytics.

Infino is a drop-in library for searching and analyzing your data. Full-text, vector,
hybrid, and SQL over the same table, with **125 µs** BM25 and **591 µs** vector top-10 warm.

It starts in memory on your laptop. When the data outgrows RAM, you change the connection
string and nothing else — the tables are Parquet on object storage, and the engine reads
them in place.

```python
db = infino.connect("memory://")            # laptop
db = infino.connect("./data")               # on-prem disk
db = infino.connect("s3://bucket/prefix")   # past RAM, same app code
```

Built by the team that created OpenSearch.

```sh
pip install infino              # Python
npm install @infino-ai/infino   # Node.js
cargo add infino                # Rust
```

## Quickstart

```python
import infino
import pyarrow as pa

db = infino.connect("memory://")

schema = pa.schema([
    pa.field("body", pa.large_utf8(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32(), 384), nullable=False),
])
docs = db.create_table(
    "docs", schema,
    infino.IndexSpec().fts("body").vector("embedding", 384, "cosine"),
)

docs.append(rows)   # list of dicts, or an Arrow RecordBatch

# One call, both signals, fused ranking. `query_vec` is your embedding.
hits = docs.hybrid_search("body", "disk full", "embedding", query_vec, k=10)
```

## Performance

Warm p50, tables on object storage:

| | 1M docs | 10M docs |
|---|---|---|
| **Vector** top-10, recall@10 0.992 | **591 µs** | **5 ms** |
| **BM25** top-10, incl. row fetch | **125 µs** | **2 ms** |
| **SQL** point lookup → crosstab | **186 µs – 7.6 ms** | **260 µs – 75 ms** |

Cold, meaning the first query on an idle table while handles open and the cache fills, is
114 ms and 314 ms for vector, 16 ms and 275 ms for BM25. The charts are log-scale because
warm and cold sit about 200× apart.

![Vector search latency, log scale, 1M and 10M documents](docs/assets/readme/vector.svg)

![BM25 full-text search latency, log scale, 1M and 10M documents](docs/assets/readme/fts.svg)

![SQL query shape latency, log scale, 1M and 10M rows](docs/assets/readme/sql.svg)

Measured on Azure Blob with 4 cores pinned ([CI run 33245831329](https://github.com/infino-ai/infino/actions/runs/33245831329))
for 1M, and on a separate run at the default 10M scale. The two scales come from different
commits and different hardware, so read each against its own baseline rather than against
the other.

<details>
<summary><b>Reproduce every chart</b> — config and exact commands</summary>

Engine behavior is configured in YAML only; environment variables never override it. Start
from the shipped defaults, which are exactly what the charts measure:

```sh
cp src/config/config.yaml infino.yaml    # or $XDG_CONFIG_HOME/infino/config.yaml
```

The `vector:` block holds probe depth, rerank codec, and cell counts. The `supertable:`
block holds commit and cache behavior. Leave both alone to reproduce the published charts.

Corpus size is the one bench knob that is an environment variable, and it takes a plain
integer (`1000000`, not `1M`). The table tier defaults to 10M:

| Chart | Command |
|---|---|
| Vector, 10M | `cargo bench -- supertable vector warm cold` |
| BM25, 10M | `cargo bench -- supertable fts warm cold` |
| SQL, 10M | `cargo bench -- supertable sql warm` |
| Any chart, 1M | prefix with `INFINO_BENCH_SUPERTABLE_DOCS=1000000` |

That runs against a local RustFS daemon, an HTTPS S3 stand-in, by default. To match CI:

```sh
INFINO_BENCH_SUPERTABLE_DOCS=1000000 \
INFINO_BENCH_STORE=azure \
INFINO_REAL_AZURE_CONTAINER=$CONTAINER \
AZURE_STORAGE_ACCOUNT_NAME=$ACCOUNT \
AZURE_STORAGE_ACCOUNT_KEY=$KEY \
  cargo bench -- supertable vector warm cold
```

Reading the output: vector is the post-drain `default` row; BM25 is `single_rare` under
**Supertable FTS — queries + cost**; the SQL shapes are `agg_max_title` (metadata),
`WHERE key = ?` (lookup), `AVG(rating) GROUP BY category` (scan), and
`COUNT(*) GROUP BY bucket, category` (crosstab). Structured results land in
`target/infino-bench/*.json`. Methodology: [benches/README.md](benches/README.md).

</details>

### Where it lands

The goal is one library that stays close to the specialized engine in each category.
Measured on the public harnesses:

- **Vector** — lowest p99 on [VectorDBBench](https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch)
  ([client](https://github.com/infino-ai/VectorDBBench/tree/main/vectordb_bench/backend/clients/infino))
- **Full-text** — within 19% of Lucene on search, and faster on count, at
  [Search Benchmark, the Game](https://tantivy-search.github.io/bench/)
  ([harness](https://github.com/quickwit-oss/search-benchmark-game))
- **SQL** — 12.8 vCPU-seconds per query on
  [ClickBench](https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino%7CDataFusion%20%28Parquet%2C%20single%29%7CSpark%7CPostgreSQL%20%28with%20indexes%29&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot),
  in the same range as DuckDB at 9.8
  ([port](https://github.com/infino-ai/clickbench/tree/add-infino/infino))

Lucene and DuckDB are excellent, and they are the right yardsticks. The goal is to be in
that range on each axis while serving all four query types from one file.

## Why it's fast

![Your app queries Infino, which caches in RAM and on disk over Parquet on object storage](docs/assets/readme/one-parquet-copy.svg)

### The index lives inside the data file

Infino writes one file per batch, and that file is ordinary Parquet with the search indexes
stored inside it. DuckDB, pyarrow, and DataFusion open it and see a normal table, with Infino
nowhere in the read path ([example](infino-python/examples/parquet_interop.py)). Infino opens
the same bytes and uses the indexes.

So there is no second artifact to build, ship, load, or keep in sync with the data, and no
"open the index" step at startup.

### A query reads a few byte ranges, not a file

The indexes are laid out by term and by vector cluster, so a top-10 resolves to a short list
of byte offsets. On object storage those are HTTP range requests — a handful of them, rather
than a download. This is the part that makes running search directly on S3 practical, since
each round trip costs 20–100 ms and the goal is to need very few.

Fetched ranges are kept in a local disk cache and memory-mapped, so the next query over the
same terms is page-cache reads with no network at all. That is where the 125 µs comes from.
The cache is reclaimable: it shrinks under memory pressure, down to nothing on an idle table,
and refills on demand.

### Scoring is SIMD over compressed vectors

Vectors are stored as 1-bit-per-dimension RaBitQ codes — `dim / 8` bytes per row, so 128
bytes for a 1024-dimension vector against 4 KB as raw float32. That is 32× less to fetch and
score, and it is small enough that a whole cluster of candidates stays in cache.

The distance kernels are hand-written intrinsics chosen at runtime: AVX-512 where the CPU
reports it, AVX2 otherwise, a portable 256-bit path elsewhere, and an int8 VNNI kernel for
the graph walk. Finalists get rescored against a higher-precision copy, which is how the
compression stays free — recall@10 is 0.992.

### Writes append, and readers never block

A table is a set of immutable files plus a small manifest listing them. Appending writes new
files and then swaps the manifest in one atomic step, so a commit publishes everything or
nothing. A query pins the manifest it started with and reads that version to completion,
which means readers never wait on a writer and never see a half-finished commit.

Nothing has to coordinate for that to hold — no lock service, no elected leader. That is what
lets Infino be a library you link in rather than a service you operate.

### Memory

Only the codes need to be resident, and they are the compressed copy:

| Vector dimensions | RaBitQ codes | Same data as float32 |
|---|---|---|
| 384 | 48 B/row | 1.5 KB/row |
| 768 | 96 B/row | 3 KB/row |
| 1024 | 128 B/row | 4 KB/row |

At 1024 dimensions, 10M vectors is about 1.3 GB of codes. The higher-precision rerank copy is
2 bytes per dimension and lives in the disk cache rather than in RAM; setting the rerank codec
to `rabitq_only` drops it entirely. Every bench table in
[benches/README.md](benches/README.md) reports peak, median, and p90 RSS alongside latency,
so the memory cost of each shape is recorded next to its speed.

## Retrieval is a relation

`bm25_search`, `vector_search`, `hybrid_search`, `token_match`, and `exact_match` are SQL
table-valued functions, so a ranked result set is an ordinary table. Retrieval, filters,
joins, and aggregation compose in one statement against one pinned snapshot. One query
replaces several, which matters when an agent pays a round trip per call.

```sql
-- a ranked search is a relation: join it, group it, aggregate it
SELECT   s.team,
         count(*)      AS hits,
         avg(h.score)  AS relevance
FROM     hybrid_search('logs', 'body', 'disk full', 'embedding', :q, 1000) AS h
JOIN     services s ON s.id = h.service_id
WHERE    h.ts > now() - interval '7 days'
GROUP BY s.team
ORDER BY hits DESC;
```

## Indexes calibrate themselves

You declare a recall target and the engine decides how to reach it. On `optimize()`, Infino
measures the corpus and picks the vector path: an IVF cell scan over 1-bit RaBitQ codes, or
a resident HNSW graph when the table fits RAM and the distribution suits a graph. A
calibrated graph that cannot reach the target is discarded and the scan path serves instead.
Per-cell probe width and depth are re-measured whenever compaction or a cell split changes
the geometry they were fitted against.

```yaml
# infino.yaml
vector:
  target_recall: 0.99      # declare the goal
  search_mode: hnsw_ivf    # default: ivf. hnsw_ivf falls back to ivf automatically
```

```python
table.optimize()    # drain, compact, recalibrate, sweep
```

Every knob, with the measurement behind each default, is documented inline in
[`src/config/config.yaml`](src/config/config.yaml).

## When not to use Infino

- **You need per-row durability.** Commit is the durability boundary. Rows are durable when
  a commit lands, not when `append()` returns.
- **You need OLTP.** Tables are append-only and time-ordered; updates are delete plus insert
  via tombstones. There are no transactions across tables.
- **You need many concurrent writers on one table.** Writes go through a single writer slot.
  Readers are unbounded and never blocked.
- **You want a server.** This crate is embedded, with a SQL and Arrow surface. There is no
  daemon to run, no REST endpoint, and no cluster to operate.

The crate is 0.x and the API can still move. The public surface is pinned by
`public-api.txt`.

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
