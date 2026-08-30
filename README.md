# Infino

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/infino-ai/infino)
[![Crates.io](https://img.shields.io/crates/v/infino.svg)](https://crates.io/crates/infino)
[![docs.rs](https://img.shields.io/docsrs/infino)](https://docs.rs/infino)
[![CI](https://github.com/infino-ai/infino/actions/workflows/ci.yml/badge.svg)](https://github.com/infino-ai/infino/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Infino is an embedded retrieval library. It runs full-text, vector, hybrid, and SQL queries
over the same table, and stores that table as ordinary Parquet. Warm top-10 on a
1M-document table is 125 µs for BM25 and 591 µs for vector search.

The storage target is a connection string. Nothing else in your code changes between them,
and a table larger than RAM works because the engine reads Parquet in place instead of
loading an index into memory.

```python
db = infino.connect("memory://")
db = infino.connect("./data")
db = infino.connect("s3://bucket/prefix")
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

# BM25 and vector in one call, fused ranking. `query_vec` is your embedding.
hits = docs.hybrid_search("body", "disk full", "embedding", query_vec, k=10)
```

## Performance

Warm p50, tables on object storage:

| | 1M docs | 10M docs |
|---|---|---|
| Vector top-10, recall@10 0.992 | 591 µs | 5 ms |
| BM25 top-10, including row fetch | 125 µs | 2 ms |
| SQL, point lookup through crosstab | 186 µs – 7.6 ms | 260 µs – 75 ms |

The first query against an idle table has to open file handles and fill the cache. That
costs 114 ms at 1M and 314 ms at 10M for vector, 16 ms and 275 ms for BM25. Warm and cold
are about 200× apart, so the charts use a log scale.

![Vector search latency, log scale, 1M and 10M documents](docs/assets/readme/vector.svg)

![BM25 full-text search latency, log scale, 1M and 10M documents](docs/assets/readme/fts.svg)

![SQL query shape latency, log scale, 1M and 10M rows](docs/assets/readme/sql.svg)

The 1M numbers are from Azure Blob with 4 cores pinned
([CI run 33245831329](https://github.com/infino-ai/infino/actions/runs/33245831329)). The
10M numbers are from a separate run at the default scale, on a different commit and
different hardware, so compare each against its own baseline rather than against the other.

<details>
<summary><b>Reproducing the charts</b></summary>

Engine behavior is configured in YAML only; environment variables never override it. The
shipped defaults are what the charts measure:

```sh
cp src/config/config.yaml infino.yaml    # or $XDG_CONFIG_HOME/infino/config.yaml
```

The `vector:` block holds probe depth, rerank codec, and cell counts. The `supertable:`
block holds commit and cache behavior. Leave both alone to reproduce the published charts.

Corpus size is the one bench knob that reads an environment variable, and it takes a plain
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
Supertable FTS; the SQL shapes are `agg_max_title` (metadata), `WHERE key = ?` (lookup),
`AVG(rating) GROUP BY category` (scan), and `COUNT(*) GROUP BY bucket, category`
(crosstab). Structured results land in `target/infino-bench/*.json`. Methodology is in
[benches/README.md](benches/README.md).

</details>

### Against the public benchmarks

Infino has the lowest p99 on
[VectorDBBench](https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch)
([client](https://github.com/infino-ai/VectorDBBench/tree/main/vectordb_bench/backend/clients/infino)).

On [Search Benchmark, the Game](https://tantivy-search.github.io/bench/) it is within 19% of
Lucene on search and faster on count
([harness](https://github.com/quickwit-oss/search-benchmark-game)).

On [ClickBench](https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino%7CDataFusion%20%28Parquet%2C%20single%29%7CSpark%7CPostgreSQL%20%28with%20indexes%29&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot)
it averages 12.8 vCPU-seconds per query, against 9.8 for DuckDB
([port](https://github.com/infino-ai/clickbench/tree/add-infino/infino)).

Lucene and DuckDB are the right yardsticks for the last two. Landing in their range while
serving all four query types from one file is the goal.

## How it works

![Your app queries Infino, which caches in RAM and on disk over Parquet on object storage](docs/assets/readme/one-parquet-copy.svg)

### The indexes live inside the Parquet file

Infino writes one Parquet file per batch, with the BM25 and vector indexes stored inside it.
DuckDB, pyarrow, and DataFusion open that file and see a normal table, with Infino nowhere in
the read path ([example](infino-python/examples/parquet_interop.py)). Infino opens the same
bytes and uses the indexes.

There is no second artifact to build, ship, or keep in sync with the data, and no step that
loads an index at startup.

### A query reads byte ranges, not files

The indexes are laid out by term and by vector cluster, so a top-10 resolves to a short list
of byte offsets. On object storage those become HTTP range requests, a handful per query
rather than a download. Each round trip to object storage costs 20–100 ms, so the number of
them is what determines whether search on S3 is usable.

Fetched ranges are kept in a local disk cache and memory-mapped. A second query over the same
terms reads from the page cache and touches the network zero times, which is where the 125 µs
comes from. The cache is reclaimable: it shrinks under memory pressure, to nothing on an idle
table, and refills on demand.

### Scoring is SIMD over quantized vectors

The routed path scores 1-bit-per-dimension RaBitQ codes, which is `dim / 8` bytes per row —
128 bytes for a 1024-dimension vector against 4 KB as float32. Finalists are rescored against
a higher-precision copy, so the quantization costs latency rather than accuracy; measured
recall@10 is 0.992.

Distance kernels are hand-written intrinsics selected at runtime: AVX-512 where the CPU
reports it, AVX2 otherwise, a portable 256-bit path elsewhere, and an int8 VNNI kernel for the
graph walk.

### Commits swap a manifest

A table is a set of immutable files plus a manifest listing them. Appending writes new files
and then replaces the manifest in one atomic operation, so a commit publishes all of its rows
or none of them. A reader pins the manifest it opened with and reads that version to
completion, so it never waits on a writer and never sees a partial commit. No lock service or
leader election participates in this.

## Vector index modes

The index is one config line. Every mode falls back to the routed scan on its own when it
cannot serve a query, so changing the line can cost recall or latency but not correctness.

What a million 1536-dimension vectors cost to keep resident:

| Mode | Resident | 1M × 1536d | vs float32 | recall@10 | warm p50 |
|---|---|---|---|---|---|
| float32, no index | 4 B/dim | 5.7 GiB | — | 1.000 | — |
| `flat_ivf` | 0.5 B/dim, pinned | 726 MiB | 8× less | 0.938 | 20 ms |
| `ivf` (default) | 2 B/dim, cached | ~2.9 GiB reclaimable, ~100 MiB pinned | ~2× less | 0.988 | 6.2 ms |
| `hnsw_ivf` | ~3.2 B/dim, pinned | ~4.6 GiB | ~1.2× less | 0.995 | 0.59 ms |

The `1M × 1536d` column is computed from the same corpus for every row, so the memory is
comparable across modes. Recall and latency come from each mode's own measurement, and those
are not the same corpus: `flat_ivf` on dbpedia at 1536 dimensions, `hnsw_ivf` on Cohere at
768 dimensions, where it pins about 2.4 GiB.

`flat_ivf` scans a 4-bit plane end to end and returns the codes' own ranking. No clusters, no
graph, no rerank plane. It fetches nothing to serve a query, so cold and warm are the same
number and the quoted latency is a worst case rather than a cache-dependent average. Cost is
linear in rows: about 1.6 ms at 100K, 20 ms at 1M. Recall is set by the codec at roughly 0.94
and does not move with scale. Cosine columns only.

`ivf` is the default and the only mode that scales past memory. The index lives on object
storage and pages into the reclaimable cache, so pinned memory stays near 100 MiB regardless
of table size and the resident set drops to nothing when the table goes idle.

`hnsw_ivf` walks a resident graph on an int8 plane and re-ranks the final beam at higher
precision. It needs the graph in RAM, which bounds it to tables of 10M rows.

At 100K rows, `flat_ivf` holds 77 MiB of plane where float32 needs 586 MiB, answers in
1.6 ms, and beats the routed path below roughly 130K rows. Serving RSS all-in, including the
manifest, is 153 MiB at 100K and 841 MiB at 1M. Build peaks are transient and released at
commit.

### Calibration

You set a recall target and `optimize()` measures the corpus to decide how to reach it. A
graph or a flat plane is published only if its calibrated recall clears the registration floor
for that index type; otherwise the routed scan serves and nothing changes for the caller.
Probe width and depth are re-fitted whenever compaction or a cell split moves the geometry
they were measured against.

```yaml
# infino.yaml
vector:
  target_recall: 0.99
  search_mode: ivf         # ivf (default) | hnsw_ivf | flat_ivf
```

```python
table.optimize()    # drain, compact, recalibrate, sweep
```

Every knob, and the measurement behind each default, is documented inline in
[`src/config/config.yaml`](src/config/config.yaml). The bench tables in
[benches/README.md](benches/README.md) report peak, median, and p90 RSS next to each latency.

## Search results are SQL tables

`bm25_search`, `vector_search`, `hybrid_search`, `token_match`, and `exact_match` are SQL
table-valued functions, so a ranked result set is a relation. Retrieval, filters, joins, and
aggregation compose in one statement against one pinned snapshot, which replaces several
round trips with one.

```sql
SELECT   s.team,
         count(*)      AS hits,
         avg(h.score)  AS relevance
FROM     hybrid_search('logs', 'body', 'disk full', 'embedding', :q, 1000) AS h
JOIN     services s ON s.id = h.service_id
WHERE    h.ts > now() - interval '7 days'
GROUP BY s.team
ORDER BY hits DESC;
```

SQL runs on Apache DataFusion. What Infino adds is the indexes it already wrote: a predicate
on an indexed text column resolves through the posting list into a Parquet row selection, so
DataFusion decodes only the matching rows, and file-level min/max and Bloom summaries drop
whole files before any bytes are read.

## Limitations

- Commit is the durability boundary. Rows are durable when a commit lands, not when
  `append()` returns.
- Tables are append-only and time-ordered. Updates are delete plus insert via tombstones, and
  there are no cross-table transactions. This is not an OLTP store.
- Writes go through a single writer slot, so there is one writer per table at a time. Readers
  are unbounded and are never blocked.
- This is a library with a SQL and Arrow surface. There is no daemon, no REST endpoint, and no
  cluster to operate.

The crate is 0.x and the API can still move. The public surface is pinned by `public-api.txt`.

## Documentation

- [Overview](docs/architecture/overview.md) — the mental model, and how this compares
- [Superfile format](docs/architecture/superfile.md) — how indexes fit inside Parquet
- [Supertable layer](docs/architecture/supertable.md) — manifest, commit, query fan-out
- [infino.ai/docs](https://infino.ai/docs) — concepts and guides

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

MSRV 1.95. Python and Node version on their own SemVer lines
([docs/versioning.md](docs/versioning.md)). See [CONTRIBUTING.md](CONTRIBUTING.md).
Licensed [Apache-2.0](LICENSE).
