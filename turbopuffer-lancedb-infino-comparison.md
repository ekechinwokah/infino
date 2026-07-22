# Turbopuffer vs LanceDB vs Infino

This compares the systems as architectural shapes, using public turbopuffer and
LanceDB documentation plus Infino's OPANN Recovery Plan. It is not a generic
"vector database" comparison.

## Side-by-Side Architecture Table

| Area | turbopuffer | LanceDB | Infino |
| --- | --- | --- | --- |
| Core thesis | Search database built directly on object storage. Query tier is stateless/cached; object storage is source of truth. | Multimodal lakehouse/vector database built on the Lance columnar format, with versioned fragments and index artifacts on local/object storage. | Search-on-Parquet retrieval engine: one immutable "superfile" is valid Parquet plus SQL/FTS/vector indexes; supertables compose superfiles through MVCC manifests. |
| Primary durable unit | Object-storage KV/LSM-style storage objects; vector clusters/postings and text index blocks are mapped into object-store-friendly physical objects. | Lance table fragments plus Lance index files. Vector index is stored as regular Lance files, including index and auxiliary vector-storage files. | Immutable Parquet superfiles plus manifest/list parts and side objects. Hidden vector centroids live in a side blob; routing/fragments live in manifest state. |
| Vector index family | OPANN-derived centroid ANN. Public docs describe centroid routing, hierarchical clustering/tree structure, and object-storage roundtrip minimization. | IVF-family indexes: `IVF_PQ`, `IVF_SQ`, `IVF_RQ`, `IVF_FLAT`, plus `IVF_HNSW_*` where HNSW is a sub-index inside IVF partitions. | OPANN/SPANN/OPANN discipline adapted to immutable superfiles: global coarse centroid grid, per-cell hierarchical centroid trees, fine-centroid runs/fragments, replica assignment, hidden resident centroid blob. |
| Routing structure | Centroid hierarchy. ANN v3 says turbopuffer extended the SPTAG/OPANN centroid index by nesting clusters hierarchically in a multidimensional tree so cold routing roundtrips are bounded by tree height. | IVF hierarchy/partitioning. Query chooses IVF partitions by centroid routing; at large centroid counts LanceDB can accelerate centroid lookup or partition search with HNSW depending on index type. | Centroid hierarchy in manifest/state: `GlobalVectorIndex.grid` / `ClusterCentroids` first, then `SpfreshRoutingIndex.cells[*].CellTree` with `CellTreeNode { centroid, left, right }` and `ClusterRef` leaves. User tail has inline fine centroids; hidden has resident fine centroids referenced by `cluster_id`. |
| Tree traversal / fanout | Query navigates the centroid hierarchy, then fetches selected clusters/postings. Public ANN v3 emphasizes tree height as the cold-query roundtrip bound. | Query fans out to selected IVF partitions. `nprobes`, HNSW `ef`, quantization, and `refine_factor` control the partition/candidate set and rerank quality. | Query first selects outer cells adaptively using centroid distance, radii, and slack. Then it traverses/scores per-cell fine-centroid trees to choose `ClusterRef` leaves, expands base+delta fragments, and range-fetches selected run bodies. |
| Data scanned after routing | Selected clusters/postings fetched from storage, then reranked/scored. Public docs emphasize fetching cluster offsets/data in one or a few massive object-storage roundtrips. | Selected IVF partitions are scanned with compressed codes (`PQ`, `SQ`, `RQ`) or exact vectors (`FLAT`), optionally refined/reranked using original vectors via `refine_factor`. | Selected `ClusterRef` fragments are range-fetched from superfiles. Each selected fine centroid may map to base+delta fragments. Scoring uses Sq8+epsilon / rerank payloads inside run bodies; duplicate replicas are deduped by stable `_id`. |
| Cold-query design | Explicitly optimized for object storage. Docs target a few object-storage roundtrips: metadata, index navigation, selected cluster data. Cached namespaces can live on NVMe/RAM. | Disk-first and object-store-capable, but public docs frame object storage as higher-latency with hundreds of ms/p95. Performance relies on Lance layout, caching, partition pruning, and index artifacts. | Object-storage-native goal. Plan requires measuring GET count, bytes, waves, fragments/centroid, and cold cost. Query should fetch only selected run ranges through existing reader/cache paths. |
| Write/freshness model | Fresh writes are committed to object storage and made searchable through the service/indexing system. OPANN conceptually supports incremental updates without full rebuild. | Data is appended as new fragments. Indexes do not automatically include all new rows forever; docs say new rows after FTS index creation fall back to flat scan until `optimize()`, and vector indexes also rely on optimize/index maintenance. | Writes append immutable user superfiles immediately searchable through user manifest trees. Hidden index drains committed user superfiles later; `drained_ranges` controls unfiltered tail handoff. |
| Mutation model | Logical in-place update/reassignment in OPANN-style index, with versions/tombstones/stale copies in the algorithmic model. Physical storage is object-storage/LSM-managed. | Append/update/delete through Lance table versions and fragments; compaction/optimize rewrites fragments and index artifacts. Stable row IDs can be enabled. | No in-place mutation of committed user superfiles. Updates are delete+insert/tombstone; hidden maintenance writes new superfiles/fragments and MVCC-swaps manifests. |
| Maintenance model | OPANN: local updater + background rebuilder. Merge, split, and reassign postings to avoid global rebuild and maintain centroid assignment quality. | Indexing, optimization, compaction, and merging are background/heavy operations. `optimize()` folds new data into indexes and compacts fragments. | Drain + compaction + split/reassign. Drain appends hidden fragments; compaction merges base+deltas; split rewrites affected neighborhoods through MVCC. |
| Boundary recall mechanism | SPANN/OPANN-style boundary replication / local reassignment keeps low-probe centroid routing high recall. Public turbopuffer docs say OPANN is centroid-based and incrementally updated; details are not fully exposed. | Recall is mainly controlled by IVF partition count, `nprobes`, quantization choice, HNSW sub-index where used, and `refine_factor`. Not OPANN boundary replication. | Explicit `(1+eps)` replica closure plus RNG pruning via `assign_replicas`, intended across the global fine-centroid set. Query dedups physical copies by stable `_id`. |
| Full-text search | BM25/inverted indexes optimized for object storage. Blog describes posting blocks as KV pairs, fixed-size blocks, and LSM compaction grouping adjacent blocks into physical objects. | FTS index with BM25 over string columns; queries can use text input. New rows after FTS index creation may require flat-scan fallback until `optimize()`. | BM25 index embedded in superfiles, plus supertable fanout. Filtered vector search uses user-table FTS predicates because hidden vector index has no FTS. |
| Scalar / metadata filters | Public architecture says exact indexes for metadata filtering. | Scalar indexes include BTREE, BITMAP, LABEL_LIST, and FM-index-style text contains support depending on use case. | Manifest pruning uses scalar stats, Bloom/term/range summaries, partition metadata, and DataFusion/Arrow row resolution for SQL/filtering. |
| SQL / analytics | Search API/database; public docs focus on vector, FTS, filters, namespaces. | Table/query interface over Lance data; integrates with Arrow ecosystem and lakehouse workflows. | SQL is first-class through DataFusion over supertables/superfiles, alongside FTS and vector search in one system. |
| Consistency / versioning | Object-storage-first service with consistency machinery; public docs mention a consistency roundtrip and optional relaxed consistency for lower latency. | Lance table versions/fragments provide versioning; object storage and catalog/manifest track table state. | MVCC manifest snapshots. Readers pin `Arc<Manifest>`; committed superfiles are immutable; centroid blobs swap through manifest pointer/`ArcSwap`. |
| Caching model | Namespace data can be cached in NVMe/RAM; cold path fetches from object storage. Query routing aims to minimize cold roundtrips. | Local disk / cache-backed reads in deployments; storage backend may be local NVMe, EBS/EFS, or object store. | Reader cache and disk cache over immutable superfile bytes/ranges; resident manifest and hidden centroid blob are in-process query state. |
| User tail / unindexed data | Public docs do not expose a separate "user tail" model like Infino's; fresh writes are handled by the service/index pipeline. | Newly appended fragments may be unindexed until optimize; search may scan unindexed fragments to preserve completeness. | Explicit user tail: committed user superfiles not yet drained into hidden index. Tail trees stay in user manifest and are required for filtered vector search. |
| Physical format openness | Proprietary service/storage engine, public docs/blogs describe object-storage KV/LSM and index ideas. | Open Lance format; Arrow-native, columnar, versioned, multimodal lakehouse format. | Open Parquet-compatible superfiles with embedded indexes; object-storage-native manifest/supertable layer. |
| Best characterization | Managed object-storage search service with OPANN-derived centroid ANN and aggressive cold-query roundtrip engineering. | Lance-format lakehouse/vector DB using IVF + quantization/HNSW variants, optimized for multimodal storage and disk/object-store access. | Parquet-native retrieval engine trying to combine SQL, BM25, and OPANN/OPANN-like vector search in immutable superfiles. |

## Key Differences That Matter

| Question | turbopuffer | LanceDB | Infino |
| --- | --- | --- | --- |
| Is it OPANN? | Publicly yes: vector indexes are based on OPANN; ANN v3 extends centroid routing hierarchically. | No. LanceDB is IVF plus quantization/HNSW variants, not OPANN. | Intended yes in discipline: SPANN/OPANN-style assignment and maintenance adapted to immutable superfiles. |
| Is object storage the source of truth? | Yes. Object-storage-first service. | Yes/optional depending deployment; LanceDB supports local and object-storage backends, with table data and index artifacts in storage. | Yes. Object storage/local FS via storage providers; superfiles and manifests are durable state. |
| Does query use graph traversal over data vectors? | Public docs contrast centroid/posting routing with graph-heavy object-storage-unfriendly approaches; ANN v3 uses hierarchy/tree over centroids. | Some index types use HNSW, but as a sub-index inside IVF or over centroid routing, not as a global HNSW-only data graph. | No global HNSW. Query routes through a centroid hierarchy: outer cells, then per-cell `CellTree` fine-centroid leaves, then selected run fragments. |
| How are new writes searched before index maintenance? | Service/index handles fresh writes incrementally; details are not fully public. | New fragments can be searched; if not indexed, fallback scan may be used until optimize folds data into indexes. | User superfile tail is searched directly from user manifest routing until hidden drain catches up. |
| What is the hardest correctness invariant? | Keep local OPANN maintenance good enough that centroid routing remains high recall under updates. | Keep IVF/quantization/refinement/index maintenance aligned with table versions and appended fragments. | Keep immutable-superfile replica assignment, hidden drain, compaction, split/reassign, and tail/hidden dedup semantically equivalent to OPANN. |

## Infino-Specific Open Checks Against the Plan

| Check | Why It Matters |
| --- | --- |
| Merge/reassign affected set | OPANN plan says compaction merge/split/reassign must re-replicate affected boundary neighborhoods. Current merge logic must be audited/fixed so rebuilt fine centroids do not leave stale boundary replica coverage. |
| Per-cell tree traversal | Infino's manifest model has `CellTreeNode { left, right }` and `ClusterRef` leaves. The comparison should treat this as a hierarchical centroid tree, not a flat fine-centroid list. The current query implementation should still be audited to ensure it uses the intended tree semantics, not just leaf scoring. |
| GET/bytes/waves bench evidence | The plan treats read amplification as first-class. A design that is theoretically centroid-routed still fails if cold queries require too many object-store waves or bytes. |
| Recall@10 >= 0.99 | Infino's bar is stricter than turbopuffer's public docs examples. It must be measured on the standard 10M bench after fixes. |
| Hidden run-size distribution | Hidden runs need to stay around the target size so each centroid probe maps to bounded range fetches. |
| User-tail filtered search | User routing trees must persist after drain because filtered vector search runs against the user table, not the hidden vector-only table. |

## Storage & cost: Infino vs Pinecone (grounded, no estimates on the Infino side)

Every Infino number is from the measured 1M × 1024 supertable-vector bench (Azure,
vector-only). Every Pinecone number is from the official Standard-plan rate card
(pinecone.io/pricing, July 2026) applied to Pinecone's own documented storage
formula. Where a number is extrapolated rather than measured, it is labeled.

### Inputs (sources)

| Input | Value | Source |
| --- | --- | --- |
| Infino stored, 1M × 1024, vector-only | 2.13 GiB = 2.29 GB (2,288 B/vector, includes RaBitQ + Sq8 rerank + centroids) | measured bench (`Stored` row) |
| Infino warm query p50 | 582–731 µs | measured bench (post-drain warm) |
| Infino warm marginal CPU | $0.0073–$0.0092 / 1M queries | bench cost model (p50 × vCPU-hour) |
| Object-store storage rate | $0.023 / GB-mo | AWS S3 Standard, us-east-1, first 50 TB |
| Pinecone billed size | `records × (ID 8 B + metadata + dims×4 B)`, f32 | Pinecone docs, "Understanding cost" |
| Pinecone storage | $0.33 / GB-mo | Pinecone pricing, Standard |
| Pinecone read units | $16 / M RU (Standard low; $18 high, $24–27 Ent.) | Pinecone pricing |
| Pinecone RU consumption | 1 RU / GB of namespace / query, min 0.25 | Pinecone docs |
| Pinecone plan floor | $50 / mo (Standard) | Pinecone pricing |

Comparison is **vector-only, no metadata**, at 1024-dim. This is conservative
toward Pinecone: their billed size and per-query RU both grow with metadata, and
Pinecone's storage formula excludes its own ANN/bitmap index overhead (they absorb
it), whereas Infino's 2.29 GB already counts its indexes.

### Storage size and cost

Pinecone/vector = 1024×4 + 8 = 4,104 B. Infino/vector = 2,288 B (measured).
Infino 10M/100M rows are **linear extrapolation** from the measured 1M byte-rate
(superfile size scales ~linearly with docs; centroid/tree overhead is sublinear,
so this slightly over-states Infino).

| Scale | Pinecone billed | Pinecone $/mo | Infino stored | Infino $/mo | Storage cost ratio |
| --- | --- | --- | --- | --- | --- |
| 1M | 4.10 GB | $1.35 | 2.29 GB (measured) | $0.053 | 25.6× |
| 10M | 41.0 GB | $13.5 | 22.9 GB (extrap) | $0.53 | 25.7× |
| 100M | 410 GB | $135 | 229 GB (extrap) | $5.26 | 25.7× |

The ~26× decomposes cleanly: **1.79× fewer bytes** (quantized-at-rest vs f32) ×
**14.3× cheaper per-GB** (raw S3 vs Pinecone's managed $0.33). Neither factor is
a modeling assumption — both are measured/published.

### Query cost (single namespace)

Pinecone's documented model: RU/query = namespace size in GB (min 0.25), so query
cost rises linearly with corpus size — the "serverless scale cliff." Infino warm
is measured only at 1M; by design the probe set is bounded, so warm CPU is
expected to stay low-single-digit-ms at 10M/100M, but that is **not yet benched**
and is left blank rather than estimated.

| Scale | Namespace | Pinecone RU/query | Pinecone $/1M queries ($16/M RU) | Infino warm $/1M queries |
| --- | --- | --- | --- | --- |
| 1M | 4.10 GB | ~4.1 | $66 | $0.0073–$0.0092 (measured) |
| 10M | 41.0 GB | ~41 | $657 | not yet benched |
| 100M | 410 GB | ~410 | $6,566 | not yet benched |

At 1M the measured warm gap is ~4 orders of magnitude, but the two rows are not
like-for-like: Infino's is **marginal self-hosted CPU** (add your own base compute
+ ops for TCO); Pinecone's is **managed all-in usage price** on top of the $50/mo
floor. The defensible, like-for-like claim is the **structural** one, straight
from Pinecone's docs: Pinecone query cost is `O(namespace GB)`; Infino warm query
work is bounded by the probe set, not the corpus.

### Caveats (scope, not hand-waves)

1. Vectors-only. Adding metadata inflates Pinecone's billed size *and* RU/query;
   it does not change Infino's index bytes. The comparison is conservative toward
   Pinecone.
2. Infino 10M/100M storage is linear extrapolation from the measured 1M; warm
   query CPU at 10M/100M is unmeasured and intentionally left blank.
3. Pinecone RU assumes one namespace. Sharding into smaller namespaces lowers
   RU/query but adds fan-out + ops; the single-namespace figure is Pinecone's own
   documented default behavior.
4. RU rate is Standard low end ($16/M). Enterprise is $24–27/M.
5. Infino query cost is marginal CPU; a full TCO must add base compute and ops
   that Pinecone bundles into its managed price.

