# Turbopuffer OPANN vs Infino OPANN

Side-by-side comparison of the published turbopuffer/SPANN/OPANN shape and
Infino's OPANN Recovery Plan implementation model.

## Main Adaptation

Turbopuffer uses OPANN as a mutable service index over postings. Infino is
adapting the same centroid/posting maintenance discipline onto immutable Parquet
superfiles, so "move a vector" becomes append new fragments, mark old copies
obsolete through manifest and tombstone state, then MVCC-swap routing.

## Detailed Comparison

| Area | Turbopuffer / OPANN | Infino OPANN Plan | Status |
| --- | --- | --- | --- |
| Base model | SPANN/OPANN: many disk/object-storage postings keyed by fine centroids, with a hot centroid router. | Same thesis: SPANN/OPANN routing expressed as immutable Parquet superfiles plus MVCC manifest state. | Aligned at the conceptual level. |
| Primary storage unit | Posting / partition, often backed by disk blocks or object-storage ranges. | Superfile with a OPANN vector subsection containing a run directory and run bodies. | Infino maps postings/runs into Parquet-compatible superfile sections. |
| Centroid residency | Centroid/index metadata is hot enough to route before fetching postings. | Outer 64-cell grid is in the manifest; hidden fine centroids live in a plain side blob loaded into process heap via the hidden table handle. | Same routing idea; Infino separates coarse manifest grid from hidden fine-centroid blob. |
| Object-store fit | Minimize cold-query roundtrips: fetch metadata/centroids, choose postings, fetch selected posting data in large batches. | Select outer cells, score fine centroids from manifest/blob, then range-fetch only selected OPANN run fragments through existing reader/cache paths. | Aligned goal; Infino must verify GETs/bytes/waves in benches. |
| Fresh writes | Foreground updater inserts vectors into nearest posting and marks deletes/version changes without global rebuild. | User superfiles are committed immediately with their own inline routing trees; hidden index catches up via drain; `drained_ranges` prevents double scanning in unfiltered global search. | Different mechanics because Infino keeps user data immutable and append-only. |
| Mutation model | In-place logical update over postings, using versions/tombstones to make old copies stale until GC. | No in-place mutation of committed user superfiles. Updates/deletes are delete+insert/tombstone; hidden maintenance writes new superfiles/fragments and MVCC-swaps manifest pointers. | Big design difference: Infino preserves immutable superfiles. |
| User tail | Not a distinct public concept in the same way; fresh inserts are part of the indexed namespace via updater. | Committed user superfiles not yet drained form the searchable tail. Tail trees stay in the user manifest for both unfiltered handoff and filtered vector search. | Infino has two routing homes: user manifest and hidden manifest/blob. |
| Boundary replication | SPANN/OPANN keeps recall by assigning boundary vectors to multiple relevant postings and deduping/staling copies by id/version. | `assign_replicas` implements `(1+eps)` closure plus RNG pruning; rows can replicate across fine centroids and outer cells; query dedups by stable `_id`. | Aligned invariant; correctness depends on every write/maintenance path using the same assignment rule. |
| Run/posting size | Postings are kept balanced to bound tail latency and search work. | Hidden runs target about 2 MB by deriving fine centroid count from row count and row stride. User superfiles write their own inline tree under the 64-cell grid. | Aligned for hidden; user path is Infino-specific tail routing. |
| Drain / flush | Updater appends new vectors to postings; background work later rebuilds local areas. | Drain reads committed user superfiles, assigns rows to hidden fine-centroid replicas, writes hidden delta/base fragments, updates hidden routing and resident centroid blob pointer. | Infino's drain is the immutable-superfile equivalent of fresh insert propagation. |
| Merge | LIRE merge removes an undersized posting/centroid and reassigns vectors from the deleted posting; neighbor checks are narrower than split. | Current plan says compaction merge of base+deltas must rewrite fragments and re-replicate affected boundary neighborhood when fine-centroid membership changes. | This is the current sensitive gap: Infino must define/implement bounded affected-set merge correctly. |
| Split | Oversized posting is split into two balanced postings; nearby postings are checked because new centroids can change NPA on both sides. | Hidden split inserts a new cell centroid, gathers the split cell plus neighboring cells, dedups by stable id, reassigns inside the affected neighborhood, and MVCC-swaps routing. | Infino already follows the split-neighborhood shape more closely than merge. |
| Deletion / stale copies | Version map marks stale/deleted vector copies; search filters stale versions; physical cleanup is deferred. | Stable `_id` plus tombstone/deleted sidecars filter user-visible deletes; replica duplicates are collapsed at query by stable id; obsolete superfiles/fragments are removed by compaction/GC. | Same logical need, different mechanism. |
| Consistency | Service controls namespace consistency and can trade consistency/latency in cached paths. | Manifest is MVCC-swapped; readers pin an `Arc<Manifest>` snapshot; centroid blob swaps through `ArcSwap`; immutable bytes make old snapshots valid. | Infino leans harder on snapshot isolation and immutable storage. |
| Filtered search | Public docs support filters; exact internal integration is not fully public. | Filtered vector search runs on the user table because hidden index has no FTS; user routing trees must persist even after drain. | Infino-specific reason user routing cannot be deleted at drain. |
| Acceptance evidence | Turbopuffer monitors/tunes recall in production and designs for few object-store roundtrips. | Plan requires recall@10 >= 0.99, boundary rows in >=2 runs, no duplicate ids, and measured GETs/bytes/waves/drain time/write amp. | Infino still needs the bench evidence after implementation fixes. |

## Shared Algorithmic Invariants

| Invariant | Meaning |
| --- | --- |
| Centroid routing first | Search the hot centroid layer before fetching vector payloads. |
| Bounded posting/run size | Keep each probe's payload bounded so tail latency does not grow with corpus size. |
| Boundary replication | Replicate vectors near centroid boundaries so low-probe routing preserves recall. |
| Local maintenance | Avoid global rebuild; merge, split, and reassign only affected local regions. |
| Dedup/stale filtering | Multiple physical copies must collapse to one live logical vector. |
