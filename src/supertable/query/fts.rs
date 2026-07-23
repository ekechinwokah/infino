// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 fan-out on [`Supertable`](super::super::Supertable).
//!
//! ## Public API
//!
//! The sync, user-facing entry points live on
//! [`Supertable`](super::super::Supertable):
//!
//! ```ignore
//! // Bare call: `_id` + `score` only — no scalar decode.
//! let ids: Vec<RecordBatch> =
//!     table.bm25_search("title", "rust async", 10, BoolMode::Or, None)?;
//!
//! // Materialize row data by naming the columns to decode.
//! let rows: Vec<RecordBatch> =
//!     table.bm25_search("title", "rust async", 10, BoolMode::Or, Some(&["_id", "title", "score"]))?;
//!
//! // Unranked candidate sets (Arrow rows, score == 0.0).
//! let any = table.token_match("title", "rust async", BoolMode::Or, None)?;
//! let exact = table.exact_match("title", "rust async", None)?;
//! ```
//!
//! Internally these drive the async kernel on the snapshot-pinned
//! [`SupertableReader`], whose `bm25_search` (rows) / `bm25_hits`
//! ([`SuperfileHit`], superfile-local) / `bm25_search_prefix` methods are
//! the engine-facing surface. Ranked results are sorted by score
//! *descending* — higher BM25 score is more relevant.
//!
//! ## Strategy
//!
//! Internally pins a snapshot reader and drives the async
//! kernel to completion via the sync→async bridge. The reader
//! holds a pinned `Arc<ManifestSnapshot>`; for each visible superfile we:
//!
//!   1. Fetch the superfile's `SuperfileReader` from the store.
//!   2. Delegate to `SuperfileReader::bm25_search` /
//!      `bm25_search_prefix` (already implemented at the superfile
//!      layer; per-superfile top-k with BlockMaxWAND skip).
//!   3. Tag each `(local_doc_id, score)` with the superfile URI.
//!   4. Concatenate across superfiles and global-top-k by score.
//!
//! Rayon fan-out runs on `options.reader_pool`. For an N-superfile
//! supertable we issue N parallel per-superfile searches; the pool
//! caps concurrency at the configured reader thread count.
//!
//! ## Score comparability across superfiles
//!
//! BM25's IDF is computed from per-superfile `n_docs` and `df`,
//! so a rare term in a small superfile can score higher than the
//! same term in a larger superfile. This is the classical sharded-
//! BM25 problem:
//! treating per-superfile scores as comparable is a documented
//! approximation, accepted in v1 because (a) global IDF would
//! require either a manifest-wide df table or a two-pass query
//! (df gather + score), both with non-trivial memory/latency
//! cost; (b) for k ≥ 10 and reasonably balanced superfiles the top-k
//! *set* converges to the global answer even if score *order*
//! within the set wiggles. Oracle tests assert set membership at
//! `k = 10` against a single-superfile ground truth.
//!
//! ManifestSnapshot-level skip pruning is wired in: each call computes a
//! per-superfile keep/prune mask from the FTS bloom (exact-term
//! mode) or the lex term range (prefix mode) before issuing
//! per-superfile work, so pruned superfiles never trigger a
//! `SuperfileReaderCache::reader` call. Vector + SQL skip remain
//! deferred (see those modules' headers).

use std::{
    borrow::Cow,
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    slice,
    sync::{Arc, Mutex},
    time::Instant,
};

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, LargeStringArray};
use dashmap::DashMap;
use roaring::RoaringBitmap;
use tokio::join;
use tracing::debug;
use uuid::Uuid;

pub use crate::superfile::fts::reader::BoolMode;
use crate::{
    InfinoError,
    superfile::{
        SuperfileReader,
        error::{FtsError, ReadError},
        fts::{
            reader::{
                ClauseLists, FtsCursorCache, OR_WINDOW_DOMINANCE_MULT, RoutedTermRow, SharedFloor,
            },
            tokenize::{AsciiLowerTokenizer, Tokenizer},
        },
    },
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{ManifestSnapshot, SuperfileEntry, list::DrainedVersionRanges},
        query::{
            SuperfileHit, dispatch,
            exec::common::{resolve_hits_named, take_rows_byte_source},
            prune::{PruneLeaf, select_superfiles},
            skip::{fts_bloom_skip, fts_prefix_skip},
            vector::{
                hits_id_score_batch, projection_is_id_score_only, user_placement_for_scalar_resolve,
            },
        },
        reader_cache::disk::ForegroundQueryGuard,
        slow_fts_state::{SlowFtsState, TermBlockMax},
        tombstones::SidecarCache,
    },
};

/// An unranked query's match set: the terms and exact phrases every
/// (`And`) or any (`Or`) of which a doc must contain. Produced by
/// `parse_and_prune` from the clause model — the must side when any
/// must exists (shoulds have no scores to raise unranked), the bare
/// side under the default operator otherwise.
struct UnrankedMatchSet {
    terms: Vec<String>,
    phrases: Vec<Vec<String>>,
    mode: BoolMode,
}

impl Default for UnrankedMatchSet {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            phrases: Vec::new(),
            mode: BoolMode::Or,
        }
    }
}

/// An unranked query's negated atoms (docs containing any are
/// excluded).
#[derive(Default)]
struct UnrankedNegatives {
    terms: Vec<String>,
    phrases: Vec<Vec<String>>,
}

/// Raise the wave's global floor with one unit's surviving scores,
/// tombstone-filtered. Sidecars were prefetched by the dispatcher, so
/// the bitmap lookup is an in-memory hit; on a cache miss/error we
/// simply don't merge (a lower floor is always safe).
fn merge_unit_scores(
    shared: &SharedTopK,
    tombstones: &Option<Arc<SidecarCache>>,
    suid: Uuid,
    now: Instant,
    hits: &[(u32, f32)],
) {
    match tombstones.as_ref().map(|c| c.bitmap_for(suid, now)) {
        Some(Ok(bitmap)) if !bitmap.is_empty() => shared.merge(
            hits.iter()
                .filter(|(d, _)| !bitmap.contains(*d))
                .map(|(_, s)| *s),
        ),
        Some(Err(_)) => {}
        _ => shared.merge(hits.iter().map(|(_, s)| *s)),
    }
}

/// Minimum bare-OR term count for the multi-term block-selected
/// kernel; a single bare term has its own dedicated selected walk.
const MULTI_SELECT_MIN_TERMS: usize = 2;
/// Minimum quantized block-bound spread for a resident row to count
/// as prunable at routing time. Bounds are ceil-quantized to u8
/// against the term's own max, so spread is in 1/255ths of that max;
/// below ~6% the top-k bar cannot separate the band — the admission
/// kernel would admit most blocks and bail mid-flight into a
/// single-threaded whole-file walk, strictly worse than the ranged
/// parallel union that routing displaced (measured as the entire
/// post-drain broad-OR gap: ten/forty_term_or 12/39 ms vs 2.1/6.2
/// pre-drain).
const MULTI_SELECT_MIN_ROW_SPREAD: u8 = 16;

/// Whether a resident block-max row can actually prune a broad-OR
/// walk: present bounds with at least [`MULTI_SELECT_MIN_ROW_SPREAD`]
/// of variance. Shared by the wave partition and the kernel
/// engagement gate — the two MUST agree, or a file that can never
/// engage loses its ranged slicing to a fallback that runs the union
/// on one thread.
fn row_can_prune(row: &TermBlockMax) -> bool {
    let min = row.quantized.iter().min().copied().unwrap_or(0);
    let max = row.quantized.iter().max().copied().unwrap_or(0);
    max - min >= MULTI_SELECT_MIN_ROW_SPREAD
}

/// Resident-row mirror of the reader's windowed-union dispatch: block
/// selection wins a bare multi-term OR only when some routed term
/// DOMINATES the score upper bounds
/// (`max > OR_WINDOW_DOMINANCE_MULT × avg`). All-prunable-but-UNIFORM
/// rows pass the engagement gate, then the kernel's bail thresholds
/// fire and the un-ranged fallback walks the shard serially — the
/// same 1-vs-8-threads cliff the any-vs-all gate fix closed for
/// never-engaging files. Each row's `scale` is its term's exact max
/// block bound, so no dequantization is needed.
fn rows_have_dominant_ub<'a>(rows: impl Iterator<Item = &'a TermBlockMax>) -> bool {
    let mut total = 0.0f32;
    let mut max = 0.0f32;
    let mut n = 0usize;
    for row in rows {
        total += row.scale;
        max = max.max(row.scale);
        n += 1;
    }
    n > 0 && max > OR_WINDOW_DOMINANCE_MULT * (total / n as f32)
}

/// Rejection message for a query with negated terms but no positive
/// anchor (e.g. `-foo`). Shared by the scored and unranked FTS paths so
/// both reject the case identically.
const NEGATION_ONLY_QUERY_MSG: &str = "only negated terms; at least one positive term is required";

/// Cross-segment top-k score sharing for the BM25 fan-out.
///
/// Every segment kernel runs an independent top-k; without
/// coordination, segment N knows nothing about the k hits segments
/// 1..N-1 already produced, so it scores blocks the global result can
/// never use. This shares the running **global kth-best score** as a
/// floor: each kernel reads it at start and seeds its pruning
/// structures (BMW block skips, the MaxScore essential boundary, AND
/// block-max bars) from it; each finishing kernel merges its surviving
/// scores back, monotonically raising the floor for the segments still
/// running.
///
/// Correctness: the floor only ever prunes docs scoring **strictly
/// below** the published kth-best (kernels apply it via
/// `floor.next_down()` comparisons), and the published floor is always
/// ≤ the final global kth-best, so every doc that could appear in the
/// merged top-k survives in some segment's result — the merged output
/// is identical to an uncoordinated run, including score ties. Only
/// the amount of *skipped work* depends on segment completion order.
struct SharedTopK {
    k: usize,
    /// Min-heap (via `Reverse`) of the best `k` scores seen so far.
    heap: Mutex<BinaryHeap<Reverse<OrdScore>>>,
    /// The live floor: `NEG_INFINITY` until `k` scores are known,
    /// then the running global kth-best. Kernels also raise and read
    /// it **mid-walk** (see `SharedFloor`), so sub-range units prune
    /// against each other while running, not only via completed-unit
    /// merges.
    floor: Arc<SharedFloor>,
}

/// Total-order f32 wrapper for the [`SharedTopK`] heap (BM25 scores
/// are finite, but `f32` still needs an `Ord` shim).
#[derive(PartialEq)]
struct OrdScore(f32);
impl Eq for OrdScore {}
impl PartialOrd for OrdScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl SharedTopK {
    fn new(k: usize) -> Arc<Self> {
        Arc::new(Self {
            k,
            heap: Mutex::new(BinaryHeap::new()),
            floor: Arc::new(SharedFloor::new()),
        })
    }

    /// The current global floor — `NEG_INFINITY` until k scores are
    /// known (merged or published mid-walk).
    fn floor(&self) -> f32 {
        self.floor.get()
    }

    /// Handle the ranged kernels use to read/raise the floor mid-walk.
    fn live_floor(&self) -> Arc<SharedFloor> {
        Arc::clone(&self.floor)
    }

    /// Merge one finished segment's (tombstone-surviving) scores and
    /// publish the new kth-best as the floor once k scores are known.
    fn merge(&self, scores: impl IntoIterator<Item = f32>) {
        let mut heap = self.heap.lock().expect("SharedTopK mutex poisoned");
        for s in scores {
            if heap.len() < self.k {
                heap.push(Reverse(OrdScore(s)));
            } else if let Some(Reverse(OrdScore(min))) = heap.peek()
                && s > *min
            {
                heap.pop();
                heap.push(Reverse(OrdScore(s)));
            }
        }
        if heap.len() == self.k
            && let Some(Reverse(OrdScore(min))) = heap.peek()
        {
            self.floor.raise(*min);
        }
    }
}

/// Ingredients of the hidden-text query route, present when the
/// hidden sibling's current epoch holds text shards (term-range
/// slices of the merged inverted index).
struct HiddenTextRoute {
    /// Reader whose store/options open the hidden table's superfiles.
    hidden_reader: SupertableReader,
    hidden_manifest: Arc<ManifestSnapshot>,
    /// Resident block-max routing (the FTS admit slab), when the
    /// epoch's generation is stamped and readable.
    fts_routing: Option<Arc<SlowFtsState>>,
    /// Stable ids deleted since the text epoch (sorted); hits are
    /// identity-filtered against it after the waves merge.
    deleted: Arc<Vec<i128>>,
    /// The epoch's residency watermark: user superfiles NOT covered
    /// by it form the tail wave.
    drained: DrainedVersionRanges,
}

/// One parsed BM25 query's six owned clause lists, shared across
/// fan-out units (and across the two hidden-text waves).
#[derive(Clone)]
struct BmClauses {
    musts: Arc<Vec<String>>,
    shoulds: Arc<Vec<String>>,
    negatives: Arc<Vec<String>>,
    must_phrases: Arc<Vec<Vec<String>>>,
    should_phrases: Arc<Vec<Vec<String>>>,
    negative_phrases: Arc<Vec<Vec<String>>>,
}

/// One BM25 fan-out wave over `kept` superfiles, opened through
/// `ctx`'s store/options. The single-wave path runs one wave over the
/// user table; the hidden-text path runs two — text shards through the
/// hidden sibling's reader, the undrained user tail through the user
/// reader — sharing `shared` so the kth-best floor crosses waves.
/// Returns the wave's top `k` hits with stable ids attached.
async fn bm25_fanout_wave(
    ctx: &SupertableReader,
    kept: Vec<Arc<SuperfileEntry>>,
    column_arc: Arc<String>,
    clauses: BmClauses,
    k: usize,
    shared: Arc<SharedTopK>,
    routing: Option<Arc<SlowFtsState>>,
) -> Result<Vec<SuperfileHit>, QueryError> {
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    let pool_threads = ctx.manifest().options.reader_pool.current_num_threads();
    let BmClauses {
        musts: must_arc,
        shoulds: should_arc,
        negatives: neg_arc,
        must_phrases: must_ph_arc,
        should_phrases: should_ph_arc,
        negative_phrases: neg_ph_arc,
    } = clauses;

    let has_phrases =
        !must_ph_arc.is_empty() || !should_ph_arc.is_empty() || !neg_ph_arc.is_empty();
    let has_negation = !neg_arc.is_empty() || !neg_ph_arc.is_empty();
    // Build the work-unit list. When the reader pool has more
    // threads than there are kept superfiles, slice each superfile
    // into doc_id sub-ranges so the fan-out can saturate every pool
    // thread — every negation-free positive shape (phrase atoms
    // included) has a range-aware kernel. Negated queries stay
    // per-superfile.
    let fanout = fanout_for(
        must_arc.len() + must_ph_arc.len(),
        should_arc.len() + should_ph_arc.len(),
        has_negation,
    );
    // Bare-term shapes whose resident block-max rows CAN PRUNE keep
    // their file as one un-ranged unit: the block-selected walks
    // visit (and, cold, fetch) fewer bytes than any parallel full
    // walk. A single bare term needs its own row prunable; a bare
    // multi-term OR engages the multi-term admission kernel when at
    // least one term's row can prune (row-less terms are small and
    // join as unrouted). Flat-bounded / row-less files slice for
    // parallelism instead.
    let single_bare_term = match (must_arc.as_slice(), should_arc.as_slice()) {
        _ if has_phrases || has_negation => None,
        ([term], []) | ([], [term]) => Some(term.as_str()),
        _ => None,
    };
    let multi_bare_or = !has_phrases
        && !has_negation
        && must_arc.is_empty()
        && should_arc.len() >= MULTI_SELECT_MIN_TERMS;
    let (selected_refs, sliceable_refs): (Vec<&Arc<SuperfileEntry>>, Vec<&Arc<SuperfileEntry>>) =
        match (single_bare_term, multi_bare_or, routing.as_ref()) {
            (Some(term), _, Some(state)) => kept.iter().partition(|e| {
                state
                    .term_block_max(e.superfile_id, &column_arc, term)
                    .is_some_and(|row| row.quantized.iter().min() < row.quantized.iter().max())
            }),
            // Mirror the kernel's engagement gate exactly (>= 1 row
            // present AND every present row prunable): a file that can
            // never engage must keep its ranged parallel slicing — the
            // any-prunable form parked near-flat merged shards on one
            // un-ranged unit whose whole-file fallback ran the union
            // single-threaded (measured: the entire post-drain
            // broad-OR gap; ten/forty_term_or 12/39 ms vs 2.1/6.2
            // pre-drain).
            (None, true, Some(state)) => kept.iter().partition(|e| {
                let rows: Vec<_> = should_arc
                    .iter()
                    .filter_map(|term| state.term_block_max(e.superfile_id, &column_arc, term))
                    .collect();
                !rows.is_empty()
                    && rows.iter().all(|row| row_can_prune(row))
                    && rows_have_dominant_ub(rows.iter().copied())
            }),
            _ => (Vec::new(), kept.iter().collect()),
        };
    let mut work_units = build_work_units(&sliceable_refs, fanout, pool_threads);
    work_units.extend(build_work_units(
        &selected_refs,
        FanOut::PerSuperfile,
        pool_threads,
    ));
    let units: Vec<(Arc<SuperfileEntry>, (Option<(u32, u32)>, Uuid))> = work_units
        .into_iter()
        .map(|u| {
            let suid = u.entry.superfile_id;
            (u.entry, (u.range, suid))
        })
        .collect();

    let tombstones = ctx.tombstone_cache.clone();
    let now = Instant::now();
    // Per-superfile shared cursor prototypes for this wave: sub-range
    // units of one file run identical term lists, so the first unit
    // builds each list (fetch + skip-table parse) once and the rest
    // walk positional-reset clones — without this, N units multiply
    // both the parse work and the cold posting-bytes fetched.
    let cursor_caches: Arc<DashMap<Uuid, Arc<FtsCursorCache>>> = Arc::new(DashMap::new());

    // One shared fan-out (`query::dispatch::fanout`) — the same
    // orchestrator the vector path uses. It warms the tombstone
    // sidecars in one batch, opens each superfile reader and runs the
    // kernel under `tokio::spawn` so cold GETs overlap, then tags +
    // tombstone-filters each unit's hits. The per-unit `params` is
    // the optional doc-id sub-range (`None` searches the whole
    // superfile) plus the superfile id for the tombstone-aware merge.
    let kernel = move |r: Arc<SuperfileReader>, (range, suid): (Option<(u32, u32)>, Uuid)| {
        let column_arc = Arc::clone(&column_arc);
        let must_arc = Arc::clone(&must_arc);
        let should_arc = Arc::clone(&should_arc);
        let neg_arc = Arc::clone(&neg_arc);
        let must_ph_arc = Arc::clone(&must_ph_arc);
        let should_ph_arc = Arc::clone(&should_ph_arc);
        let neg_ph_arc = Arc::clone(&neg_ph_arc);
        let shared = Arc::clone(&shared);
        let tombstones = tombstones.clone();
        let routing = routing.clone();
        let cursor_caches = Arc::clone(&cursor_caches);
        async move {
            // Share the global kth-best floor with every superfile —
            // single-term queries included — so each prunes its scored
            // scan against the running top-k instead of returning a full
            // local top-k for the merge to re-sort. Without this the
            // fan-out churns ~(superfiles × k) candidates through the
            // merge heap at large k, which dominates high-k latency.
            // Ties stay correct: the floor prunes only scores strictly
            // below the published kth-best (kernels compare via
            // `floor.next_down()`), so the merged top-k — score ties
            // included — matches an uncoordinated run; only the amount
            // of skipped work depends on segment completion order.
            // The deterministic choice among kth-score ties is made
            // once, at the merge, by [`select_top_k_stable`].
            let n_terms = must_arc.len() + should_arc.len();
            let phrase_free = must_ph_arc.is_empty() && should_ph_arc.is_empty();
            let floor = shared.floor();
            // Resident-routed block selection: a single bare term with
            // a resident admit row visits blocks best-first by bound
            // and fetches exactly those (the FTS cell-read analog).
            // Everything else keeps the whole-term kernels.
            if range.is_none()
                && n_terms == 1
                && phrase_free
                && neg_arc.is_empty()
                && neg_ph_arc.is_empty()
                && let Some(state) = routing.as_ref()
            {
                let term = must_arc.first().or_else(|| should_arc.first());
                if let Some(term) = term
                    && let Some(row) = state.term_block_max(suid, &column_arc, term)
                    // Flat bounds can't prune: the ranged parallel
                    // walk (sub-range units) beats a serial selected
                    // walk that must visit everything anyway.
                    && row.quantized.iter().min() < row.quantized.iter().max()
                {
                    let hits = r
                        .bm25_single_term_block_selected(
                            &column_arc,
                            k,
                            floor,
                            row.metadata_offset,
                            &row.quantized,
                            row.scale,
                        )
                        .await
                        .map_err(fts_read_error)?;
                    merge_unit_scores(&shared, &tombstones, suid, now, &hits);
                    return Ok(hits);
                }
            }
            // Multi-term bare-OR admission: an un-ranged unit whose
            // file has at least one prunable resident row runs the
            // (term, block) best-first kernel — fetching only blocks
            // that can beat the live bar — instead of walking every
            // term's whole merged list. Falls through to the plain
            // walk when the kernel declines (stale routing).
            if range.is_none()
                && n_terms >= MULTI_SELECT_MIN_TERMS
                && must_arc.is_empty()
                && phrase_free
                && neg_arc.is_empty()
                && neg_ph_arc.is_empty()
                && let Some(state) = routing.as_ref()
            {
                let rows: Vec<Option<&TermBlockMax>> = should_arc
                    .iter()
                    .map(|term| state.term_block_max(suid, &column_arc, term))
                    .collect();
                let engage = rows.iter().flatten().count() > 0
                    && rows.iter().flatten().all(|row| row_can_prune(row));
                if engage {
                    let mut routed_rows = Vec::new();
                    let mut unrouted: Vec<&str> = Vec::new();
                    for (term, row) in should_arc.iter().zip(&rows) {
                        match row {
                            Some(row) => routed_rows.push(RoutedTermRow {
                                metadata_offset: row.metadata_offset,
                                quantized: &row.quantized,
                                scale: row.scale,
                            }),
                            None => unrouted.push(term.as_str()),
                        }
                    }
                    let live = shared.live_floor();
                    if let Some(hits) = r
                        .bm25_multi_term_or_block_selected(
                            &column_arc,
                            k,
                            floor,
                            &routed_rows,
                            &unrouted,
                            Some(&live),
                        )
                        .await
                        .map_err(fts_read_error)?
                    {
                        merge_unit_scores(&shared, &tombstones, suid, now, &hits);
                        return Ok(hits);
                    }
                }
            }
            let hits = match range {
                // Ranged units exist for every negation-free positive
                // shape, phrase atoms included (`fanout_for`).
                Some((start, end)) => {
                    let must_refs: Vec<&str> = must_arc.iter().map(|s| s.as_str()).collect();
                    let should_refs: Vec<&str> = should_arc.iter().map(|s| s.as_str()).collect();
                    let cache = Arc::clone(&*cursor_caches.entry(suid).or_default());
                    let live = shared.live_floor();
                    r.bm25_search_clauses_range_with_floor(
                        &column_arc,
                        ClauseLists {
                            musts: &must_refs,
                            shoulds: &should_refs,
                            negatives: &[],
                            must_phrases: &must_ph_arc,
                            should_phrases: &should_ph_arc,
                            negative_phrases: &[],
                        },
                        k,
                        start,
                        end,
                        floor,
                        Some(&cache),
                        Some(&live),
                    )
                    .await
                    .map_err(fts_read_error)?
                }
                None => {
                    let must_refs: Vec<&str> = must_arc.iter().map(|s| s.as_str()).collect();
                    let should_refs: Vec<&str> = should_arc.iter().map(|s| s.as_str()).collect();
                    let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                    r.bm25_search_clauses(
                        &column_arc,
                        ClauseLists {
                            musts: &must_refs,
                            shoulds: &should_refs,
                            negatives: &neg_refs,
                            must_phrases: &must_ph_arc,
                            should_phrases: &should_ph_arc,
                            negative_phrases: &neg_ph_arc,
                        },
                        k,
                        floor,
                    )
                    .await
                    .map_err(fts_read_error)?
                }
            };
            merge_unit_scores(&shared, &tombstones, suid, now, &hits);
            Ok(hits)
        }
    };
    let per_unit = dispatch::fanout_local_hits(ctx, units, kernel).await?;
    select_top_k_stable(ctx, per_unit, k).await
}

/// One unranked token-match fan-out wave over `kept` superfiles
/// opened through `ctx`'s store/options — the `token_match` sibling of
/// [`bm25_fanout_wave`]. Returns every matching row (no scoring, no
/// ordering) with stable ids attached.
#[allow(clippy::too_many_arguments)]
async fn token_match_wave(
    ctx: &SupertableReader,
    kept: Vec<Arc<SuperfileEntry>>,
    column_arc: Arc<String>,
    term_arc: Arc<Vec<String>>,
    phrase_arc: Arc<Vec<Vec<String>>>,
    neg_arc: Arc<Vec<String>>,
    neg_ph_arc: Arc<Vec<Vec<String>>>,
    match_mode: BoolMode,
) -> Result<Vec<SuperfileHit>, QueryError> {
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    let has_negatives = !neg_arc.is_empty() || !neg_ph_arc.is_empty();
    let phrase_involved = !phrase_arc.is_empty() || !neg_ph_arc.is_empty();
    let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();
    let kernel = move |r: Arc<SuperfileReader>, _: ()| {
        let column_arc = Arc::clone(&column_arc);
        let term_arc = Arc::clone(&term_arc);
        let phrase_arc = Arc::clone(&phrase_arc);
        let neg_arc = Arc::clone(&neg_arc);
        let neg_ph_arc = Arc::clone(&neg_ph_arc);
        async move {
            let refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
            // Any phrase atom (match or negated) takes the
            // phrase-aware walk; plain-token queries keep the
            // optimized token_match path unchanged.
            let docs = match phrase_involved {
                true => r
                    .atoms_match_ids(&column_arc, &refs, &phrase_arc, match_mode)
                    .await
                    .map_err(fts_read_error)?,
                false => r
                    .token_match(&column_arc, &refs, match_mode)
                    .await
                    .map_err(fts_read_error)?,
            };
            // Drop any positive match that also carries a negated
            // atom (union of the negatives). The df / count fast
            // paths can't express exclusion, so negation forces a
            // materialized walk over both sets.
            let docs = if has_negatives {
                let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                let excluded: RoaringBitmap = match neg_ph_arc.is_empty() {
                    true => r
                        .token_match(&column_arc, &neg_refs, BoolMode::Or)
                        .await
                        .map_err(fts_read_error)?,
                    false => r
                        .atoms_match_ids(&column_arc, &neg_refs, &neg_ph_arc, BoolMode::Or)
                        .await
                        .map_err(fts_read_error)?,
                }
                .into_iter()
                .collect();
                docs.into_iter()
                    .filter(|d| !excluded.contains(*d))
                    .collect::<Vec<_>>()
            } else {
                docs
            };
            Ok(docs.into_iter().map(|d| (d, 0.0f32)).collect::<Vec<_>>())
        }
    };
    let per_unit = dispatch::fanout_local_hits(ctx, units, kernel).await?;
    // Exact pre-size: `Flatten`'s size_hint is opaque, and growth
    // reallocations copy the whole hit vec repeatedly at 1M hits.
    let total: usize = per_unit.iter().map(Vec::len).sum();
    let mut hits: Vec<SuperfileHit> = Vec::with_capacity(total);
    for unit in per_unit {
        hits.extend(unit);
    }
    dispatch::attach_stable_ids_to_hits(ctx, &mut hits).await?;
    Ok(hits)
}

/// One unranked match-count fan-out wave over `kept` superfiles
/// opened through `ctx`'s store/options — the counting sibling of
/// [`token_match_wave`]. Returns the wave's total match count,
/// tombstone-filtered per superfile.
#[allow(clippy::too_many_arguments)]
async fn token_match_count_wave(
    ctx: &SupertableReader,
    kept: Vec<Arc<SuperfileEntry>>,
    column_arc: Arc<String>,
    term_arc: Arc<Vec<String>>,
    phrase_arc: Arc<Vec<Vec<String>>>,
    neg_arc: Arc<Vec<String>>,
    neg_ph_arc: Arc<Vec<Vec<String>>>,
    match_mode: BoolMode,
) -> Result<u64, QueryError> {
    if kept.is_empty() {
        return Ok(0);
    }
    let single_term = term_arc.len() == 1 && phrase_arc.is_empty();
    let has_negatives = !neg_arc.is_empty() || !neg_ph_arc.is_empty();
    let phrase_involved = !phrase_arc.is_empty() || !neg_ph_arc.is_empty();
    let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();

    // Shared fan-out (`dispatch::fanout_with`): warms tombstones,
    // spawns + opens each superfile concurrently, and short-circuits
    // on the first error. The per-superfile body returns this
    // superfile's match count; the totals are summed.
    let per_superfile = dispatch::fanout_with(
        ctx,
        units,
        true,
        true,
        move |r, entry, tombstone_cache, now, _params: ()| {
            let column_arc = Arc::clone(&column_arc);
            let term_arc = Arc::clone(&term_arc);
            let phrase_arc = Arc::clone(&phrase_arc);
            let neg_arc = Arc::clone(&neg_arc);
            let neg_ph_arc = Arc::clone(&neg_ph_arc);
            async move {
                // Tombstone bitmap for this superfile (None = no deletes).
                let tomb = match tombstone_cache.as_ref() {
                    Some(c) => {
                        let b = c
                            .bitmap_for(entry.superfile_id, now)
                            .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
                        if b.is_empty() { None } else { Some(b) }
                    }
                    None => None,
                };
                let refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
                // Negated terms or deletes both force materialization:
                // the df read and the bare match count can't subtract
                // excluded or tombstoned docs. Materialize the positive
                // matches, then drop any doc carrying a negated term
                // (union of the negatives) or a tombstone.
                if has_negatives || tomb.is_some() {
                    let docs = match phrase_involved {
                        true => r
                            .atoms_match_ids(&column_arc, &refs, &phrase_arc, match_mode)
                            .await
                            .map_err(fts_read_error)?,
                        false => r
                            .token_match(&column_arc, &refs, match_mode)
                            .await
                            .map_err(fts_read_error)?,
                    };
                    let excluded: RoaringBitmap = if has_negatives {
                        let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                        match neg_ph_arc.is_empty() {
                            true => r
                                .token_match(&column_arc, &neg_refs, BoolMode::Or)
                                .await
                                .map_err(fts_read_error)?,
                            false => r
                                .atoms_match_ids(&column_arc, &neg_refs, &neg_ph_arc, BoolMode::Or)
                                .await
                                .map_err(fts_read_error)?,
                        }
                        .into_iter()
                        .collect()
                    } else {
                        RoaringBitmap::new()
                    };
                    let n = docs
                        .iter()
                        .filter(|d| {
                            !excluded.contains(**d)
                                && tomb.as_ref().is_none_or(|b| !b.contains(**d))
                        })
                        .count() as u64;
                    return Ok::<u64, QueryError>(n);
                }
                // No negatives and no deletes (the common case): count
                // without materializing ids — a single token resolves
                // O(1) from the stored df, multi-token tallies the
                // match walk through the counting sink.
                let n = if single_term {
                    r.term_df(&column_arc, &term_arc[0])
                        .await
                        .map_err(fts_read_error)?
                } else if phrase_involved {
                    r.atoms_match_count(&column_arc, &refs, &phrase_arc, match_mode)
                        .await
                        .map_err(fts_read_error)?
                } else {
                    r.token_match_count(&column_arc, &refs, match_mode)
                        .await
                        .map_err(fts_read_error)?
                };
                Ok(n)
            }
        },
    )
    .await?;
    Ok(per_superfile.into_iter().sum())
}

/// One prefix-expanded BM25 fan-out wave over `kept` superfiles
/// opened through `ctx`'s store/options — the prefix sibling of
/// [`bm25_fanout_wave`] (no cross-unit floor: the per-superfile
/// prefix kernels never shared one). Returns the wave's top `k` hits
/// with stable ids attached.
async fn bm25_prefix_wave(
    ctx: &SupertableReader,
    kept: Vec<Arc<SuperfileEntry>>,
    column_arc: Arc<String>,
    prefix_arc: Arc<String>,
    k: usize,
) -> Result<Vec<SuperfileHit>, QueryError> {
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    let pool_threads = ctx.manifest().options.reader_pool.current_num_threads();
    let kept_refs: Vec<&Arc<SuperfileEntry>> = kept.iter().collect();
    // Prefix expansion is always multi-term OR with no negation, so
    // it is directly sub-range eligible.
    let work_units = build_work_units(&kept_refs, FanOut::SubRanges, pool_threads);
    let units: Vec<(Arc<SuperfileEntry>, Option<(u32, u32)>)> =
        work_units.into_iter().map(|u| (u.entry, u.range)).collect();

    // Shared fan-out — see `bm25_search` for the rationale; the
    // kernel differs only in calling the prefix search variants.
    let kernel = move |r: Arc<SuperfileReader>, range: Option<(u32, u32)>| {
        let column_arc = Arc::clone(&column_arc);
        let prefix_arc = Arc::clone(&prefix_arc);
        async move {
            match range {
                Some((start, end)) => r
                    .bm25_search_prefix_range(&column_arc, &prefix_arc, k, start, end)
                    .await
                    .map_err(fts_read_error),
                None => r
                    .bm25_search_prefix(&column_arc, &prefix_arc, k)
                    .await
                    .map_err(fts_read_error),
            }
        }
    };
    let per_unit = dispatch::fanout_local_hits(ctx, units, kernel).await?;
    select_top_k_stable(ctx, per_unit, k).await
}

impl SupertableReader {
    /// Whether the hidden sibling's current epoch holds text shards
    /// (flat-view probe, zero I/O). Gates both the hidden-text route
    /// and the row wrappers' placement pass: the two must agree, so a
    /// lazy consumer whose hidden flat view hasn't hydrated falls back
    /// to the (always-correct) user path on both sides consistently.
    fn hidden_epoch_has_text(&self) -> bool {
        self.vector_index_table().is_some_and(|vit| {
            vit.pinned_reader()
                .manifest()
                .get_all_superfiles()
                .iter()
                .any(|e| !e.fts_summary.is_empty() && e.vector_summary.is_empty())
        })
    }

    /// The hidden-text route, when this table's hidden sibling holds
    /// text shards: `Some` ⇒ callers run two waves (text shards +
    /// undrained user tail); `None` ⇒ single-wave user path (never
    /// configured, pre-first-drain, or no text shards yet). A
    /// present-but-broken hidden index fails loud — the vector path's
    /// policy (see `vector_search_global_index_async`).
    async fn hidden_text_route(&self) -> Result<Option<HiddenTextRoute>, QueryError> {
        let Some(vit) = self.vector_index_table() else {
            if let Some(reason) = self.hidden_index_open_error() {
                return Err(QueryError::Execute(format!(
                    "hidden index present but failed to open: {reason}"
                )));
            }
            return Ok(None);
        };
        let hidden_reader = vit.pinned_reader();
        let hidden_manifest = Arc::clone(hidden_reader.manifest());
        if !self.hidden_epoch_has_text() {
            return Ok(None);
        }
        // Fast delete set: stable ids deleted since the text epoch.
        vit.ensure_fresh_async().await;
        let deleted = vit
            .pinned_reader()
            .hidden_deleted_ids()
            .map_err(|error| QueryError::Execute(error.to_string()))?;
        let drained = hidden_manifest.get_drained_ranges();
        let fts_routing = hidden_reader.slow_fts_state_resident().await;
        Ok(Some(HiddenTextRoute {
            hidden_reader,
            hidden_manifest,
            fts_routing,
            deleted,
            drained,
        }))
    }

    /// Wave-1 superfiles: the hidden epoch's text shards surviving
    /// `prune_leaf`. The prune keeps entries with no FTS info at all
    /// (always-keep), so the vector family is filtered out here, not
    /// there.
    async fn text_shards_pruned(
        route: &HiddenTextRoute,
        prune_leaf: &PruneLeaf,
    ) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
        Ok(
            select_superfiles(route.hidden_manifest.as_ref(), slice::from_ref(prune_leaf))
                .await?
                .into_iter()
                .filter(|e| !e.fts_summary.is_empty() && e.vector_summary.is_empty())
                .collect(),
        )
    }

    /// Wave-2 superfiles: user files newer than the text epoch's
    /// watermark, masked by the same term-presence prune the text wave
    /// used (the two-tier `select_superfiles` walk already ran for
    /// wave 1; the tail only needs the per-entry mask).
    async fn undrained_tail_pruned(
        &self,
        route: &HiddenTextRoute,
        column: &str,
        terms: &[String],
        mode: BoolMode,
    ) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
        let tail = self
            .manifest()
            .get_undrained_superfiles_loaded(&route.drained)
            .await
            .map_err(QueryError::ManifestLoad)?;
        let term_refs: Vec<&str> = terms.iter().map(|t| t.as_str()).collect();
        let mask = fts_bloom_skip(&tail, column, &term_refs, mode);
        Ok(tail
            .into_iter()
            .zip(mask)
            .filter_map(|(e, keep)| keep.then_some(e))
            .collect())
    }

    /// Single-column BM25 search across the pinned manifest's
    /// superfiles. Returns up to `k` highest-scoring hits, sorted
    /// descending by score.
    ///
    /// `query` is tokenized by the v1 [`AsciiLowerTokenizer`] —
    /// the same tokenizer used at index time. Returns
    /// [`QueryError::Store`] if any superfile is unreachable, or
    /// [`QueryError::Parquet`] if a superfile's bytes can't be
    /// queried (column missing from the superfile's FTS index, etc.).
    ///
    /// Empty supertable (no superfiles) returns an empty `Vec`
    /// without consulting the store.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`SupertableReader::bm25_search`], which drives this via the
    /// sync→async bridge.
    ///
    /// [`AsciiLowerTokenizer`]: crate::superfile::fts::tokenize::AsciiLowerTokenizer
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, mode = ?mode))
    )]
    pub(crate) async fn bm25_search_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let column_owned = column.to_owned();

        // Parse the query once here, not per superfile, resolving the
        // bare tokens' polarity from the default operator (`And` ⇒
        // must, `Or` ⇒ should). The fan-out closures below need owned
        // ('static) data for tokio::spawn, so this is the one place
        // the tokens are copied — the prune and every per-superfile
        // search reuse them.
        let clauses = AsciiLowerTokenizer.parse(query).into_clauses(mode);
        let musts: Vec<String> = clauses.musts.into_iter().map(Cow::into_owned).collect();
        let shoulds: Vec<String> = clauses.shoulds.into_iter().map(Cow::into_owned).collect();
        let negatives: Vec<String> = clauses.negatives.into_iter().map(Cow::into_owned).collect();
        let own_phrases = |phrases: Vec<Vec<Cow<'_, str>>>| -> Vec<Vec<String>> {
            phrases
                .into_iter()
                .map(|p| p.into_iter().map(Cow::into_owned).collect())
                .collect()
        };
        let must_phrases = own_phrases(clauses.must_phrases);
        let should_phrases = own_phrases(clauses.should_phrases);
        let negative_phrases = own_phrases(clauses.negative_phrases);
        let has_musts = !musts.is_empty() || !must_phrases.is_empty();

        if !has_musts && shoulds.is_empty() && should_phrases.is_empty() {
            // No scorable clause at all. Empty / punctuation-only
            // queries match nothing (not an error); negation-only
            // (e.g. `-foo`) has no anchor to rank — reject up front so
            // the per-superfile kernel never has to, and so the
            // unranked count / token_match path surfaces the identical
            // error (see `parse_and_prune`).
            if negatives.is_empty() && negative_phrases.is_empty() {
                return Ok(Vec::new());
            }
            return Err(QueryError::InvalidQuery(NEGATION_ONLY_QUERY_MSG.to_owned()));
        }

        // Pick the superfiles to search, via the shared two-tier bloom
        // prune. Musts prune hardest: every match contains all of
        // them — a phrase's member terms included, since a phrase
        // match requires every member present — so a superfile
        // lacking any is skipped regardless of `mode`. A pure should
        // query prunes as the flat term list did (phrase members join
        // the union: a doc matching the phrase contains each member).
        // Negated atoms never prune, and shoulds never prune once a
        // must exists, since they only affect scores.
        let (mut prune_terms, prune_mode) = if !has_musts {
            (shoulds.clone(), mode)
        } else {
            (musts.clone(), BoolMode::And)
        };
        match has_musts {
            true => {
                for p in &must_phrases {
                    prune_terms.extend(p.iter().cloned());
                }
            }
            false => {
                for p in &should_phrases {
                    prune_terms.extend(p.iter().cloned());
                }
            }
        }
        let prune_leaf = PruneLeaf::TermPresence {
            column: column_owned.clone(),
            terms: prune_terms,
            mode: prune_mode,
        };
        let clauses = BmClauses {
            musts: Arc::new(musts),
            shoulds: Arc::new(shoulds),
            negatives: Arc::new(negatives),
            must_phrases: Arc::new(must_phrases),
            should_phrases: Arc::new(should_phrases),
            negative_phrases: Arc::new(negative_phrases),
        };
        let column_arc = Arc::new(column_owned);
        // ---- Hidden-text route: two waves over (text shards, undrained
        // user tail), mirroring `vector_search_global_index_async`.
        // Fetching k + |deleted| in one pass guarantees k live
        // survivors when they exist (at most |deleted| hits can be
        // identity-filtered), so no refill loop is needed — FTS fetch
        // cost grows only with heap sizes, unlike the vector path's
        // per-candidate rerank.
        if let Some(route) = self.hidden_text_route().await? {
            let text_entries = Self::text_shards_pruned(&route, &prune_leaf).await?;
            let (leaf_terms, leaf_mode) = match &prune_leaf {
                PruneLeaf::TermPresence { terms, mode, .. } => (terms.clone(), *mode),
                _ => unreachable!("bm25 builds a TermPresence leaf above"),
            };
            let tail = self
                .undrained_tail_pruned(&route, &column_arc, &leaf_terms, leaf_mode)
                .await?;
            let k_fetch = k.saturating_add(route.deleted.len());
            // Cross-segment threshold sharing spans BOTH waves: the
            // shared kth-best floor a text shard establishes prunes
            // tail blocks and vice versa.
            let shared = SharedTopK::new(k_fetch);
            let text_wave = bm25_fanout_wave(
                &route.hidden_reader,
                text_entries,
                Arc::clone(&column_arc),
                clauses.clone(),
                k_fetch,
                Arc::clone(&shared),
                route.fts_routing.clone(),
            );
            let tail_wave = bm25_fanout_wave(
                self,
                tail,
                Arc::clone(&column_arc),
                clauses.clone(),
                k_fetch,
                Arc::clone(&shared),
                None,
            );
            let (text_hits, tail_hits) = join!(text_wave, tail_wave);
            // Each wave returns its own id-attached top-`k_fetch` in the
            // (score desc, `_id` asc) total order, so the cross-wave
            // combine is a plain merge under the same order —
            // deterministic regardless of wave completion timing.
            let mut combined = text_hits?;
            combined.extend(tail_hits?);
            sort_hits_by_score_then_id(&mut combined);
            combined.retain(|hit| {
                hit.stable_id
                    .is_some_and(|id| route.deleted.binary_search(&id).is_err())
            });
            combined.truncate(k);
            return Ok(combined);
        }

        // ---- Single-wave user path (no hidden text index).
        let kept = select_superfiles(manifest.as_ref(), slice::from_ref(&prune_leaf)).await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let shared = SharedTopK::new(k);

        bm25_fanout_wave(self, kept, column_arc, clauses, k, shared, None).await
    }

    /// Prefix-expanded BM25 search across the pinned manifest's
    /// superfiles. The prefix is ASCII-lowercased before expansion
    /// (matching the v1 tokenizer) and expanded per-superfile to the
    /// concrete term list before `BoolMode::Or` BM25 scoring.
    ///
    /// Returns up to `k` highest-scoring hits, sorted descending
    /// by score.
    ///
    /// Empty supertable (no superfiles) and `k == 0` short-circuit
    /// to an empty `Vec`.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`SupertableReader::bm25_search_prefix`].
    pub(crate) async fn bm25_search_prefix_async(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let column_owned = column.to_owned();
        let prefix_owned = prefix.to_owned();

        // ManifestSnapshot-level term-range skip uses the same
        // lowercased prefix bytes the v1 tokenizer +
        // FST-expansion path use, so the skip's
        // lex-range overlap test exactly matches the
        // tokenizer's interpretation of the prefix.
        let prefix_lower = prefix_owned.to_ascii_lowercase();

        let prune_leaf = PruneLeaf::Prefix {
            column: column_owned.clone(),
            prefix: prefix_lower.as_bytes().to_vec(),
        };
        let column_arc = Arc::new(column_owned);
        let prefix_arc = Arc::new(prefix_owned);

        // Hidden-text route: two waves, prefix-routed by the text
        // shards' lex term ranges (wave 1) and the tail's per-entry
        // range mask (wave 2); post-epoch deletes identity-filtered
        // out of the k + |deleted| combined fetch.
        if let Some(route) = self.hidden_text_route().await? {
            let text_entries = Self::text_shards_pruned(&route, &prune_leaf).await?;
            let tail = self
                .manifest()
                .get_undrained_superfiles_loaded(&route.drained)
                .await
                .map_err(QueryError::ManifestLoad)?;
            let mask = fts_prefix_skip(&tail, &column_arc, prefix_lower.as_bytes());
            let tail: Vec<Arc<SuperfileEntry>> = tail
                .into_iter()
                .zip(mask)
                .filter_map(|(e, keep)| keep.then_some(e))
                .collect();
            let k_fetch = k.saturating_add(route.deleted.len());
            let text_wave = bm25_prefix_wave(
                &route.hidden_reader,
                text_entries,
                Arc::clone(&column_arc),
                Arc::clone(&prefix_arc),
                k_fetch,
            );
            let tail_wave = bm25_prefix_wave(
                self,
                tail,
                Arc::clone(&column_arc),
                Arc::clone(&prefix_arc),
                k_fetch,
            );
            let (text_hits, tail_hits) = join!(text_wave, tail_wave);
            // Same cross-wave merge contract as `bm25_search_async`:
            // waves are id-attached and (score desc, `_id` asc)-ordered.
            let mut combined = text_hits?;
            combined.extend(tail_hits?);
            sort_hits_by_score_then_id(&mut combined);
            combined.retain(|hit| {
                hit.stable_id
                    .is_some_and(|id| route.deleted.binary_search(&id).is_err())
            });
            combined.truncate(k);
            return Ok(combined);
        }

        // Superfile selection via the shared two-tier prune — the
        // single-`Prefix`-leaf case (part-level term-range skip →
        // lazy-load surviving parts → per-superfile term-range skip).
        let kept = select_superfiles(manifest.as_ref(), slice::from_ref(&prune_leaf)).await?;
        bm25_prefix_wave(self, kept, column_arc, prefix_arc, k).await
    }

    /// Parse `query` into the unranked **match set**, negatives, and
    /// the term-presence prune leaf — with no manifest walk, so each
    /// route prunes its own wave(s) against the right manifest.
    /// `None` = empty/whitespace query (matches nothing, not an
    /// error); negation-only (e.g. `-foo`) is rejected like the
    /// scored path.
    ///
    /// The leaf keys on the **positives only** — a negated term must
    /// never drop a superfile: a superfile lacking it excludes
    /// nothing, and under `And` keying on it would wrongly prune
    /// every superfile that doesn't carry it. Unranked matching has
    /// no scores for a should clause to raise, so the match set is
    /// the musts' intersection whenever any must exists, keeping
    /// `token_match` / `count` consistent with which docs the scored
    /// search returns.
    #[allow(clippy::type_complexity)]
    fn parse_unranked(
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Option<(UnrankedMatchSet, UnrankedNegatives, PruneLeaf)>, QueryError> {
        let clauses = AsciiLowerTokenizer.parse(query).into_clauses(mode);
        let musts: Vec<String> = clauses.musts.into_iter().map(Cow::into_owned).collect();
        let shoulds: Vec<String> = clauses.shoulds.into_iter().map(Cow::into_owned).collect();
        let negatives: Vec<String> = clauses.negatives.into_iter().map(Cow::into_owned).collect();
        let own_phrases = |phrases: Vec<Vec<Cow<'_, str>>>| -> Vec<Vec<String>> {
            phrases
                .into_iter()
                .map(|p| p.into_iter().map(Cow::into_owned).collect())
                .collect()
        };
        let must_phrases = own_phrases(clauses.must_phrases);
        let should_phrases = own_phrases(clauses.should_phrases);
        let negative_phrases = own_phrases(clauses.negative_phrases);
        let negs = UnrankedNegatives {
            terms: negatives,
            phrases: negative_phrases,
        };
        let has_musts = !musts.is_empty() || !must_phrases.is_empty();
        if !has_musts && shoulds.is_empty() && should_phrases.is_empty() {
            if negs.terms.is_empty() && negs.phrases.is_empty() {
                // No tokens at all (empty/whitespace query) — nothing to
                // match, not an error.
                return Ok(None);
            }
            // Negation-only (e.g. `-foo`): reject, matching the scored
            // search path, which has no positive anchor to rank or match.
            return Err(QueryError::InvalidQuery(NEGATION_ONLY_QUERY_MSG.to_owned()));
        }
        // Unranked matching has no scores for a should to raise, so
        // the match set is the must side whenever any must exists.
        let match_set = match has_musts {
            true => UnrankedMatchSet {
                terms: musts,
                phrases: must_phrases,
                mode: BoolMode::And,
            },
            false => UnrankedMatchSet {
                terms: shoulds,
                phrases: should_phrases,
                mode,
            },
        };
        // Prune on the match set's terms plus its phrases' members —
        // a phrase match requires every member present.
        let mut prune_terms = match_set.terms.clone();
        for p in &match_set.phrases {
            prune_terms.extend(p.iter().cloned());
        }
        let prune_leaf = PruneLeaf::TermPresence {
            column: column.to_owned(),
            terms: prune_terms,
            mode: match_set.mode,
        };
        Ok(Some((match_set, negs, prune_leaf)))
    }

    /// Unranked token match across the pinned snapshot. Returns
    /// every row matching `query`'s tokens under `mode` (`Or` = any
    /// token, `And` = every token) as [`SuperfileHit`]s — **no scoring**
    /// (`score` is left `0.0`; these results are unordered). Superfile
    /// skip uses the same term-bloom prune as BM25.
    ///
    /// With a `+must` clause, the match set is the musts' intersection
    /// and bare (should) tokens are ignored — they only affect scores,
    /// and there are none here (see [`Self::parse_and_prune`]).
    ///
    /// `pub(crate)` async kernel; the public surface is the sync
    /// [`SupertableReader::token_match`].
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, mode = ?mode))
    )]
    pub(crate) async fn token_match_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let Some((match_set, negatives, prune_leaf)) = Self::parse_unranked(column, query, mode)?
        else {
            return Ok(Vec::new());
        };
        let match_mode = match_set.mode;
        let column_arc = Arc::new(column.to_owned());
        let term_arc: Arc<Vec<String>> = Arc::new(match_set.terms);
        let phrase_arc: Arc<Vec<Vec<String>>> = Arc::new(match_set.phrases);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives.terms);
        let neg_ph_arc: Arc<Vec<Vec<String>>> = Arc::new(negatives.phrases);

        // Hidden-text route: unranked matching needs no floor sharing
        // (no scores) — the waves just union, then post-epoch deletes
        // are identity-filtered (every match is returned, so no
        // fetch-size padding is needed either).
        if let Some(route) = self.hidden_text_route().await? {
            let text_entries = Self::text_shards_pruned(&route, &prune_leaf).await?;
            let (leaf_terms, leaf_mode) = match &prune_leaf {
                PruneLeaf::TermPresence { terms, mode, .. } => (terms.clone(), *mode),
                _ => unreachable!("parse_unranked builds a TermPresence leaf"),
            };
            let tail = self
                .undrained_tail_pruned(&route, &column_arc, &leaf_terms, leaf_mode)
                .await?;
            let text_wave = token_match_wave(
                &route.hidden_reader,
                text_entries,
                Arc::clone(&column_arc),
                Arc::clone(&term_arc),
                Arc::clone(&phrase_arc),
                Arc::clone(&neg_arc),
                Arc::clone(&neg_ph_arc),
                match_mode,
            );
            let tail_wave = token_match_wave(
                self,
                tail,
                Arc::clone(&column_arc),
                Arc::clone(&term_arc),
                Arc::clone(&phrase_arc),
                Arc::clone(&neg_arc),
                Arc::clone(&neg_ph_arc),
                match_mode,
            );
            let (text_hits, tail_hits) = join!(text_wave, tail_wave);
            let mut hits = text_hits?;
            hits.extend(tail_hits?);
            hits.retain(|hit| {
                hit.stable_id
                    .is_some_and(|id| route.deleted.binary_search(&id).is_err())
            });
            return Ok(hits);
        }

        let kept =
            select_superfiles(self.manifest().as_ref(), slice::from_ref(&prune_leaf)).await?;
        token_match_wave(
            self, kept, column_arc, term_arc, phrase_arc, neg_arc, neg_ph_arc, match_mode,
        )
        .await
    }

    /// Count documents whose `column` matches `query`'s tokens under
    /// `mode` (`Or` = any token, `And` = every token), over this reader's
    /// pinned snapshot — **count only, no scoring and no row
    /// materialization**.
    ///
    /// With a `+must` clause, the count is the musts' intersection
    /// cardinality — bare (should) tokens affect only scores, so they
    /// never change which docs are counted (see
    /// [`Self::parse_and_prune`]). `count("+climate policy")` is the
    /// number of docs containing `climate`.
    ///
    /// Fast path: a single-token query against a superfile with no
    /// tombstones resolves from the term dictionary's stored document
    /// frequency ([`SuperfileReader::term_df`]) — O(1) per superfile, no
    /// posting decode. A multi-token query, or a superfile with deletes,
    /// falls back to materializing the matching local doc ids and
    /// counting those not tombstoned. Tombstoned (deleted) rows are
    /// always excluded so the count matches what a search would return.
    pub(crate) async fn token_match_count_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<u64, QueryError> {
        let Some((match_set, negatives, prune_leaf)) = Self::parse_unranked(column, query, mode)?
        else {
            return Ok(0);
        };
        let match_mode = match_set.mode;
        let column_arc = Arc::new(column.to_owned());
        let term_arc: Arc<Vec<String>> = Arc::new(match_set.terms);
        let phrase_arc: Arc<Vec<Vec<String>>> = Arc::new(match_set.phrases);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives.terms);
        let neg_ph_arc: Arc<Vec<Vec<String>>> = Arc::new(negatives.phrases);

        // Hidden-text route — only when the fast delete set is empty:
        // a count can't be identity-filtered (there are no ids to
        // subtract), so any post-epoch delete falls back to the
        // always-correct user path until the next drain purges it.
        if let Some(route) = self.hidden_text_route().await?
            && route.deleted.is_empty()
        {
            let text_entries = Self::text_shards_pruned(&route, &prune_leaf).await?;
            let (leaf_terms, leaf_mode) = match &prune_leaf {
                PruneLeaf::TermPresence { terms, mode, .. } => (terms.clone(), *mode),
                _ => unreachable!("parse_unranked builds a TermPresence leaf"),
            };
            let tail = self
                .undrained_tail_pruned(&route, &column_arc, &leaf_terms, leaf_mode)
                .await?;
            let text_wave = token_match_count_wave(
                &route.hidden_reader,
                text_entries,
                Arc::clone(&column_arc),
                Arc::clone(&term_arc),
                Arc::clone(&phrase_arc),
                Arc::clone(&neg_arc),
                Arc::clone(&neg_ph_arc),
                match_mode,
            );
            let tail_wave = token_match_count_wave(
                self,
                tail,
                Arc::clone(&column_arc),
                Arc::clone(&term_arc),
                Arc::clone(&phrase_arc),
                Arc::clone(&neg_arc),
                Arc::clone(&neg_ph_arc),
                match_mode,
            );
            let (text_n, tail_n) = join!(text_wave, tail_wave);
            return Ok(text_n? + tail_n?);
        }

        let kept =
            select_superfiles(self.manifest().as_ref(), slice::from_ref(&prune_leaf)).await?;
        token_match_count_wave(
            self, kept, column_arc, term_arc, phrase_arc, neg_arc, neg_ph_arc, match_mode,
        )
        .await
    }

    /// Unranked two-pass exact match of the **raw string** `value`
    /// against `column` across the pinned snapshot. Returns the rows
    /// whose stored value equals `value` exactly as [`SuperfileHit`]s —
    /// **no scoring**. See [`crate::superfile::SuperfileReader::exact_match`]
    /// for the per-superfile two-pass (token-AND prune + raw verify).
    ///
    /// Deliberately NOT routed through the hidden text shards: the
    /// verify pass reads the stored column text, which text superfiles
    /// don't carry (their Parquet body is an `_id` stub) — the user
    /// path is the one that can compare raw strings.
    ///
    /// `pub(crate)` async kernel; the public surface is the sync
    /// [`SupertableReader::exact_match`].
    pub(crate) async fn exact_match_async(
        &self,
        column: &str,
        value: &str,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let manifest = self.manifest();
        let term_strings: Vec<String> = AsciiLowerTokenizer.tokenize(value).collect();
        // Tokens prune superfiles via the term bloom (AND); a token-less
        // value (e.g. punctuation only) can't prune, so keep all.
        let leaves = if term_strings.is_empty() {
            Vec::new()
        } else {
            vec![PruneLeaf::TermPresence {
                column: column.to_owned(),
                terms: term_strings.clone(),
                mode: BoolMode::And,
            }]
        };
        let kept = select_superfiles(manifest.as_ref(), &leaves).await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();
        let column_arc = Arc::new(column.to_owned());
        let value_arc = Arc::new(value.to_owned());
        let tokens_arc = Arc::new(term_strings);
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         _: ()| {
            let column_arc = Arc::clone(&column_arc);
            let value_arc = Arc::clone(&value_arc);
            let tokens_arc = Arc::clone(&tokens_arc);
            async move {
                let candidates: Vec<u32> = if tokens_arc.is_empty() {
                    (0..r.n_docs() as u32).collect()
                } else {
                    let refs: Vec<&str> = tokens_arc.iter().map(String::as_str).collect();
                    r.token_match(&column_arc, &refs, BoolMode::And)
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string()))?
                };
                if candidates.is_empty() {
                    return Ok(Vec::new());
                }
                let batch = if r.can_take_by_local_doc_ids() {
                    r.take_by_local_doc_ids(&candidates, &[column_arc.as_str()])
                        .map_err(|e| QueryError::Parquet(e.to_string()))?
                } else {
                    take_rows_byte_source(&r, &candidates, &[column_arc.as_str()])
                        .await
                        .map_err(|e| QueryError::Execute(e.to_string()))?
                };
                let values = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        QueryError::Execute(format!(
                            "exact_match column '{}' is not LargeUtf8",
                            column_arc
                        ))
                    })?;
                let mut hits: Vec<SuperfileHit> = candidates
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        !values.is_null(*index) && values.value(*index) == value_arc.as_str()
                    })
                    .map(|(_, &local_doc_id)| SuperfileHit {
                        superfile: entry.uri,
                        local_doc_id,
                        score: 0.0,
                        stable_id: None,
                    })
                    .collect();
                dispatch::apply_tombstone_filter(tombstone_cache.as_ref(), &entry, &mut hits, now)?;
                Ok(hits)
            }
        };
        let per_unit = dispatch::fanout_with(self, units, true, true, body).await?;
        let mut hits: Vec<SuperfileHit> = per_unit.into_iter().flatten().collect();
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        Ok(hits)
    }
}

impl SupertableReader {
    /// Single-column BM25 search over this reader's pinned snapshot,
    /// materialized as Arrow rows.
    ///
    /// This is the user-facing row-returning path. It runs the same
    /// BM25 hit kernel the SQL TVF uses, then resolves those top-k hits
    /// through the shared row materializer. Returned batches include
    /// `_id`, every visible scalar column, and a trailing `score` column.
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(async {
            let hits = self.bm25_search_async(column, query, k, mode).await?;
            // Bare projection needs no scalar decode: every hit already
            // carries its stable `_id` (the waves stamp and require it),
            // so build `_id` + `score` directly — the vector path's
            // fast-path contract. Skipping the placement pass below
            // matters: relocating hidden hits into a merged user file
            // with a gapped id span decodes that file's whole `_id`
            // column per query (~60 ms flat post-compact at 1M).
            let id_column = self.options().id_column.as_str();
            if projection_is_id_score_only(projection, id_column) {
                let batch = hits_id_score_batch(self, &hits)?;
                return Ok(vec![batch]);
            }
            // Hidden text-shard hits carry no scalar data; relocate them
            // to their user-table placement by stable `_id` before the
            // decode (user-table hits pass through unchanged). Gated on
            // the same probe the route uses: when no text shards can
            // have produced hits, the pass is skipped — its per-hit
            // manifest lookups would force lazy parts to load.
            let hits = match self.hidden_epoch_has_text() {
                true => user_placement_for_scalar_resolve(self, &hits).await?,
                false => hits,
            };
            // `projection` selects columns by name (any of `_id`, the
            // visible scalar columns, or the trailing `score`); `None`
            // returns `_id` + `score` only. The shared resolver decodes
            // only the projected columns.
            let batch = resolve_hits_named(self, &hits, projection, "bm25_search")
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?;
            Ok(vec![batch])
        })
    }

    /// Low-level BM25 search over this reader's pinned snapshot.
    ///
    /// Drives the internal async kernel to completion via the
    /// sync→async bridge ([`SupertableReader::block_on`]). Returns up
    /// to `k` hits sorted by BM25 score *descending*.
    ///
    /// ## Query clauses (`+term`, `-term`)
    ///
    /// A `+`-prefixed term is a **must**: every hit contains it. A
    /// `-`-prefixed term is a **must-not**: docs containing it are
    /// excluded, regardless of score. Bare terms take their polarity
    /// from `mode`, the default operator — `And` requires them like
    /// musts; `Or` makes them scoring-only **shoulds** when a must
    /// exists (`"+climate policy"` matches the docs containing
    /// `climate`, ranking those that also mention `policy` higher)
    /// and a plain union when none does. A query with only negated
    /// terms is an error.
    pub fn bm25_hits(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.bm25_search_async(column, query, k, mode))
    }

    /// Prefix-expanded BM25 search — see [`SupertableReader::bm25_search`]
    /// for the bridge semantics.
    pub fn bm25_search_prefix(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.bm25_search_prefix_async(column, prefix, k))
    }

    /// Unranked token match over this reader's pinned snapshot. Returns
    /// every row whose `column` matches `query`'s tokens under `mode`
    /// (`Or` = any token, `And` = every token). With a `+must` clause
    /// the match set is the musts' intersection and bare terms are
    /// ignored — unranked matching has no scores for a should to
    /// raise; `-term` exclusions apply. The returned hits are
    /// **unranked** — `score` is `0.0` and order is unspecified — unlike
    /// the ranked [`SupertableReader::bm25_search`]. Drives the async
    /// kernel via the sync→async bridge ([`SupertableReader::block_on`]).
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.token_match_async(column, query, mode))
    }

    /// Count documents matching `query`'s tokens under `mode` over this
    /// reader's pinned snapshot — count only, no scoring or row
    /// materialization. A single-token query on a delete-free superfile
    /// resolves in O(1) from the stored document frequency. Drives the
    /// async kernel via the sync→async bridge.
    pub fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.token_match_count_async(column, query, mode))
    }

    /// Unranked exact match of the raw string `value` against `column`
    /// over this reader's pinned snapshot — the two-pass index-pruned,
    /// text-verified match (see
    /// [`SuperfileReader::exact_match`](crate::superfile::SuperfileReader::exact_match)).
    /// Returns the rows whose stored value equals `value` exactly;
    /// hits are **unranked** (`score` is `0.0`).
    pub fn exact_match(&self, column: &str, value: &str) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.exact_match_async(column, value))
    }
}

/// One unit of per-superfile search work scheduled into the reader
/// pool's `par_iter`. `range == None` means "the whole superfile" and
/// dispatches to the un-ranged BM25 API; `range == Some((start,
/// end))` means "only doc_ids in [start, end)" and dispatches to
/// the range-aware OR path.
struct WorkUnit {
    entry: Arc<SuperfileEntry>,
    range: Option<(u32, u32)>,
}

/// Minimum docs per sub-range. Below this width, splitting adds
/// more pool-scheduling + per-shard top-K-merge overhead than it
/// saves in scoring work. Tuned to be coarse — the heuristic only
/// needs to avoid splitting toy superfiles; production superfiles at
/// the scales we benchmark (1.25M docs/superfile after 10M × cpus/2
/// row-shard) are well above this floor.
const SUBRANGE_MIN_DOCS: u32 = 50_000;

/// Map a per-superfile FTS read error to the query-layer error. A
/// phrase query against a column indexed without positions, or a query
/// with no positive clause to rank, is a malformed *request* — surface
/// it as [`QueryError::InvalidQuery`] so the caller sees a bad-input
/// error, not a storage/scan failure. Everything else is a genuine
/// read error and stays [`QueryError::Parquet`].
fn fts_read_error(e: ReadError) -> QueryError {
    match &e {
        ReadError::Fts(fts)
            if matches!(
                fts.as_ref(),
                FtsError::PositionsUnavailable { .. } | FtsError::NegationOnly
            ) =>
        {
            QueryError::InvalidQuery(e.to_string())
        }
        _ => QueryError::Parquet(e.to_string()),
    }
}

/// How a query fans out over the kept superfiles.
enum FanOut {
    /// One un-ranged unit per superfile.
    PerSuperfile,
    /// Additionally slice big superfiles into doc-id sub-ranges when the
    /// reader pool has spare threads.
    SubRanges,
}

/// Pick the fan-out for a term query: every phrase-free,
/// negation-free positive shape (single term, AND, must+should, and
/// the multi-should union) has a range-aware kernel, so those slice;
/// negation forces the un-ranged walk (the ranged kernels carry no
/// exclusion in v1). `build_work_units` still slices only big files
/// with spare pool threads, so many-small-file tables keep the one
/// unit per superfile shape either way — the slicing win is the
/// hidden index's few large merged shards.
fn fanout_for(n_musts: usize, n_shoulds: usize, has_negatives: bool) -> FanOut {
    if (n_musts + n_shoulds) >= 1 && !has_negatives {
        FanOut::SubRanges
    } else {
        FanOut::PerSuperfile
    }
}

/// Slice the kept superfiles into parallel work units — one
/// [`WorkUnit`] per (superfile, doc_id sub-range) tuple.
///
/// `FanOut::SubRanges` slices only when:
///   1. The reader pool has more threads than kept superfiles —
///      otherwise every thread is already saturated by one superfile
///      and splitting just adds overhead.
///   2. The candidate sub-range width is at least
///      `SUBRANGE_MIN_DOCS` — below that, BMM bookkeeping +
///      cross-sub-range top-K merge dominate the parallel win.
///
/// Otherwise each kept superfile becomes a single un-ranged work unit
/// — identical to the original `par_iter` over superfiles shape.
fn build_work_units(
    kept: &[&Arc<SuperfileEntry>],
    fanout: FanOut,
    pool_threads: usize,
) -> Vec<WorkUnit> {
    let want_subranges = pool_threads.div_ceil(kept.len().max(1)).max(1);
    if matches!(fanout, FanOut::PerSuperfile) || want_subranges <= 1 {
        return kept
            .iter()
            .map(|e| WorkUnit {
                entry: Arc::clone(e),
                range: None,
            })
            .collect();
    }

    let mut units: Vec<WorkUnit> = Vec::with_capacity(kept.len() * want_subranges);
    for entry in kept {
        let n_docs = entry.n_docs as u32;
        if n_docs == 0 {
            continue;
        }
        // Round the sub-range count down to avoid producing
        // narrower-than-floor slices. With `want_subranges = 2` on
        // a 1.25M-doc superfile, stride = 625K (well above floor) so
        // both sub-ranges fire. With a tiny superfile (e.g., 10K
        // docs, well below `SUBRANGE_MIN_DOCS`), the division
        // collapses to 1 sub-range = full superfile.
        let cap_by_floor = (n_docs / SUBRANGE_MIN_DOCS).max(1) as usize;
        let n_sub = want_subranges.min(cap_by_floor);
        if n_sub <= 1 {
            units.push(WorkUnit {
                entry: Arc::clone(entry),
                range: None,
            });
            continue;
        }
        let stride = n_docs.div_ceil(n_sub as u32);
        let mut start: u32 = 0;
        while start < n_docs {
            let end = start.saturating_add(stride).min(n_docs);
            units.push(WorkUnit {
                entry: Arc::clone(entry),
                range: Some((start, end)),
            });
            start = end;
        }
    }
    units
}

/// Total order for merged, id-attached hits: score descending, then
/// stable `_id` ascending — deterministic and invariant across
/// compaction (unlike physical superfile/offset keys). Shared by the
/// single-wave merge ([`select_top_k_stable`]) and the two-wave
/// hidden-text combines, so both routes rank ties identically.
fn sort_hits_by_score_then_id(hits: &mut [SuperfileHit]) {
    hits.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then(a.stable_id.cmp(&b.stable_id))
    });
}

/// Select the global top-k deterministically and compaction-stably: order
/// by score descending, breaking ties on the stable `_id` (ascending).
///
/// A plain score-only merge leaves the choice among
/// score-tied hits to segment completion order — the cross-superfile floor
/// changes which ties each segment returns, so the surviving tied docs vary
/// run to run. Physical keys (superfile uuid + local offset) would break the
/// tie but shift on every compaction. The stable `_id` is invariant across
/// compaction, so tie-breaking on it yields the same top-k as a
/// single-segment engine's docid-ordered ties, independent of layout or
/// completion order. `_id`s are resolved up front here — cheap because the
/// shared floor caps the candidate set near k.
async fn select_top_k_stable(
    tr: &SupertableReader,
    per_unit: Vec<Vec<SuperfileHit>>,
    k: usize,
) -> Result<Vec<SuperfileHit>, QueryError> {
    let mut cands: Vec<SuperfileHit> = per_unit.into_iter().flatten().collect();
    // Narrow to the top-k *by score plus its boundary ties* before touching
    // `_id`. `_id` resolution costs a decode per hit, so it must stay
    // top-k-sized (never per-candidate — that's what the fan-out defers).
    // Partition at the k-th best score, then keep everything scoring at or
    // above it: the strictly-better hits are always in, and the ties at the
    // k-th score are the only ones whose inclusion the `_id` order decides.
    if cands.len() > k {
        cands.select_nth_unstable_by(k - 1, |a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        });
        let kth_score = cands[k - 1].score;
        cands.retain(|c| c.score >= kth_score);
    }
    dispatch::attach_stable_ids_to_hits(tr, &mut cands).await?;
    sort_hits_by_score_then_id(&mut cands);
    cands.truncate(k);
    Ok(cands)
}

impl Supertable {
    /// Single-column BM25 search over the current snapshot, returning
    /// Arrow rows best-score-first (BM25 relevance, higher is better).
    ///
    /// The query string carries lucene-style clause sigils: `+term`
    /// is a must (every hit contains it), `-term` a must-not (hard
    /// exclusion), and bare terms take their polarity from `mode`,
    /// the default operator (`And` ⇒ must, `Or` ⇒ scoring-only should
    /// once any must exists). `"+climate policy"` under `Or` matches
    /// the docs containing `climate` and ranks those also mentioning
    /// `policy` higher.
    ///
    /// A double-quoted run of words is an **exact phrase** atom: the
    /// words must appear adjacent and in order, verified against
    /// token positions. A phrase takes any clause polarity —
    /// `"new york" hotel`, `+"new york" +hotel`, `-"new york"` — and
    /// scores as one BM25 atom whose `tf` is the number of phrase
    /// occurrences and whose `idf` is the sum of its members'. Phrase
    /// queries require the column to be indexed with token positions
    /// (the `positions` flag on the column's FTS build config, off by
    /// default); against a positionless column they return a typed
    /// error rather than silently degrading to a bag-of-words match.
    /// A single-word phrase (`"york"`) is just that term.
    ///
    /// `score` is a similarity (higher is better) — the opposite
    /// direction from [`Supertable::vector_search`]'s distance. Fuse the
    /// two with [`Supertable::hybrid_search`], not by raw score.
    ///
    /// Pins a fresh reader (applying the read-consistency policy), runs
    /// the BM25 fan-out, and resolves the top-`k` hits to Arrow rows.
    ///
    /// `projection` selects output columns by name (any of `_id`, the
    /// visible scalar columns, or the trailing `score`); `None` returns
    /// the engine-native result — `_id` + `score` only. Only the
    /// projected scalar columns are decoded, so materializing row data
    /// is an explicit opt-in by column name.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, BoolMode, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(
    /// #     schema, vec![Arc::new(LargeStringArray::from(vec!["the quick brown fox"]))])?)?;
    /// // Bare call → `_id` + `score`, no scalar decode:
    /// let hits = posts.bm25_search("body", "fox", 10, BoolMode::Or, None)?;
    /// assert_eq!(hits[0].num_columns(), 2);
    /// // Name columns to materialize row data:
    /// let rows = posts.bm25_search("body", "fox", 10, BoolMode::Or, Some(&["_id", "body", "score"]))?;
    /// assert_eq!(rows[0].num_columns(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, mode = ?mode))
    )]
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, k, mode = ?mode, "bm25_search");
        self.reader()
            .bm25_search(column, query, k, mode, projection)
            .map_err(InfinoError::from)
            .map_err(|e| e.with_context("bm25_search", None))
    }

    /// Unranked token match over one FTS column: every row whose
    /// `column` matches `query`'s tokens under `mode` (`Or` = any token,
    /// `And` = every token). With a `+must` clause the match set is
    /// the musts' intersection and bare terms are ignored (no scores
    /// for a should to raise); `-term` exclusions apply. Quoted
    /// phrases participate as atoms exactly as in
    /// [`Supertable::bm25_search`]: an exact-adjacency match against
    /// token positions, requiring a positions-indexed column. Returns
    /// Arrow rows like [`Supertable::bm25_search`], but the `score`
    /// column is `0.0` and row order is unspecified — a candidate
    /// set, not a ranking. `projection` follows the same rules as
    /// `bm25_search`.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, mode = ?mode))
    )]
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, mode = ?mode, "token_match");
        let reader = self.reader();
        let hits = reader
            .token_match(column, query, mode)
            .map_err(|e| InfinoError::from(e).with_context("token_match", None))?;
        let batch = self
            .block_on_query(async {
                // Bare projection: build `_id` + `score` directly from the
                // stable-id stamps, skipping the placement pass — same
                // fast-path contract as `bm25_search` (a gapped merged
                // user file makes the pass decode its whole `_id` column
                // per query).
                let id_column = reader.options().id_column.as_str();
                if projection_is_id_score_only(projection, id_column) {
                    return hits_id_score_batch(&reader, &hits);
                }
                // Hidden text-shard hits carry no scalar data; relocate
                // them to their user-table placement by stable `_id`
                // before the decode. Gated on the same probe the route
                // uses: when no text shards can have produced hits, the
                // pass is skipped — its per-hit manifest lookups would
                // force lazy parts to load.
                let hits = match reader.hidden_epoch_has_text() {
                    true => user_placement_for_scalar_resolve(&reader, &hits).await?,
                    false => hits,
                };
                resolve_hits_named(&reader, &hits, projection, "token_match")
                    .await
                    .map_err(|e| QueryError::Execute(e.to_string()))
            })
            .map_err(|e| InfinoError::Query(e.to_string()).with_context("token_match", None))?;
        Ok(vec![batch])
    }

    /// Unranked exact match: rows whose `column` value equals `value`
    /// exactly (index-pruned, then text-verified). Returns Arrow rows
    /// like [`Supertable::bm25_search`], with `score` fixed at `0.0` and
    /// unspecified row order. `projection` follows the same rules as
    /// `bm25_search`.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column))
    )]
    pub fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, "exact_match");
        let reader = self.reader();
        let hits = reader
            .exact_match(column, value)
            .map_err(|e| InfinoError::from(e).with_context("exact_match", None))?;
        let batch = self
            .block_on_query(async {
                // Bare projection: `_id` + score straight from the
                // stable-id stamps — same fast path as `bm25_search`.
                let id_column = reader.options().id_column.as_str();
                if projection_is_id_score_only(projection, id_column) {
                    return hits_id_score_batch(&reader, &hits);
                }
                // Hidden text-shard hits carry no scalar data; relocate
                // them to their user-table placement by stable `_id`
                // before the decode — this pass was missing here (only
                // `bm25_search`/`token_match` had it), so scalar
                // projections mis-resolved hidden hits.
                let hits = match reader.hidden_epoch_has_text() {
                    true => user_placement_for_scalar_resolve(&reader, &hits).await?,
                    false => hits,
                };
                resolve_hits_named(&reader, &hits, projection, "exact_match")
                    .await
                    .map_err(|e| QueryError::Execute(e.to_string()))
            })
            .map_err(|e| InfinoError::Query(e.to_string()).with_context("exact_match", None))?;
        Ok(vec![batch])
    }

    /// Count documents whose `column` matches `query`'s tokens under
    /// `mode` (`Or` = any token, `And` = every token) over the current
    /// snapshot — count only, no scoring or row materialization. A
    /// single-token query on a delete-free snapshot resolves in O(1) per
    /// superfile from the term dictionary's document frequency, so
    /// counting a high-frequency term is cheap.
    ///
    /// With a `+must` clause the count is the musts' intersection
    /// cardinality — bare (should) terms affect only scores, never
    /// which docs count, so `count("+climate policy")` is the number
    /// of docs containing `climate`. A lone must keeps the O(1) df
    /// fast path. `-term` exclusions apply as in search. Quoted
    /// phrases count exact-adjacency matches (verified against token
    /// positions, so the column must be positions-indexed) — every
    /// match is verified, giving exact phrase counts.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, BoolMode, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(
    /// #     schema,
    /// #     vec![Arc::new(LargeStringArray::from(vec!["the quick brown fox", "a lazy dog"]))],
    /// # )?)?;
    /// let n = posts.count("body", "fox", BoolMode::Or)?;
    /// assert_eq!(n, 1);
    /// // `+must` defines the count; bare terms are scoring-only:
    /// let n = posts.count("body", "+quick lazy", BoolMode::Or)?;
    /// assert_eq!(n, 1); // docs containing `quick`
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError> {
        self.reader()
            .count(column, query, mode)
            .map_err(InfinoError::from)
            .map_err(|e| e.with_context("count", None))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, future::Future, sync::Arc};

    use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use datafusion::prelude::{col, lit};
    use tokio::runtime::Builder;

    use super::{BoolMode, FanOut, build_work_units, fanout_for, hits_id_score_batch};
    use crate::{
        config::{CompactionSettings, DEFAULT_STALE_SEAL_TIMEOUT_MS},
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            vector::layout::VectorLayout,
        },
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            handle::SupertableReader,
            options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        },
        test_helpers::default_tokenizer as tok,
    };

    /// Drive an async future to completion on a throwaway current-thread
    /// runtime. Used only for the single-superfile `SuperfileReader`
    /// oracle, whose search surface is async-only; the supertable
    /// reader's own search methods are sync and need no runtime here.
    fn block_on<F: Future>(fut: F) -> F::Output {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(fut)
    }

    fn schema_id_title() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn options_one_superfile_per_commit() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    fn build_batch(_start: u64, titles: &[&str]) -> RecordBatch {
        let titles_arr = LargeStringArray::from(titles.to_vec());
        RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles_arr)]).expect("batch")
    }

    /// Build a single SuperfileBuilder containing the same docs as
    /// the supertable across all superfiles. Used as the oracle for
    /// per-superfile-vs-global BM25 set-membership tests.
    fn build_oracle_superfile(titles: &[&str]) -> Arc<SuperfileReader> {
        // The oracle path goes directly through SuperfileBuilder
        // (not through Supertable::append's auto-injection), so
        // we build the effective schema by hand: `_id` is
        // `Decimal128(38, 0)`, ids are 0..n.
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "_id",
                DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
                false,
            ),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "_id",
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let n = titles.len();
        let ids = Decimal128Array::from((0..n as i128).collect::<Vec<_>>())
            .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
            .expect("decimal128");
        let titles_arr = LargeStringArray::from(titles.to_vec());
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles_arr)]).expect("batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = Bytes::from(b.finish().expect("finish"));
        Arc::new(SuperfileReader::open(bytes).expect("open"))
    }

    #[test]
    fn negation_excludes_across_superfiles() {
        // 3 commits → 3 superfiles. "alpha -beta" must drop the one doc
        // containing beta and keep the other two alpha docs.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["beta gamma"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r
            .bm25_hits("title", "alpha -beta", 10, BoolMode::Or)
            .expect("negation search");
        assert_eq!(hits.len(), 2, "alpha minus beta: {hits:?}");

        // Positive-only stays untouched: all three alpha docs.
        let hits = r
            .bm25_hits("title", "alpha", 10, BoolMode::Or)
            .expect("positive search");
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn negated_term_does_not_prune_superfiles() {
        // "delta" exists only in superfile 2. Under And, if the negated
        // term leaked into the bloom prune, superfiles 1 and 3 (no delta)
        // would be wrongly dropped and the result would be empty; the
        // correct answer is superfile 1's two alpha docs.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha one", "alpha two"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["gamma three"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r
            .bm25_hits("title", "alpha -delta", 10, BoolMode::And)
            .expect("negation search");
        assert_eq!(hits.len(), 2, "alpha minus delta: {hits:?}");
    }

    #[test]
    fn negation_only_query_errors() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let res = r.bm25_hits("title", "-alpha", 10, BoolMode::Or);
        assert!(res.is_err(), "negation-only must error; got {res:?}");
    }

    #[test]
    fn count_and_token_match_negation_only_query_errors() {
        // The unranked count / token_match surfaces reject a negation-only
        // query (`-foo`) the same way the scored path does — there is no
        // positive anchor to match against. A token-less query (empty /
        // whitespace) is still 0 / empty, not an error.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta"])).expect("append");
        w.commit().expect("commit");
        let r = st.reader();

        for mode in [BoolMode::Or, BoolMode::And] {
            assert!(
                r.count("title", "-alpha", mode).is_err(),
                "negation-only count must error ({mode:?})"
            );
            assert!(
                r.token_match("title", "-alpha", mode).is_err(),
                "negation-only token_match must error ({mode:?})"
            );
        }
        // No positive anchor across several negated terms either.
        assert!(r.count("title", "-alpha -beta", BoolMode::Or).is_err());
        // Token-less queries stay non-error, 0 / empty.
        assert_eq!(r.count("title", "", BoolMode::Or).expect("empty"), 0);
        assert!(
            r.token_match("title", "   ", BoolMode::Or)
                .expect("blank")
                .is_empty()
        );
    }

    #[test]
    fn bm25_search_empty_supertable_returns_empty_without_store_calls() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 5, BoolMode::Or)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 0, BoolMode::Or)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_search_returns_descending_score_order() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &[
                "rust rust rust async",
                "rust async runtime",
                "rust embedded",
                "python data",
            ],
        ))
        .expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 4, BoolMode::Or)
            .expect("query");
        // Should return 3 hits (the python doc has no `rust`).
        assert_eq!(hits.len(), 3);
        // Strictly descending.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn bm25_search_carries_superfile_uri_for_each_hit() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust rust async"])).expect("a1");
        w.commit().expect("c1");
        w.append(&build_batch(10, &["rust runtime"])).expect("a2");
        w.commit().expect("c2");

        let r = st.reader();
        assert_eq!(r.n_superfiles(), 2);
        let hits = r
            .bm25_hits("title", "rust", 5, BoolMode::Or)
            .expect("query");
        assert_eq!(hits.len(), 2);
        // Both superfile URIs should appear.
        let mut uris: Vec<_> = hits.iter().map(|h| h.superfile).collect();
        uris.sort();
        let expected: Vec<_> = {
            let mut v: Vec<_> = r.manifest().superfiles.iter().map(|e| e.uri).collect();
            v.sort();
            v
        };
        assert_eq!(uris, expected);
    }

    #[test]
    fn bm25_search_oracle_top_k_set_matches_single_superfile() {
        // Plant a corpus where the top-k under BM25 is unambiguous
        // regardless of per-superfile-vs-global IDF variation: 3 docs
        // contain the rare term `nimblefox`, distributed across 3
        // superfiles; the other 9 docs share only generic terms with
        // each other and with the query, so they score zero against
        // `nimblefox`. The set membership check survives even
        // though per-superfile IDF for `nimblefox` differs from
        // global IDF (it's `df=1` in each superfile vs `df=3` global).
        let titles = vec![
            "lookup nimblefox special token",   // 0  — match
            "ordinary common everyday text",    // 1
            "more usual filler corpus copy",    // 2
            "something boring without it",      // 3
            "mid corpus another nimblefox row", // 4  — match
            "generic page that adds nothing",   // 5
            "another stuffer no rare terms",    // 6
            "more padding here for filler",     // 7
            "tail nimblefox final superfile",   // 8  — match
            "another tail row",                 // 9
            "yet another normal title",         // 10
            "wrapping up the corpus today",     // 11
        ];

        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for chunk_start in (0..titles.len()).step_by(4) {
            let end = (chunk_start + 4).min(titles.len());
            let chunk = &titles[chunk_start..end];
            w.append(&build_batch(chunk_start as u64, chunk))
                .expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(st.reader().n_superfiles(), 3);

        let oracle = build_oracle_superfile(&titles);
        // Single-superfile `SuperfileReader` oracle: async-only search,
        // driven on a throwaway runtime. The supertable reader below
        // uses its sync public API.
        let oracle_hits = block_on(oracle.bm25_hits_async("title", "nimblefox", 5, BoolMode::Or))
            .expect("oracle");
        // Oracle should find exactly 3 docs containing `nimblefox`.
        assert_eq!(oracle_hits.len(), 3);
        let oracle_set: HashSet<u32> = oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_set, [0u32, 4, 8].iter().copied().collect());

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_hits("title", "nimblefox", 5, BoolMode::Or)
            .expect("supertable query");
        assert_eq!(st_hits.len(), 3);
        // Resolve supertable hits to global doc-ids via superfile
        // ordering (superfiles appear in append order; chunk size = 4).
        let manifest = st_reader.manifest();
        let st_globals: HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.superfile)
                    .expect("superfile in manifest");
                (seg_idx as u32) * 4 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_globals, oracle_set);
    }

    #[test]
    fn bm25_search_prefix_oracle_top_k_set_matches_single_superfile() {
        let titles = vec![
            "rust async runtime",
            "rust embedded systems",
            "ruby gemfile config",
            "rustacean conference",
            "python machine learning",
            "python web framework",
            "rusty pipe rebuild",
            "go concurrency model",
        ];
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for chunk_start in (0..titles.len()).step_by(2) {
            let end = (chunk_start + 2).min(titles.len());
            let chunk = &titles[chunk_start..end];
            w.append(&build_batch(chunk_start as u64, chunk))
                .expect("append");
            w.commit().expect("commit");
        }

        let oracle = build_oracle_superfile(&titles);
        let oracle_hits = block_on(oracle.bm25_search_prefix("title", "rust", 5)).expect("oracle");
        let oracle_globals: HashSet<u32> = oracle_hits.iter().map(|(d, _)| *d).collect();

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_search_prefix("title", "rust", 5)
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.superfile)
                    .expect("superfile in manifest");
                (seg_idx as u32) * 2 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_hits.len(), oracle_hits.len());
        assert_eq!(st_globals, oracle_globals);
        // Prefix-expansion sanity: we should hit "rust*" and
        // "rusty*" / "rustacean*" but not "ruby*".
        assert!(st_hits.len() >= 4);
    }

    #[test]
    fn bm25_search_prefix_unmatched_prefix_returns_empty() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r.bm25_search_prefix("title", "zzzz", 10).expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_search_prefix_lowercases_input() {
        // Index stores tokenized terms (lowercased); user provides
        // mixed-case prefix; we lowercase before expansion so the
        // FST walk finds the matching subtree.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["Rust async runtime"]))
            .expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r.bm25_search_prefix("title", "RUST", 5).expect("query");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn bm25_search_unknown_column_errors() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let err = r
            .bm25_hits("missing_column", "rust", 5, BoolMode::Or)
            .expect_err("expected error");
        assert!(matches!(err, QueryError::Parquet(_)), "got {err:?}");
    }

    #[test]
    fn bm25_search_results_global_top_k_caps_at_k() {
        // 4 superfiles × 1 doc each = 4 hits; ask for k=2; expect 2.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for i in 0..4 {
            w.append(&build_batch(i * 10, &["rust async runtime"]))
                .expect("a");
            w.commit().expect("c");
        }
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 2, BoolMode::Or)
            .expect("query");
        assert_eq!(hits.len(), 2);
    }

    fn seeded_three_doc_supertable() -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["the quick brown fox", "a lazy dog", "quick thinking"],
        ))
        .expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn supertable_bm25_search_rows_default_and_projected() {
        let st = seeded_three_doc_supertable();

        // Bare call → `_id` + `score` only (no scalar decode).
        let bare = st
            .bm25_search("title", "fox", 10, BoolMode::Or, None)
            .expect("bm25 rows");
        assert_eq!(bare.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        assert_eq!(bare[0].num_columns(), 2, "_id + score");

        // Named projection materializes the requested columns.
        let rows = st
            .bm25_search(
                "title",
                "fox",
                10,
                BoolMode::Or,
                Some(&["_id", "title", "score"]),
            )
            .expect("bm25 projected rows");
        assert_eq!(rows[0].num_columns(), 3);
    }

    #[test]
    fn supertable_token_match_and_exact_match_rows() {
        let st = seeded_three_doc_supertable();

        // token_match: any row containing "quick" (Or over one token).
        let tm = st
            .token_match("title", "quick", BoolMode::Or, None)
            .expect("token_match");
        assert_eq!(tm.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

        // exact_match: only the row equal to the raw string.
        let em = st
            .exact_match("title", "a lazy dog", Some(&["_id", "title"]))
            .expect("exact_match");
        assert_eq!(em.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        assert_eq!(em[0].num_columns(), 2);
    }

    #[test]
    fn reader_token_match_and_exact_match_hits() {
        let st = seeded_three_doc_supertable();
        let r = st.reader();

        // token_match And requires every token to be present.
        let any = r.token_match("title", "quick", BoolMode::And).expect("tm");
        assert_eq!(any.len(), 2);

        // Token-less value (punctuation only) prunes nothing and matches
        // no stored row exactly.
        let none = r.exact_match("title", "!!!").expect("em punctuation");
        assert!(none.is_empty());

        // Exact verify against a real row.
        let one = r.exact_match("title", "quick thinking").expect("em");
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn token_match_empty_query_short_circuits() {
        let st = seeded_three_doc_supertable();
        let r = st.reader();
        // A query that tokenizes to nothing returns empty without
        // touching the store.
        let hits = r
            .token_match("title", "   ", BoolMode::Or)
            .expect("tm empty");
        assert!(hits.is_empty());
    }

    /// Two-superfile fixture for the clause model: `climate` docs are
    /// split across superfiles, and one superfile has no `climate` at
    /// all (so the must prune drops it).
    fn seeded_clause_supertable() -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["climate change policy", "climate science report"],
        ))
        .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(
            10,
            &["policy analysis quarterly", "climate policy summit"],
        ))
        .expect("append");
        w.commit().expect("commit");
        st
    }

    /// Positional twin of the options fixture, for phrase queries.
    fn options_positional_one_superfile_per_commit() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig {
                column: "title".into(),
                positions: true,
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Two superfiles with controlled "new york" adjacency: docs in
    /// the first commit match (0, 1), the second commit has both
    /// words non-adjacent plus one more match.
    fn seeded_phrase_supertable() -> Supertable {
        let st = Supertable::create(options_positional_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["new york city", "the new york times"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(10, &["york loves new haven", "big new york"]))
            .expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn phrase_query_end_to_end() {
        let st = seeded_phrase_supertable();
        let r = st.reader();

        // Ranked: exactly the adjacent-in-order docs across both
        // superfiles.
        let hits = r
            .bm25_hits("title", r#""new york""#, 10, BoolMode::Or)
            .expect("phrase hits");
        assert_eq!(hits.len(), 3, "three docs contain the phrase");

        // Count = the phrase match set.
        let n = r
            .count("title", r#""new york""#, BoolMode::Or)
            .expect("phrase count");
        assert_eq!(n, 3);
        // The non-adjacent doc is the difference vs the token AND.
        let and_count = r
            .count("title", "+new +york", BoolMode::Or)
            .expect("token and count");
        assert_eq!(and_count, 4);

        // Phrase composed with clauses: must-phrase + must-term.
        let hits = r
            .bm25_hits("title", r#"+"new york" +the"#, 10, BoolMode::Or)
            .expect("phrase + term");
        assert_eq!(hits.len(), 1);

        // Negated phrase: docs with `york` minus the phrase docs.
        let n = r
            .count("title", r#"york -"new york""#, BoolMode::Or)
            .expect("negated phrase count");
        assert_eq!(n, 1);
    }

    #[test]
    fn phrase_on_positionless_table_errors() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        let err = r
            .bm25_hits("title", r#""climate change""#, 10, BoolMode::Or)
            .expect_err("typed error expected");
        // A phrase on a positionless column is a bad *request*, not a
        // read failure — it surfaces as InvalidQuery, and the message
        // explains the missing positions.
        assert!(
            matches!(err, QueryError::InvalidQuery(_)),
            "phrase on positionless column should be InvalidQuery, got {err:?}"
        );
        assert!(
            err.to_string().contains("positions"),
            "error should say positions are missing: {err}"
        );
        let err = r
            .count("title", r#""climate change""#, BoolMode::Or)
            .expect_err("count errors too");
        assert!(
            matches!(err, QueryError::InvalidQuery(_)),
            "count phrase on positionless column should be InvalidQuery, got {err:?}"
        );
        assert!(err.to_string().contains("positions"));
    }

    #[test]
    fn must_should_match_set_and_count_across_superfiles() {
        let st = seeded_clause_supertable();
        let r = st.reader();

        // 3 docs contain `climate`; `policy` is scoring-only and must
        // not pull in "policy analysis quarterly".
        let hits = r
            .bm25_hits("title", "+climate policy", 10, BoolMode::Or)
            .expect("bm25 +climate policy");
        assert_eq!(hits.len(), 3, "match set is the must set");

        // Count agrees with the scored match set and ignores shoulds.
        let n = r
            .count("title", "+climate policy", BoolMode::Or)
            .expect("count +climate policy");
        assert_eq!(n, 3);
        // Flat OR over the same tokens is the union — strictly bigger.
        let union = r
            .count("title", "climate policy", BoolMode::Or)
            .expect("count union");
        assert_eq!(union, 4);

        // Docs matching must+should outrank must-only docs: both
        // climate∧policy docs come first.
        let top2: Vec<f32> = hits.iter().take(2).map(|h| h.score).collect();
        let third = hits[2].score;
        assert!(
            top2.iter().all(|s| *s > third),
            "climate∧policy docs must outrank climate-only: {hits:?}"
        );
    }

    #[test]
    fn must_should_token_match_matches_musts_only() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        // Unranked matching has no scores for the should to raise —
        // the match set is exactly the must set.
        let tm = r
            .token_match("title", "+climate policy", BoolMode::Or)
            .expect("tm +climate policy");
        assert_eq!(tm.len(), 3);
    }

    #[test]
    fn must_should_with_negation_across_superfiles() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        // Negation still excludes: drop the summit doc from the
        // climate must set.
        let hits = r
            .bm25_hits("title", "+climate policy -summit", 10, BoolMode::Or)
            .expect("bm25 with negation");
        assert_eq!(hits.len(), 2);
        let n = r
            .count("title", "+climate policy -summit", BoolMode::Or)
            .expect("count with negation");
        assert_eq!(n, 2);
    }

    #[test]
    fn absent_must_prunes_every_superfile() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        // The must term exists nowhere: bloom-prune (or the empty
        // intersection) yields no hits despite the common should.
        let hits = r
            .bm25_hits("title", "+zzzabsent policy", 10, BoolMode::Or)
            .expect("bm25 absent must");
        assert!(hits.is_empty());
        let n = r
            .count("title", "+zzzabsent policy", BoolMode::Or)
            .expect("count absent must");
        assert_eq!(n, 0);
    }

    #[test]
    fn token_match_no_match_returns_empty() {
        let st = seeded_three_doc_supertable();
        let r = st.reader();
        let hits = r
            .token_match("title", "nonexistentterm", BoolMode::Or)
            .expect("tm");
        assert!(hits.is_empty());
    }

    #[test]
    fn fanout_for_slices_every_negation_free_positive_shape() {
        // Every phrase-free, negation-free positive shape has a
        // range-aware kernel: multi-should OR, single term, AND,
        // and must+should all slice.
        assert!(matches!(fanout_for(0, 2, false), FanOut::SubRanges));
        assert!(matches!(fanout_for(0, 1, false), FanOut::SubRanges));
        assert!(matches!(fanout_for(2, 0, false), FanOut::SubRanges));
        assert!(matches!(fanout_for(1, 1, false), FanOut::SubRanges));
        // Negation disables sub-ranges (the ranged kernels carry no
        // exclusion in v1).
        assert!(matches!(fanout_for(0, 2, true), FanOut::PerSuperfile));
        assert!(matches!(fanout_for(1, 0, true), FanOut::PerSuperfile));
    }

    #[test]
    fn build_work_units_per_superfile_is_one_unranged_unit_each() {
        use std::collections::HashMap;

        use uuid::Uuid;

        use crate::supertable::manifest::{SuperfileEntry, SuperfileUri};

        fn entry(n_docs: u64) -> Arc<SuperfileEntry> {
            let id = Uuid::new_v4();
            Arc::new(SuperfileEntry {
                birth_version: 0,
                superfile_id: id,
                uri: SuperfileUri(id),
                n_docs,
                id_min: 0,
                id_max: n_docs.saturating_sub(1) as i128,
                scalar_stats: HashMap::new(),
                row_group_stats: None,
                fts_summary: HashMap::new(),
                vector_summary: HashMap::new(),
                partition_key: Vec::new(),
                partition_hint: None,
                vector_layout: VectorLayout::Ivf,
                subsection_offsets: None,
            })
        }

        let e0 = entry(100);
        let e1 = entry(200);
        let kept = vec![&e0, &e1];

        // PerSuperfile always yields exactly one un-ranged unit per kept
        // superfile regardless of pool width.
        let units = build_work_units(&kept, FanOut::PerSuperfile, 8);
        assert_eq!(units.len(), 2);
        assert!(units.iter().all(|u| u.range.is_none()));

        // SubRanges with one pool thread collapses to per-superfile too
        // (no spare threads to slice across).
        let units = build_work_units(&kept, FanOut::SubRanges, 1);
        assert_eq!(units.len(), 2);
        assert!(units.iter().all(|u| u.range.is_none()));

        // Tiny superfiles below SUBRANGE_MIN_DOCS never slice even with
        // spare threads.
        let units = build_work_units(&kept, FanOut::SubRanges, 16);
        assert_eq!(units.len(), 2);
        assert!(units.iter().all(|u| u.range.is_none()));
    }

    #[test]
    fn build_work_units_slices_large_superfiles_when_threads_spare() {
        use std::collections::HashMap;

        use uuid::Uuid;

        use crate::supertable::manifest::{SuperfileEntry, SuperfileUri};

        let id = Uuid::new_v4();
        // One large superfile, well above SUBRANGE_MIN_DOCS (50k).
        let big = Arc::new(SuperfileEntry {
            birth_version: 0,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 200_000,
            id_min: 0,
            id_max: 199_999,
            scalar_stats: HashMap::new(),
            row_group_stats: None,
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        });
        let kept = vec![&big];
        // 4 spare threads, 1 superfile → slice into multiple ranged units
        // that tile [0, n_docs) without gaps.
        let units = build_work_units(&kept, FanOut::SubRanges, 4);
        assert!(units.len() > 1, "large superfile sliced into sub-ranges");
        let mut cursor = 0u32;
        for u in &units {
            let (start, end) = u.range.expect("ranged unit");
            assert_eq!(start, cursor);
            cursor = end;
        }
        assert_eq!(cursor, 200_000, "sub-ranges tile the whole superfile");
    }

    #[test]
    fn count_single_term_sums_df_across_superfiles() {
        // 3 commits → 3 superfiles. Single-term count takes the O(1)
        // term_df fast path (no deletes) and sums across superfiles.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["beta gamma"])).expect("append");
        w.commit().expect("commit");

        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("count"), 3);
        assert_eq!(st.count("title", "beta", BoolMode::Or).expect("count"), 2);
        assert_eq!(st.count("title", "gamma", BoolMode::Or).expect("count"), 2);
        assert_eq!(st.count("title", "absent", BoolMode::Or).expect("count"), 0);
    }

    #[test]
    fn count_multi_term_sums_across_superfiles() {
        // 3 commits → 3 superfiles. Multi-term queries take the general
        // `token_match` branch (not the single-term df fast path), so this
        // exercises summing per-superfile match counts across superfiles
        // for both OR (union spans all three) and AND (intersection lands
        // in one). Doc ids are globally unique across commits.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["beta gamma", "delta"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(4, &["alpha delta", "beta"]))
            .expect("append");
        w.commit().expect("commit");

        // OR "alpha beta": alpha∪beta matches in all three superfiles
        // (2 + 1 + 2) — proves the per-superfile counts are summed.
        assert_eq!(st.count("title", "alpha beta", BoolMode::Or).expect("c"), 5);
        // OR "gamma delta": 1 + 2 + 1 across the three superfiles.
        assert_eq!(
            st.count("title", "gamma delta", BoolMode::Or).expect("c"),
            4
        );
        // AND "alpha beta": both terms only in the first superfile's
        // "alpha beta" doc → 1 (the other superfiles contribute 0).
        assert_eq!(
            st.count("title", "alpha beta", BoolMode::And).expect("c"),
            1
        );
        // AND "alpha delta": both terms only in the third superfile.
        assert_eq!(
            st.count("title", "alpha delta", BoolMode::And).expect("c"),
            1
        );

        // Cross-check every shape against token_match cardinality.
        let r = st.reader();
        for (q, mode) in [
            ("alpha beta", BoolMode::Or),
            ("gamma delta", BoolMode::Or),
            ("alpha beta", BoolMode::And),
            ("alpha delta", BoolMode::And),
        ] {
            let c = r.count("title", q, mode).expect("count");
            let n = r.token_match("title", q, mode).expect("token_match").len() as u64;
            assert_eq!(c, n, "count vs token_match for {q:?} {mode:?}");
        }
    }

    #[test]
    fn count_honors_or_and_modes() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha beta", "alpha gamma", "beta delta"],
        ))
        .expect("append");
        w.commit().expect("commit");

        // OR: docs containing alpha OR delta → all three.
        assert_eq!(
            st.count("title", "alpha delta", BoolMode::Or).expect("c"),
            3
        );
        // AND: docs containing both alpha AND beta → just "alpha beta".
        assert_eq!(
            st.count("title", "alpha beta", BoolMode::And).expect("c"),
            1
        );
        // AND with no doc holding both → 0.
        assert_eq!(
            st.count("title", "gamma delta", BoolMode::And).expect("c"),
            0
        );
    }

    #[test]
    fn count_agrees_with_token_match_len() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha beta", "alpha gamma", "beta delta"],
        ))
        .expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        for (q, mode) in [
            ("alpha", BoolMode::Or),
            ("alpha delta", BoolMode::Or),
            ("alpha beta", BoolMode::And),
        ] {
            let c = r.count("title", q, mode).expect("count");
            let n = r.token_match("title", q, mode).expect("token_match").len() as u64;
            assert_eq!(c, n, "count vs token_match for {q:?} {mode:?}");
        }
    }

    #[test]
    fn count_empty_query_and_empty_supertable_are_zero() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        // Empty supertable: nothing matches.
        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("c"), 0);
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta"])).expect("append");
        w.commit().expect("commit");
        // Token-less queries produce no terms → 0.
        assert_eq!(st.count("title", "", BoolMode::Or).expect("c"), 0);
        assert_eq!(st.count("title", "   ", BoolMode::Or).expect("c"), 0);
    }

    #[test]
    fn count_excludes_tombstoned_docs() {
        // Storage-backed so delete (tombstones) is available. After a
        // delete, the single-term count must drop the term_df fast path
        // and subtract the tombstone — df would over-count.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit().with_storage(storage))
            .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha one", "alpha two", "alpha three"]))
            .expect("append");
        w.commit().expect("commit");
        drop(w); // release the writer slot so `delete` can acquire it

        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("count"), 3);

        let stats = st
            .delete(col("title").eq(lit("alpha two")))
            .expect("delete");
        assert_eq!(stats.matched(), 1);

        // term_df still says 3; the count must subtract the tombstone → 2.
        assert_eq!(
            st.count("title", "alpha", BoolMode::Or)
                .expect("count after delete"),
            2
        );
    }

    #[test]
    fn count_excludes_negated_terms() {
        // A count query with a negated term must drop the docs matching
        // that term, the same way a scored search does. The earlier count
        // path tokenized "alpha -beta" into ["alpha", "beta"] and counted
        // "beta" as a positive, so it over-counted instead of excluding.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["beta gamma"])).expect("append");
        w.commit().expect("commit");

        // "alpha" matches three docs across the superfiles; "-beta" drops
        // the one that also contains beta → 2. Mirrors the search-side
        // `negation_excludes_across_superfiles`.
        assert_eq!(
            st.count("title", "alpha -beta", BoolMode::Or)
                .expect("count"),
            2
        );
        // Positive-only count is unchanged: all three alpha docs.
        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("count"), 3);
        // A negated term absent from the corpus excludes nothing.
        assert_eq!(
            st.count("title", "alpha -absent", BoolMode::Or)
                .expect("count"),
            3
        );
    }

    #[test]
    fn count_with_negation_agrees_with_token_match() {
        // The count↔token_match invariant must hold for negated queries
        // too, across OR / AND and single- vs multi-positive shapes.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha beta", "alpha gamma", "beta delta", "gamma delta"],
        ))
        .expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        for (q, mode) in [
            ("alpha -beta", BoolMode::Or),
            ("alpha gamma -delta", BoolMode::Or),
            ("alpha -gamma", BoolMode::And),
            ("beta -alpha", BoolMode::Or),
        ] {
            let c = r.count("title", q, mode).expect("count");
            let n = r.token_match("title", q, mode).expect("token_match").len() as u64;
            assert_eq!(c, n, "count vs token_match for {q:?} {mode:?}");
        }
    }

    #[test]
    fn count_excludes_negated_terms_and_tombstones() {
        // Negation and deletes compose: the materialized count drops both
        // negated-term docs and tombstoned docs in one pass.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit().with_storage(storage))
            .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha one", "alpha two", "alpha beta", "alpha three"],
        ))
        .expect("append");
        w.commit().expect("commit");
        drop(w); // release the writer slot so `delete` can acquire it

        // 4 alpha docs minus the one also containing beta → 3.
        assert_eq!(
            st.count("title", "alpha -beta", BoolMode::Or)
                .expect("count"),
            3
        );

        // Delete one of the surviving alpha docs; the count drops it too.
        let stats = st
            .delete(col("title").eq(lit("alpha two")))
            .expect("delete");
        assert_eq!(stats.matched(), 1);
        assert_eq!(
            st.count("title", "alpha -beta", BoolMode::Or)
                .expect("count after delete"),
            2
        );
    }
    /// Diagnostic, not a gate — run explicitly:
    /// `cargo test --lib supertable::query::fts::tests::diag_hidden_route_overhead -- --ignored --nocapture`
    /// Breaks the hidden-route per-query overhead (the post-drain
    /// `single_df1` gap: ~8µs pre-drain vs ~86µs post-drain at 1M on
    /// Azure) into route construction, the two prune walks, and the
    /// full query, on a LocalFs table small enough that kernel time
    /// is negligible and setup cost dominates.
    #[test]
    #[ignore = "diagnostic; run with -- --ignored --nocapture"]
    fn diag_hidden_route_overhead() {
        use std::{slice, time::Instant};

        use tempfile::TempDir;

        use crate::supertable::query::prune::PruneLeaf;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_storage(Arc::clone(&storage))
        .with_writer_pool(Arc::clone(&pool));
        let st = Supertable::create(options).expect("create");

        // Two commits; "uniqterm" planted once (df = 1).
        for c in 0..2u32 {
            let texts: Vec<String> = (0..300u32)
                .map(|i| match (c, i) {
                    (0, 7) => "uniqterm filler".to_string(),
                    _ => format!("common filler{c}x{i}"),
                })
                .collect();
            let titles = LargeStringArray::from(texts);
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(titles) as _]).expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        /// Iterations per timed section — enough for stable µs math.
        const ITERS: u32 = 2000;
        let time = |label: &str, mut f: Box<dyn FnMut() + '_>| {
            for _ in 0..50 {
                f();
            }
            let t = Instant::now();
            for _ in 0..ITERS {
                f();
            }
            println!("{label:>40}: {:>10.2?}/iter", t.elapsed() / ITERS);
        };

        let reader = st.reader();
        time(
            "PRE-DRAIN full bm25_search(single_df1)",
            Box::new(|| {
                let hits = reader
                    .bm25_search("title", "uniqterm", 10, BoolMode::Or, None)
                    .expect("query");
                assert_eq!(hits[0].num_rows(), 1);
            }),
        );

        st.drain_vectors_to_cells_sync().expect("drain");

        let reader = st.reader();
        time(
            "POST-DRAIN full bm25_search(single_df1)",
            Box::new(|| {
                let hits = reader
                    .bm25_search("title", "uniqterm", 10, BoolMode::Or, None)
                    .expect("query");
                assert_eq!(hits[0].num_rows(), 1);
            }),
        );
        // Decomposition through the REAL bridge (the test-local
        // `block_on` builds a runtime per call, a ~20µs artifact on
        // the route/prune rows below): async kernel alone, then the
        // id+score batch build — their gap to the full call is the
        // sync→async bridge + guard overhead.
        time(
            "POST-DRAIN kernel via real bridge",
            Box::new(|| {
                let hits = reader
                    .block_on(reader.bm25_search_async("title", "uniqterm", 10, BoolMode::Or))
                    .expect("query");
                assert_eq!(hits.len(), 1);
            }),
        );
        let kernel_hits = reader
            .block_on(reader.bm25_search_async("title", "uniqterm", 10, BoolMode::Or))
            .expect("query");
        time(
            "POST-DRAIN hits_id_score_batch alone",
            Box::new(|| {
                let batch = hits_id_score_batch(&reader, &kernel_hits).expect("batch");
                assert_eq!(batch.num_rows(), 1);
            }),
        );
        time(
            "hidden_text_route() alone",
            Box::new(|| {
                let route = block_on(reader.hidden_text_route())
                    .expect("route")
                    .expect("hidden epoch present");
                drop(route);
            }),
        );
        let route = block_on(reader.hidden_text_route())
            .expect("route")
            .expect("hidden epoch present");
        let leaf = PruneLeaf::TermPresence {
            column: "title".to_string(),
            terms: vec!["uniqterm".to_string()],
            mode: BoolMode::Or,
        };
        time(
            "text_shards_pruned()",
            Box::new(|| {
                let shards =
                    block_on(SupertableReader::text_shards_pruned(&route, &leaf)).expect("prune");
                assert_eq!(shards.len(), 1);
            }),
        );
        time(
            "undrained_tail_pruned()",
            Box::new(|| {
                let tail = block_on(reader.undrained_tail_pruned(
                    &route,
                    "title",
                    slice::from_ref(&"uniqterm".to_string()),
                    BoolMode::Or,
                ))
                .expect("tail prune");
                assert!(tail.is_empty(), "everything drained");
            }),
        );
    }
    /// SCRATCH: shared corpus for the warm-profile diags — a LocalFs
    /// supertable with `docs` ten-term docs appended across `commits`
    /// commits. Returns the tempdir (keep it alive) and the handle.
    fn build_ten_term_profile_table(docs: u32, commits: u32) -> (tempfile::TempDir, Supertable) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            DataType::LargeUtf8,
            false,
        )]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("pool"),
        );
        let dir = tempfile::TempDir::new_in("/mnt/scratch/tmp").expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "body".into(),
                    positions: false,
                }],
                vec![],
                Some(tok()),
            )
            .expect("options")
            .with_storage(storage)
            .with_writer_pool(pool),
        )
        .expect("create");

        let per = docs / commits;
        for c in 0..commits {
            let texts: Vec<String> = (0..per)
                .map(|i| {
                    let d = c * per + i;
                    format!(
                        "t0w{} t1x t2y{} t3z t4a{} t5b t6c t7d t8e t9f fill{d}",
                        d % 3,
                        d % 5,
                        d % 2
                    )
                })
                .collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeStringArray::from(texts)) as _],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
            println!("commit {}/{commits}", c + 1);
        }
        (dir, st)
    }

    /// SCRATCH (uncommitted): full-path profile target. LocalFs
    /// supertable, 1M docs, drained; loops warm ten-term OR through
    /// the public bm25_search so `perf` sees the whole hidden route.
    #[test]
    #[ignore = "scratch profiling target"]
    fn diag_supertable_ten_term_profile() {
        use std::time::Instant;

        const DOCS: u32 = 1_000_000;
        const COMMITS: u32 = 16;
        const WARM_ITERS: u32 = 2000;

        let (_dir, st) = build_ten_term_profile_table(DOCS, COMMITS);
        let t = Instant::now();
        st.drain_vectors_to_cells_sync().expect("drain");
        println!("drain done in {:?}", t.elapsed());

        let reader = st.reader();
        let query = "t0w0 t1x t2y0 t3z t4a0 t5b t6c t7d t8e t9f";
        // Warm up.
        for _ in 0..20 {
            reader
                .bm25_search("body", query, 10, BoolMode::Or, None)
                .expect("query");
        }
        println!("PROFILE_NOW pid={}", std::process::id());
        let t = Instant::now();
        for _ in 0..WARM_ITERS {
            reader
                .bm25_search("body", query, 10, BoolMode::Or, None)
                .expect("query");
        }
        println!("warm ten-term OR: {:?}/query", t.elapsed() / WARM_ITERS);
    }

    /// SCRATCH (uncommitted): post-compact warm profile. Same corpus
    /// as [`diag_supertable_ten_term_profile`], but after the drain it
    /// runs a full `optimize()` (compact + gc) and re-times the warm
    /// loop — reproducing the bench's post-compact flat ~60 ms/query
    /// tax locally if the mechanism is structural rather than store-
    /// or budget-dependent. Also probes the undrained tail directly:
    /// post-optimize it must be empty (the merged file inherits the
    /// oldest drained birth_version); a non-empty tail means wave 2
    /// re-walks the merged corpus on every query.
    #[test]
    #[ignore = "scratch profiling target"]
    fn diag_post_compact_warm_profile() {
        use std::{slice, time::Instant};

        use crate::config::OptimizeOptions;

        const DOCS: u32 = 1_000_000;
        const COMMITS: u32 = 16;
        const ITERS: usize = 50;

        let (_dir, st) = build_ten_term_profile_table(DOCS, COMMITS);
        let t = Instant::now();
        st.drain_vectors_to_cells_sync().expect("drain");
        println!("drain done in {:?}", t.elapsed());

        let query = "t0w0 t1x t2y0 t3z t4a0 t5b t6c t7d t8e t9f";
        let warm_window = |label: &str| {
            let reader = st.reader();
            for _ in 0..10 {
                reader
                    .bm25_search("body", query, 10, BoolMode::Or, None)
                    .expect("query");
            }
            let mut samples: Vec<u128> = (0..ITERS)
                .map(|_| {
                    let t = Instant::now();
                    reader
                        .bm25_search("body", query, 10, BoolMode::Or, None)
                        .expect("query");
                    t.elapsed().as_micros()
                })
                .collect();
            samples.sort_unstable();
            println!(
                "{label}: p50 {}µs p90 {}µs min {}µs max {}µs",
                samples[ITERS / 2],
                samples[ITERS * 9 / 10],
                samples[0],
                samples[ITERS - 1],
            );
        };

        warm_window("POST-DRAIN warm ten_term_or");

        // Mirror the bench lifecycle exactly: an undrained delta batch
        // lands between the drain and the optimize, so optimize's
        // inner drain extends the watermark right before compaction
        // stamps merged birth_versions.
        let delta_texts: Vec<String> = (0..1000u32)
            .map(|i| {
                let d = DOCS + i;
                format!(
                    "t0w{} t1x t2y{} t3z t4a{} t5b t6c t7d t8e t9f fill{d}",
                    d % 3,
                    d % 5,
                    d % 2
                )
            })
            .collect();
        let schema = st.schema();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(LargeStringArray::from(delta_texts)) as _],
        )
        .expect("delta batch");
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("delta append");
        w.commit().expect("delta commit");
        println!("delta committed (1000 docs, undrained)");

        let t = Instant::now();
        st.optimize(&OptimizeOptions::default()).expect("optimize");
        println!("optimize done in {:?}", t.elapsed());
        println!(
            "post-optimize user superfiles: {}",
            st.reader().manifest().get_all_superfiles().len()
        );

        // Watermark probe: the merged user file must stay on the
        // drained side of the two-wave split.
        let reader = st.reader();
        let route = block_on(reader.hidden_text_route())
            .expect("route")
            .expect("hidden epoch present");
        let tail = block_on(reader.undrained_tail_pruned(
            &route,
            "body",
            slice::from_ref(&"t1x".to_string()),
            BoolMode::Or,
        ))
        .expect("tail prune");
        println!(
            "post-optimize undrained tail: {} entries, birth versions {:?}",
            tail.len(),
            tail.iter().map(|e| e.birth_version).collect::<Vec<_>>(),
        );

        println!("PROFILE_NOW pid={}", std::process::id());
        warm_window("POST-COMPACT warm ten_term_or");
        warm_window("POST-COMPACT warm ten_term_or (2nd window)");
    }

    /// Aggressive compaction settings for the placement regression
    /// test below: 1 MB target + 1% fill floor force a merge job for
    /// any handful of small commits.
    const PLACEMENT_TEST_COMPACTION: CompactionSettings = CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        max_memory_mb: 64,
        stale_seal_timeout_ms: DEFAULT_STALE_SEAL_TIMEOUT_MS,
    };

    /// Post-compact placement regression: compaction merges several
    /// commits into ONE user superfile whose stable-id span is gapped
    /// (Snowflake ids jump between commits), knocking scalar placement
    /// off the `id_min + local` arithmetic path. Guards two fixes:
    /// the bare projection must skip the placement pass entirely
    /// (vector-parity fast path — before it, every projection-`None`
    /// query decoded the merged file's whole `_id` column, the flat
    /// +60 ms/query post-compact tax at 1M), and scalar projections
    /// must locate rows through the sorted-`_id` bisection — both
    /// checked against ground truth (the row text names its doc).
    #[test]
    fn post_compact_gapped_placement_and_bare_projection() {
        use arrow_array::Float32Array;
        use tempfile::TempDir;

        use crate::config::OptimizeOptions;

        const COMMITS: u32 = 4;
        const PER: u32 = 300;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            DataType::LargeUtf8,
            false,
        )]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new_in("/mnt/scratch/tmp").expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "body".into(),
                    positions: false,
                }],
                vec![],
                Some(tok()),
            )
            .expect("options")
            .with_storage(storage)
            .with_writer_pool(pool),
        )
        .expect("create");

        for c in 0..COMMITS {
            let texts: Vec<String> = (0..PER)
                .map(|i| {
                    let d = c * PER + i;
                    format!("uniq{d} common filler")
                })
                .collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeStringArray::from(texts)) as _],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");
        st.optimize(&OptimizeOptions::compact(PLACEMENT_TEST_COMPACTION))
            .expect("optimize");

        let reader = st.reader();
        assert!(reader.hidden_epoch_has_text(), "text shards must exist");
        let user_files = reader.manifest().get_all_superfiles().len();
        assert!(
            user_files < COMMITS as usize,
            "compaction must have merged user files (got {user_files})"
        );

        // Scalar projection: placement must locate each hit's row in
        // the merged gapped file — the body text names its doc, so a
        // mis-placement returns another row's text.
        for d in [0u32, 1, PER - 1, PER, 2 * PER + 7, COMMITS * PER - 1] {
            let term = format!("uniq{d}");
            let rows = st
                .bm25_search(
                    "body",
                    &term,
                    3,
                    BoolMode::Or,
                    Some(&["_id", "body", "score"]),
                )
                .expect("scalar search");
            assert_eq!(rows[0].num_rows(), 1, "term {term} matches one doc");
            let body = rows[0]
                .column(1)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("body col");
            assert_eq!(body.value(0), format!("uniq{d} common filler"));

            // Bare projection must agree with the scalar path on `_id`
            // and score (it takes the stamp-only fast path).
            let bare = st
                .bm25_search("body", &term, 3, BoolMode::Or, None)
                .expect("bare search");
            assert_eq!(bare[0].num_rows(), 1);
            let id_of = |b: &RecordBatch, col: usize| {
                b.column(col)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("_id col")
                    .value(0)
            };
            assert_eq!(id_of(&bare[0], 0), id_of(&rows[0], 0), "term {term}");
            let score_of = |b: &RecordBatch, col: usize| {
                b.column(col)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("score col")
                    .value(0)
            };
            assert_eq!(score_of(&bare[0], 1), score_of(&rows[0], 2));
        }

        // Multi-hit scalar projection: every returned row's text must
        // pair with its own doc (exercises multi-id bisection).
        let rows = st
            .bm25_search(
                "body",
                "common",
                10,
                BoolMode::Or,
                Some(&["_id", "body", "score"]),
            )
            .expect("multi search");
        assert_eq!(rows[0].num_rows(), 10);
        let body = rows[0]
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("body col");
        for r in 0..10 {
            assert!(
                body.value(r).starts_with("uniq") && body.value(r).ends_with("common filler"),
                "row {r} text {:?} not a valid doc",
                body.value(r)
            );
        }

        // token_match and exact_match must honor the same placement +
        // fast-path contracts (exact_match historically skipped the
        // placement pass entirely, mis-resolving hidden hits under a
        // scalar projection).
        let expect_body = |rows: &Vec<RecordBatch>, label: &str| {
            assert_eq!(rows[0].num_rows(), 1, "{label} matches one doc");
            let body = rows[0]
                .column(1)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("body col");
            assert_eq!(body.value(0), "uniq7 common filler", "{label}");
        };
        let rows = st
            .token_match(
                "body",
                "uniq7",
                BoolMode::Or,
                Some(&["_id", "body", "score"]),
            )
            .expect("token_match scalar");
        expect_body(&rows, "token_match");
        let rows = st
            .exact_match(
                "body",
                "uniq7 common filler",
                Some(&["_id", "body", "score"]),
            )
            .expect("exact_match scalar");
        expect_body(&rows, "exact_match");
        let bare = st
            .exact_match("body", "uniq7 common filler", None)
            .expect("exact_match bare");
        assert_eq!(bare[0].num_rows(), 1);
    }

    /// End-to-end bigram flow through the drain: the merged text
    /// shards must carry drain-generated adjacent-pair terms (members
    /// above the df floor), and post-drain phrase queries — which
    /// rewrite onto those pair postings — must return the planted
    /// ground truth: exact phrase counts and the highest-phrase-tf doc
    /// on top.
    #[test]
    fn drained_shards_carry_bigrams_and_phrase_results_match() {
        use tempfile::TempDir;

        /// Docs in the corpus; members reach df ≥ the default 1024
        /// bigram floor.
        const DOCS: u32 = 3000;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            DataType::LargeUtf8,
            false,
        )]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new_in("/mnt/scratch/tmp").expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "body".into(),
                    positions: true,
                }],
                vec![],
                Some(tok()),
            )
            .expect("options")
            .with_storage(storage)
            .with_writer_pool(pool),
        )
        .expect("create");

        // Three commits; adjacency planted in every 2nd doc, members
        // present-but-separated in every 3rd, one doc with phrase
        // tf = 3 (the top hit by contract), filler for dl variance.
        let mut expected_matches: u64 = 0;
        for c in 0..3u32 {
            let texts: Vec<String> = (0..DOCS / 3)
                .map(|i| {
                    let d = c * (DOCS / 3) + i;
                    let mut t = String::new();
                    if d == 7 {
                        t.push_str("quick brown quick brown quick brown ");
                    } else if d.is_multiple_of(2) {
                        t.push_str("quick brown ");
                    } else if d.is_multiple_of(3) {
                        t.push_str("quick sep brown ");
                    }
                    t.push_str(&format!("fill{d} tail"));
                    t
                })
                .collect();
            expected_matches += texts.iter().filter(|t| t.contains("quick brown")).count() as u64;
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeStringArray::from(texts)) as _],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");

        // (a) The text shards carry the pair term with the exact
        // adjacency df.
        let vit = st
            .reader()
            .vector_index_table()
            .cloned()
            .expect("hidden sibling");
        let hidden_reader = vit.pinned_reader();
        let hidden_store = vit.inner().options.store.clone();
        let mut bigram_df: u64 = 0;
        for entry in hidden_reader.manifest().get_all_superfiles() {
            if entry.fts_summary.is_empty() || !entry.vector_summary.is_empty() {
                continue;
            }
            let reader = hidden_store.reader(&entry.uri).expect("shard reader");
            let fts = reader.fts().expect("text shard has FTS");
            bigram_df += block_on(fts.term_df("body", "quick\u{1f}brown")).expect("df");
        }
        assert_eq!(
            bigram_df, expected_matches,
            "drain must emit the pair term with adjacency df"
        );

        // (b) Post-drain phrase semantics against planted truth.
        let phrase = "\"quick brown\"";
        let count = st.count("body", phrase, BoolMode::Or).expect("count");
        assert_eq!(count, expected_matches);
        let hits = st
            .reader()
            .bm25_search("body", phrase, 3, BoolMode::Or, None)
            .expect("phrase search");
        assert_eq!(hits[0].num_rows(), 3);
        // Highest phrase tf wins under the contract; doc 7 planted 3
        // occurrences. Verify by materializing its text.
        let top = st
            .bm25_search(
                "body",
                phrase,
                1,
                BoolMode::Or,
                Some(&["_id", "body", "score"]),
            )
            .expect("top hit");
        let body = top[0]
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("body col");
        assert!(
            body.value(0)
                .starts_with("quick brown quick brown quick brown"),
            "top hit must be the tf=3 doc, got {:?}",
            body.value(0)
        );
    }

    /// SCRATCH (uncommitted): merged-shard BM25 statistics audit.
    /// Near-universal terms make idf ~= gap/N, so any df/n_docs/avgdl
    /// accounting drift in the drain shows up as a ratio shift
    /// between the user files and the drained shard.
    #[test]
    #[ignore = "scratch diagnostic"]
    fn diag_drained_shard_stats_audit() {
        use tempfile::TempDir;

        use crate::supertable::storage::LocalFsStorageProvider;

        const FILES: u32 = 4;
        const PER: u32 = 1000;
        /// Docs per file missing the near-universal term.
        const GAP: u32 = 20;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            DataType::LargeUtf8,
            false,
        )]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new_in("/mnt/scratch/tmp").expect("tempdir");
        let storage: Arc<dyn crate::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "body".into(),
                    positions: true,
                }],
                vec![],
                Some(tok()),
            )
            .expect("options")
            .with_storage(storage)
            .with_writer_pool(pool),
        )
        .expect("create");

        for f in 0..FILES {
            let texts: Vec<String> = (0..PER)
                .map(|i| {
                    let d = f * PER + i;
                    if i < GAP {
                        format!("filler{d} pad pad")
                    } else {
                        format!("alpha beta filler{d}")
                    }
                })
                .collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeStringArray::from(texts)) as _],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");

        // Expected: per-file df = PER-GAP, N_f = PER; global df =
        // FILES*(PER-GAP), N = FILES*PER — identical ratios.
        let reader = st.reader();
        let hidden = reader.vector_index_table().expect("hidden").reader();
        let shard = hidden
            .manifest()
            .superfiles
            .iter()
            .find(|e| !e.fts_summary.is_empty() && e.vector_summary.is_empty())
            .expect("text shard")
            .clone();
        println!("shard n_docs={} (expect {})", shard.n_docs, FILES * PER);
        // Pull kernel-visible stats: score a 1-doc-tf query and
        // back-derive. Simpler: bm25 scores of \"alpha beta\" phrase on
        // user path vs hidden path must MATCH (same corpus stats).
        let user_rows = reader
            .bm25_search("body", "\"alpha beta\"", 5, BoolMode::Or, None)
            .expect("hidden-route query");
        let batches = &user_rows;
        let score_col = batches[0]
            .column(batches[0].num_columns() - 1)
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .expect("score col");
        println!("post-drain top score = {}", score_col.value(0));
    }
    /// SCRATCH (uncommitted): does the per-file idf scope mis-rank
    /// phrase results pre-drain? Files with skewed member df score in
    /// incomparable idf domains; the drained shard scores globally.
    /// If the pre-drain top-k differs from post-drain on identical
    /// data, the post-drain result is the textbook-correct one and
    /// the phrase "regression" partially prices a correctness fix.
    #[test]
    #[ignore = "scratch diagnostic"]
    fn diag_phrase_idf_scope_ranking() {
        use tempfile::TempDir;

        use crate::supertable::storage::LocalFsStorageProvider;

        const FILES: u32 = 4;
        const PER: u32 = 2000;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            DataType::LargeUtf8,
            false,
        )]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new_in("/mnt/scratch/tmp").expect("tempdir");
        let storage: Arc<dyn crate::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "body".into(),
                    positions: true,
                }],
                vec![],
                Some(tok()),
            )
            .expect("options")
            .with_storage(storage)
            .with_writer_pool(pool),
        )
        .expect("create");

        // Files 0..2: "alpha beta" ubiquitous (per-file idf tiny).
        // File 3: "alpha beta" rare within the file (per-file idf
        // large) — its docs get inflated scores under per-file idf,
        // beating docs in files 0..2 whose global evidence is equal.
        for f in 0..FILES {
            let texts: Vec<String> = (0..PER)
                .map(|i| {
                    let d = f * PER + i;
                    let rare_file = f == FILES - 1;
                    if rare_file && i >= PER / 20 {
                        format!("gamma delta fill{d}")
                    } else {
                        // tf varies so scores are not exact ties.
                        format!("{}fill{d}", "alpha beta ".repeat(((d % 3) + 1) as usize))
                    }
                })
                .collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeStringArray::from(texts)) as _],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let pre = st
            .reader()
            .bm25_hits("body", "\"alpha beta\"", 10, BoolMode::Or)
            .expect("pre-drain hits");
        st.drain_vectors_to_cells_sync().expect("drain");
        let post = st
            .reader()
            .bm25_hits("body", "\"alpha beta\"", 10, BoolMode::Or)
            .expect("post-drain hits");
        println!(
            "pre-drain  top10: {:?}",
            pre.iter()
                .map(|h| (h.superfile, h.local_doc_id, h.score))
                .collect::<Vec<_>>()
        );
        println!(
            "post-drain top10: {:?}",
            post.iter()
                .map(|h| (h.superfile, h.local_doc_id, h.score))
                .collect::<Vec<_>>()
        );
    }
    /// SCRATCH (uncommitted): reproduce the post-drain SQL
    /// point-aggregate hang seen in the supertable_sql bench
    /// ("aggregate shapes over a token_match candidate set" battery,
    /// 2026-07-22: all threads futex-parked, main in nanosleep).
    #[test]
    #[ignore = "scratch diagnostic"]
    fn diag_sql_point_aggregate_after_drain() {
        use std::time::Duration;

        use tempfile::TempDir;

        use crate::supertable::storage::LocalFsStorageProvider;

        const DOCS: u32 = 20_000;

        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("key", DataType::LargeUtf8, false),
            Field::new("rating", DataType::Int64, false),
        ]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new_in("/mnt/scratch/tmp").expect("tempdir");
        let storage: Arc<dyn crate::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "title".into(),
                    positions: false,
                }],
                vec![],
                Some(tok()),
            )
            .expect("options")
            .with_storage(storage)
            .with_writer_pool(pool),
        )
        .expect("create");

        for c in 0..4u32 {
            let n = DOCS / 4;
            let titles: Vec<String> = (0..n)
                .map(|i| format!("word{} text", (c * n + i) % 97))
                .collect();
            let keys: Vec<String> = (0..n).map(|i| format!("key{:06}", c * n + i)).collect();
            let ratings: Vec<i64> = (0..n).map(|i| (i % 100) as i64).collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(LargeStringArray::from(titles)) as _,
                    Arc::new(LargeStringArray::from(keys)) as _,
                    Arc::new(arrow_array::Int64Array::from(ratings)) as _,
                ],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");

        // Watchdog: the hang parks everything, so a plain test would
        // sit forever — abort loudly instead.
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watchdog = {
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                for _ in 0..120 {
                    std::thread::sleep(Duration::from_millis(500));
                    if done.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                }
                eprintln!("DEADLOCK REPRODUCED: query_sql hung 60s");
                std::process::abort();
            })
        };
        for q in [
            "SELECT COUNT(*) AS a FROM supertable WHERE key = 'key000042'",
            "SELECT SUM(rating) AS a FROM supertable WHERE key = 'key000042'",
            "SELECT MAX(rating) AS a FROM supertable WHERE key = 'key000042'",
        ] {
            let batches = st.reader().query_sql(q).expect("query_sql");
            assert!(!batches.is_empty());
            eprintln!("ok: {q}");
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        watchdog.join().expect("watchdog");
    }
}

#[cfg(test)]
mod cold_read_probe {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;

    use super::BoolMode;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::builder::FtsConfig,
        supertable::{Supertable, SupertableOptions},
        test_helpers::default_tokenizer as tok,
    };

    /// PROBE: does a df=1 bare-term search on a freshly OPENED consumer
    /// (new process-equivalent: new handle, new reader cache) read any
    /// bytes from the storage provider?
    #[test]
    fn df1_cold_search_reads_storage_probe() {
        const COMMITS: u32 = 2;
        const PER: u32 = 300;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            DataType::LargeUtf8,
            false,
        )]));
        let dir = TempDir::new_in("/mnt/scratch/tmp").expect("tempdir");
        let provider = Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let storage: Arc<dyn StorageProvider> = provider.clone();
        let mk_options = || {
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "body".into(),
                    positions: false,
                }],
                vec![],
                Some(tok()),
            )
            .expect("options")
            .with_storage(storage.clone())
        };
        let st = Supertable::create(mk_options()).expect("create");
        for c in 0..COMMITS {
            let texts: Vec<String> = (0..PER)
                .map(|i| format!("uniq{} common filler", c * PER + i))
                .collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeStringArray::from(texts)) as _],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        drop(st);

        let consumer = Supertable::open(mk_options()).expect("reopen");
        let before = provider.usage_meter().snapshot();
        let rows = consumer
            .bm25_search("body", "uniq0", 10, BoolMode::Or, None)
            .expect("df1 search");
        let io = provider.usage_meter().snapshot().since(&before);
        let hits: usize = rows.iter().map(|b| b.num_rows()).sum();
        eprintln!("PROBE df1: {} hit(s), {} GET during search", hits, io.get_count);
        assert_eq!(hits, 1, "df1 term matches exactly one doc");
    }
}
