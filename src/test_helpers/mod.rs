// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Convenience builders for test fixtures.
//!
//! Three test contexts share these helpers:
//!
//! - **Unit tests** (`#[cfg(test)] mod tests` inside `src/`)
//!   reach this module via `crate::test_helpers::...` —
//!   `cfg(test)` always enables it.
//! - **Integration tests** (`tests/...`) reach it via
//!   `infino::test_helpers::...` — the `test-helpers` Cargo
//!   feature is auto-enabled by the `dev-dependencies` self-
//!   reference in `Cargo.toml`.
//! - **Benches** (`benches/...`) reach it the same way.
//!
//! Scope: small atomic idioms that repeat across dozens of
//! test / bench fixtures (Decimal128 id construction, default
//! tokenizer, default vector config). Higher-level "build a
//! test corpus" / "build a full test superfile" stays in the
//! test files themselves — those vary too much per scenario
//! to share usefully.
//!
//! [`brute_force_bm25`] is the textbook BM25 reference impl
//! used as the FTS correctness oracle.

pub mod brute_force_bm25;
pub mod cas_conformance;
pub mod fault_storage;

/// Observability probe for the vector query path's EFFECTIVE served
/// shortlist budget — the regression guard for the serve-the-law scope
/// bug (#520 review): recall-floor tests are insensitive to a re-shadowed
/// `options` (the `rm=256` constant yields MORE survivors, so recall
/// stays equal-or-better and only latency regresses), but the served
/// budget is not. The pooled warm arm records the limit and cell floor
/// it passes to the global shortlist; tests read them after a query.
/// Observability probes for the hidden vector path's two routing stages
/// (#515 diagnosis): the 1-bit admit window (which cells get exact fine
/// rescoring) and the exact-fine cell ranking selection serves from.
/// Append logs — drain between queries; race-tolerant under test
/// parallelism (a concurrent test adds entries, never removes yours).
pub mod admit_trace {
    use std::sync::Mutex;

    static ADMITS: Mutex<Vec<Vec<u32>>> = Mutex::new(Vec::new());
    static FINES: Mutex<Vec<Vec<(u32, f32)>>> = Mutex::new(Vec::new());

    /// The admit shortlist (global cell ids) chosen by the 1-bit window.
    pub fn record_admit(cells: Vec<u32>) {
        ADMITS.lock().expect("admit probe lock").push(cells);
    }

    /// The exact-fine cell ranking (cell id, best exact fine score).
    pub fn record_fine(ranked: Vec<(u32, f32)>) {
        FINES.lock().expect("fine probe lock").push(ranked);
    }

    /// Drain both logs recorded since the last drain.
    #[allow(clippy::type_complexity)]
    pub fn drain() -> (Vec<Vec<u32>>, Vec<Vec<(u32, f32)>>) {
        (
            std::mem::take(&mut *ADMITS.lock().expect("admit probe lock")),
            std::mem::take(&mut *FINES.lock().expect("fine probe lock")),
        )
    }
}

pub mod served_shortlist_probe {
    use std::sync::Mutex;

    static RECORDS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

    /// Called by the query path (test-helpers builds only).
    pub fn record(limit: usize, cell_floor: usize) {
        RECORDS
            .lock()
            .expect("shortlist probe lock")
            .push((limit, cell_floor));
    }

    /// Drain every `(limit, cell_floor)` recorded since the last drain.
    /// Append-log semantics keep the probe race-tolerant under the test
    /// binary's parallelism: a concurrent test adds tuples but can never
    /// remove this test's — assert with `contains`, not equality.
    pub fn drain() -> Vec<(usize, usize)> {
        std::mem::take(&mut *RECORDS.lock().expect("shortlist probe lock"))
    }
}

/// Counterfactual plane-truncation probe for the 1-bit LUT cell scan.
/// Enabled by `INFINO_DIAG_PLANE_TRUNC=1` in the environment (read
/// once) — a diagnostics switch in the `INFINO_DIAG_*` family, not an
/// engine-behavior knob: search results are identical either way, the
/// probe only measures what a checkpointed early-exit kernel WOULD
/// have skipped against the cell's evolving shortlist floor. Counters
/// are cumulative and process-global; a JSON snapshot line prefixed
/// `PLANE_TRUNC` goes to stderr every [`EMIT_EVERY_CELLS`] recorded
/// cell scans, so the last line of a run carries the totals.
pub mod plane_trunc_probe {
    use std::{
        env,
        sync::{
            OnceLock,
            atomic::{AtomicU64, Ordering},
        },
    };

    /// Bound variants measured side by side: index 0 is the exact
    /// suffix-max bound; indexes 1.. are the probabilistic z-sigma
    /// family (the scan side defines the z ladder).
    pub const VARIANTS: usize = 5;
    /// Emit a cumulative stderr snapshot every this many cell scans.
    const EMIT_EVERY_CELLS: u64 = 64;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    static ENFORCE: OnceLock<Option<usize>> = OnceLock::new();
    static CELLS: AtomicU64 = AtomicU64::new(0);
    static CLUSTERS: AtomicU64 = AtomicU64::new(0);
    static PLANE_UNITS: AtomicU64 = AtomicU64::new(0);
    static UNFLOORED_UNITS: AtomicU64 = AtomicU64::new(0);
    static SKIPPED: [AtomicU64; VARIANTS] = new_counters();
    static CLUSTERS_SKIPPED: [AtomicU64; VARIANTS] = new_counters();
    static UNSAFE_LANES: [AtomicU64; VARIANTS] = new_counters();
    static ENFORCED_ROWS: AtomicU64 = AtomicU64::new(0);

    const fn new_counters() -> [AtomicU64; VARIANTS] {
        [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ]
    }

    /// Counters-only switch for the PRODUCTION prune path
    /// (`INFINO_DIAG_PLANE_PROD=1`): unlike [`enabled`] it does not force
    /// the serial scan arm and runs no shadow, so it observes exactly what
    /// shipped code does — how often a bar exists, and how many rows the
    /// kernels actually skipped.
    pub fn prod_enabled() -> bool {
        static PROD: OnceLock<bool> = OnceLock::new();
        *PROD.get_or_init(|| env::var_os("INFINO_DIAG_PLANE_PROD").is_some_and(|v| v == "1"))
    }

    static PROD_CLUSTERS: AtomicU64 = AtomicU64::new(0);
    static PROD_WITH_BAR: AtomicU64 = AtomicU64::new(0);
    static PROD_ROWS: AtomicU64 = AtomicU64::new(0);
    static PROD_ROWS_SKIPPED: AtomicU64 = AtomicU64::new(0);

    /// One scanned cluster on the production path: `rows` in it,
    /// `pushed` rows that survived to the accumulator, and whether a
    /// pruning bar existed at all.
    pub fn add_prod_cluster(rows: u64, pushed: u64, had_bar: bool) {
        PROD_CLUSTERS.fetch_add(1, Ordering::Relaxed);
        if had_bar {
            PROD_WITH_BAR.fetch_add(1, Ordering::Relaxed);
        }
        PROD_ROWS.fetch_add(rows, Ordering::Relaxed);
        PROD_ROWS_SKIPPED.fetch_add(rows.saturating_sub(pushed), Ordering::Relaxed);
    }

    static PROD_SCANS: AtomicU64 = AtomicU64::new(0);
    static PROD_SCANS_WITH_BAR_OBJ: AtomicU64 = AtomicU64::new(0);
    static PROD_MAX_ACC: AtomicU64 = AtomicU64::new(0);
    static PROD_LIMIT: AtomicU64 = AtomicU64::new(0);

    static PROD_FLOOR_SEEN: AtomicU64 = AtomicU64::new(0);
    static PROD_GAIN_ZERO: AtomicU64 = AtomicU64::new(0);
    static PROD_BAR_NONE_OTHER: AtomicU64 = AtomicU64::new(0);

    /// Why a cluster ended up without a prune bar: no floor published yet,
    /// a zero residual gain (kappa * max norm), or the LUT declining.
    pub fn add_prod_bar_reason(floor_seen: bool, gain_zero: bool, other: bool) {
        if floor_seen {
            PROD_FLOOR_SEEN.fetch_add(1, Ordering::Relaxed);
        }
        if gain_zero {
            PROD_GAIN_ZERO.fetch_add(1, Ordering::Relaxed);
        }
        if other {
            PROD_BAR_NONE_OTHER.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One `scan_shortlist` call: whether it was handed a shared bar at
    /// all, the shortlist depth it was asked for, and how many rows its
    /// accumulator actually reached. Distinguishes "the bar never got
    /// published" from "this code path never had a bar to begin with".
    pub fn add_prod_scan(had_bar_obj: bool, limit: u64, acc_rows: u64) {
        PROD_SCANS.fetch_add(1, Ordering::Relaxed);
        if had_bar_obj {
            PROD_SCANS_WITH_BAR_OBJ.fetch_add(1, Ordering::Relaxed);
        }
        PROD_MAX_ACC.fetch_max(acc_rows, Ordering::Relaxed);
        PROD_LIMIT.store(limit, Ordering::Relaxed);
    }

    static PROD_PLANES: AtomicU64 = AtomicU64::new(0);
    static PROD_PLANES_POSSIBLE: AtomicU64 = AtomicU64::new(0);

    /// Plane-scan units the kernels actually consumed vs what a full scan
    /// would have cost — the only measure that says whether early exits
    /// are saving BYTES, not just skipping row pushes.
    pub fn add_prod_planes(scanned: u64, possible: u64) {
        PROD_PLANES.fetch_add(scanned, Ordering::Relaxed);
        PROD_PLANES_POSSIBLE.fetch_add(possible, Ordering::Relaxed);
    }

    /// Dump the production counters (called per cell scan; cheap).
    pub fn emit_prod() {
        let cells = CELLS.fetch_add(1, Ordering::Relaxed) + 1;
        if !cells.is_multiple_of(EMIT_EVERY_CELLS) {
            return;
        }
        eprintln!(
            "PLANE_PROD {{\"clusters\":{},\"with_bar\":{},\"rows\":{},\"rows_skipped\":{},\"scans\":{},\"scans_with_bar_obj\":{},\"max_acc\":{},\"limit\":{},\"floor_seen\":{},\"gain_zero\":{},\"bar_none_other\":{},\"planes\":{},\"planes_possible\":{}}}",
            PROD_CLUSTERS.load(Ordering::Relaxed),
            PROD_WITH_BAR.load(Ordering::Relaxed),
            PROD_ROWS.load(Ordering::Relaxed),
            PROD_ROWS_SKIPPED.load(Ordering::Relaxed),
            PROD_SCANS.load(Ordering::Relaxed),
            PROD_SCANS_WITH_BAR_OBJ.load(Ordering::Relaxed),
            PROD_MAX_ACC.load(Ordering::Relaxed),
            PROD_LIMIT.load(Ordering::Relaxed),
            PROD_FLOOR_SEEN.load(Ordering::Relaxed),
            PROD_GAIN_ZERO.load(Ordering::Relaxed),
            PROD_BAR_NONE_OTHER.load(Ordering::Relaxed),
            PROD_PLANES.load(Ordering::Relaxed),
            PROD_PLANES_POSSIBLE.load(Ordering::Relaxed),
        );
    }

    pub fn enabled() -> bool {
        *ENABLED.get_or_init(|| env::var_os("INFINO_DIAG_PLANE_TRUNC").is_some_and(|v| v == "1"))
    }

    /// Which bound variant the scan should ACT on, from
    /// `INFINO_DIAG_PLANE_TRUNC_ENFORCE=<variant index>` — rows in blocks
    /// that variant declares dead are dropped from the shortlist, exactly
    /// as a checkpointed early-exit kernel would drop them. This is how
    /// the recall cost of a probabilistic bound gets measured before any
    /// kernel is written: the arithmetic is the shadow's, the effect on
    /// results is the real one. Variant 0 (the exact bound) must leave
    /// recall untouched by construction — the built-in control.
    pub fn enforce_variant() -> Option<usize> {
        *ENFORCE.get_or_init(|| {
            env::var_os("INFINO_DIAG_PLANE_TRUNC_ENFORCE")
                .and_then(|v| v.to_str().and_then(|s| s.parse::<usize>().ok()))
                .filter(|v| *v < VARIANTS)
        })
    }

    /// One shadowed cluster: `units` = plane-scan units the real kernel
    /// spent (blocks x planes), `skipped[v]` = units variant `v` would
    /// have saved, `fully[v]` = variant `v` killed every block before
    /// its first plane (the whole cluster was skippable).
    pub fn add_cluster(
        units: u64,
        skipped: &[u64; VARIANTS],
        fully: &[bool; VARIANTS],
        unsafe_lanes: &[u64; VARIANTS],
    ) {
        CLUSTERS.fetch_add(1, Ordering::Relaxed);
        PLANE_UNITS.fetch_add(units, Ordering::Relaxed);
        for v in 0..VARIANTS {
            SKIPPED[v].fetch_add(skipped[v], Ordering::Relaxed);
            if fully[v] {
                CLUSTERS_SKIPPED[v].fetch_add(1, Ordering::Relaxed);
            }
            UNSAFE_LANES[v].fetch_add(unsafe_lanes[v], Ordering::Relaxed);
        }
    }

    /// Plane units scanned before the shortlist floor existed (no bound
    /// can prune there); kept separate so skip fractions stay honest.
    pub fn add_unfloored(units: u64) {
        UNFLOORED_UNITS.fetch_add(units, Ordering::Relaxed);
    }

    /// Rows actually withheld from a shortlist under
    /// [`Self::enforce_variant`] — the enforced counterpart of the
    /// counterfactual skip counters.
    pub fn add_enforced_rows(rows: u64) {
        ENFORCED_ROWS.fetch_add(rows, Ordering::Relaxed);
    }

    /// Mark one cell scan finished; periodically emit the totals.
    pub fn cell_done() {
        let cells = CELLS.fetch_add(1, Ordering::Relaxed) + 1;
        if cells % EMIT_EVERY_CELLS == 0 {
            emit();
        }
    }

    fn emit() {
        let fmt_counts = |counters: &[AtomicU64; VARIANTS]| {
            let each: Vec<String> = counters
                .iter()
                .map(|c| c.load(Ordering::Relaxed).to_string())
                .collect();
            format!("[{}]", each.join(","))
        };
        eprintln!(
            "PLANE_TRUNC {{\"cells\":{},\"clusters\":{},\"plane_units\":{},\"unfloored_units\":{},\"skipped\":{},\"clusters_skipped\":{},\"unsafe_lanes\":{},\"enforce\":{},\"enforced_rows\":{}}}",
            CELLS.load(Ordering::Relaxed),
            CLUSTERS.load(Ordering::Relaxed),
            PLANE_UNITS.load(Ordering::Relaxed),
            UNFLOORED_UNITS.load(Ordering::Relaxed),
            fmt_counts(&SKIPPED),
            fmt_counts(&CLUSTERS_SKIPPED),
            fmt_counts(&UNSAFE_LANES),
            enforce_variant().map_or(-1i64, |v| v as i64),
            ENFORCED_ROWS.load(Ordering::Relaxed),
        );
    }
}

/// Test-only override of the adaptive-stopping band floor
/// (`STOP_BAND_MIN_ROWS`): the banded rerank only engages naturally at
/// 10M-scale shortlists, so unit-scale tests lower the floor to walk the
/// band loop on a small fixture. Process-global — tests that set it must
/// not assume exclusivity (assert on their own table's results, never on
/// the counter of another test's query).
pub mod stop_band_floor_override {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `0` = no override (the compiled constant serves).
    static ROWS: AtomicUsize = AtomicUsize::new(0);

    pub fn set(rows: usize) {
        ROWS.store(rows, Ordering::Relaxed);
    }

    pub fn clear() {
        ROWS.store(0, Ordering::Relaxed);
    }

    /// Read by the query path (test-helpers builds only).
    pub fn get() -> Option<usize> {
        match ROWS.load(Ordering::Relaxed) {
            0 => None,
            n => Some(n),
        }
    }
}

use std::{collections::HashSet, path::Path, sync::Arc};

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use rayon::ThreadPoolBuilder;

use crate::{
    storage::StorageProvider,
    superfile::{
        builder::{FtsConfig, VectorConfig},
        fts::tokenize::{AsciiLowerTokenizer, Tokenizer},
        vector::{distance::Metric, rerank_codec::RerankCodec},
    },
    supertable::{
        SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
    },
};

/// 1 GiB disk-cache budget for tests.
const TEST_DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams for the test disk cache.
const TEST_COLD_FETCH_STREAMS: usize = 4;
/// Cold-fetch range chunk (1 MiB) for the test disk cache.
const TEST_COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;

/// Build a `DiskCacheStore` with the standard test config: 1 GiB budget,
/// hybrid-with-prefetch cold fetch, mmap sweep timers disabled, LRU eviction,
/// CRC-on-open, and a no-op pin set (pinning is a perf optimization, not a
/// correctness requirement — an `Arc<SuperfileReader>` keeps the mmap alive
/// past eviction). Shared by the storage / query / disk-cache tests.
pub fn default_disk_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: TEST_DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: TEST_COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: TEST_COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("test disk cache")
}

/// A `DiskCacheStore` in `LazyForegroundWithBackgroundFill`: the foreground
/// query reads through an `open_lazy` `StorageRangeSource`, so the superfile's
/// bytes stay non-resident and every cold read is an object-store GET. That is
/// the path the connection budget gates. `default_disk_cache`
/// (`HybridWithPrefetch`) instead collects the cold responses into a resident
/// in-memory reader, which reads warm and reserves nothing. Same budget,
/// timers, and eviction otherwise.
pub fn lazy_foreground_disk_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: TEST_DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
        cold_fetch_streams: TEST_COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: TEST_COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("test lazy-foreground disk cache")
}

/// Build a `Decimal128Array(38, 0)` from `u64` ids.
///
/// Centralizes the verbose three-step construction that
/// every test fixture reinvents:
///
/// ```ignore
/// Decimal128Array::from(ids.into_iter().map(|v| v as i128).collect::<Vec<_>>())
///     .with_precision_and_scale(38, 0)
///     .expect("decimal128")
/// ```
pub fn decimal128_ids<I: IntoIterator<Item = u64>>(ids: I) -> Decimal128Array {
    Decimal128Array::from(ids.into_iter().map(|v| v as i128).collect::<Vec<_>>())
        .with_precision_and_scale(38, 0)
        .expect("Decimal128(38, 0) is a valid precision/scale pair")
}

/// `Field` for the primary-key id column — `Decimal128(38, 0)`,
/// non-nullable. Caller supplies the column name (typically
/// `"_id"` at the supertable layer or `"doc_id"` in lower-level
/// superfile fixtures).
pub fn decimal128_id_field(name: &str) -> Field {
    Field::new(name, DataType::Decimal128(38, 0), false)
}

/// The default tokenizer used in tests + benches:
/// `AsciiLowerTokenizer` wrapped in `Arc<dyn Tokenizer>`.
///
/// Callers passing this into `BuilderOptions::new` wrap in
/// `Some(...)` at the call site:
///
/// ```ignore
/// BuilderOptions::new(schema, "doc_id", fts_cols, vec_cols,
///                     Some(default_tokenizer()));
/// ```
pub fn default_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(AsciiLowerTokenizer)
}

/// Default `VectorConfig` for test fixtures: `dim=16`,
/// `n_cent=4`, `metric=Cosine`. Caller supplies the column
/// name and `rot_seed` — the only fields tests vary.
///
/// For realistic-scale vectors (e.g. `dim=384` in benches),
/// callers construct `VectorConfig` directly with their own
/// values.
pub fn default_vector_config(column: &str, rot_seed: u64) -> VectorConfig {
    VectorConfig {
        column: column.into(),
        dim: 16,
        rot_seed,
        metric: Metric::Cosine,
        rerank_codec: RerankCodec::Fp32,
        provided_centroids: None,
        residual_codes: false,
    }
}

/// Single-column user schema with `title: LargeUtf8`.
///
/// Mirrors the supertable's auto-`_id` model: the supertable
/// layer prepends `_id: Decimal128(38, 0)` automatically at
/// append time, so the user-facing schema only declares the
/// payload columns. Dozens of supertable tests reconstruct
/// this exact schema; centralizing keeps the
/// supertable-auto-injects-id contract in one place.
pub fn schema_id_title() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

/// Build a single-column `RecordBatch` of titles matching
/// [`schema_id_title`]. Caller supplies the title strings;
/// the rest is fixed.
pub fn build_title_batch(titles: &[&str]) -> RecordBatch {
    let titles_arr = LargeStringArray::from(titles.to_vec());
    RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles_arr)])
        .expect("RecordBatch shape matches schema_id_title")
}

/// `SupertableOptions` with the test-fixture defaults:
/// [`schema_id_title`] schema, a single FTS column `title`,
/// no vector columns, and a 1-thread rayon writer pool.
///
/// Caller chains `.with_storage(...)` / `.with_disk_cache(...)`
/// / `.with_*(...)` for whatever the specific test needs.
/// Returning the un-storage-d shape lets each test decide
/// explicitly whether to attach storage.
pub fn default_supertable_options() -> SupertableOptions {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("rayon ThreadPoolBuilder with num_threads(1) builds"),
    );
    SupertableOptions::new(
        schema_id_title(),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(default_tokenizer()),
    )
    .expect("SupertableOptions::new with default test fixture args")
    .with_writer_pool(pool)
}
