// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS query-path diagnostic — splits the scored top-k path into the FTS
//! kernel vs the supertable `_id`-resolution + result assembly.
//!
//! The serving path (`bm25_search(.., None)`) is *kernel + resolve*: the
//! kernel scores and returns superfile-local hits; resolution turns each
//! hit into its stable `_id` and builds the Arrow batch. The kernel alone
//! is `bm25_hits`. So:
//!
//!   resolve/assembly = full (`bm25_search`) − kernel (`bm25_hits`)
//!
//! Resolution scales with the number of hits returned (≤ k) and sits
//! *above* the FTS kernel, so kernel-side scoring changes can't move it.
//! At large k this diagnostic shows whether the top-k cost lives in the
//! kernel or in resolution — the split that decides where to optimize.
//!
//! Shares the build + config with the SQL diagnostic (see
//! [`crate::diag_common`]): one corpus, one scale knob
//! (`INFINO_BENCH_SUPERTABLE_DOCS`), one iters knob (`INFINO_DIAG_ITERS`).
//!
//! ```text
//! cargo bench -- fts-diag
//! INFINO_BENCH_SUPERTABLE_DOCS=1000000 cargo bench -- fts-diag
//! INFINO_DIAG_ITERS=30 cargo bench -- fts-diag
//! ```

use std::{sync::Arc, time::Instant};

use infino::{
    OptimizeOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::fts::reader::BoolMode,
    supertable::Supertable,
};

use crate::{diag_common, markdown::fmt_count};

/// Large-k retrieval — the regime where resolution cost, proportional to
/// hits returned, is most exposed.
const K: usize = 1000;

/// FTS column planted by [`diag_common::build_supertable`].
const COLUMN: &str = "title";

/// Skip-spread dump: only terms at/above the routing df floor matter
/// (the block-selected kernels only ever see routed terms).
const SKIP_SPREAD_DF_FLOOR: u32 = 1024;
/// Skip-spread dump: heaviest terms per shard to print.
const SKIP_SPREAD_TOP_TERMS: usize = 12;

/// One query shape measured across the kernel and full paths.
struct FtsShape {
    name: &'static str,
    query: &'static str,
    mode: BoolMode,
}

/// Shapes chosen to span the two regimes the split matters for: a small
/// intersection (matches ≤ k ⇒ no pruning, every match scored *and*
/// resolved), a large intersection (heavy pruning), and a union.
const SHAPES: &[FtsShape] = &[
    FtsShape {
        name: "single_common",
        query: "term00001",
        mode: BoolMode::Or,
    },
    FtsShape {
        name: "small_and",
        query: "term00500 term01000",
        mode: BoolMode::And,
    },
    FtsShape {
        name: "large_and",
        query: "term00001 term00050",
        mode: BoolMode::And,
    },
    FtsShape {
        name: "union",
        query: "term00050 term00051 term00052",
        mode: BoolMode::Or,
    },
    FtsShape {
        name: "phrase_two_common",
        query: "\"term00001 term00002\"",
        mode: BoolMode::Or,
    },
];

pub fn run() {
    let cfg = diag_common::config();
    eprintln!(
        "[fts-diag] kernel vs resolve/assembly split: n_docs={} iters={} k={K} \
         (knobs: INFINO_BENCH_SUPERTABLE_DOCS, INFINO_DIAG_ITERS)",
        fmt_count(cfg.n_docs),
        cfg.iters,
    );

    eprintln!("[fts-diag] building supertable...");
    let build_t0 = Instant::now();
    let (table, _batches) = diag_common::build_supertable(&cfg);
    let reader = table.reader();
    eprintln!(
        "[fts-diag] built in {:.1}s ({} superfile(s) after optimize)",
        build_t0.elapsed().as_secs_f64(),
        reader.manifest().superfiles.len(),
    );

    // Exact skip-bound spread of drained shards' heaviest terms —
    // decides whether bound-based block admission can prune this
    // corpus (flat exact maxima: whole-list walks are inherent for
    // those terms; spread: the flat resident rows are a quantization
    // artifact and an exact-bound tier recovers pruning). The hidden
    // sibling needs real storage, so a second LocalFs table is built
    // from the same batches, drained, and dumped; the timed diag
    // table above stays untouched.
    {
        let dir = tempfile::TempDir::new().expect("skip-spread tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("skip-spread localfs"));
        let fs_table = Supertable::create(diag_common::diag_options().with_storage(storage))
            .expect("skip-spread table");
        {
            let mut writer = fs_table.writer().expect("skip-spread writer");
            for batch in &_batches {
                writer.append(batch).expect("skip-spread append");
            }
            writer.commit().expect("skip-spread commit");
        }
        let drain_t0 = Instant::now();
        fs_table
            .drain_vectors_to_cells_sync()
            .expect("skip-spread drain");
        eprintln!(
            "[fts-diag] skip-spread table drained in {:.1}s",
            drain_t0.elapsed().as_secs_f64()
        );
        fs_table.dump_text_skip_spread(COLUMN, SKIP_SPREAD_DF_FLOOR, SKIP_SPREAD_TOP_TERMS);

        // Hidden-index kernel timings on the REAL corpus, per shape:
        // the same battery as the (user-table) main diag below, over
        // the drained shards and again after optimize — the direct
        // user-vs-hidden A/B at kernel granularity that the synthetic
        // src-tree diag corpus cannot reproduce (its uniform tf/dl
        // profile hides the bench shapes' regressions).
        let drained_battery = |tag: &str| {
            let fs_reader = fs_table.reader();
            for s in SHAPES {
                let _ = fs_reader
                    .bm25_search(COLUMN, s.query, K, s.mode, None)
                    .expect("drained warm-up");
            }
            for s in SHAPES {
                let mut count = Vec::with_capacity(cfg.iters);
                let mut kernel = Vec::with_capacity(cfg.iters);
                let mut full = Vec::with_capacity(cfg.iters);
                for _ in 0..cfg.iters {
                    let t = Instant::now();
                    let c = fs_reader
                        .count(COLUMN, s.query, s.mode)
                        .expect("drained count");
                    count.push(t.elapsed());
                    std::hint::black_box(c);
                    let t = Instant::now();
                    let h = fs_reader
                        .bm25_hits(COLUMN, s.query, K, s.mode)
                        .expect("drained bm25_hits");
                    kernel.push(t.elapsed());
                    std::hint::black_box(h);
                    let t = Instant::now();
                    let out = fs_reader
                        .bm25_search(COLUMN, s.query, K, s.mode, None)
                        .expect("drained bm25_search");
                    full.push(t.elapsed());
                    std::hint::black_box(out);
                }
                // traverse (count) → +score/heap → +resolve/route (full).
                // On the hidden path `full − kernel` also carries the
                // two-wave + HDEL + id-attach overhead, so this split
                // separates "slow leapfrog" from "fixed per-query tax".
                let cp = diag_common::percentile(&mut count, 50);
                let kp = diag_common::percentile(&mut kernel, 50);
                let fp = diag_common::percentile(&mut full, 50);
                eprintln!(
                    "[fts-diag/{tag}] {:<18} count {:>9.2?}  kernel {:>9.2?}  full {:>9.2?}  \
                     (route+resolve {:>9.2?})",
                    s.name,
                    cp,
                    kp,
                    fp,
                    fp.saturating_sub(kp),
                );
            }
        };
        drained_battery("post-drain");
        let opt_t0 = Instant::now();
        fs_table
            .optimize(&OptimizeOptions::default())
            .expect("skip-spread optimize");
        eprintln!(
            "[fts-diag] skip-spread table optimized in {:.1}s",
            opt_t0.elapsed().as_secs_f64()
        );
        drained_battery("post-compact");
    }

    // Warm both paths for every shape (cache-hot before timing).
    for s in SHAPES {
        let _ = reader
            .bm25_search(COLUMN, s.query, K, s.mode, None)
            .expect("warm-up bm25_search");
    }

    // Warm the count path too.
    for s in SHAPES {
        let _ = reader
            .count(COLUMN, s.query, s.mode)
            .expect("warm-up count");
    }

    // Decompose the scored path into three additive layers:
    //   count  = posting traversal + block decode (no score, no heap)
    //   kernel = count + BM25 scoring + top-k heap   (= bm25_hits)
    //   full   = kernel + _id-resolution + result assembly (= bm25_search)
    // so score+heap = kernel − count, and resolve = full − kernel. (At
    // k=1000 over a large superfile the scored path prunes little, so
    // count is a fair traverse/decode floor for it.)
    eprintln!();
    eprintln!(
        "[fts-diag] {:<15}{:>8}{:>12}{:>12}{:>12}{:>13}{:>12}",
        "shape", "hits", "count", "kernel", "full", "score+heap", "resolve"
    );
    for s in SHAPES {
        let hits = reader
            .bm25_hits(COLUMN, s.query, K, s.mode)
            .expect("bm25_hits")
            .len();

        let mut count = Vec::with_capacity(cfg.iters);
        for _ in 0..cfg.iters {
            let t = Instant::now();
            let out = reader.count(COLUMN, s.query, s.mode).expect("count");
            count.push(t.elapsed());
            std::hint::black_box(out);
        }

        let mut kernel = Vec::with_capacity(cfg.iters);
        for _ in 0..cfg.iters {
            let t = Instant::now();
            let out = reader
                .bm25_hits(COLUMN, s.query, K, s.mode)
                .expect("kernel bm25_hits");
            kernel.push(t.elapsed());
            std::hint::black_box(out);
        }

        let mut full = Vec::with_capacity(cfg.iters);
        for _ in 0..cfg.iters {
            let t = Instant::now();
            let out = reader
                .bm25_search(COLUMN, s.query, K, s.mode, None)
                .expect("full bm25_search");
            full.push(t.elapsed());
            std::hint::black_box(out);
        }

        let cp = diag_common::percentile(&mut count, 50);
        let kp = diag_common::percentile(&mut kernel, 50);
        let fp = diag_common::percentile(&mut full, 50);
        let score_heap = kp.saturating_sub(cp);
        let resolve = fp.saturating_sub(kp);
        eprintln!(
            "[fts-diag] {:<15}{:>8}{:>12}{:>12}{:>12}{:>13}{:>12}",
            s.name,
            hits,
            diag_common::fmt(cp),
            diag_common::fmt(kp),
            diag_common::fmt(fp),
            diag_common::fmt(score_heap),
            diag_common::fmt(resolve),
        );
    }
}
