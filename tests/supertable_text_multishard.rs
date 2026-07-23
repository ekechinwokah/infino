// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cross-shard query correctness over MULTIPLE hidden text shards.
//!
//! Term-range shard partitioning silently broke every query shape
//! that needs several dictionary keys in one walk (AND intersections,
//! OR scoring, k>=3 phrase chains) once a corpus outgrew one shard:
//! per-shard kernels can't intersect keys living in different shards.
//! Shards are doc-partitioned now — each is a complete index over its
//! doc slice, one shard per drain worker (the writer pool, pinned to
//! 4 threads here so the count is runner-independent) — and this
//! binary is the regression gate the term layout never had: it
//! asserts the shapes that die under term partitioning.
//!
//! One `#[test]` fn: the cwd/config trick is process-global.

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    BoolMode,
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions, storage::LocalFsStorageProvider},
    test_helpers::default_tokenizer,
};
use tempfile::TempDir;

/// Commits in the corpus; one drain run per commit
/// (`drain_batch_superfiles(1)`), so shard grouping has several runs
/// to slice.
const COMMITS: u32 = 6;
/// Docs per commit — with ~150 bytes of text per doc each run's blob
/// lands well past the 1 MiB shard target, forcing one-or-more run
/// per group and >= 2 shards overall.
const PER: u32 = 4000;
/// Top-k large enough to sample phrase coverage across shards.
const K_SAMPLE: usize = 20;

fn ids_of(batches: &[RecordBatch]) -> Vec<i128> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("_id column")
                .values()
                .to_vec()
        })
        .collect()
}

#[test]
fn multishard_and_or_phrase_correctness() {
    // Shard count follows the drain's worker budget (the writer
    // pool, one shard per worker — the vector scheme). A multi-thread
    // writer pool over a multi-commit corpus therefore produces
    // several doc-partitioned shards without any config.

    let schema = Arc::new(Schema::new(vec![Field::new(
        "body",
        DataType::LargeUtf8,
        false,
    )]));
    let table_dir = TempDir::new().expect("table dir");
    let storage = Arc::new(LocalFsStorageProvider::new(table_dir.path()).expect("provider"));
    let st = Supertable::create(
        SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "body".into(),
                positions: true,
            }],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("options")
        .with_storage(storage)
        .with_writer_pool(Arc::new(
            // Pin the drain's worker budget so the shard count (one
            // per worker) is deterministic on any runner.
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("pool"),
        ))
        .with_drain_batch_superfiles(1),
    )
    .expect("create");

    // Every doc: a corpus-wide phrase ("alpha beta gamma"), a df=1
    // token, and a rotating token — the witnesses for cross-shard
    // phrases, ANDs, and OR dedup respectively.
    for c in 0..COMMITS {
        let texts: Vec<String> = (0..PER)
            .map(|i| {
                let d = c * PER + i;
                {
                    // ~30 tokens/doc so the run bytes clear several
                    // 1 MiB shard groups at this corpus size.
                    let mut t = format!("alpha beta gamma uniq{d} tail{}", d % 7);
                    for j in 0..25u32 {
                        t.push_str(&format!(" pad{}", (d + j * 13) % 97));
                    }
                    t
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

    // Precondition: the tiny target actually produced several shards —
    // otherwise every assertion below degenerates to the single-shard
    // case the term layout also passed.
    let hidden = st.vector_index_table().expect("hidden table");
    let n_shards = hidden
        .reader()
        .manifest()
        .get_all_superfiles()
        .iter()
        .filter(|e| !e.fts_summary.is_empty() && e.vector_summary.is_empty())
        .count();
    assert!(
        n_shards >= 2,
        "corpus must span several text shards (got {n_shards}); \
         raise PER or lower the target"
    );

    // 1. AND whose terms have wildly different df: a df=1 token
    //    intersected with a corpus-wide token. Under term
    //    partitioning these keys usually live in different shards and
    //    the intersection came back EMPTY.
    let and_hits = st
        .bm25_search("body", "uniq123 alpha", 5, BoolMode::And, None)
        .expect("and search");
    let and_ids = ids_of(&and_hits);
    assert_eq!(
        and_ids.len(),
        1,
        "uniq123 AND alpha matches exactly doc 123"
    );

    // 2. Three-word phrase: chains two pair cursors, which under term
    //    partitioning can straddle shards and verify to nothing.
    let phrase_hits = st
        .bm25_search("body", "\"alpha beta gamma\"", K_SAMPLE, BoolMode::Or, None)
        .expect("phrase search");
    assert_eq!(
        ids_of(&phrase_hits).len(),
        K_SAMPLE,
        "the corpus-wide phrase must fill the top-k from every shard"
    );

    // 3. OR mixing a df=1 term with a broad term: the matching doc
    //    must surface exactly once with both terms' contribution —
    //    under term partitioning a doc's terms in different shards
    //    produced duplicate partial-score hits.
    let or_hits = st
        .bm25_search("body", "uniq77 tail0", 4000, BoolMode::Or, None)
        .expect("or search");
    let or_ids = ids_of(&or_hits);
    let distinct: HashSet<i128> = or_ids.iter().copied().collect();
    assert_eq!(
        or_ids.len(),
        distinct.len(),
        "no doc may appear twice in an OR top-k"
    );
    // doc 77 matches BOTH terms (77 % 7 == 0) — its combined score
    // must beat every single-term tail0 doc, putting it first.
    let top = or_ids.first().copied().expect("non-empty OR result");
    let full = st
        .bm25_search("body", "uniq77", 1, BoolMode::Or, None)
        .expect("uniq77 search");
    let doc77 = ids_of(&full).first().copied().expect("doc 77 exists");
    assert_eq!(
        top, doc77,
        "the two-term doc must rank first with its full combined score"
    );
}
