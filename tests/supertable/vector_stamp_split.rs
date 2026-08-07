// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Stamp-driven cell splits — the grid must refine on UNIMODAL corpora.
//!
//! Real embedding corpora never trip the modality (Ashman-D) split
//! trigger: k-means cells are compact blobs by construction, so every
//! cell scores at the unimodal baseline and the grid stays at its
//! bootstrap size while the calibrator inflates the rerank budget to
//! compensate (measured at 10M/256 cells: 26.7K-row budget at k=100,
//! ~69 ms warm). The fix bounds the per-cell row cap by the table's own
//! stamped rerank budget (`opann::effective_cell_row_cap`), so a cell
//! holding more rows than a query must exactly score splits on SIZE.
//!
//! This regression pins that end to end on a smooth (unimodal-per-cell)
//! corpus: pre-fix, `optimize()` refuses to split it (D below
//! threshold, config cap out of reach) and the assertion fails.

#![deny(clippy::unwrap_used)]

use std::{collections::HashMap, sync::Arc};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    OptimizeOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::{default_tokenizer, default_vector_config},
};
use tempfile::TempDir;

/// Matches `default_vector_config`'s dimension.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 41;
/// Rows: on the 2-cell pinned grid below, ~3000 rows per cell — far above
/// any rerank budget a corpus this size stamps, so the stamped cap must
/// bite. The precondition is asserted, not assumed.
const N_ROWS: usize = 6000;
/// Pinned grids: a deliberately coarse hidden grid so cells overflow the
/// stamped budget (the situation a bulk-loaded real corpus lands in).
const PINNED_CELLS: usize = 2;
/// Optimize passes allowed for convergence (each pass is bounded by the
/// engine's per-pass split allowance).
const MAX_OPTIMIZE_PASSES: usize = 5;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn vector_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_vector_cell_counts(PINNED_CELLS, PINNED_CELLS)
}

/// Deterministic SMOOTH corpus: one blob per grid half, rows spread
/// continuously (no planted discrete modes), so the modality trigger sees
/// unimodal cells — exactly the shape real embeddings present.
fn row_vec(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| {
            let noise = ((i * 31 + d * 17 + 7) % 97) as f32 / 97.0;
            let lobe = if i.is_multiple_of(2) { 0.5 } else { -0.5 };
            lobe + noise * 0.45
        })
        .collect()
}

fn corpus_batch(schema: Arc<Schema>) -> RecordBatch {
    let mut flat = Vec::<f32>::with_capacity(N_ROWS * DIM);
    let mut titles = Vec::with_capacity(N_ROWS);
    for i in 0..N_ROWS {
        flat.extend(row_vec(i));
        titles.push(format!("row{i:05}"));
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(titles)) as ArrayRef,
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

/// Live rows per cell via the stable-id -> cell diag map.
async fn cells_with_counts(st: &Supertable) -> HashMap<u32, u64> {
    let reader = st.reader().expect("reader");
    let map = reader
        .diag_hidden_stable_cell_map("emb")
        .await
        .expect("cell map");
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for cell in map.values() {
        *counts.entry(*cell).or_default() += 1;
    }
    counts
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn over_budget_unimodal_cells_split_on_optimize() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st =
        Supertable::create(vector_options().with_storage(Arc::clone(&storage))).expect("create");

    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&corpus_batch(schema)).expect("append");
    w.commit().expect("commit");
    drop(w);
    st.drain_vectors_to_cells_sync().expect("drain");

    let reader = st.reader().expect("reader");
    let (_, _, rerank) = reader
        .diag_hidden_probe_laws()
        .expect("laws stamped at drain");
    // The stamped budget at the recall@10 knot (index 1 of the [1, 10,
    // 100, 1000] knots), floored exactly as the engine floors it.
    let budget = (rerank[1] as u64).max(256);
    let before = cells_with_counts(&st).await;
    let max_before = before.values().copied().max().expect("cells");
    assert!(
        max_before > budget,
        "fixture precondition: cells ({max_before} rows) must exceed the \
         stamped budget ({budget}) or this regression tests nothing"
    );

    // Pre-fix behavior: the modality trigger scores this smooth corpus
    // unimodal and the 500K config cap is out of reach, so optimize()
    // leaves the grid at its bootstrap size forever. Post-fix the stamped
    // budget is the effective cap and the over-budget cells split.
    for _ in 0..MAX_OPTIMIZE_PASSES {
        st.optimize(&OptimizeOptions::default()).expect("optimize");
        let counts = cells_with_counts(&st).await;
        if counts.values().all(|&n| n <= budget) {
            break;
        }
    }
    let after = cells_with_counts(&st).await;
    let max_after = after.values().copied().max().expect("cells");
    assert!(
        after.len() > before.len(),
        "optimize must split over-budget cells: {} cells before, {} after \
         (max cell {} rows vs budget {})",
        before.len(),
        after.len(),
        max_after,
        budget
    );
    assert!(
        max_after <= budget,
        "grid must converge to the stamped budget: max cell {max_after} \
         rows vs budget {budget} across {} cells",
        after.len()
    );
}
