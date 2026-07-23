// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The library must stay silent on stderr when the host installs no
//! `tracing` subscriber. All progress and warning output goes through
//! the `tracing` facade (see `examples/logging.rs`: the library emits
//! events, installing a subscriber is the consumer's job). Regression
//! for the unconditional `eprintln!` lines the hidden-index drain used
//! to write.
//!
//! Parent spawns a child copy of this test binary (the crash-test
//! pattern): the child drives the chattiest maintenance path — create
//! a vector table on LocalFS, commit twice, drain into the hidden
//! per-cell index — with no subscriber installed, and the parent
//! asserts the child's stderr is empty. A subprocess is load-bearing
//! here: the drain executes on shared runtime worker threads, whose
//! output libtest's in-process capture does not intercept, so only a
//! child's real stderr shows what an embedding host would see.

#![deny(clippy::unwrap_used)]

use std::{
    env,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    supertable::{
        Supertable, SupertableOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::default_vector_config,
};
use tempfile::TempDir;

/// Directory handoff to the child; presence flips this binary into
/// child mode.
const ENV_DIR: &str = "INFINO_DRAIN_STDERR_DIR";
/// `default_vector_config` is dim=16, cosine, n_cent=4.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 7;
/// Rows per commit; two commits give the drain two source superfiles.
const ROWS_PER_COMMIT: usize = 32;
/// Both source superfiles drain in one batch.
const DRAIN_BATCH_SUPERFILES: i64 = 2;
/// Child stdout sentinel proving the drain path actually completed.
const CHILD_OK: &str = "DRAIN-STDERR-CHILD-OK";

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// `n` rows; row `i` is one-hot at dim `(seed + i) % DIM`, so the two
/// commits plant distinct clusters for the drain to route.
fn vector_batch(schema: Arc<Schema>, n: usize, seed: usize) -> RecordBatch {
    let mut flat = Vec::<f32>::with_capacity(n * DIM);
    for i in 0..n {
        let active = (seed + i) % DIM;
        for d in 0..DIM {
            flat.push(if d == active { 1.0 } else { 0.0 });
        }
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(schema, vec![Arc::new(fsl)]).expect("batch")
}

/// Child body: exercise create → commit ×2 → drain end-to-end, then
/// exit before libtest can print its own summary. Anything the library
/// writes to stderr along the way is the regression.
fn run_child(dir: PathBuf) -> ! {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "emb",
        fixed_list_f32(DIM),
        false,
    )]));
    let options = SupertableOptions::new(
        schema.clone(),
        vec![],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        None,
    )
    .expect("valid options")
    .with_storage(storage)
    .with_drain_batch_superfiles(DRAIN_BATCH_SUPERFILES);
    let table = Supertable::create(options).expect("create");
    for seed in 0..2 {
        let mut writer = table.writer().expect("writer");
        writer
            .append(&vector_batch(schema.clone(), ROWS_PER_COMMIT, seed))
            .expect("append");
        writer.commit().expect("commit");
    }
    table.drain_vectors_to_cells_sync().expect("drain");
    println!("{CHILD_OK}");
    std::process::exit(0);
}

#[test]
fn drain_writes_nothing_to_stderr_without_a_subscriber() {
    if let Ok(dir) = env::var(ENV_DIR) {
        run_child(PathBuf::from(dir));
    }

    let dir = TempDir::new().expect("tempdir");
    let exe = env::current_exe().expect("current_exe");
    let output = Command::new(&exe)
        .args([
            "--exact",
            "--test-threads=1",
            "--nocapture",
            "drain_writes_nothing_to_stderr_without_a_subscriber",
        ])
        .env(ENV_DIR, dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn child");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "child failed ({:?});\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains(CHILD_OK),
        "child never completed the drain;\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.is_empty(),
        "library wrote to stderr with no subscriber installed:\n{stderr}"
    );
}
