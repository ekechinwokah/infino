// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `SupertableWriter` — the single-writer append + commit path.
//!
//! **Naming convention.** `SupertableWriter` is a long-lived
//! append handle — `append×N → commit`, repeated across many
//! commits over its lifetime. Contrast
//! [`crate::superfile::SuperfileBuilder`], which is a single-shot
//! factory consuming `self` to produce one immutable artifact.
//! Each `commit` here internally spawns many superfile builders,
//! one per shard.
//!
//! Acquired via [`Supertable::writer`](super::Supertable::writer);
//! at most one writer is outstanding per supertable at a time
//! (enforced by the inner state's `writer_outstanding` flag, with
//! release on `Drop`). Holds an in-memory buffer of
//! `(scalar_batch, vectors_per_column)` payloads that
//! [`SupertableWriter::commit`] partitions across the writer
//! pool's rayon workers — each worker constructs its own
//! [`SuperfileBuilder`], feeds its slice, and emits one
//! self-contained superfile. All resulting superfiles are published
//! in a single `ArcSwap` of the manifest at the end.
//!
//! ## Flow
//!
//! - `append(batch)` runs schema + null validation via
//!   `vector_split`, pushes a `BufferedBatch` onto the writer's
//!   buffer, and triggers an internal `commit()` if the running
//!   buffer-byte estimate crosses the configured threshold.
//! - `commit()` drains the buffer, partitions across the writer
//!   pool, runs each shard build in parallel, and publishes all
//!   shards as new superfiles in one manifest swap. Idempotent on
//!   an empty buffer (no-op return Ok). The writer slot is
//!   released on `Drop`; callers don't need a separate `finish()`
//!   call.
//!
//! ## Buffer ownership
//!
//! Vectors arrive from the input `RecordBatch` as
//! `FixedSizeListArray` columns; `vector_split` views them as
//! `&[f32]` slices. To keep the buffer ownership clean across
//! `append` calls (each input batch can be dropped by the caller
//! once `append` returns), we Arc-clone the underlying
//! `Float32Array` payloads into the buffer. At commit time we
//! re-derive `&[f32]` slices from the Arc'd arrays for the
//! per-shard `SuperfileBuilder::add_batch` call. No bytes copied;
//! just Arc reference counts.

use std::{
    cmp,
    collections::{HashMap, HashSet},
    fmt, io,
    marker::PhantomData,
    mem,
    sync::{Arc, atomic::Ordering},
    time,
};

use arrow::{
    compute::{concat_batches, take},
    ipc::writer::StreamWriter,
};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array,
};
use bytes::Bytes;
use chrono::Utc;
use datafusion::prelude::Expr;
use futures::{
    future::try_join_all,
    stream::{self, StreamExt},
};
use object_store::{PutPayload, UploadPart};
use rayon::prelude::*;
use tokio::time::sleep;
use tracing::{debug, error};

use super::{
    build::fanout_shards,
    error::BuildError,
    handle::{GLOBAL_VECTOR_KMEANS_ITERS, GLOBAL_VECTOR_KMEANS_SEED, Supertable, SupertableInner},
    manifest::{
        CellVectorSummary, FtsSummaryAgg, ManifestSnapshot, ScalarStatsAgg, SubsectionOffsets,
        SuperfileEntry, SuperfileUri, VectorSummary, bloom::BloomBuilder,
    },
    mutations::{
        CommitError, CommitResult, MAX_TARGETS_PER_MUTATION, MutationError, MutationStats,
        PendingDelete, PendingUpdate,
    },
    opann,
    options::{DECIMAL128_PRECISION, DECIMAL128_SCALE, SupertableOptions},
    utils::vector_split::split_vectors,
    wal::{
        WalStore,
        pipeline::{self, TombstonePhaseOutcome},
        state_doc::{
            IdSpan, OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId,
            WalState, WalStateDoc,
        },
    },
};
#[cfg(test)]
use crate::superfile::ReadError;
use crate::{
    InfinoError,
    config::{self, CentroidAlignment, DrainConsolidate, ThreadCount},
    runtime_bridge::bridge_on_runtime,
    storage::{StorageError, StorageProvider},
    superfile::{
        SuperfileReader,
        builder::{SuperfileBuilder, VectorConfig},
        format::{
            CRC_BYTES,
            footer::read_kv_metadata,
            fts::{HEADER_SIZE as FTS_HEADER_SIZE, U64_BYTES, hdr},
            kv,
            vec::{
                CELL_DIR_ENTRY_SIZE, CLUSTER_IDX_ENTRY_BYTES, DIR_ENTRY_SIZE, DOC_ID_BYTES,
                OUTER_HEADER_SIZE, STABLE_ID_BYTES, SUB_HEADER_SIZE, U32_BYTES, cell_dir_entry,
                dir_entry, outer_hdr, sub_hdr,
            },
        },
        reader::vector_layout_from_kv,
        vector::{
            builder::{
                build_merged_subsection_from_fp32, build_merged_subsection_from_materialized,
            },
            cell_posting::{EncodedCellRow, MaterializedIvfRow},
            distance::{Metric, transpose_centroids_cluster_major},
            ivf_merge::{MergedIvfSubsection, route_clusters_into_cells},
            kmeans::kmeans_with_assignments,
            layout::VectorLayout,
            reader::VectorReader,
            rerank_codec::RerankCodec,
            spill::CellRowAccumulator,
        },
    },
    supertable::{
        CommitError as SupertableCommitError, ManifestLoadError,
        error::ManifestError,
        hidden_deleted::{self, encode_deleted_ids},
        manifest::{
            ClusterCentroids,
            commit::get_current_manifest_etag,
            list::PartitionStrategy,
            part::{self as part_mod, PartId},
        },
        query::{dispatch::open_reader, vector::stable_ids_by_local_for_routing},
        reader_cache::DiskCacheStore,
        slow_vector_state,
    },
};

/// Target bytes per fine IVF run inside one global cell. Fine-centroid count
/// is derived from this target; it is not copied from the outer/global grid or
/// repeated as a fixed count for every small commit delta.
const DRAIN_FINE_RUN_TARGET_BYTES: usize = 2 * 1024 * 1024;

pub struct SupertableWriter {
    inner: Arc<SupertableInner>,
    /// Accumulated input from append() calls. The writer (not the
    /// SuperfileBuilder) owns the buffer so commit() can rayon-
    /// shard it across workers, each running its own builder.
    buffer: Vec<BufferedBatch>,
    /// Estimated byte cost of `buffer` so append() can auto-flush
    /// when the buffer crosses the configured threshold.
    buffer_bytes: usize,
    /// Pending update entries, in buffer order. Each is
    /// fully-resolved at `update()` call time (predicate
    /// captured, `_id` range minted, IPC sidecar bytes encoded);
    /// `commit()` drives them through the WAL pipeline in order.
    pending_updates: Vec<PendingUpdateEntry>,
    /// Pending delete entries, in buffer order. Each carries
    /// the call-time resolved `target_ids` + a pre-minted
    /// `wal_id`; `commit()` builds the WAL state doc and drives
    /// the tombstone phase.
    pending_deletes: Vec<PendingDeleteEntry>,
}

/// One buffered update. Resources here are all reserved at the
/// `update()` call so the writer can drop the `RecordBatch`
/// after IPC-encoding it (the `ipc_bytes` are what the WAL
/// sidecar carries).
struct PendingUpdateEntry {
    wal_id: WalId,
    target_ids: Vec<i128>,
    preallocated_superfile_id: uuid::Uuid,
    minted_id_spans: Vec<IdSpan>,
    new_row_count: u32,
    new_row_content_hash: String,
    ipc_bytes: Bytes,
}

/// One buffered delete. Just the call-time resolved target_ids
/// + a pre-minted `wal_id`.
struct PendingDeleteEntry {
    wal_id: WalId,
    target_ids: Vec<i128>,
}

impl fmt::Debug for SupertableWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SupertableWriter")
            .field("buffered_batches", &self.buffer.len())
            .field("buffered_bytes", &self.buffer_bytes)
            .field("manifest_id", &self.inner.manifest.load().manifest_id)
            .finish()
    }
}

/// One buffered append-call payload. Vectors stored as
/// `Arc<Float32Array>` so the buffer owns its data outright;
/// per-shard builders re-derive `&[f32]` slices via
/// [`Float32Array::values`] without copying.
struct BufferedBatch {
    scalar: RecordBatch,
    vectors: Vec<Arc<Float32Array>>,
}

/// Row-balanced split of the writer's buffered batches into
/// `n_shards` shard inputs, each shaped as a `Vec<BufferedBatch>`
/// that [`build_one_shard_with_layout`] can consume directly. The split walks
/// rows across the original buffer in order and emits zero-copy
/// Arrow slices (`RecordBatch::slice` + `Float32Array::slice` —
/// adjust buffer offsets only; underlying memory stays Arc-counted),
/// so no payload bytes are copied even when a shard boundary falls
/// in the middle of a `BufferedBatch`.
///
/// Row imbalance across shards is ≤ 1: with `total_rows = q·n + r`,
/// the first `r` shards get `q+1` rows and the rest get `q`.
///
/// Trailing empty shards (only possible when `total_rows < n_shards`)
/// are dropped before return; callers see exactly the shards that
/// will produce a non-empty superfile.
fn split_buffer_into_row_shards(
    buffer: Vec<BufferedBatch>,
    n_shards: usize,
    vector_dims: &[usize],
) -> Vec<Vec<BufferedBatch>> {
    debug_assert!(n_shards > 0);
    let total_rows: usize = buffer.iter().map(|b| b.scalar.num_rows()).sum();
    if total_rows == 0 {
        return Vec::new();
    }
    let base = total_rows / n_shards;
    let remainder = total_rows % n_shards;
    let target = |i: usize| if i < remainder { base + 1 } else { base };

    let mut shards: Vec<Vec<BufferedBatch>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut shard_idx = 0usize;
    let mut shard_remaining = target(0);

    for batch in buffer {
        let n_rows = batch.scalar.num_rows();
        if n_rows == 0 {
            continue;
        }
        let mut row_cursor = 0;
        while row_cursor < n_rows {
            // Skip ahead over any zero-target shards (only happens
            // when total_rows < n_shards, leaving trailing shards
            // with target == 0).
            while shard_remaining == 0 && shard_idx + 1 < n_shards {
                shard_idx += 1;
                shard_remaining = target(shard_idx);
            }
            let take = cmp::min(shard_remaining, n_rows - row_cursor);
            let scalar = batch.scalar.slice(row_cursor, take);
            let vectors: Vec<Arc<Float32Array>> = batch
                .vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dim = vector_dims[i];
                    Arc::new(v.slice(row_cursor * dim, take * dim))
                })
                .collect();
            shards[shard_idx].push(BufferedBatch { scalar, vectors });
            row_cursor += take;
            shard_remaining -= take;
        }
    }
    shards.retain(|s| !s.is_empty());
    shards
}

/// After a manifest swap that drops superfile references, schedule a deferred
/// GC sweep instead of inline `storage.delete`. Inline delete races snapshot-
/// pinned readers that may still cold-fetch superseded bytes.
fn schedule_background_storage_reclaim(inner: Arc<SupertableInner>) {
    if inner.options.storage.is_none() {
        return;
    }
    // Integration tests that need reclaim call `Supertable::gc()` explicitly
    // (see `tests/supertable/compact_gc.rs`). Spawning here from a
    // `current_thread` tokio test runtime panics in `block_in_place`.
    #[cfg(not(test))]
    {
        let rt = inner.query_runtime();
        rt.spawn(async move {
            sleep(super::gc::DEFAULT_SUPERFILE_RECLAIM_GRACE).await;
            if let Err(e) = super::gc::gc_storage_sweep_for_inner(
                &inner,
                super::gc::DEFAULT_SUPERFILE_RECLAIM_GRACE,
            )
            .await
            {
                tracing::debug!("supertable: deferred storage reclaim: {e}");
            }
        });
    }
    #[cfg(test)]
    {
        let _ = inner;
    }
}

/// Sq8+ε IVF rows aligned to scalar `_id` row order. Optional tombstone bitmap
/// skips deleted locals (cell maintenance); incoming routing passes `None`.
async fn materialized_ivf_rows_in_doc_order(
    vec_reader: &VectorReader,
    column: &str,
    stable_ids_by_local: &[i128],
    tombstones: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let mut rows = vec_reader
        .materialized_index_rows_async(column)
        .await
        .ok_or_else(|| {
            BuildError::Store(format!(
                "IVF maintenance: column '{column}' missing Sq8Residual index"
            ))
        })?;
    let n_rows = stable_ids_by_local.len();
    let mut by_local = vec![None; n_rows];
    for row in &mut rows {
        if tombstones.is_some_and(|bm| bm.contains(row.local_doc_id)) {
            continue;
        }
        let slot = row.local_doc_id as usize;
        if slot < n_rows {
            // Cell superfiles inline the stable `_id` in the IVF blob, so the
            // read-back already carries it (nonzero). Region-less incoming
            // superfiles return 0 here and fall back to the scalar `_id` column
            // resolved into `stable_ids_by_local`.
            if row.stable_id == 0 {
                row.stable_id = stable_ids_by_local[slot];
                row.encoded.stable_id = row.stable_id;
            }
            by_local[slot] = Some(row.clone());
        }
    }
    Ok(by_local
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| {
            r.map(|mut row| {
                row.local_doc_id = i as u32;
                row
            })
        })
        .collect())
}

/// Split buffered rows into per-cell shards based on nearest centroid.
/// Each shard carries all rows assigned to one cell; the caller stamps
/// `partition_hint` on the resulting superfile entries.
fn split_buffer_by_vector_cell(
    buffer: Vec<BufferedBatch>,
    cells: &ClusterCentroids,
    metric: Metric,
    vec_col_idx: usize,
) -> Vec<(u32, Vec<BufferedBatch>)> {
    let k = cells.n_cent as usize;
    let mut cell_batches: Vec<Vec<BufferedBatch>> = (0..k).map(|_| Vec::new()).collect();
    for batch in buffer {
        let n_rows = batch.scalar.num_rows();
        if n_rows == 0 {
            continue;
        }
        let vecs = batch.vectors[vec_col_idx].values();
        let mut assignments = vec![0u32; n_rows];
        cells.assign_rows(metric, vecs, &mut assignments);
        let mut per_cell_rows: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
        for (row, &cell) in assignments.iter().enumerate() {
            per_cell_rows[cell as usize].push(row);
        }
        for (cell_id, rows) in per_cell_rows.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let indices = UInt32Array::from(rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
            let scalar_cols: Vec<ArrayRef> = (0..batch.scalar.num_columns())
                .map(|col_idx| {
                    arrow::compute::take(batch.scalar.column(col_idx), &indices, None)
                        .expect("take column")
                })
                .collect();
            let scalar_batch =
                RecordBatch::try_new(batch.scalar.schema(), scalar_cols).expect("rebuild batch");
            let vectors: Vec<Arc<Float32Array>> = batch
                .vectors
                .iter()
                .map(|v| {
                    let vdim = v.len() / n_rows;
                    let mut out = Vec::with_capacity(rows.len() * vdim);
                    for &r in &rows {
                        out.extend_from_slice(&v.values()[r * vdim..(r + 1) * vdim]);
                    }
                    std::sync::Arc::new(Float32Array::from(out))
                })
                .collect();
            cell_batches[cell_id].push(BufferedBatch {
                scalar: scalar_batch,
                vectors,
            });
        }
    }
    cell_batches
        .into_iter()
        .enumerate()
        .filter(|(_, batches)| !batches.is_empty())
        .map(|(cell_id, batches)| (cell_id as u32, batches))
        .collect()
}

/// The public folded `update` / `delete` buffer exactly one mutation
/// before committing, so `CommitResult.outcomes` carries exactly one
/// entry; surface it (or a backend error if, impossibly, none landed).
fn single_outcome(res: CommitResult) -> Result<MutationStats, InfinoError> {
    res.outcomes
        .into_iter()
        .next()
        .ok_or_else(|| InfinoError::Backend("commit produced no mutation outcome".to_string()))
}

impl Supertable {
    /// Append one batch of rows and commit — durable when this returns.
    ///
    /// Folds the buffered writer + commit into a single call: one
    /// `append` == one commit == one sealed superfile, so callers batch
    /// rows per call rather than calling once per row.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// let batch = RecordBatch::try_new(
    ///     schema,
    ///     vec![Arc::new(LargeStringArray::from(vec!["hello world"]))],
    /// )?;
    /// posts.append(&batch)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError> {
        let mut w = self.writer()?;
        w.append(batch)?;
        w.commit()?;
        Ok(())
    }

    /// Replace every row matching `predicate` with `new_rows`, then
    /// commit. `new_rows.num_rows()` must equal the match count.
    /// Durable when this returns.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use datafusion::prelude::{col, lit};
    /// # use infino::{connect, IndexSpec};
    /// # let dir = tempfile::tempdir()?; // update/delete need durable storage
    /// # let db = connect(dir.path().to_str().expect("utf8 path"))?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # let row = |s: &str| RecordBatch::try_new(
    /// #     schema.clone(), vec![Arc::new(LargeStringArray::from(vec![s]))]).expect("batch");
    /// # posts.append(&row("draft"))?;
    /// let stats = posts.update(col("body").eq(lit("draft")), &row("published"))?;
    /// assert_eq!(stats.matched(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn update(
        &self,
        predicate: Expr,
        new_rows: &RecordBatch,
    ) -> Result<MutationStats, InfinoError> {
        let mut w = self.writer()?;
        w.update(predicate, new_rows.clone())?;
        single_outcome(w.commit()?)
    }

    /// Tombstone every row matching `predicate`, then commit. Durable
    /// when this returns.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use datafusion::prelude::{col, lit};
    /// # use infino::{connect, IndexSpec};
    /// # let dir = tempfile::tempdir()?; // update/delete need durable storage
    /// # let db = connect(dir.path().to_str().expect("utf8 path"))?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(
    /// #     schema, vec![Arc::new(LargeStringArray::from(vec!["spam"]))])?)?;
    /// let stats = posts.delete(col("body").eq(lit("spam")))?;
    /// assert_eq!(stats.n_tombstoned(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError> {
        let mut w = self.writer()?;
        w.delete(predicate)?;
        single_outcome(w.commit()?)
    }

    test_visible! {
    /// Acquire the single writer for this supertable.
    ///
    /// Returns [`BuildError::SupertableInUse`] if another
    /// `SupertableWriter` is already outstanding (drop it before
    /// acquiring a new one). Each `Supertable` has exactly one
    /// active writer slot at a time, enforced atomically; when
    /// the writer is dropped, the slot is released and a
    /// subsequent `writer()` call succeeds.
    fn writer(&self) -> Result<SupertableWriter, BuildError> {
        match self.inner().writer_outstanding.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(SupertableWriter {
                inner: Arc::clone(self.inner()),
                buffer: Vec::new(),
                buffer_bytes: 0,
                pending_updates: Vec::new(),
                pending_deletes: Vec::new(),
            }),
            Err(_) => Err(BuildError::SupertableInUse),
        }
    }
    }
}

fn bootstrap_centroids_from_batch(
    batches: &[BufferedBatch],
    vec_dim: usize,
    n_cells: usize,
) -> Option<ClusterCentroids> {
    let mut vectors = Vec::new();
    for batch in batches {
        if batch.vectors.is_empty() {
            continue;
        }
        let vecs = batch.vectors[0].values();
        let n_rows = batch.scalar.num_rows();
        for row in 0..n_rows {
            vectors.extend_from_slice(&vecs[row * vec_dim..(row + 1) * vec_dim]);
        }
    }
    let n_docs = vectors.len() / vec_dim;
    if n_docs == 0 {
        return None;
    }
    let k = n_cells.min(n_docs).max(1);
    let (centroids, assignments) = kmeans_with_assignments(
        &vectors,
        vec_dim,
        k,
        GLOBAL_VECTOR_KMEANS_ITERS,
        GLOBAL_VECTOR_KMEANS_SEED,
    );
    let mut counts = vec![0u32; k];
    for &a in &assignments {
        counts[a as usize] += 1;
    }
    Some(ClusterCentroids::from_fp32(
        k as u32,
        vec_dim as u32,
        &centroids,
        counts,
    ))
}

impl SupertableWriter {
    /// Number of buffered batches not yet committed. Useful for
    /// tests + diagnostics; not part of the production hot path.
    pub fn buffered_batches(&self) -> usize {
        self.buffer.len()
    }

    /// Estimated bytes of buffered (un-committed) data.
    pub fn buffered_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Add one batch to the in-memory buffer. Triggers an
    /// internal `commit()` if the running buffer-byte estimate
    /// crosses the configured threshold (or returns immediately
    /// if `commit_threshold_size_mb == 0`).
    ///
    /// The supplied batch's schema must match
    /// [`SupertableOptions::user_schema`] — i.e., it must NOT
    /// contain the id column. This method injects the id column
    /// unconditionally; the buffered batch's schema therefore
    /// matches [`SupertableOptions::scalar_schema`] with the
    /// id column at position 0.
    pub fn append(&mut self, batch: &RecordBatch) -> Result<(), BuildError> {
        let options = &self.inner.options;

        // Validate + split. Batch schema is user_schema (no id col).
        let (scalar_no_id, _vector_slices) = split_vectors(batch, options)?;

        // Re-derive owned Arc<Float32Array> handles for each
        // vector column. We can't keep the &[f32] slices from
        // split_vectors in the buffer (their lifetime is tied to
        // `batch`, which the caller reclaims after this returns).
        // The Arc<Float32Array> shares the same underlying buffer
        // — no bytes copied.
        let mut vectors = Vec::with_capacity(options.vector_columns.len());
        for vc in &options.vector_columns {
            let col_idx = batch
                .schema()
                .index_of(&vc.column)
                .map_err(|_| BuildError::BatchSchemaMismatch)?;
            let fsl = batch
                .column(col_idx)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or(BuildError::BatchSchemaMismatch)?;
            let values = fsl.values();
            let f32_arr = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(BuildError::BatchSchemaMismatch)?
                .clone();
            vectors.push(Arc::new(f32_arr));
        }

        // Mint one id per row and prepend the id column. Lock
        // is uncontended in practice (writer-slot exclusivity
        // serializes append per supertable handle); held only
        // long enough to drain N ids into the Vec.
        let n_rows = scalar_no_id.num_rows();
        let mut ids: Vec<i128> = Vec::with_capacity(n_rows);
        {
            let generator = self
                .inner
                .id_generator
                .lock()
                .expect("id_generator mutex poisoned");
            for _ in 0..n_rows {
                ids.push(generator.next_id());
            }
        }
        let id_array = Decimal128Array::from(ids)
            .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
            .expect(
                "invariant: precision 38 + scale 0 always valid \
                 for any i128 payload",
            );
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(scalar_no_id.num_columns() + 1);
        columns.push(Arc::new(id_array));
        columns.extend(scalar_no_id.columns().iter().cloned());
        let scalar = RecordBatch::try_new(options.scalar_schema(), columns)
            .map_err(|_| BuildError::BatchSchemaMismatch)?;

        // Estimate byte cost: Arrow scalar columns + f32 vector
        // payload. RecordBatch::get_array_memory_size accounts
        // for buffer allocations (rough but good enough for
        // threshold gating).
        let bytes = scalar.get_array_memory_size()
            + vectors
                .iter()
                .map(|v| v.len() * mem::size_of::<f32>())
                .sum::<usize>();

        self.buffer.push(BufferedBatch { scalar, vectors });
        self.buffer_bytes += bytes;

        // Auto-flush if over threshold.
        let threshold = (options.commit_threshold_size_mb as usize)
            .saturating_mul(1024)
            .saturating_mul(1024);
        if threshold > 0 && self.buffer_bytes >= threshold {
            self.commit_appends_internal()?;
        }

        Ok(())
    }

    /// Buffer a delete operation. Every row whose `_id`
    /// matches `predicate` at call time will be tombstoned by
    /// the next [`commit`] call.
    ///
    /// `predicate` is evaluated **immediately** against the
    /// current manifest snapshot (the same ArcSwap-backed view
    /// queries use). The resolved `_id` set is captured on the
    /// writer's pending-deletes buffer; rows that newly match
    /// `predicate` between this call and `commit()` (because of
    /// an interleaving append on this or another writer) are
    /// NOT tombstoned — only the captured `_id` list is.
    ///
    /// **Does NOT make the change durable.** Buffered deletes
    /// are lost on writer drop until the next successful
    /// `commit()`. Symmetric with buffered `append()`s.
    ///
    /// [`commit`]: SupertableWriter::commit
    pub fn delete(&mut self, predicate: Expr) -> Result<PendingDelete, MutationError> {
        // Pre-flight: storage must be attached for the WAL
        // pipeline to drive this op at commit time.
        let _ = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?;

        // Resolve the predicate against the current manifest
        // snapshot. NOTE: the writer's pending-appends buffer
        // is NOT flushed here. Captured-at-call semantics mean
        // the delete sees the manifest as it stood at this
        // call's instant; rows the caller appended in the same
        // writer session are not yet in the manifest.
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let target_ids = supertable
            .reader()
            .scan_ids_matching(predicate)
            .map_err(MutationError::PredicateEval)?;
        let matched = target_ids.len();
        if matched > MAX_TARGETS_PER_MUTATION {
            return Err(MutationError::MatchCountExceedsCap {
                matched,
                cap: MAX_TARGETS_PER_MUTATION,
            });
        }

        // Pre-mint the wal_id so we can surface it at commit
        // time even on a partial-failure path (the recovery
        // sweep on a fresh open completes any WAL whose id
        // already landed in storage).
        let wal_id_value = self
            .inner
            .id_generator
            .lock()
            .expect("id_generator mutex poisoned")
            .next_id();

        self.pending_deletes.push(PendingDeleteEntry {
            wal_id: WalId(wal_id_value),
            target_ids,
        });
        Ok(PendingDelete { matched })
    }

    /// Buffer a 1:1-cardinality update: at the next [`commit`],
    /// `new_rows` is appended as the replacement payload AND
    /// every row whose `_id` matched `predicate` at call entry
    /// is tombstoned.
    ///
    /// `predicate` is evaluated **immediately** against the
    /// current manifest snapshot; the resolved `_id` set + the
    /// IPC-encoded payload + a pre-reserved `_id` range + a
    /// preallocated superfile UUID are captured on the writer's
    /// pending-updates buffer. `commit()` drives each entry
    /// through its WAL pipeline (append → tombstone).
    ///
    /// **Cardinality:** `new_rows.num_rows()` MUST equal the
    /// predicate's resolved match count. Mismatch returns
    /// `CardinalityMismatch` and nothing is buffered.
    ///
    /// **Does NOT make the change durable.** Symmetric with
    /// buffered `append()` / `delete()`s.
    ///
    /// [`commit`]: SupertableWriter::commit
    pub fn update(
        &mut self,
        predicate: Expr,
        new_rows: RecordBatch,
    ) -> Result<PendingUpdate, MutationError> {
        // Pre-flight: storage attached.
        let _ = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?;

        // Schema check (no _id column on the user-facing path).
        if new_rows.schema().as_ref() != self.inner.options.schema.as_ref() {
            return Err(MutationError::SchemaMismatch(format!(
                "expected {:?}, got {:?}",
                self.inner.options.schema.fields(),
                new_rows.schema().fields()
            )));
        }

        // Resolve predicate against the manifest snapshot.
        // Captured-at-call semantics: appends still in this
        // writer's buffer don't count toward the match set.
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let target_ids = supertable
            .reader()
            .scan_ids_matching(predicate)
            .map_err(MutationError::PredicateEval)?;
        let matched = target_ids.len();
        if matched > MAX_TARGETS_PER_MUTATION {
            return Err(MutationError::MatchCountExceedsCap {
                matched,
                cap: MAX_TARGETS_PER_MUTATION,
            });
        }
        let new_row_count = new_rows.num_rows();
        if matched != new_row_count {
            return Err(MutationError::CardinalityMismatch {
                matched,
                new_rows: new_row_count,
            });
        }

        // Cardinality 0 is a structurally-impossible update —
        // the WAL pipeline needs `preallocated_superfile_id`
        // and at least one minted id span. We mint a wal_id so
        // the caller's `PendingUpdate` is comparable to the
        // non-zero shape, but skip buffering. The commit's
        // `CommitResult.outcomes` will reflect `matched: 0` if
        // the caller routes through the buffer instead.
        if matched == 0 {
            return Ok(PendingUpdate { matched: 0 });
        }

        // Reserve _id range + preallocate superfile id + mint
        // wal_id under one lock so the relative ordering is
        // deterministic and visible to any recovery replay.
        let (wal_id_value, minted_id_spans, preallocated_superfile_id) = {
            let idgen = self.inner.id_generator.lock().expect("idgen mutex");
            let spans = idgen
                .reserve_range(matched as u32)
                .into_iter()
                .map(|(first, last)| IdSpan {
                    first: RowId(first),
                    last: RowId(last),
                })
                .collect::<Vec<_>>();
            let wal_id_value = idgen.next_id();
            let preallocated = uuid::Uuid::new_v4();
            (wal_id_value, spans, preallocated)
        };

        // IPC-encode the new_rows batch + blake3. Doing this at
        // call time (rather than commit time) means the caller
        // can drop the `RecordBatch` immediately — the buffer
        // owns the bytes from here on.
        let ipc_bytes = encode_record_batch_ipc(&new_rows).map_err(|e| {
            MutationError::Storage(StorageError::Permanent {
                uri: "ipc encode".into(),
                source: Box::new(io::Error::other(e)),
            })
        })?;
        let content_hash = blake3::hash(&ipc_bytes).to_hex().to_string();

        self.pending_updates.push(PendingUpdateEntry {
            wal_id: WalId(wal_id_value),
            target_ids,
            preallocated_superfile_id,
            minted_id_spans,
            new_row_count: matched as u32,
            new_row_content_hash: content_hash,
            ipc_bytes,
        });
        Ok(PendingUpdate { matched })
    }

    /// Flush every buffered operation atomically (from the
    /// caller's perspective):
    ///
    /// 1. Pending appends → built into superfiles, manifest
    ///    swap committed.
    /// 2. Pending updates, in buffer order → per-op WAL
    ///    pipeline (append phase + tombstone phase).
    /// 3. Pending deletes, in buffer order → per-op WAL
    ///    pipeline (tombstone phase only).
    ///
    /// On success returns a [`CommitResult`] with one
    /// [`MutationStats`] per buffered mutation (in buffer
    /// order). On a mid-flush mutation failure surfaces
    /// [`CommitError::PartialCommit`] listing the WALs that DID
    /// land durably; the remaining buffered ops stay on the
    /// writer for retry, and the recovery sweep on the next
    /// supertable open completes the listed WALs if this
    /// process dies before retrying.
    ///
    /// [`CommitResult`]: crate::supertable::mutations::CommitResult
    /// [`MutationStats`]: crate::supertable::mutations::MutationStats
    /// [`CommitError::PartialCommit`]: crate::supertable::mutations::CommitError::PartialCommit
    pub fn commit(&mut self) -> Result<CommitResult, CommitError> {
        // Step 1: flush appends. A failure here is atomic —
        // the buffer is preserved and no mutation WAL has
        // landed yet.
        if !self.buffer.is_empty() {
            self.commit_appends_internal()
                .map_err(CommitError::AppendFlush)?;
        }

        let total_mutations = self.pending_updates.len() + self.pending_deletes.len();
        let mut committed_wal_ids: Vec<WalId> = Vec::with_capacity(total_mutations);
        let mut outcomes: Vec<MutationStats> = Vec::with_capacity(total_mutations);

        // Step 2: drive pending updates in buffer order. On
        // mid-loop failure, the failed entry is dropped (its
        // WAL may already be on storage; recovery sweep
        // completes it on the next open) and the unattempted
        // entries stay on `self.pending_updates` for retry.
        let mut updates_to_run = mem::take(&mut self.pending_updates);
        let mut update_cursor = 0usize;
        while update_cursor < updates_to_run.len() {
            let entry = &updates_to_run[update_cursor];
            match self.drive_one_update(entry) {
                Ok(outcome) => {
                    committed_wal_ids.push(outcome.wal_id);
                    outcomes.push(outcome);
                    update_cursor += 1;
                }
                Err(cause) => {
                    // Drop the failed entry + put the rest
                    // back on the buffer.
                    let remaining: Vec<PendingUpdateEntry> =
                        updates_to_run.split_off(update_cursor + 1);
                    self.pending_updates = remaining;
                    error!(
                        committed = outcomes.len(),
                        total = total_mutations,
                        error = %cause,
                        "partial commit: update failed mid-flush"
                    );
                    // Don't lose the not-yet-attempted deletes
                    // either — they stay where they were on
                    // self.pending_deletes (we hadn't taken
                    // them yet).
                    return Err(CommitError::PartialCommit {
                        committed_wal_ids,
                        committed: outcomes.len(),
                        total: total_mutations,
                        cause: Box::new(cause),
                    });
                }
            }
        }

        // Step 3: drive pending deletes in buffer order.
        let mut deletes_to_run = mem::take(&mut self.pending_deletes);
        let mut delete_cursor = 0usize;
        while delete_cursor < deletes_to_run.len() {
            let entry = &deletes_to_run[delete_cursor];
            match self.drive_one_delete(entry) {
                Ok(outcome) => {
                    committed_wal_ids.push(outcome.wal_id);
                    outcomes.push(outcome);
                    delete_cursor += 1;
                }
                Err(cause) => {
                    let remaining: Vec<PendingDeleteEntry> =
                        deletes_to_run.split_off(delete_cursor + 1);
                    self.pending_deletes = remaining;
                    error!(
                        committed = outcomes.len(),
                        total = total_mutations,
                        error = %cause,
                        "partial commit: delete failed mid-flush"
                    );
                    return Err(CommitError::PartialCommit {
                        committed_wal_ids,
                        committed: outcomes.len(),
                        total: total_mutations,
                        cause: Box::new(cause),
                    });
                }
            }
        }

        Ok(CommitResult {
            wal_ids: committed_wal_ids,
            outcomes,
        })
    }

    /// Drive one pending update entry through its full WAL
    /// pipeline. Returns the per-op outcome on success.
    fn drive_one_update(&self, entry: &PendingUpdateEntry) -> Result<MutationStats, MutationError> {
        let storage = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?
            .clone();

        let wal_doc = WalStateDoc {
            wal_id: entry.wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "writer.update()".into(),
            target_ids: entry.target_ids.iter().map(|&v| RowId(v)).collect(),
            new_row_count: Some(entry.new_row_count),
            new_row_content_hash: Some(entry.new_row_content_hash.clone()),
            preallocated_superfile_id: Some(entry.preallocated_superfile_id),
            minted_id_spans: entry.minted_id_spans.clone(),
            tombstone_progress: entry
                .target_ids
                .iter()
                .map(|&v| TombstoneEntry {
                    target_id: RowId(v),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                })
                .collect(),
        };

        let wal_store = WalStore::new(Arc::clone(&storage));
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let wal_id = entry.wal_id;
        let ipc_bytes = entry.ipc_bytes.clone();
        let drive = async move {
            wal_store
                .put_arrow(wal_id, ipc_bytes)
                .await
                .map_err(MutationError::WalStore)?;
            let etag = wal_store
                .create(&wal_doc)
                .await
                .map_err(MutationError::WalStore)?;
            let (_outcome, doc_after_append, etag_after_append) =
                pipeline::run_append_phase(&supertable, &wal_store, &wal_doc, &etag).await?;
            let (outcome, _post, _post_etag) = pipeline::run_tombstone_phase(
                &supertable,
                &wal_store,
                &doc_after_append,
                &etag_after_append,
            )
            .await?;
            let (n_t, n_nf) = match outcome {
                TombstonePhaseOutcome::Applied {
                    n_tombstoned,
                    n_not_found,
                }
                | TombstonePhaseOutcome::AlreadyComplete {
                    n_tombstoned,
                    n_not_found,
                } => (n_tombstoned, n_not_found),
            };
            // Best-effort cleanup of the WAL artifacts.
            let _ = wal_store.delete_arrow(wal_id).await;
            let _ = wal_store.delete_state(wal_id).await;
            Ok::<_, MutationError>((n_t, n_nf))
        };
        let (n_tombstoned, n_not_found) = bridge_on_runtime(drive, &self.inner.query_runtime())?;
        Ok(MutationStats {
            wal_id: entry.wal_id,
            matched: entry.target_ids.len(),
            n_tombstoned,
            n_not_found,
        })
    }

    /// Drive one pending delete entry through its tombstone
    /// phase. Returns the per-op outcome on success.
    fn drive_one_delete(&self, entry: &PendingDeleteEntry) -> Result<MutationStats, MutationError> {
        let storage = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?
            .clone();

        let wal_doc = WalStateDoc {
            wal_id: entry.wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "writer.delete()".into(),
            target_ids: entry.target_ids.iter().map(|&v| RowId(v)).collect(),
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: entry
                .target_ids
                .iter()
                .map(|&v| TombstoneEntry {
                    target_id: RowId(v),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                })
                .collect(),
        };

        let wal_store = WalStore::new(Arc::clone(&storage));
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let wal_id = entry.wal_id;
        // The hidden vector-index cells are not rewritten on a user delete, so
        // the deleted rows stay physically present in them. Record the resolved
        // user `_id`s into the hidden index's resident deleted-set so vector
        // search drops them in memory (zero per-cell tombstone GETs).
        let hidden_inner = self
            .inner
            .vector_index_table
            .as_ref()
            .map(|vit| Arc::clone(vit.inner()));
        let deleted_ids: Vec<i128> = entry.target_ids.clone();
        let drive = async move {
            let etag = wal_store
                .create(&wal_doc)
                .await
                .map_err(MutationError::WalStore)?;
            let (outcome, _post, _post_etag) =
                pipeline::run_tombstone_phase(&supertable, &wal_store, &wal_doc, &etag).await?;
            let (n_t, n_nf) = match outcome {
                TombstonePhaseOutcome::Applied {
                    n_tombstoned,
                    n_not_found,
                }
                | TombstonePhaseOutcome::AlreadyComplete {
                    n_tombstoned,
                    n_not_found,
                } => (n_tombstoned, n_not_found),
            };
            let _ = wal_store.delete_state(wal_id).await;
            if let Some(hi) = hidden_inner
                && let Err(e) = record_hidden_deleted_ids(&hi, &deleted_ids).await
            {
                tracing::warn!(
                    "supertable: hidden vector-index deleted-set record failed: {e} \
                     (user-table delete is durable; vector search may transiently \
                     return deleted rows until the next successful record)"
                );
            }
            Ok::<_, MutationError>((n_t, n_nf))
        };
        let (n_tombstoned, n_not_found) = bridge_on_runtime(drive, &self.inner.query_runtime())?;
        Ok(MutationStats {
            wal_id: entry.wal_id,
            matched: entry.target_ids.len(),
            n_tombstoned,
            n_not_found,
        })
    }

    /// [`SupertableWriter::commit`] calls this first before
    /// driving pending mutations.
    ///
    /// Rows are balanced evenly across shards regardless of the
    /// caller's `append()` cadence — many small appends followed by
    /// one `commit` produce the same shard layout as one large append.
    fn commit_appends_internal(&mut self) -> Result<(), BuildError> {
        if self.buffer.is_empty() {
            return Ok::<(), BuildError>(());
        }
        let buffer = mem::take(&mut self.buffer);
        self.buffer_bytes = 0;

        // Phase A — bootstrap the global cell grid from the FIRST committed batch
        // into THIS (the user) table's manifest, which is the source of truth for
        // the grid. Gated on VECTORS being in the input (a vector column), not on
        // any sibling table: the packed user build below and the query both read
        // the grid from here, and it persists with this commit (`ManifestSnapshot::update`
        // carries `global_vector_index` through). A hidden cell-index sibling, when
        // present, gets the grid as a derived copy at drain time — but its absence
        // must not change how user superfiles are laid out. Idempotent: only
        // trains while absent.
        if self
            .inner
            .manifest
            .load()
            .get_global_vector_index()
            .is_none()
            && !buffer.is_empty()
            && let Some(vc) = self.inner.options.vector_columns.first()
            && let Some(grid) = bootstrap_centroids_from_batch(
                &buffer,
                vc.dim,
                super::handle::global_vector_cell_count(),
            )
        {
            let index = super::manifest::GlobalVectorIndex {
                column: vc.column.clone(),
                grid,
            };
            self.inner.manifest.store(Arc::new(
                self.inner.manifest.load().with_global_vector_index(index),
            ));
        }

        let total_rows: usize = buffer.iter().map(|b| b.scalar.num_rows()).sum();
        if total_rows == 0 {
            return Ok::<(), BuildError>(());
        }

        // Vector commit: same row-shard fanout as the legacy path. Each writer
        // assigns its rows to cells, calls drain's pack
        // (`build_merged_subsection_from_fp32` → materialized pack: sampled
        // fine k-means + Sq8), overlapped with Parquet+FTS, then splices IVF
        // blobs into the superfile and publishes. Drain does not write/S3 on
        // this path. No slow CAS.
        if !self.inner.options.vector_columns.is_empty() {
            let commit_t0 = time::Instant::now();
            let Some(global) = self.inner.manifest.load().get_global_vector_index() else {
                return Err(BuildError::Store(
                    "vector columns present but global cell grid missing after Phase A".into(),
                ));
            };
            let metric = self
                .inner
                .options
                .vector_columns
                .first()
                .map(|vc| vc.metric)
                .unwrap_or(Metric::L2Sq);
            let (outputs, cell_hints) =
                commit_shards_via_drain(buffer, &self.inner, &global.grid, metric)?;
            let build_elapsed = commit_t0.elapsed();
            let output_bytes: usize = outputs.iter().map(|output| output.bytes.len()).sum();
            let user_batch = prepare_user_superfile_batch(&self.inner, outputs, cell_hints)?;
            let prepare_elapsed = commit_t0.elapsed().saturating_sub(build_elapsed);
            let data_put_bytes: usize = user_batch
                .pending_storage_writes
                .iter()
                .map(|(_, bytes)| bytes.len())
                .sum();
            let publish_t0 = time::Instant::now();
            bridge_on_runtime(
                persist_superfile_publish_batch_async(&self.inner, user_batch),
                &self.inner.query_runtime(),
            )?;
            if crate::storage::io_counters::timeline_enabled() {
                eprintln!(
                    "[supertable commit] build {:.1}ms ({:.1} MiB output) + prepare {:.1}ms + \
                     publish {:.1}ms ({:.1} MiB data PUT)",
                    build_elapsed.as_secs_f64() * 1e3,
                    output_bytes as f64 / (1u64 << 20) as f64,
                    prepare_elapsed.as_secs_f64() * 1e3,
                    publish_t0.elapsed().as_secs_f64() * 1e3,
                    data_put_bytes as f64 / (1u64 << 20) as f64,
                );
            }
            if self.inner.options.storage.is_some() {
                schedule_background_storage_reclaim(Arc::clone(&self.inner));
            }
            return Ok(());
        }

        let writer_pool = Arc::clone(&self.inner.options.writer_pool);
        let n_threads = writer_pool.current_num_threads().max(1);
        let n_shards = n_threads.min(total_rows);

        let vector_dims: Vec<usize> = self
            .inner
            .options
            .vector_columns
            .iter()
            .map(|vc| vc.dim)
            .collect();
        // VectorCell strategy: pre-shard by nearest centroid instead of
        // round-robin. Each shard becomes one superfile in its cell-partition.
        let (shards, cell_hints): (Vec<Vec<BufferedBatch>>, Vec<Option<u32>>) =
            if let Some(PartitionStrategy::VectorCell { ref clusters, .. }) =
                self.inner.options.partition_strategy
            {
                let metric = self
                    .inner
                    .options
                    .vector_columns
                    .first()
                    .map(|vc| vc.metric)
                    .unwrap_or(Metric::L2Sq);
                if clusters.n_cent > 0 && clusters.dim > 0 {
                    // Run on the build pool: `split_buffer_by_vector_cell` →
                    // `assign_rows` is a CPU wave (per-row nearest-cell scoring)
                    // and must dispatch to `writer_pool`, not the global rayon
                    // pool, per the rayon-owns-CPU concurrency contract.
                    let cell_shards = writer_pool
                        .install(|| split_buffer_by_vector_cell(buffer, clusters, metric, 0));
                    let hints: Vec<Option<u32>> = cell_shards
                        .iter()
                        .map(|(cell_id, _)| Some(*cell_id))
                        .collect();
                    let shards: Vec<Vec<BufferedBatch>> = cell_shards
                        .into_iter()
                        .map(|(_, batches)| batches)
                        .collect();
                    (shards, hints)
                } else {
                    let shards = split_buffer_into_row_shards(buffer, n_shards, &vector_dims);
                    let hints = vec![None; shards.len()];
                    (shards, hints)
                }
            } else {
                let shards = split_buffer_into_row_shards(buffer, n_shards, &vector_dims);
                let hints = vec![None; shards.len()];
                (shards, hints)
            };

        // Parallel create: user superfile build + hidden incoming build
        // share one writer-pool install (rayon::join, no nested install).
        // Parallel publish: user + hidden manifest/storage commits overlap.
        let user_inner = Arc::clone(&self.inner);
        let user_options = Arc::clone(&self.inner.options);
        // A/B knob (`vector.user_centroids: global`): build user superfiles
        // aligned to the GLOBAL cell grid (cluster c == cell c) instead of local
        // k-means — so the splice/kmeans drain routes cluster c → cell c
        // doc-correctly. The grid is read from THIS table's manifest (bootstrapped
        // above, Phase A). Default `local` is unchanged.
        let user_global_centroids: Option<std::sync::Arc<[f32]>> =
            if config::global().vector.user_centroids == CentroidAlignment::Global {
                self.inner
                    .manifest
                    .load()
                    .get_global_vector_index()
                    .filter(|g| g.grid.n_cent > 0 && g.grid.dim > 0)
                    .map(|g| g.grid.to_fp32().into())
            } else {
                None
            };

        // Phase B: user-only build + publish. No hidden incoming build/publish;
        // the hidden cell index is drained later straight from these user
        // superfiles, and pre-drain queries fall back to them.
        let outputs = fanout_shards(&writer_pool, &shards, |slice| {
            build_one_shard_with_layout(
                slice.as_slice(),
                &user_options,
                user_options.vector_layout,
                user_global_centroids.clone(),
            )
        })?;
        let superfiles = outputs.len();
        let user_batch = prepare_user_superfile_batch(&self.inner, outputs, cell_hints)?;
        bridge_on_runtime(
            persist_superfile_publish_batch_async(&user_inner, user_batch),
            &self.inner.query_runtime(),
        )?;
        if self.inner.options.storage.is_some() {
            schedule_background_storage_reclaim(Arc::clone(&self.inner));
        }
        debug!(superfiles, "published appended superfiles");
        Ok(())
    }
}

impl Drop for SupertableWriter {
    fn drop(&mut self) {
        // Release the writer slot. Uncommitted buffer is
        // intentionally lost — callers must invoke commit()
        // explicitly to publish.
        self.inner
            .writer_outstanding
            .store(false, Ordering::Release);
    }
}

/// Output of one rayon shard worker.
///
/// FTS + vector summaries are derived in `prepare_user_superfile_batch` from
/// the cached `SuperfileReader` (cheaper than re-walking buffered
/// batches). `scalar_stats` is computed here, before the buffer is
/// dropped, since the post-store `SuperfileReader` only exposes
/// parquet row groups — Arrow batch min/max would require a full
/// re-decode through DataFusion or parquet-rs's stats reader.
pub struct ShardOutput {
    bytes: Bytes,
    n_docs: u64,
    /// `id_min` / `id_max`: only meaningful when `n_docs > 0`.
    /// For a 0-doc shard (empty slice — shouldn't happen given
    /// chunk sizing, but defensive), both are 0. Stored as
    /// `i128` to carry the 128-bit Snowflake-shaped ids
    /// produced by [`crate::supertable::utils::idgen::IdGenerator`].
    id_min: i128,
    id_max: i128,
    /// Per-scalar-column min/max for skip pruning. Computed from
    /// the shard's `BufferedBatch` slice via Arrow per-type
    /// aggregate kernels; types whose ordering isn't well-defined
    /// (FixedSizeList, struct, etc.) are absent and treated as
    /// "can't prune" by the skip planner.
    scalar_stats: HashMap<String, ScalarStatsAgg>,
}

impl ShardOutput {
    pub fn new_with_params(
        bytes: Bytes,
        n_docs: u64,
        id_min: i128,
        id_max: i128,
        scalar_stats: HashMap<String, ScalarStatsAgg>,
    ) -> Self {
        Self {
            bytes,
            n_docs,
            id_min,
            id_max,
            scalar_stats,
        }
    }
}

/// Build one superfile from one slice of buffered batches with an explicit
/// vector layout override. Runs on a rayon worker thread inside the writer
/// pool's `install`. The commit path always passes an explicit layout +
/// optional global centroids.
fn build_one_shard_with_layout(
    slice: &[BufferedBatch],
    options: &SupertableOptions,
    vector_layout: crate::superfile::vector::layout::VectorLayout,
    provided_centroids: Option<std::sync::Arc<[f32]>>,
) -> Result<ShardOutput, BuildError> {
    let mut builder = SuperfileBuilder::new(
        options
            .builder_options()
            .with_vector_layout(vector_layout)
            .with_vector_centroids(provided_centroids),
    )?;

    let scalar_schema = options.scalar_schema();
    // The supertable always prepends the id column at index 0
    // via `SupertableOptions::scalar_schema`, so we can skip
    // the schema lookup here.
    let id_idx = 0;

    let mut id_min = i128::MAX;
    let mut id_max = i128::MIN;
    let mut n_docs: u64 = 0;

    for buffered in slice {
        let id_col = buffered
            .scalar
            .column(id_idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                BuildError::IdColumnWrongType(
                    options.id_column.clone(),
                    "<id column not Decimal128 at runtime>".to_string(),
                )
            })?;
        for i in 0..id_col.len() {
            let v = id_col.value(i);
            id_min = id_min.min(v);
            id_max = id_max.max(v);
        }
        n_docs += id_col.len() as u64;

        // Float32Array::values() returns &ScalarBuffer<f32>;
        // ScalarBuffer derefs to &[f32], so AsRef does the slice
        // view without a copy.
        let vector_slices: Vec<&[f32]> = buffered
            .vectors
            .iter()
            .map(|fa| fa.values().as_ref())
            .collect();
        builder.add_batch(&buffered.scalar, &vector_slices)?;
    }

    // Compute per-scalar-column min/max BEFORE moving `slice`'s
    // batches into the builder via `finish`. We pass references —
    // `from_batches` doesn't take ownership.
    let scalar_batches: Vec<&RecordBatch> = slice.iter().map(|b| &b.scalar).collect();
    let scalar_stats = ScalarStatsAgg::from_batches(&scalar_schema, &scalar_batches);

    let bytes = Bytes::from(builder.finish()?);

    let (id_min, id_max) = if n_docs == 0 {
        (0, 0)
    } else {
        (id_min, id_max)
    };

    Ok(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    })
}

/// Pull the superfile's `(total_size, vec_off/len, fts_off/len)`
/// out of the freshly-written parquet KV metadata so the manifest
/// can carry it forward as a [`SubsectionOffsets`]. Returns `None`
/// if the bytes don't parse — that path falls back to the
/// 2-RTT cold open shape rather than failing the publish.
pub(crate) fn build_subsection_offsets(bytes: &Bytes) -> Option<SubsectionOffsets> {
    let kvs = read_kv_metadata(bytes).ok()?;
    let get = |k: &str| -> Option<u64> { kvs.get(k).and_then(|s| s.parse::<u64>().ok()) };
    let vec = match (get(kv::VEC_OFFSET), get(kv::VEC_LENGTH)) {
        (Some(o), Some(l)) if l > 0 => Some((o, l)),
        _ => None,
    };
    let fts = match (get(kv::FTS_OFFSET), get(kv::FTS_LENGTH)) {
        (Some(o), Some(l)) if l > 0 => Some((o, l)),
        _ => None,
    };
    let total_size = bytes.len() as u64;
    // Derive the layout from the `kvs` already parsed above rather than
    // re-reading the footer via `read_vector_layout_from_bytes`.
    let layout = vector_layout_from_kv(&kvs);
    if layout == VectorLayout::CellPosting {
        // Cell-posting hidden superfiles are read in bulk (a full-cell scan of
        // the contiguous vec blob) and served resident from the disk cache.
        // Staging their bytes into the manifest `open_blob` would replicate the
        // entire vector index into the manifest — its size would grow with the
        // whole dataset (memory + cold-load GET cost), since the open overlay
        // captures each superfile's vec blob *and* parquet tail. Skip the
        // inline overlay entirely; the vec subsection is fetched on demand
        // (and cached) via `fetch_cell_posting_blob`. Offsets are still carried
        // so that fetch knows where to read.
        return Some(SubsectionOffsets {
            total_size,
            vec,
            fts,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        });
    }
    let vec_open_ranges = vec
        .and_then(|(off, len)| vector_open_ranges(bytes, off, len))
        .unwrap_or_default();
    let fts_open_ranges = fts
        .and_then(|(off, len)| fts_open_ranges(bytes, off, len))
        .unwrap_or_default();

    // capture the open-time batch bytes (parquet
    // footer tail + vector open ranges + FTS open ranges) so the
    // reader can resolve a superfile's open metadata straight from
    // the manifest part, issuing zero per-superfile open GETs.
    let open_blob = build_open_blob(bytes, total_size, &vec_open_ranges, &fts_open_ranges);

    Some(SubsectionOffsets {
        total_size,
        vec,
        fts,
        vec_open_ranges,
        fts_open_ranges,
        open_blob,
    })
}

/// Slice the bytes for the superfile's open-time batch out of the
/// freshly-written superfile so the manifest can carry them
/// inline. Mirrors the cold-fetch open batch in
/// `DiskCacheStore::cold_fetch_lazy_with_hints`: the parquet
/// footer tail (matching the 64 KiB speculation length) plus each
/// vector / FTS open range. Returns `(absolute_offset, bytes)`
/// tuples; an empty `Vec` disables the inline-open fast path for
/// this superfile.
fn build_open_blob(
    bytes: &Bytes,
    total_size: u64,
    vec_open_ranges: &[(u64, u64)],
    fts_open_ranges: &[(u64, u64)],
) -> Vec<(u64, Vec<u8>)> {
    // Must match `cold_fetch_lazy_with_hints`'s parquet tail
    // speculation length so the overlay covers `source.tail()`.
    const PARQUET_TAIL_SPEC: u64 = 64 * 1024;
    let mut blob: Vec<(u64, Vec<u8>)> =
        Vec::with_capacity(1 + vec_open_ranges.len() + fts_open_ranges.len());

    let parquet_tail_len = PARQUET_TAIL_SPEC.min(total_size);
    let parquet_tail_start = total_size.saturating_sub(parquet_tail_len);
    let slice = |off: u64, len: u64| -> Option<Vec<u8>> {
        let start = off as usize;
        let end = start.checked_add(len as usize)?;
        bytes.get(start..end).map(|s| s.to_vec())
    };
    if parquet_tail_len > 0 {
        match slice(parquet_tail_start, parquet_tail_len) {
            Some(b) => blob.push((parquet_tail_start, b)),
            None => return Vec::new(),
        }
    }
    for &(off, len) in vec_open_ranges.iter().chain(fts_open_ranges.iter()) {
        match slice(off, len) {
            Some(b) => blob.push((off, b)),
            // A range we can't satisfy means the capture is
            // inconsistent; disable the fast path rather than ship
            // a partial overlay.
            None => return Vec::new(),
        }
    }
    blob
}

fn vector_open_ranges(bytes: &Bytes, off: u64, len: u64) -> Option<Vec<(u64, u64)>> {
    let start = off as usize;
    let end = start.checked_add(len as usize)?;
    let blob = bytes.get(start..end)?;
    if blob.len() < OUTER_HEADER_SIZE + CRC_BYTES {
        return None;
    }
    let version =
        read_u32_le(blob.get(outer_hdr::VERSION_OFF..outer_hdr::VERSION_OFF + U32_BYTES)?);
    if version == crate::superfile::format::vec::VERSION_MULTI_CELL {
        return vector_open_ranges_multi_cell(blob, off);
    }
    let n_columns =
        read_u32_le(blob.get(outer_hdr::N_COLUMNS_OFF..outer_hdr::N_COLUMNS_OFF + U32_BYTES)?)
            as usize;
    let dir_offset =
        read_u64_le(blob.get(outer_hdr::DIR_OFFSET_OFF..outer_hdr::DIR_OFFSET_OFF + U64_BYTES)?)
            as usize;
    let dir_size = n_columns.checked_mul(DIR_ENTRY_SIZE)?;
    let dir_end = dir_offset.checked_add(dir_size)?.checked_add(CRC_BYTES)?;
    let dir = blob.get(dir_offset..dir_offset + dir_size)?;

    let mut ranges = vec![(off + dir_offset as u64, (dir_size + CRC_BYTES) as u64)];
    ranges.push((off, OUTER_HEADER_SIZE as u64));
    for i in 0..n_columns {
        let entry = i * DIR_ENTRY_SIZE;
        let subsection_off = read_u64_le(dir.get(
            entry + dir_entry::SUBSECTION_OFF_OFF
                ..entry + dir_entry::SUBSECTION_OFF_OFF + U64_BYTES,
        )?) as usize;
        let subsection_len = read_u64_le(dir.get(
            entry + dir_entry::SUBSECTION_LEN_OFF
                ..entry + dir_entry::SUBSECTION_LEN_OFF + U64_BYTES,
        )?) as usize;
        let codec_meta_off = read_u32_le(dir.get(
            entry + dir_entry::CODEC_META_OFF_OFF
                ..entry + dir_entry::CODEC_META_OFF_OFF + U32_BYTES,
        )?) as usize;
        let codec_meta_size = read_u32_le(dir.get(
            entry + dir_entry::CODEC_META_SIZE_OFF
                ..entry + dir_entry::CODEC_META_SIZE_OFF + U32_BYTES,
        )?) as usize;
        if subsection_off.checked_add(SUB_HEADER_SIZE)? > blob.len()
            || subsection_off.checked_add(subsection_len)? > blob.len()
        {
            return None;
        }
        ranges.push((off + subsection_off as u64, SUB_HEADER_SIZE as u64));
        let sub = blob.get(subsection_off..subsection_off + subsection_len)?;
        let centroids_off = read_u64_le(
            sub.get(sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + U64_BYTES)?,
        ) as usize;
        let cluster_idx_off = read_u64_le(
            sub.get(sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES)?,
        ) as usize;
        let cluster_idx_end = cluster_idx_off.checked_add(
            CLUSTER_IDX_ENTRY_BYTES
                * read_u32_le(dir.get(
                    entry + dir_entry::N_CENT_OFF..entry + dir_entry::N_CENT_OFF + U32_BYTES,
                )?) as usize,
        )?;
        if centroids_off < SUB_HEADER_SIZE || cluster_idx_end > subsection_len {
            return None;
        }
        // Stage only [cluster_idx .. cluster_idx_end]. The fp32 centroids that
        // precede it are read solely by the rare fallback per-segment `nprobe`
        // path (segments lacking a manifest cluster summary), which range-GETs
        // them from the superfile on demand — they remain on disk. The hot
        // cluster-probe path reads only `cluster_idx`, so keeping centroids out
        // of the open_blob makes the manifest-inline open footprint independent
        // of `n_cent` (centroids are ~99% of it at high `n_cent`).
        ranges.push((
            off + subsection_off as u64 + cluster_idx_off as u64,
            (cluster_idx_end - cluster_idx_off) as u64,
        ));
        if codec_meta_size > 0 {
            let meta_end = codec_meta_off.checked_add(codec_meta_size)?;
            if meta_end > subsection_len {
                return None;
            }
        }
    }
    if dir_end > blob.len() {
        return None;
    }
    Some(merge_ranges(ranges))
}

/// Open-time ranges for a v2 multi-cell vector blob: outer header, cell
/// directory, and each cell's open-time region (sub-header through
/// `per_cluster_blocks_off`).
fn vector_open_ranges_multi_cell(blob: &[u8], off: u64) -> Option<Vec<(u64, u64)>> {
    use crate::superfile::format::vec::U64_BYTES;
    let n_cells =
        read_u32_le(blob.get(outer_hdr::N_CELLS_OFF..outer_hdr::N_CELLS_OFF + U32_BYTES)?) as usize;
    let dir_offset =
        read_u64_le(blob.get(outer_hdr::DIR_OFFSET_OFF..outer_hdr::DIR_OFFSET_OFF + U64_BYTES)?)
            as usize;
    let dir_size = n_cells.checked_mul(CELL_DIR_ENTRY_SIZE)?;
    let dir_end = dir_offset.checked_add(dir_size)?.checked_add(CRC_BYTES)?;
    if dir_end > blob.len() {
        return None;
    }
    let dir = blob.get(dir_offset..dir_offset + dir_size)?;
    let mut ranges = vec![
        (off, OUTER_HEADER_SIZE as u64),
        (off + dir_offset as u64, (dir_size + CRC_BYTES) as u64),
    ];
    for i in 0..n_cells {
        let entry = i * CELL_DIR_ENTRY_SIZE;
        let subsection_off = read_u64_le(dir.get(
            entry + cell_dir_entry::SUBSECTION_OFF_OFF
                ..entry + cell_dir_entry::SUBSECTION_OFF_OFF + U64_BYTES,
        )?) as usize;
        let subsection_len = read_u64_le(dir.get(
            entry + cell_dir_entry::SUBSECTION_LEN_OFF
                ..entry + cell_dir_entry::SUBSECTION_LEN_OFF + U64_BYTES,
        )?) as usize;
        if subsection_off.checked_add(SUB_HEADER_SIZE)? > blob.len()
            || subsection_off.checked_add(subsection_len)? > blob.len()
        {
            return None;
        }
        let sub = blob.get(subsection_off..subsection_off + subsection_len)?;
        let per_cluster_blocks_off = read_u64_le(sub.get(
            sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF..sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF + U64_BYTES,
        )?) as usize;
        if per_cluster_blocks_off < SUB_HEADER_SIZE || per_cluster_blocks_off > subsection_len {
            return None;
        }
        ranges.push((off + subsection_off as u64, per_cluster_blocks_off as u64));
    }
    Some(merge_ranges(ranges))
}

fn fts_open_ranges(bytes: &Bytes, off: u64, len: u64) -> Option<Vec<(u64, u64)>> {
    let start = off as usize;
    let end = start.checked_add(len as usize)?;
    let blob = bytes.get(start..end)?;
    if blob.len() < FTS_HEADER_SIZE {
        return None;
    }
    let postings_offset =
        read_u64_le(blob.get(hdr::POSTINGS_OFFSET_OFF..hdr::POSTINGS_OFFSET_OFF + U64_BYTES)?)
            as usize;
    let doc_lengths_offset =
        read_u64_le(blob.get(hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES)?)
            as usize;
    if postings_offset > blob.len()
        || doc_lengths_offset > blob.len()
        || postings_offset > doc_lengths_offset
    {
        return None;
    }
    Some(merge_ranges(vec![
        (off, postings_offset as u64),
        (
            off + doc_lengths_offset as u64,
            (blob.len() - doc_lengths_offset) as u64,
        ),
    ]))
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|&(_, len)| len > 0);
    ranges.sort_unstable_by_key(|&(off, _)| off);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (off, len) in ranges {
        let end = off + len;
        if let Some((last_off, last_len)) = merged.last_mut() {
            let last_end = *last_off + *last_len;
            if off <= last_end {
                *last_len = (*last_len).max(end - *last_off);
                continue;
            }
        }
        merged.push((off, len));
    }
    merged
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("u32 slice length"))
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("u64 slice length"))
}

/// Per-shard publish artifacts produced in parallel before the
/// serial manifest swap. One entry per non-empty shard.
pub(crate) struct PreparedSuperfile {
    pub(crate) entry: Arc<SuperfileEntry>,
    /// Bytes destined for the in-memory superfile store. `Some` on
    /// the in-memory-only path and the storage-without-cache
    /// path; `None` on the cache-attached path (the disk cache
    /// hydrates lazily from storage).
    pub(crate) bytes_for_store: Option<(SuperfileUri, Bytes)>,
    pub(crate) bytes_for_storage: Option<(SuperfileUri, Bytes)>,
    pub(crate) bytes_for_cache: Option<(SuperfileUri, Bytes)>,
}

impl PreparedSuperfile {
    /// Open a `SuperfileReader` directly on this superfile's bytes.
    /// Returns `None` if no bytes are held (cache-attached path with
    /// no prepopulation — bytes went to storage only).
    #[cfg(test)]
    pub(crate) fn open_reader(&self) -> Option<Result<SuperfileReader, ReadError>> {
        let bytes = self
            .bytes_for_store
            .as_ref()
            .or(self.bytes_for_storage.as_ref())
            .or(self.bytes_for_cache.as_ref())
            .map(|(_, b)| b.clone())?;
        Some(SuperfileReader::open(bytes))
    }
}

/// Build the per-shard publish artifacts: open a `SuperfileReader`
/// on the shard bytes, derive FTS + vector summaries, and decide
/// the bytes-disposition triplet. Pure per-shard work — no shared
/// mutable state, safe to run in parallel across shards.
pub(super) fn prepare_superfile(
    inner: &SupertableInner,
    shard: ShardOutput,
) -> Result<Option<PreparedSuperfile>, BuildError> {
    prepare_superfile_with_uri(inner, shard, None)
}

pub(super) fn prepare_superfile_with_uri(
    inner: &SupertableInner,
    shard: ShardOutput,
    reuse_uri: Option<SuperfileUri>,
) -> Result<Option<PreparedSuperfile>, BuildError> {
    if shard.n_docs == 0 {
        return Ok(None);
    }

    let uri = reuse_uri.unwrap_or_else(SuperfileUri::new_v4);

    let bytes_for_storage = inner.options.storage.is_some().then(|| shard.bytes.clone());
    let cache_attached = inner.options.disk_cache.is_some() && inner.options.storage.is_some();
    // `bytes_for_store` (in-memory tier) is gated only on cache attachment —
    // a cache-attached producer keeps superfile bytes out of the unbounded
    // in-memory store regardless of whether we pre-populate the disk cache.
    let bytes_for_store = (!cache_attached).then(|| shard.bytes.clone());
    // Always warm-fill the disk cache when attached: commits are durable in
    // object storage first, then mirrored locally so maintenance/compaction
    // can merge from mmap-resident bytes without re-fetching whole objects.
    let bytes_for_cache = cache_attached.then(|| shard.bytes.clone());

    // Open the reader directly on shard bytes (not via the
    // in-memory `SuperfileReaderCache`). This lets the cache-attached
    // path skip the in-memory tier entirely — the bytes can go
    // straight to object storage without a RAM detour, which is
    // what removes the 100GB OOM trap (the in-memory cache doesn't
    // evict, so a long-running writer with cache + storage would
    // otherwise accumulate every superfile's bytes in RAM forever).
    let reader =
        SuperfileReader::open_with(shard.bytes.clone(), inner.options.superfile_open_options())
            .map_err(|e| BuildError::Store(format!("opening superfile for summary: {e}")))?;

    let mut fts_summary: HashMap<String, FtsSummaryAgg> = HashMap::new();
    if let Some(fts_reader) = reader.fts() {
        for fc in &inner.options.fts_columns {
            let terms = fts_reader
                .iter_column_terms(&fc.column)
                .expect("FST bytes valid: superfile just built");
            let n_terms_distinct = terms.len() as u32;
            let (min_term, max_term) = match (terms.first(), terms.last()) {
                (Some(min), Some(max)) => (min.clone(), max.clone()),
                _ => (Vec::new(), Vec::new()),
            };
            let mut bloom_builder = BloomBuilder::new();
            for term in &terms {
                bloom_builder.insert(term);
            }
            fts_summary.insert(
                fc.column.clone(),
                FtsSummaryAgg::new_with_params(
                    bloom_builder.finish(),
                    n_terms_distinct,
                    (min_term, max_term),
                ),
            );
        }
    }

    let mut vector_summary: HashMap<String, VectorSummary> = HashMap::new();
    if let Some(vec_reader) = reader.vec() {
        for vc in &inner.options.vector_columns {
            if let Some(centroid) = vec_reader.summary(&vc.column) {
                // Stage the per-cluster centroids (Sq8) into the
                // manifest so a query can rank this superfile's clusters
                // globally without opening the superfile.
                let cells = vec_reader
                    .cluster_centroids_by_cell(&vc.column)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(cell_id, n_cent, dim, fp32, counts)| CellVectorSummary {
                        cell_id,
                        clusters: ClusterCentroids::from_fp32(n_cent, dim, &fp32, counts),
                    })
                    .collect();
                vector_summary.insert(vc.column.clone(), VectorSummary { centroid, cells });
            }
        }
    }

    // capture `(total_size, vec_off/len, fts_off/len)`
    // from the freshly-written bytes' parquet KV metadata. Caching
    // these on the manifest lets `DiskCacheStore::reader_with_hints`
    // fire the parquet-footer, vector, and FTS subsection GETs in
    // parallel on cold open (1 RTT instead of 2 sequential).
    let subsection_offsets = build_subsection_offsets(&shard.bytes);
    let vector_layout = read_vector_layout_from_bytes(&shard.bytes);
    if vector_layout == VectorLayout::CellPosting
        && subsection_offsets.as_ref().and_then(|o| o.vec).is_none()
    {
        let kvs = crate::superfile::format::footer::read_kv_metadata(shard.bytes.as_ref())
            .map(|kvs| kvs.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        return Err(BuildError::Store(format!(
            "cell-posting superfile missing inf.vec offset/length; kv_keys={kvs:?}"
        )));
    }

    let entry = Arc::new(SuperfileEntry {
        // Hidden cell superfile; stamped by the hidden manifest's own
        // `update`. Irrelevant to the user-side drain watermark.
        birth_version: 0,
        superfile_id: uuid::Uuid::new_v4(),
        uri,
        n_docs: shard.n_docs,
        id_min: shard.id_min,
        id_max: shard.id_max,
        scalar_stats: shard.scalar_stats,
        fts_summary,
        vector_summary,
        // Partition assignment populated by the per-shard
        // `PartitionStrategy` wiring elsewhere; superfiles
        // emitted here remain unpartitioned (default).
        partition_key: Vec::new(),
        partition_hint: None,
        subsection_offsets,
        vector_layout,
    });

    Ok(Some(PreparedSuperfile {
        entry,
        bytes_for_store: bytes_for_store.map(|b| (uri, b)),
        bytes_for_storage: bytes_for_storage.map(|b| (uri, b)),
        bytes_for_cache: bytes_for_cache.map(|b| (uri, b)),
    }))
}

/// Insert each shard's bytes into the superfile store, derive
/// per-superfile summaries from the stored `SuperfileReader`, and
/// publish all entries in one `ArcSwap` of the manifest.
///
/// Per-shard work (reader open, FTS bloom build, vector summary,
/// `SuperfileEntry` construction) runs in parallel across the
/// writer pool — for an FTS supertable the bloom build alone is
/// O(n_terms_distinct) per FTS column per shard, which at 10M
/// docs × 4 superfiles is the dominant cost. ManifestSnapshot swap +
/// storage write-through stay serial after the join.
fn finish_superfile_entry(
    entry: Arc<SuperfileEntry>,
    hint: Option<u32>,
) -> Result<Arc<SuperfileEntry>, BuildError> {
    let old = entry.as_ref();
    let staged = SuperfileEntry {
        birth_version: old.birth_version,
        superfile_id: old.superfile_id,
        uri: old.uri,
        n_docs: old.n_docs,
        id_min: old.id_min,
        id_max: old.id_max,
        scalar_stats: old.scalar_stats.clone(),
        fts_summary: old.fts_summary.clone(),
        vector_summary: old.vector_summary.clone(),
        // Partition key is now stamped by manifest update at commit time.
        partition_key: Vec::new(),
        partition_hint: hint.or(old.partition_hint),
        subsection_offsets: old.subsection_offsets.clone(),
        vector_layout: old.vector_layout,
    };
    Ok(Arc::new(staged))
}

/// Collected superfile entries + pending storage/cache writes for one publish.
struct SuperfilePublishBatch {
    new_entries: Vec<Arc<SuperfileEntry>>,
    to_remove: Vec<Arc<SuperfileEntry>>,
    pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    pending_cache_inserts: Vec<(SuperfileUri, Bytes)>,
}

fn collect_prepared_superfiles(
    inner: &SupertableInner,
    prepared: Vec<PreparedSuperfile>,
) -> Result<SuperfilePublishBatch, BuildError> {
    let mut new_entries: Vec<Arc<SuperfileEntry>> = Vec::with_capacity(prepared.len());
    let mut pending_storage_writes: Vec<(SuperfileUri, Bytes)> = Vec::new();
    let mut pending_cache_inserts: Vec<(SuperfileUri, Bytes)> = Vec::new();
    for p in prepared {
        if let Some((uri, b)) = p.bytes_for_store {
            inner
                .options
                .store
                .insert(uri, b)
                .map_err(|e| BuildError::Store(e.to_string()))?;
        }
        if let Some(t) = p.bytes_for_storage {
            pending_storage_writes.push(t);
        }
        if let Some(t) = p.bytes_for_cache {
            pending_cache_inserts.push(t);
        }
        new_entries.push(p.entry);
    }
    Ok(SuperfilePublishBatch {
        new_entries,
        to_remove: Vec::new(),
        pending_storage_writes,
        pending_cache_inserts,
    })
}

fn prepare_user_superfile_batch_in_scope(
    inner: &SupertableInner,
    outputs: Vec<ShardOutput>,
    hints: Vec<Option<u32>>,
) -> Result<SuperfilePublishBatch, BuildError> {
    let prepared: Vec<PreparedSuperfile> = outputs
        .into_par_iter()
        .zip(hints.into_par_iter())
        .filter_map(|(shard, hint)| match prepare_superfile(inner, shard) {
            Ok(Some(p)) => {
                Some(
                    finish_superfile_entry(p.entry, hint).map(|entry| PreparedSuperfile {
                        entry,
                        bytes_for_store: p.bytes_for_store,
                        bytes_for_storage: p.bytes_for_storage,
                        bytes_for_cache: p.bytes_for_cache,
                    }),
                )
            }
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    collect_prepared_superfiles(inner, prepared)
}

fn prepare_user_superfile_batch(
    inner: &SupertableInner,
    outputs: Vec<ShardOutput>,
    hints: Vec<Option<u32>>,
) -> Result<SuperfilePublishBatch, BuildError> {
    inner
        .options
        .writer_pool
        .install(|| prepare_user_superfile_batch_in_scope(inner, outputs, hints))
}

async fn persist_superfile_publish_batch_async(
    inner: &SupertableInner,
    batch: SuperfilePublishBatch,
) -> Result<(), BuildError> {
    if batch.new_entries.is_empty() {
        return Ok(());
    }
    if let Some(storage) = inner.options.storage.as_ref().cloned() {
        let new_manifest = persist_commit_async(
            inner,
            storage,
            batch.new_entries,
            &batch.to_remove,
            batch.pending_storage_writes,
            Vec::new(),
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
        inner.manifest.store(Arc::new(new_manifest));
        // Already async — await the warm-cache fill directly. Do NOT call
        // `warm_cache_after_commit` here: its sync `block_in_place` + nested
        // `block_on` inside the `tokio::join!` commit future deadlocks the
        // runtime (main thread parked, all workers idle).
        if let Some(cache) = inner.options.disk_cache.as_ref() {
            warm_cache_inserts(cache, batch.pending_cache_inserts).await;
        }
        if let (Some(cache), Some(budget)) = (
            inner.options.disk_cache.as_ref(),
            inner.options.memory_budget_bytes,
        ) {
            cache.sweep_for_budget(budget);
        }
        return Ok(());
    }
    let old = inner.manifest.load();
    let new = old.with_appended(batch.new_entries);
    inner.manifest.store(Arc::new(new));
    Ok(())
}

/// Single-thread rayon pool for incoming-routing CPU work (cell assignment + per-cell
/// superfile encode). Installing the build under this pool pins all its nested
/// `par_iter`/`join` to one thread instead of fanning out across every core, so
/// routing can't starve foreground ingest CPU.
static MAINT_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

fn maint_pool() -> &'static rayon::ThreadPool {
    MAINT_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "hidden-maint-cpu".into())
            .build()
            .expect("hidden maintenance rayon pool")
    })
}

/// No-staging drain: splice each user-table vector superfile's cluster `c`
/// (the user superfiles are global-aligned, so cluster `c` == cell `c`) into
/// the hidden index as packed multi-cell superfiles (`cell_id % N` shards,
/// `N = writer_pool`). Reads from `user_inner`, writes to `hidden_inner`;
/// user superfiles are the durable source and are NOT removed. No dual-write,
/// no staging copy, no decode/re-k-means.
///
/// Processes user superfiles in BOUNDED BATCHES (`drain_batch_superfiles`) so
/// working-set RAM stays O(batch); each batch still builds per-cell IVFs, then
/// packs them into ≤N shard objects. **Incremental**: skips user commits whose
/// `birth_version` is already in the hidden manifest's `drained_ranges`, and
/// advances `drained_ranges` atomically with each batch's commit — so re-running
/// (or running periodically) drains only newly-ingested commits, never
/// duplicating cells. Pre-drain queries see an empty hidden index (0 results)
/// until this runs.
///
/// Batch size comes from `vector.drain_batch_superfiles`, which
/// [`SupertableOptions::apply_config`] copies into the option below; per-table
/// callers can still override it via `with_drain_batch_superfiles`.
fn drain_batch_superfiles(opts: &SupertableOptions) -> i64 {
    opts.drain_batch_superfiles
}

/// Working-set cap for one wave of the drain's end-of-run per-cell builds.
const DRAIN_BUILD_GROUP_BYTES: u64 = 4 * (1 << 30);

/// Rough per-row byte estimate for wave sizing when rows are already in RAM
/// (prefix + RaBitQ + Sq8 codes/residuals). Matches the spill layout's
/// variable payload closely enough for the 4 GiB wave budget.
const DRAIN_MEMORY_ROW_BYTES_ESTIMATE: u64 = 29 + 128 + 2 * 1024;

/// Drain replica factor at or below which no boundary replicas are added.
const DEFAULT_DRAIN_REPLICA_TARGET_FACTOR: f32 = 1.0;

/// Target storage amplification for boundary-only drain replication. For
/// example, `1.2` means the drain may add at most `0.2 * rows` extra row copies,
/// selected from rows closest to a Voronoi boundary. Values `<= 1.0` disable
/// replication; the default drain path is unchanged. Sourced from
/// `vector.drain_replica_target_factor`.
fn drain_replica_target_factor() -> f32 {
    let factor = config::global().vector.drain_replica_target_factor;
    if factor.is_finite() && factor > DEFAULT_DRAIN_REPLICA_TARGET_FACTOR {
        factor
    } else {
        DEFAULT_DRAIN_REPLICA_TARGET_FACTOR
    }
}

fn drain_replica_extra_budget(n_rows: usize, target_factor: f32) -> usize {
    if n_rows == 0 || target_factor <= DEFAULT_DRAIN_REPLICA_TARGET_FACTOR {
        return 0;
    }
    let target_rows = (n_rows as f64 * target_factor as f64).ceil() as usize;
    target_rows.saturating_sub(n_rows).min(n_rows)
}

async fn materialized_user_rows_for_drain(
    reader: &SuperfileReader,
    column: &str,
    stable_ids: &[i128],
    tombstones: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let vec_reader = reader
        .vec()
        .ok_or_else(|| BuildError::Store("user superfile missing vector index".into()))?;
    if vec_reader.is_multi_cell() {
        let cells = vec_reader
            .materialized_cells_rows_async(None)
            .await
            .ok_or_else(|| {
                BuildError::Store(format!(
                    "drain materialize: multi-cell column '{column}' missing Sq8Residual index"
                ))
            })?;
        // One physical row per `_id`: boundary stubs share the primary's
        // stable_id, so `or_insert` keeps the first posting seen.
        let mut by_id: HashMap<i128, MaterializedIvfRow> = HashMap::new();
        for (_, rows) in cells {
            for row in rows {
                by_id.entry(row.stable_id).or_insert(row);
            }
        }
        // Tombstones address Parquet primary locals (not IVF file-locals /
        // stubs). Resolve deleted locals → `_id`, then drop by stable_id.
        if let Some(bm) = tombstones
            && !bm.is_empty()
        {
            let locals: Vec<u32> = bm.iter().collect();
            let id_column = reader.id_column();
            let batch = reader
                .take_by_local_doc_ids(&locals, &[id_column])
                .map_err(|e| BuildError::Store(e.to_string()))?;
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| BuildError::Store("_id column missing".into()))?;
            let deleted: HashSet<i128> = array.values().iter().copied().collect();
            by_id.retain(|stable_id, _| !deleted.contains(stable_id));
        }
        let mut rows: Vec<MaterializedIvfRow> = by_id.into_values().collect();
        rows.sort_by_key(|row| row.stable_id);
        for (local, row) in rows.iter_mut().enumerate() {
            row.local_doc_id = local as u32;
        }
        return Ok(rows);
    }
    materialized_ivf_rows_in_doc_order(vec_reader, column, stable_ids, tombstones).await
}

pub(in crate::supertable) async fn drain_user_superfiles_to_hidden_cells(
    user_inner: Arc<SupertableInner>,
    hidden_inner: Arc<SupertableInner>,
) -> Result<(), BuildError> {
    // Single-flight on the hidden side.
    if hidden_inner
        .compaction_outstanding
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Ok(());
    }
    struct Slot<'a>(&'a std::sync::atomic::AtomicBool);
    impl Drop for Slot<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let _slot = Slot(&hidden_inner.compaction_outstanding);

    // The global cell grid is owned by the USER manifest (bootstrapped at the
    // first commit). The hidden cell index is the derived copy this drain writes.
    let Some(gvi) = user_inner.manifest.load_full().get_global_vector_index() else {
        return Ok(());
    };
    let clusters = gvi.grid;
    let column = gvi.column;
    if clusters.n_cent == 0 || clusters.dim == 0 {
        return Ok(());
    }
    // Preserve any existing hidden-side query tuning (`routing`) across drains.
    let routing = match hidden_inner.manifest.load_full().get_partition_strategy() {
        PartitionStrategy::VectorCell { routing, .. } => routing,
        _ => Default::default(),
    };

    // Source: every user-table vector superfile, processed in BOUNDED BATCHES so
    // drain working-set RAM stays O(batch) instead of O(corpus) (the >3M memory
    // wall). Each batch opens its readers, builds its cell superfiles, publishes
    // them (append — one file per touched cell), then frees its working set.
    // Batch size is `vector.drain_batch_superfiles`: `0` = skip, `-1` =
    // unbounded single merge.
    let user_manifest = user_inner.manifest.load_full();
    // A cold-open user manifest is parts-backed and may have an empty flat
    // view. Drain must hydrate the authoritative user parts; reading only
    // `get_all_superfiles()` silently turns drain into a no-op after reopen.
    let sources = user_manifest
        .get_all_superfiles_loaded()
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
    if sources.is_empty() {
        return Ok(());
    }
    let batch_cfg = drain_batch_superfiles(&user_inner.options);
    if batch_cfg == 0 {
        eprintln!("[supertable drain] skipped (drain_batch_superfiles = 0)");
        return Ok(());
    }

    // INCREMENTAL: skip user commits already consumed into cells. The hidden
    // manifest's `drained_ranges` records drained user `birth_version`s — and
    // because the manifest-pointer CAS serializes every commit (across all
    // writers/hosts) into one gap-free version sequence, that's the only total
    // order safe to filter on under concurrent ingest. Group the undrained
    // superfiles by birth_version: a commit is atomic and indivisible, so the
    // version is the unit a batch must never split.
    let drained = hidden_inner.manifest.load_full().get_drained_ranges();
    let mut by_version: std::collections::BTreeMap<u64, Vec<Arc<SuperfileEntry>>> =
        std::collections::BTreeMap::new();
    for sf in &sources {
        if drained.contains(sf.birth_version) {
            continue;
        }
        by_version
            .entry(sf.birth_version)
            .or_default()
            .push(Arc::clone(sf));
    }
    if by_version.is_empty() {
        eprintln!(
            "[supertable drain] nothing to drain: all {} user superfile(s) already drained",
            sources.len()
        );
        return Ok(());
    }

    // Version-aligned BOUNDED batches: accumulate WHOLE versions until the
    // superfile budget is reached, then cut at the version boundary — so the
    // watermark only ever advances on a commit boundary (never mid-commit,
    // which would risk re-drain or exclusion on crash). `drain_batch_superfiles`
    // is thus a TARGET, not a hard cap: a single commit larger than the budget
    // becomes its own oversized batch. `-1` → everything in one batch.
    let budget = if batch_cfg < 0 {
        usize::MAX
    } else {
        (batch_cfg as usize).max(1)
    };
    let mut batches: Vec<(Vec<u64>, Vec<Arc<SuperfileEntry>>)> = Vec::new();
    let mut cur_v: Vec<u64> = Vec::new();
    let mut cur_sf: Vec<Arc<SuperfileEntry>> = Vec::new();
    for (version, sfs) in by_version {
        if !cur_sf.is_empty() && cur_sf.len().saturating_add(sfs.len()) > budget {
            batches.push((std::mem::take(&mut cur_v), std::mem::take(&mut cur_sf)));
        }
        cur_v.push(version);
        cur_sf.extend(sfs);
        if cur_sf.len() >= budget {
            batches.push((std::mem::take(&mut cur_v), std::mem::take(&mut cur_sf)));
        }
    }
    if !cur_sf.is_empty() {
        batches.push((cur_v, cur_sf));
    }

    let store = user_inner.options.store.clone();
    let storage_opt = user_inner.options.storage.clone();
    let storage = hidden_inner
        .options
        .storage
        .clone()
        .ok_or_else(|| BuildError::Store("hidden drain requires storage".into()))?;
    let metric = hidden_inner
        .options
        .vector_columns
        .first()
        .map(|c| c.metric)
        .unwrap_or(Metric::L2Sq);
    // A/B the per-cell consolidation op (`vector.drain_consolidate`):
    //   `kmeans` (default) — materialize each superfile's rows, assign to the
    //     nearest global cell, re-cluster per cell → few clean clusters per cell.
    //   `splice` — route each superfile's LOCAL clusters to their nearest global
    //     cell, keep them verbatim as multi-cluster fragments (no re-cluster).
    let mode = match config::global().vector.drain_consolidate {
        DrainConsolidate::Kmeans => "kmeans",
        DrainConsolidate::Splice => "splice",
    };
    let is_splice = mode == "splice";
    // assign-skip: with global-aligned user superfiles (`vector.user_centroids:
    // global`) cluster c == cell c, so group by the row's own cluster ordinal
    // instead of the O(n·n_cent) per-row nearest-cell scoring.
    let assign_skip = config::global().vector.user_centroids == CentroidAlignment::Global;
    let column_name = column.clone();

    let drain_t0 = std::time::Instant::now();
    let drain_rss0 = proc_rss_mib();
    let n_batches = batches.len();
    // Carries per-cell counts cumulatively across batches; the centroids
    // are immutable (owned by the user manifest), so each batch's
    // `apply_cell_updates` builds on the prior batches' running totals.
    let mut running_clusters = clusters;
    // Keep the trusted drain semantics: every bounded materialize batch gets
    // its own assignment + boundary-replica budget, then its assigned rows are
    // accumulated by cell. Packing/publish still happens once after the final
    // batch. A single batch stays in RAM; multi-batch drains spill to scratch.
    let prefer_memory_accum = !is_splice && n_batches <= 1;
    let drain_scratch = if is_splice || prefer_memory_accum {
        None
    } else {
        Some(
            tempfile::tempdir()
                .map_err(|e| BuildError::Store(format!("drain spill scratch dir: {e}")))?,
        )
    };
    let mut cell_rows: CellRowAccumulator = if prefer_memory_accum {
        CellRowAccumulator::memory()
    } else if let Some(ref scratch) = drain_scratch {
        CellRowAccumulator::disk(scratch.path().to_path_buf())
    } else {
        CellRowAccumulator::memory()
    };
    let mut added_per_cell: HashMap<u32, u32> = HashMap::new();

    for (batch_idx, (batch_versions, batch_sources)) in batches.iter().enumerate() {
        let batch_t0 = std::time::Instant::now();
        // Zero the I/O timeline so the readout below reflects only this batch's
        // superfile reads (INFINO_IO_TIMELINE; a no-op otherwise).
        if crate::storage::io_counters::timeline_enabled() {
            let _ = crate::storage::io_counters::take();
            crate::storage::io_counters::timeline_reset();
        }
        let read_concurrency = drain_read_concurrency();
        // Open this batch's user superfiles FULLY RESIDENT: the splice/materialize
        // read via `try_get_range_sync` on rayon workers, which needs the whole
        // superfile in memory — a lazy reader yields VectorReadError. Reuse a
        // resident cached reader if present, else fetch the full bytes + open.
        // `buffer_unordered` yields each open as it completes, so one straggler
        // read can't stall the fan-out window (order is irrelevant — rows are
        // bucketed by cell downstream). Routing-id resolution is resident (no
        // object-store I/O), so it rides each open's future and overlaps the
        // other reads' in-flight bytes.
        let readers: Vec<(Arc<SuperfileReader>, Vec<i128>)> =
            stream::iter(batch_sources.iter().map(|entry| {
                let entry = Arc::clone(entry);
                let store = Arc::clone(&store);
                let storage_opt = storage_opt.clone();
                let manifest = Arc::clone(&user_manifest);
                async move {
                    let reader = match store.reader(&entry.uri) {
                        Ok(r) if r.parquet_bytes().is_some() => r,
                        _ => {
                            let storage = storage_opt.as_ref().ok_or_else(|| {
                                BuildError::Store(
                                    "drain requires storage to load user superfiles".into(),
                                )
                            })?;
                            let (bytes, _) = storage
                                .get(&entry.uri.storage_path())
                                .await
                                .map_err(|e| BuildError::Store(e.to_string()))?;
                            Arc::new(
                                SuperfileReader::open(bytes)
                                    .map_err(|e| BuildError::Store(e.to_string()))?,
                            )
                        }
                    };
                    let stable_ids = stable_ids_by_local_for_routing(&manifest, &entry, &reader)
                        .await
                        .map_err(|e| BuildError::Store(e.to_string()))?;
                    Ok::<_, BuildError>((reader, stable_ids))
                }
            }))
            .buffer_unordered(read_concurrency)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, BuildError>>()?;

        // The batch's superfile reads land here (opens are fully resident). The
        // timeline distinguishes a serial dependent chain (concurrency ~1x) from
        // overlapped reads (concurrency ~ buffered fan-out) — the lever for the
        // materialize phase. Gated on INFINO_IO_TIMELINE.
        if crate::storage::io_counters::timeline_enabled() {
            let spans = crate::storage::io_counters::timeline_take();
            let (range_gets, _, _, _) = crate::storage::io_counters::take();
            let min_start = spans.iter().map(|s| s.start_us).min().unwrap_or(0);
            let max_end = spans.iter().map(|s| s.end_us).max().unwrap_or(0);
            let wall_us = max_end.saturating_sub(min_start);
            let sum_us: u64 = spans
                .iter()
                .map(|s| s.end_us.saturating_sub(s.start_us))
                .sum();
            let bytes: u64 = spans.iter().map(|s| s.len).sum();
            let concurrency = if wall_us > 0 {
                sum_us as f64 / wall_us as f64
            } else {
                0.0
            };
            eprintln!(
                "[supertable drain] batch {}/{} materialize I/O: {} object reads, {:.1} MiB, wall {:.1}ms, Σdur {:.1}ms, implied concurrency {:.1}x ({} range-gets)",
                batch_idx + 1,
                n_batches,
                spans.len(),
                bytes as f64 / (1u64 << 20) as f64,
                wall_us as f64 / 1e3,
                sum_us as f64 / 1e3,
                concurrency,
                range_gets,
            );
        }

        let mut prepared: Vec<PreparedSuperfile> = Vec::new();
        let mut cell_updates: HashMap<u32, u32> = HashMap::new();

        if mode == "splice" {
            let column_name_ref = column_name.as_str();
            let stable_ids_per_input: Vec<Vec<i128>> =
                readers.iter().map(|(_, ids)| ids.clone()).collect();
            let routed: HashMap<u32, (MergedIvfSubsection, Vec<i128>)> =
                hidden_inner.options.writer_pool.install(
                    || -> Result<HashMap<u32, (MergedIvfSubsection, Vec<i128>)>, BuildError> {
                        let inputs: Vec<(&VectorReader, &str)> = readers
                            .iter()
                            .map(|(r, _)| {
                                r.vec()
                                    .ok_or_else(|| {
                                        BuildError::Store(
                                            "user superfile missing vector index".into(),
                                        )
                                    })
                                    .map(|vr| (vr, column_name_ref))
                            })
                            .collect::<Result<_, _>>()?;
                        let clusters_ref = &running_clusters;
                        route_clusters_into_cells(
                            &inputs,
                            &stable_ids_per_input,
                            |centroid: &[f32]| {
                                let mut assign = [0u32];
                                clusters_ref.assign_rows(metric, centroid, &mut assign);
                                vec![assign[0]]
                            },
                        )
                        .map_err(|e| e.into())
                    },
                )?;
            let n_cells = routed.len();
            let n_shards = packed_cell_shard_count(&hidden_inner.options);
            let cell_subs: Vec<(u32, (MergedIvfSubsection, Vec<i128>))> =
                routed.into_iter().collect();
            for (cell_id, (subsection, _)) in &cell_subs {
                let added = subsection.n_docs;
                let base = running_clusters
                    .counts
                    .get(*cell_id as usize)
                    .copied()
                    .unwrap_or(0);
                cell_updates.insert(*cell_id, base.saturating_add(added));
            }
            let packed_shards = group_cells_by_packed_shard(cell_subs, n_shards);
            let n_objects = packed_shards.len();
            let shard_prepared: Vec<PreparedSuperfile> =
                hidden_inner.options.writer_pool.install(|| {
                    packed_shards
                        .into_par_iter()
                        .map(|(shard_id, cells)| {
                            let packed: Vec<(u32, MergedIvfSubsection, Vec<i128>)> = cells
                                .into_iter()
                                .map(|(cell_id, (sub, ids))| (cell_id, sub, ids))
                                .collect();
                            build_prepared_from_packed_cells(&hidden_inner, shard_id, packed)
                        })
                        .collect::<Result<Vec<_>, BuildError>>()
                })?;
            prepared.extend(shard_prepared);
            eprintln!(
                "[supertable drain] batch {}/{} ({} sf, splice): route+build {:.1}ms, {} cell(s) -> {} packed object(s)",
                batch_idx + 1,
                n_batches,
                batch_sources.len(),
                batch_t0.elapsed().as_secs_f64() * 1e3,
                n_cells,
                n_objects,
            );
        } else {
            // kmeans: materialize THIS batch, assign with the shared core, then
            // accumulate the assigned (primary + replica) postings by cell.
            let column_for_mat = column_name.clone();
            let tombstone_cache = user_inner.tombstone_cache.clone();
            let now = time::Instant::now();
            let row_sets: Vec<Vec<MaterializedIvfRow>> =
                stream::iter(readers.iter().zip(batch_sources.iter()).map(
                    |((reader, stable_ids), entry)| {
                        let column_for_mat = column_for_mat.clone();
                        let tombstone_cache = tombstone_cache.clone();
                        let entry = Arc::clone(entry);
                        async move {
                            let bitmap = tombstone_cache
                                .as_ref()
                                .map(|t| t.bitmap_for(entry.superfile_id, now))
                                .transpose()
                                .map_err(|e| BuildError::Store(e.to_string()))?;
                            materialized_user_rows_for_drain(
                                reader,
                                &column_for_mat,
                                stable_ids,
                                bitmap.as_deref(),
                            )
                            .await
                        }
                    },
                ))
                .buffered(commit_write_concurrency())
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, BuildError>>()?;
            let t_mat = batch_t0.elapsed().as_secs_f64() * 1e3;

            let all_rows: Vec<MaterializedIvfRow> = row_sets.into_iter().flatten().collect();
            let n_batch_rows = all_rows.len();
            // Batch boundary: reset disk writers' Arc-pointer dedup (no-op in RAM).
            cell_rows.begin_batch();
            let replica_target = drain_replica_target_factor();
            let assigned_groups = hidden_inner.options.writer_pool.install(|| {
                let row_refs: Vec<PackRow<'_>> =
                    all_rows.iter().map(PackRow::Materialized).collect();
                assign_cells(
                    &row_refs,
                    &running_clusters,
                    metric,
                    assign_skip,
                    replica_target,
                )
            })?;
            for group in assigned_groups {
                let added = u32::try_from(group.members.len()).unwrap_or(u32::MAX);
                let count = added_per_cell.entry(group.cell_id).or_insert(0);
                *count = count.saturating_add(added);
                for (_, _, row) in group.members {
                    let PackRow::Materialized(row) = row else {
                        unreachable!("drain assignment contains only materialized rows");
                    };
                    cell_rows.append(group.cell_id, row)?;
                }
            }
            let t_accum = batch_t0.elapsed().as_secs_f64() * 1e3;
            let accum_label = if cell_rows.prefer_memory() {
                "memory"
            } else {
                "spill"
            };
            eprintln!(
                "[supertable drain] batch {}/{} ({} sf, kmeans): materialize {:.1}ms + assign+accum {:.1}ms, {} batch row(s) -> {} {} cell store(s)",
                batch_idx + 1,
                n_batches,
                batch_sources.len(),
                t_mat,
                t_accum - t_mat,
                n_batch_rows,
                cell_rows.len(),
                accum_label,
            );
        }

        if prepared.is_empty() {
            // kmeans mode spills rows above and publishes once after the loop.
            continue;
        }
        // Publish this batch's cell superfiles (append — no removals; the user
        // superfiles stay as the durable source). In the SAME hidden commit,
        // advance the derived grid (counts) AND mark this batch's user
        // versions drained — so cells-written and watermark-advanced are one
        // atomic CAS (a crash can only drop uncommitted work, never re-drive or
        // exclude committed work). Re-read drained_ranges so sequential batches
        // (and a prior drainer's progress) compose.
        let publish_batch = collect_prepared_superfiles(&hidden_inner, prepared)?;
        running_clusters = opann::apply_cell_updates(&running_clusters, &cell_updates);
        let mut new_drained = hidden_inner.manifest.load_full().get_drained_ranges();
        // Advance as a CONTIGUOUS range up to this batch's max version, starting
        // from just past the existing genesis-anchored prefix. This absorbs
        // vacuous version gaps — commit-versions with no superfile (deletes,
        // compaction outputs with an older birth_version, empty commits) — which
        // are nothing-to-drain and whose version numbers are never reused. Folding
        // them keeps `drained_ranges` a single interval under single-flight drain
        // instead of fragmenting once per superfile-less commit. (Falls back to
        // the batch's own min when there's no anchored prefix yet — e.g. a
        // parallel drainer working a high slice.)
        let batch_max = batch_versions.iter().copied().max().unwrap_or(0);
        // Single-flight drain processes ALL undrained superfiles in ascending
        // version order, so the first batch is the global-minimum undrained work
        // — nothing below it is undrained — and we can anchor the prefix at
        // genesis (0), absorbing every vacuous version below `batch_max`.
        // Subsequent batches extend from the prefix end. (When parallel
        // version-claims land, a drainer working a high slice would use its own
        // `batch_min` here instead of 0.)
        let lo = new_drained.prefix_end().map(|end| end + 1).unwrap_or(0);
        new_drained.insert_range(lo.min(batch_max), batch_max);
        hidden_inner.manifest.store(Arc::new(
            hidden_inner
                .manifest
                .load()
                .with_partition_strategy(PartitionStrategy::VectorCell {
                    column: column.clone(),
                    clusters: running_clusters.clone(),
                    routing,
                })
                .with_drained_ranges(new_drained),
        ));
        let no_removals: Vec<Arc<SuperfileEntry>> = Vec::new();
        // Cheap write-side readout: wall + bytes across the publish (cell-file
        // PUTs + manifest CAS). A single timer, no per-put provider hooks — enough
        // to tell whether the publish is I/O throughput or commit overhead, and
        // whether it's a lever. Gated on INFINO_IO_TIMELINE.
        let publish_bytes: u64 = publish_batch
            .pending_storage_writes
            .iter()
            .map(|(_, b)| b.len() as u64)
            .sum();
        let n_puts = publish_batch.pending_storage_writes.len();
        let publish_t0 = std::time::Instant::now();
        let mut pending_writes = publish_batch.pending_storage_writes;
        let mut pending_replaces: Vec<(SuperfileUri, Bytes)> = Vec::new();
        write_superfile_list_with_threshold(
            &storage,
            &hidden_inner.options,
            DRAIN_PUT_MULTIPART_THRESHOLD_BYTES,
            &mut pending_writes,
            &mut pending_replaces,
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
        let new_manifest = persist_commit_async(
            &hidden_inner,
            Arc::clone(&storage),
            publish_batch.new_entries,
            &no_removals,
            Vec::new(),
            Vec::new(),
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
        hidden_inner.manifest.store(Arc::new(new_manifest));
        if crate::storage::io_counters::timeline_enabled() {
            let ms = publish_t0.elapsed().as_secs_f64() * 1e3;
            let mib = publish_bytes as f64 / (1u64 << 20) as f64;
            let rate = if ms > 0.0 { mib / (ms / 1e3) } else { 0.0 };
            eprintln!(
                "[supertable drain] batch {}/{} publish: {} puts + CAS, {:.1} MiB, wall {:.1}ms → {:.0} MiB/s",
                batch_idx + 1,
                n_batches,
                n_puts,
                mib,
                ms,
                rate,
            );
        }
    }

    // kmeans: rows were assigned per bounded batch above (trusted drain
    // semantics). Pack their per-cell accumulators in memory-bounded waves.
    if !cell_rows.is_empty() {
        let build_t0 = time::Instant::now();
        let mut cell_subs = cell_rows.into_cell_rows()?;
        cell_subs.sort_unstable_by_key(|(cell_id, _)| *cell_id);
        let total_rows: u64 = cell_subs.iter().map(|(_, rows)| rows.len() as u64).sum();
        let n_cells_total = cell_subs.len();
        let vc = hidden_inner
            .options
            .vector_columns
            .first()
            .cloned()
            .ok_or_else(|| BuildError::Store("drain pack requires a vector column".into()))?;

        crate::superfile::vector::builder::build_phase_timers::reset();
        let mut new_entries: Vec<Arc<SuperfileEntry>> = Vec::with_capacity(n_cells_total);
        let mut wave: Vec<(u32, Vec<MaterializedIvfRow>)> = Vec::new();
        let mut wave_bytes = 0u64;
        let mut n_waves = 0usize;
        let mut cells_iter = cell_subs.into_iter().peekable();
        while let Some((cell_id, rows)) = cells_iter.next() {
            wave_bytes += rows.len() as u64 * DRAIN_MEMORY_ROW_BYTES_ESTIMATE;
            wave.push((cell_id, rows));
            let flush = wave_bytes >= DRAIN_BUILD_GROUP_BYTES || cells_iter.peek().is_none();
            if !flush {
                continue;
            }
            n_waves += 1;
            let wave_cells = std::mem::take(&mut wave);
            wave_bytes = 0;
            let prepared_wave: Vec<PreparedSuperfile> =
                hidden_inner.options.writer_pool.install(|| {
                    let n_shards = packed_cell_shard_count(&hidden_inner.options);
                    let packed_shards = group_cells_by_packed_shard(wave_cells, n_shards);
                    packed_shards
                        .into_par_iter()
                        .map(|(shard_id, cells)| {
                            let packed: Vec<(u32, MergedIvfSubsection, Vec<i128>)> = cells
                                .into_iter()
                                .map(|(cell_id, rows)| {
                                    drain_pack_materialized_cell(cell_id, rows, &vc)
                                        .map(|p| p.into_hidden_triple())
                                })
                                .collect::<Result<Vec<_>, BuildError>>()?;
                            build_prepared_from_packed_cells(&hidden_inner, shard_id, packed)
                        })
                        .collect::<Result<Vec<_>, BuildError>>()
                })?;
            let publish = collect_prepared_superfiles(&hidden_inner, prepared_wave)?;
            let mut pending_writes = publish.pending_storage_writes;
            let mut pending_replaces: Vec<(SuperfileUri, Bytes)> = Vec::new();
            write_superfile_list_with_threshold(
                &storage,
                &hidden_inner.options,
                DRAIN_PUT_MULTIPART_THRESHOLD_BYTES,
                &mut pending_writes,
                &mut pending_replaces,
            )
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
            new_entries.extend(publish.new_entries);
        }
        let mut cell_updates: HashMap<u32, u32> = HashMap::new();
        for (cell, added) in &added_per_cell {
            let base = running_clusters
                .counts
                .get(*cell as usize)
                .copied()
                .unwrap_or(0);
            cell_updates.insert(*cell, base.saturating_add(*added));
        }
        running_clusters = opann::apply_cell_updates(&running_clusters, &cell_updates);
        let mut new_drained = hidden_inner.manifest.load_full().get_drained_ranges();
        let drained_max = batches
            .iter()
            .flat_map(|(versions, _)| versions.iter().copied())
            .max()
            .unwrap_or(0);
        let lo = new_drained.prefix_end().map(|end| end + 1).unwrap_or(0);
        new_drained.insert_range(lo.min(drained_max), drained_max);
        hidden_inner.manifest.store(Arc::new(
            hidden_inner
                .manifest
                .load()
                .with_partition_strategy(PartitionStrategy::VectorCell {
                    column: column.clone(),
                    clusters: running_clusters.clone(),
                    routing,
                })
                .with_drained_ranges(new_drained),
        ));
        let no_removals: Vec<Arc<SuperfileEntry>> = Vec::new();
        let new_manifest = persist_commit_async(
            &hidden_inner,
            Arc::clone(&storage),
            new_entries,
            &no_removals,
            Vec::new(),
            Vec::new(),
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
        hidden_inner.manifest.store(Arc::new(new_manifest));
        eprintln!(
            "[supertable drain] cell build: {} row(s) -> {} cell(s) packed into shard objects in {} wave(s), {:.1}ms",
            total_rows,
            n_cells_total,
            n_waves,
            build_t0.elapsed().as_secs_f64() * 1e3,
        );
        if crate::superfile::vector::builder::build_phase_timers::enabled() {
            let (train_ms, assign_ms, calib_ms) =
                crate::superfile::vector::builder::build_phase_timers::snapshot_ms();
            eprintln!(
                "[supertable drain] cell build phases (summed CPU, {n_cells_total} cells): train {train_ms:.1}ms + assign {assign_ms:.1}ms + calibrate {calib_ms:.1}ms",
            );
        }
    }

    eprintln!(
        "[supertable drain] done ({mode}, {} batch(es), budget {} sf): total {:.1}ms; RSS {} -> {} MiB",
        n_batches,
        batch_cfg,
        drain_t0.elapsed().as_secs_f64() * 1e3,
        drain_rss0
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|| "?".into()),
        proc_rss_mib()
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|| "?".into()),
    );
    // Membership has settled: publish the slow-CAS entry blob and stamp its
    // ref (the per-batch `update`s cleared it). Hidden tables have no manifest
    // parts, so publication is required for reopen and cannot degrade to a
    // warning.
    refresh_slow_vector_state(&hidden_inner).await?;
    schedule_background_storage_reclaim(Arc::clone(&hidden_inner));
    Ok(())
}

/// Load Sq8+ε IVF rows from one hidden superfile.
///
/// - Legacy one-cell-per-file (`Ivf`): all rows (file == cell).
/// - Packed multi-cell (`MultiCellIvf`): only cells in `only_cells` when
///   provided; otherwise every cell in the directory. Rows keep cell-local
///   `local_doc_id`s; stable ids come from the inline region.
async fn load_materialized_rows_from_ivf_superfile(
    inner: &SupertableInner,
    entry: &Arc<SuperfileEntry>,
    column: &str,
    now: time::Instant,
    only_cells: Option<&[u32]>,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or_else(|| BuildError::Store("cell maintenance requires storage".into()))?;
    let store = inner.options.store.clone();
    let disk_cache = inner.options.disk_cache.as_ref();

    let bitmap = inner
        .tombstone_cache
        .as_ref()
        .map(|t| t.bitmap_for(entry.superfile_id, now))
        .transpose()
        .map_err(|e| BuildError::Store(e.to_string()))?;

    let reader = open_reader(&store, disk_cache, Some(storage), entry)
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;

    let vec_reader = reader
        .vec()
        .ok_or_else(|| BuildError::Store("IVF cell superfile missing vector index".into()))?;

    if vec_reader.is_multi_cell() {
        let cells = vec_reader
            .materialized_cells_rows_async(only_cells)
            .await
            .ok_or_else(|| {
                BuildError::Store(format!(
                    "IVF maintenance: multi-cell column '{column}' missing Sq8Residual index"
                ))
            })?;
        // File-local doc bases follow cell-directory order (same as parquet).
        let mut file_doc_base_by_cell: HashMap<u32, u32> = HashMap::new();
        let mut running = 0u32;
        for (ci, &cell_id) in vec_reader.packed_cell_ids().iter().enumerate() {
            file_doc_base_by_cell.insert(cell_id, running);
            let n = vec_reader
                .vector_columns_config()
                .nth(ci)
                .map(|c| c.n_docs)
                .unwrap_or(0);
            running = running.saturating_add(n);
        }
        let mut out = Vec::new();
        for (cell_id, mut rows) in cells {
            let base = file_doc_base_by_cell.get(&cell_id).copied().unwrap_or(0);
            if let Some(bm) = bitmap.as_deref() {
                rows.retain(|r| !bm.contains(base + r.local_doc_id));
            }
            out.append(&mut rows);
        }
        return Ok(out);
    }

    let manifest = inner.manifest.load_full();
    let stable_ids = stable_ids_by_local_for_routing(&manifest, entry, &reader)
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
    materialized_ivf_rows_in_doc_order(vec_reader, column, &stable_ids, bitmap.as_deref()).await
}

/// Per-cell doc counts from a packed (or legacy) entry. Legacy returns one
/// `(partition_hint_or_0, n_docs)` pair.
async fn cell_doc_counts_for_entry(
    inner: &SupertableInner,
    entry: &Arc<SuperfileEntry>,
) -> Result<Vec<(u32, u32)>, BuildError> {
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or_else(|| BuildError::Store("cell maintenance requires storage".into()))?;
    let reader = open_reader(
        &inner.options.store,
        inner.options.disk_cache.as_ref(),
        Some(storage),
        entry,
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    let v = reader
        .vec()
        .ok_or_else(|| BuildError::Store("IVF entry missing vector index".into()))?;
    if v.is_multi_cell() {
        Ok(v.packed_cell_ids()
            .iter()
            .filter_map(|&cell| {
                let n = v.packed_cell_n_docs(cell)?;
                Some((cell, n))
            })
            .collect())
    } else {
        let cell = entry.partition_hint.unwrap_or(0);
        Ok(vec![(cell, entry.n_docs as u32)])
    }
}

/// Build one Sq8 IVF superfile via the normal superfile/vector builder.
fn build_prepared_ivf_from_materialized(
    inner: &SupertableInner,
    partition_hint: u32,
    rows: Vec<MaterializedIvfRow>,
) -> Result<PreparedSuperfile, BuildError> {
    if rows.is_empty() {
        return Err(BuildError::NoDocsToBuild);
    }
    let shard = build_one_shard_from_materialized(&rows, &inner.options, VectorLayout::Ivf)?;
    let prepared = prepare_superfile(inner, shard)?.ok_or(BuildError::NoDocsToBuild)?;
    let entry = finish_superfile_entry(prepared.entry, Some(partition_hint))?;
    Ok(PreparedSuperfile {
        entry,
        bytes_for_store: prepared.bytes_for_store,
        bytes_for_storage: prepared.bytes_for_storage,
        bytes_for_cache: prepared.bytes_for_cache,
    })
}

/// Coarse current RSS in MiB from `/proc/self/status` (Linux); `None` elsewhere
/// or on parse failure. Drain instrumentation only — not a hot path.
fn proc_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

/// Drain packed-layout shard count: align with the writer pool width
/// (same rule as ingest's per-commit shard count).
fn packed_cell_shard_count(options: &SupertableOptions) -> usize {
    options.writer_pool.current_num_threads().max(1)
}

/// Shared cell → packed-shard mapping: `cell_id % shard_count`.
fn packed_cell_shard(cell: u32, shard_count: usize) -> usize {
    debug_assert!(shard_count > 0);
    (cell as usize) % shard_count
}

/// Group `(cell_id, payload)` into `shard_count` buckets by `cell % N`.
fn group_cells_by_packed_shard<T>(
    cells: Vec<(u32, T)>,
    shard_count: usize,
) -> Vec<(u32, Vec<(u32, T)>)> {
    debug_assert!(shard_count > 0);
    let mut buckets: Vec<Vec<(u32, T)>> = (0..shard_count).map(|_| Vec::new()).collect();
    for (cell, payload) in cells {
        buckets[packed_cell_shard(cell, shard_count)].push((cell, payload));
    }
    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, cells)| !cells.is_empty())
        .map(|(shard, mut cells)| {
            cells.sort_unstable_by_key(|(cell, _)| *cell);
            (shard as u32, cells)
        })
        .collect()
}

#[derive(Clone, Copy)]
enum PackRow<'a> {
    Materialized(&'a MaterializedIvfRow),
    Fp32 { stable_id: i128, vector: &'a [f32] },
}

/// One cell after boundary assignment, before IVF subsection build.
/// Packing (k-means + encode) belongs in the parallel shard stage — not here.
struct AssignedCellGroup<'a> {
    cell_id: u32,
    /// `(stable_id, is_primary, row)` sorted primary-first then by id.
    members: Vec<(i128, bool, PackRow<'a>)>,
}

/// One packed cell IVF. Primary-vs-stub markers live on
/// [`AssignedCellGroup::members`]; the commit writer consumes them **before**
/// pack (Parquet keeps primaries only), and the hidden drain indexes every
/// posting, so the packed group carries no separate marker copy.
struct PackedCellGroup {
    cell_id: u32,
    subsection: MergedIvfSubsection,
    stable_ids: Vec<i128>,
}

impl PackedCellGroup {
    fn into_hidden_triple(self) -> (u32, MergedIvfSubsection, Vec<i128>) {
        (self.cell_id, self.subsection, self.stable_ids)
    }
}

fn pack_row_stable_id(row: PackRow<'_>) -> i128 {
    match row {
        PackRow::Materialized(row) => row.stable_id,
        PackRow::Fp32 { stable_id, .. } => stable_id,
    }
}

/// Shared drain/commit assignment core: rows in, boundary assignment and
/// replica budget applied once, cell buckets out. Does **not** build IVF
/// subsections — that runs in the shard-stage pack (parallel). Boundary
/// replicas are vector postings only; callers decide which primaries become
/// Parquet rows.
fn assign_cells<'a>(
    rows: &[PackRow<'a>],
    clusters: &ClusterCentroids,
    metric: Metric,
    assign_skip: bool,
    replica_target_factor: f32,
) -> Result<Vec<AssignedCellGroup<'a>>, BuildError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let replica_extra_budget = drain_replica_extra_budget(rows.len(), replica_target_factor);
    let needs_boundary = replica_extra_budget > 0 || !assign_skip;
    let transposed = needs_boundary.then(|| {
        transpose_centroids_cluster_major(
            &clusters.centroids,
            clusters.n_cent as usize,
            clusters.dim as usize,
        )
    });
    let assignments: Vec<opann::BoundaryAssignment> = rows
        .iter()
        .map(|row| match *row {
            PackRow::Materialized(row) if assign_skip && replica_extra_budget == 0 => {
                opann::BoundaryAssignment {
                    primary: row.cluster,
                    neighbor: None,
                }
            }
            PackRow::Materialized(row) if assign_skip => {
                let mut assignment = opann::boundary_assignment_encoded_with_transposed(
                    clusters,
                    transposed.as_deref().unwrap_or(&[]),
                    metric,
                    &row.encoded,
                );
                assignment.primary = row.cluster;
                assignment
            }
            PackRow::Materialized(row) => opann::boundary_assignment_encoded_with_transposed(
                clusters,
                transposed.as_deref().unwrap_or(&[]),
                metric,
                &row.encoded,
            ),
            PackRow::Fp32 { vector, .. } => {
                opann::boundary_assignment_fp32(clusters, transposed.as_deref(), metric, vector)
            }
        })
        .collect();

    let mut replica_candidates: Vec<(usize, u32, f32)> = assignments
        .iter()
        .enumerate()
        .filter_map(|(row_idx, assignment)| {
            assignment
                .neighbor
                .map(|(neighbor, margin)| (row_idx, neighbor, margin))
        })
        .collect();
    replica_candidates.sort_by(|a, b| a.2.total_cmp(&b.2));

    let mut buckets: HashMap<u32, Vec<(i128, bool, PackRow<'a>)>> = HashMap::new();
    for (row_idx, cell, _) in replica_candidates.into_iter().take(replica_extra_budget) {
        let row = rows[row_idx];
        buckets
            .entry(cell)
            .or_default()
            .push((pack_row_stable_id(row), false, row));
    }
    for (row, assignment) in rows.iter().zip(&assignments) {
        buckets
            .entry(assignment.primary)
            .or_default()
            .push((pack_row_stable_id(*row), true, *row));
    }

    let mut out = Vec::with_capacity(buckets.len());
    for (cell_id, mut members) in buckets {
        members.sort_by_key(|(stable_id, is_primary, _)| (!*is_primary, *stable_id));
        out.push(AssignedCellGroup { cell_id, members });
    }
    out.sort_unstable_by_key(|group| group.cell_id);
    Ok(out)
}

/// Call drain's cell IVF pack on one assigned bucket.
///
/// This is the **only** pack entry: it forwards to drain's APIs —
/// [`build_merged_subsection_from_fp32`] (commit fp32 / in-memory stream) or
/// [`build_merged_subsection_from_materialized`] (hidden drain Sq8 rows).
/// No commit-local IVF logic. Returns buffers only (no S3 / no superfile write).
fn drain_pack_materialized_cell(
    cell_id: u32,
    mut rows: Vec<MaterializedIvfRow>,
    cfg: &VectorConfig,
) -> Result<PackedCellGroup, BuildError> {
    if rows.is_empty() {
        return Err(BuildError::Store(format!(
            "cell {cell_id}: drain pack received no rows"
        )));
    }
    rows.sort_by_key(|row| row.stable_id);
    for (local, row) in rows.iter_mut().enumerate() {
        row.local_doc_id = local as u32;
    }
    let stable_ids: Vec<i128> = rows.iter().map(|row| row.stable_id).collect();
    let cell_cfg = drain_cell_vector_config(cfg, rows.len());
    let subsection = build_merged_subsection_from_materialized(cell_cfg, rows)?;
    Ok(PackedCellGroup {
        cell_id,
        subsection,
        stable_ids,
    })
}

/// Size one cell's fine IVF so one run is approximately
/// [`DRAIN_FINE_RUN_TARGET_BYTES`]. The stride counts every per-row byte in
/// the packed IVF: RaBitQ estimate code, local id, Sq8+epsilon rerank bytes,
/// inline stable id, and the conservative norm word.
fn drain_cell_vector_config(cfg: &VectorConfig, n_rows: usize) -> VectorConfig {
    debug_assert!(n_rows > 0);
    let dim = cfg.dim;
    let rabitq_bytes = dim.div_ceil(u8::BITS as usize);
    let rerank_bytes = RerankCodec::Sq8Residual.per_vector_bytes(dim);
    let row_stride =
        rabitq_bytes + DOC_ID_BYTES + rerank_bytes + STABLE_ID_BYTES + mem::size_of::<f32>();
    let rows_per_run = (DRAIN_FINE_RUN_TARGET_BYTES / row_stride.max(1)).max(1);
    let n_cent = n_rows.div_ceil(rows_per_run).clamp(1, n_rows);
    VectorConfig {
        n_cent,
        rerank_codec: RerankCodec::Sq8Residual,
        provided_centroids: None,
        ..cfg.clone()
    }
}

fn drain_pack_assigned_cell(
    group: AssignedCellGroup<'_>,
    cfg: &VectorConfig,
) -> Result<PackedCellGroup, BuildError> {
    let AssignedCellGroup { cell_id, members } = group;
    if members.is_empty() {
        return Err(BuildError::Store(format!(
            "cell {cell_id}: assign produced an empty bucket"
        )));
    }
    let dim = cfg.dim;
    let cell_cfg = drain_cell_vector_config(cfg, members.len());
    let stable_ids: Vec<i128> = members.iter().map(|(stable_id, _, _)| *stable_id).collect();
    let subsection = match members[0].2 {
        PackRow::Materialized(_) => {
            let materialized: Vec<MaterializedIvfRow> = members
                .iter()
                .map(|(_, _, row)| match *row {
                    PackRow::Materialized(row) => row.clone(),
                    PackRow::Fp32 { .. } => unreachable!("mixed pack row kinds"),
                })
                .collect();
            return drain_pack_materialized_cell(cell_id, materialized, cfg);
        }
        PackRow::Fp32 { .. } => {
            let mut corpus = Vec::with_capacity(members.len() * dim);
            for (_, _, row) in &members {
                match *row {
                    PackRow::Fp32 { vector, .. } => corpus.extend_from_slice(vector),
                    PackRow::Materialized(_) => unreachable!("mixed pack row kinds"),
                }
            }
            // Drain's fp32 in-memory stream pack (why fp32 support exists).
            build_merged_subsection_from_fp32(cell_cfg, Arc::new(corpus), &stable_ids)?
        }
    };
    Ok(PackedCellGroup {
        cell_id,
        subsection,
        stable_ids,
    })
}

/// Build one multi-cell packed superfile: many complete cell-IVFs in one
/// Parquet object, `partition_hint = shard_id`.
fn build_one_shard_from_packed_cells(
    cells: Vec<(u32, MergedIvfSubsection, Vec<i128>)>,
    options: &SupertableOptions,
) -> Result<ShardOutput, BuildError> {
    if cells.is_empty() {
        return Err(BuildError::NoDocsToBuild);
    }
    let mut stable_ids: Vec<i128> = Vec::new();
    let mut subsections: Vec<(u32, MergedIvfSubsection)> = Vec::with_capacity(cells.len());
    for (cell_id, sub, ids) in cells {
        if ids.len() != sub.n_docs as usize {
            return Err(BuildError::Store(format!(
                "cell {cell_id}: stable_ids len {} != subsection n_docs {}",
                ids.len(),
                sub.n_docs
            )));
        }
        stable_ids.extend_from_slice(&ids);
        subsections.push((cell_id, sub));
    }
    let id_array = Decimal128Array::from_iter_values(stable_ids.iter().copied())
        .with_precision_and_scale(
            crate::supertable::options::DECIMAL128_PRECISION,
            crate::supertable::options::DECIMAL128_SCALE,
        )
        .expect("invariant: precision 38 + scale 0 always valid for any i128 payload");
    let scalar = RecordBatch::try_new(
        options.scalar_schema(),
        vec![Arc::new(id_array) as ArrayRef],
    )
    .map_err(|_| BuildError::BatchSchemaMismatch)?;

    let mut builder = SuperfileBuilder::new(
        options
            .builder_options()
            .with_vector_layout(VectorLayout::MultiCellIvf),
    )?;
    builder.add_batch_ids_only(&scalar)?;
    builder.set_prebuilt_multi_cell_ivfs(subsections)?;

    let id_min = stable_ids.iter().copied().min().unwrap_or(0);
    let id_max = stable_ids.iter().copied().max().unwrap_or(0);
    let n_docs = stable_ids.len() as u64;
    let scalar_stats = ScalarStatsAgg::from_batches(&options.scalar_schema(), &[&scalar]);
    let bytes = Bytes::from(builder.finish()?);

    Ok(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    })
}

/// Prepare a packed multi-cell shard for publish (`partition_hint = shard_id`).
fn build_prepared_from_packed_cells(
    inner: &SupertableInner,
    shard_id: u32,
    cells: Vec<(u32, MergedIvfSubsection, Vec<i128>)>,
) -> Result<PreparedSuperfile, BuildError> {
    let shard = build_one_shard_from_packed_cells(cells, &inner.options)?;
    let prepared = prepare_superfile(inner, shard)?.ok_or(BuildError::NoDocsToBuild)?;
    let entry = finish_superfile_entry(prepared.entry, Some(shard_id))?;
    Ok(PreparedSuperfile {
        entry,
        bytes_for_store: prepared.bytes_for_store,
        bytes_for_storage: prepared.bytes_for_storage,
        bytes_for_cache: prepared.bytes_for_cache,
    })
}

/// Commit vector path — drain's flow with a Parquet+FTS finish:
///
/// 1. assign the **whole buffer** to global cells in one pass (drain's core;
///    the boundary-replica budget is batch-global, exactly like drain),
/// 2. group whole cells into ≤ `n_writers` shard files (`cell % N` — drain's
///    [`group_cells_by_packed_shard`]),
/// 3. each writer: `rayon::join` — drain pack (fp32→Sq8→materialized fine
///    IVF) ‖ Parquet+FTS for that shard's primary rows — then splice + finish.
///
/// Rows are resharded by centroid distance instead of arrival time; drain
/// never writes superfiles or touches S3 here — the writer publishes through
/// the normal batch path.
fn commit_shards_via_drain(
    buffer: Vec<BufferedBatch>,
    inner: &SupertableInner,
    clusters: &ClusterCentroids,
    metric: Metric,
) -> Result<(Vec<ShardOutput>, Vec<Option<u32>>), BuildError> {
    let stage_t0 = time::Instant::now();
    let vc = inner
        .options
        .vector_columns
        .first()
        .cloned()
        .ok_or_else(|| BuildError::Store("drain-commit requires a vector column".into()))?;
    let dim = vc.dim;
    if dim != clusters.dim as usize {
        return Err(BuildError::Store(format!(
            "commit vector dim {dim} does not match global grid dim {}",
            clusters.dim
        )));
    }

    // Flatten the buffer once (ids + vectors + scalar batches).
    let mut stable_ids: Vec<i128> = Vec::new();
    let mut flat_vectors: Vec<Vec<f32>> = vec![Vec::new(); inner.options.vector_columns.len()];
    let mut scalar_batches: Vec<&RecordBatch> = Vec::with_capacity(buffer.len());
    for buffered in &buffer {
        let id_col = buffered
            .scalar
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                BuildError::IdColumnWrongType(
                    inner.options.id_column.clone(),
                    "<id column not Decimal128 at runtime>".to_string(),
                )
            })?;
        for i in 0..id_col.len() {
            stable_ids.push(id_col.value(i));
        }
        for (col_idx, fa) in buffered.vectors.iter().enumerate() {
            flat_vectors[col_idx].extend_from_slice(fa.values());
        }
        scalar_batches.push(&buffered.scalar);
    }
    if stable_ids.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let primary_vectors = flat_vectors
        .first()
        .ok_or_else(|| BuildError::Store("drain-commit missing vector values".into()))?;
    if primary_vectors.len() != stable_ids.len() * dim {
        return Err(BuildError::Store(format!(
            "commit vector len {} != rows {} * dim {dim}",
            primary_vectors.len(),
            stable_ids.len()
        )));
    }

    let scalar_schema = inner.options.scalar_schema();
    let source_scalar = concat_batches(&scalar_schema, scalar_batches.iter().copied())
        .map_err(|err| BuildError::Store(err.to_string()))?;
    let local_by_id: HashMap<i128, u32> = stable_ids
        .iter()
        .enumerate()
        .map(|(local, &id)| (id, local as u32))
        .collect();
    let flatten_elapsed = stage_t0.elapsed();

    // One global assign over the batch (drain's core; runs on the writer pool).
    let rows: Vec<PackRow<'_>> = stable_ids
        .iter()
        .enumerate()
        .map(|(local, &stable_id)| {
            let start = local * dim;
            PackRow::Fp32 {
                stable_id,
                vector: &primary_vectors[start..start + dim],
            }
        })
        .collect();
    let replica_target = drain_replica_target_factor();
    let assigned = inner
        .options
        .writer_pool
        .install(|| assign_cells(&rows, clusters, metric, false, replica_target))?;
    let assign_elapsed = stage_t0.elapsed().saturating_sub(flatten_elapsed);
    let assigned_cells: Vec<(u32, AssignedCellGroup<'_>)> = assigned
        .into_iter()
        .map(|group| (group.cell_id, group))
        .collect();
    let packed_shards =
        group_cells_by_packed_shard(assigned_cells, packed_cell_shard_count(&inner.options));

    let options = &inner.options;
    let shard_outputs = fanout_shards(&inner.options.writer_pool, &packed_shards, |task| {
        let (shard_id, cells) = task;
        build_one_packed_shard_via_drain(
            cells,
            &source_scalar,
            &flat_vectors,
            &local_by_id,
            options,
            &vc,
        )
        .map(|output| output.map(|output| (*shard_id, output)))
    })?;
    let fanout_elapsed = stage_t0
        .elapsed()
        .saturating_sub(flatten_elapsed)
        .saturating_sub(assign_elapsed);
    if crate::storage::io_counters::timeline_enabled() {
        eprintln!(
            "[supertable commit] flatten {:.1}ms + assign {:.1}ms + shard pack/finish {:.1}ms",
            flatten_elapsed.as_secs_f64() * 1e3,
            assign_elapsed.as_secs_f64() * 1e3,
            fanout_elapsed.as_secs_f64() * 1e3,
        );
    }

    let mut outputs = Vec::with_capacity(shard_outputs.len());
    let mut cell_hints = Vec::with_capacity(shard_outputs.len());
    for entry in shard_outputs.into_iter().flatten() {
        cell_hints.push(Some(entry.0));
        outputs.push(entry.1);
    }
    Ok((outputs, cell_hints))
}

/// One writer, one packed shard (a group of whole cells): drain pack of the
/// cells' IVF blobs ‖ Parquet+FTS of the cells' primary rows, then splice.
///
/// Parquet row order = IVF primary order (cells ascending, primaries in
/// member order within each cell) so Parquet local `l` and vector file-local
/// `l` carry the same `_id`; boundary stubs stay vector-only postings.
/// Returns `None` for a stub-only shard (no primary rows — the primaries live
/// in their home cells' files; dropping the replica copies loses nothing).
fn build_one_packed_shard_via_drain(
    cells: &[(u32, AssignedCellGroup<'_>)],
    source_scalar: &RecordBatch,
    flat_vectors: &[Vec<f32>],
    local_by_id: &HashMap<i128, u32>,
    options: &SupertableOptions,
    vc: &VectorConfig,
) -> Result<Option<ShardOutput>, BuildError> {
    let mut ordered_locals: Vec<u32> = Vec::new();
    for (_, group) in cells {
        for (member_id, is_primary, _) in &group.members {
            if !*is_primary {
                continue;
            }
            let local = local_by_id.get(member_id).copied().ok_or_else(|| {
                BuildError::Store(format!(
                    "primary stable_id {member_id} missing from commit rows"
                ))
            })?;
            ordered_locals.push(local);
        }
    }
    if ordered_locals.is_empty() {
        return Ok(None);
    }

    // Drain packs this shard's cell IVFs; Parquet+FTS build overlaps it.
    let (packed_groups, body_and_fts) = rayon::join(
        || {
            cells
                .iter()
                .map(|(cell_id, group)| {
                    let owned = AssignedCellGroup {
                        cell_id: *cell_id,
                        members: group.members.clone(),
                    };
                    drain_pack_assigned_cell(owned, vc)
                })
                .collect::<Result<Vec<_>, BuildError>>()
        },
        || build_shard_parquet_and_fts(source_scalar, flat_vectors, &ordered_locals, options),
    );
    let packed_groups = packed_groups?;
    let (mut builder, id_min, id_max, n_docs, scalar_stats) = body_and_fts?;

    let subsections: Vec<(u32, MergedIvfSubsection)> = packed_groups
        .into_iter()
        .map(|g| (g.cell_id, g.subsection))
        .collect();
    builder.set_prebuilt_multi_cell_ivfs(subsections)?;
    let bytes = Bytes::from(builder.finish()?);

    Ok(Some(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    }))
}

/// Parquet body + FTS for one shard, rows reordered to `ordered_locals`
/// (primaries in IVF order). MultiCell has no streaming VectorBuilder, so
/// `add_batch` builds scalars + FTS and only validates the vector slices;
/// IVF subsections arrive from drain via `set_prebuilt_multi_cell_ivfs`.
#[allow(clippy::type_complexity)]
fn build_shard_parquet_and_fts(
    source_scalar: &RecordBatch,
    flat_vectors: &[Vec<f32>],
    ordered_locals: &[u32],
    options: &SupertableOptions,
) -> Result<
    (
        SuperfileBuilder,
        i128,
        i128,
        u64,
        HashMap<String, ScalarStatsAgg>,
    ),
    BuildError,
> {
    let take_indices = UInt32Array::from(ordered_locals.to_vec());
    let columns: Vec<ArrayRef> = source_scalar
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &take_indices, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| BuildError::Store(err.to_string()))?;
    let scalar = RecordBatch::try_new(source_scalar.schema(), columns)
        .map_err(|_| BuildError::BatchSchemaMismatch)?;

    let mut ordered_vectors: Vec<Vec<f32>> = Vec::with_capacity(flat_vectors.len());
    for (col_idx, source) in flat_vectors.iter().enumerate() {
        let dim = options.vector_columns[col_idx].dim;
        let mut ordered = Vec::with_capacity(ordered_locals.len() * dim);
        for &local in ordered_locals {
            let start = local as usize * dim;
            let Some(slice) = source.get(start..start + dim) else {
                return Err(BuildError::Store(format!(
                    "shard vector local {local} out of bounds for column {col_idx}"
                )));
            };
            ordered.extend_from_slice(slice);
        }
        ordered_vectors.push(ordered);
    }
    let vector_slices: Vec<&[f32]> = ordered_vectors.iter().map(Vec::as_slice).collect();

    let mut builder = SuperfileBuilder::new(
        options
            .builder_options()
            .with_vector_layout(VectorLayout::MultiCellIvf),
    )?;
    builder.add_batch(&scalar, &vector_slices)?;

    let scalar_schema = options.scalar_schema();
    let scalar_stats = ScalarStatsAgg::from_batches(&scalar_schema, &[&scalar]);

    let id_col = scalar
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| {
            BuildError::IdColumnWrongType(
                options.id_column.clone(),
                "<id column not Decimal128 at runtime>".to_string(),
            )
        })?;
    let mut id_min = i128::MAX;
    let mut id_max = i128::MIN;
    for i in 0..id_col.len() {
        let v = id_col.value(i);
        id_min = id_min.min(v);
        id_max = id_max.max(v);
    }
    let n_docs = id_col.len() as u64;
    let (id_min, id_max) = if n_docs == 0 {
        (0, 0)
    } else {
        (id_min, id_max)
    };
    Ok((builder, id_min, id_max, n_docs, scalar_stats))
}

/// Same as [`build_one_shard_with_layout`] but feeds Sq8+ε materialized IVF rows
/// into the normal vector builder — no fp32 corpus decode.
fn build_one_shard_from_materialized(
    rows: &[MaterializedIvfRow],
    options: &SupertableOptions,
    vector_layout: crate::superfile::vector::layout::VectorLayout,
) -> Result<ShardOutput, BuildError> {
    let id_array = Decimal128Array::from_iter_values(rows.iter().map(|r| r.stable_id))
        .with_precision_and_scale(
            crate::supertable::options::DECIMAL128_PRECISION,
            crate::supertable::options::DECIMAL128_SCALE,
        )
        .expect("invariant: precision 38 + scale 0 always valid for any i128 payload");
    let scalar = RecordBatch::try_new(
        options.scalar_schema(),
        vec![Arc::new(id_array) as ArrayRef],
    )
    .map_err(|_| BuildError::BatchSchemaMismatch)?;

    let mut builder =
        SuperfileBuilder::new(options.builder_options().with_vector_layout(vector_layout))?;
    builder.add_batch_ids_only(&scalar)?;
    builder.load_materialized_ivf_rows(rows.to_vec())?;

    let id_min = rows.iter().map(|r| r.stable_id).min().unwrap_or(0);
    let id_max = rows.iter().map(|r| r.stable_id).max().unwrap_or(0);
    let n_docs = rows.len() as u64;
    let scalar_stats = ScalarStatsAgg::from_batches(&options.scalar_schema(), &[&scalar]);
    let bytes = Bytes::from(builder.finish()?);

    Ok(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    })
}

/// Minimum overflow rows required to split a cell into two sub-cells — a split
/// needs at least one row per side, so fewer than this is a no-op.
const MIN_ROWS_TO_SPLIT_CELL: usize = 2;

/// OPANN steps 7–9 after hidden compaction: find the overflow **global cell**
/// via the merged file's cell directory (not `partition_hint`, which is a
/// shard id for packed files), Sq8-split it, pull neighborhood cells out of
/// packed/legacy files by cell directory, redrive those rows through
/// `INCOMING_VECTOR_CELL`, and republish any non-neighborhood cells that
/// shared a packed shard.
pub(in crate::supertable) async fn split_overflow_cell_after_compaction(
    inner: Arc<SupertableInner>,
    merged_entry: &Arc<SuperfileEntry>,
) -> Result<(), BuildError> {
    let manifest = inner.manifest.load_full();
    let (clusters, column, routing, metric, _vec_dim) = match manifest.get_partition_strategy() {
        PartitionStrategy::VectorCell {
            clusters,
            column,
            routing,
        } => {
            let Some(vec_col) = inner.options.vector_columns.first() else {
                return Ok(());
            };
            (clusters, column, routing, vec_col.metric, vec_col.dim)
        }
        _ => return Ok(()),
    };
    if clusters.n_cent == 0 || clusters.dim == 0 {
        return Ok(());
    }

    let now = time::Instant::now();
    let cell_counts = cell_doc_counts_for_entry(&inner, merged_entry).await?;
    // Overflow cell = largest cell in this merged object that exceeds the cap.
    // Packed shards may hold many cells; never treat partition_hint (shard id)
    // as a cell id.
    let Some((split_cell, split_n_docs)) = cell_counts
        .iter()
        .copied()
        .filter(|(cell, _)| *cell != super::handle::INCOMING_VECTOR_CELL)
        .filter(|(_, n)| opann::split_overflow_needed(u64::from(*n)))
        .max_by_key(|(_, n)| *n)
    else {
        return Ok(());
    };
    if (split_n_docs as usize) < MIN_ROWS_TO_SPLIT_CELL {
        return Ok(());
    }

    let storage = inner
        .options
        .storage
        .clone()
        .ok_or_else(|| BuildError::Store("cell split requires storage".into()))?;

    let overflow_materialized = load_materialized_rows_from_ivf_superfile(
        &inner,
        merged_entry,
        &column,
        now,
        Some(&[split_cell]),
    )
    .await?;
    if overflow_materialized.len() < MIN_ROWS_TO_SPLIT_CELL {
        return Ok(());
    }
    let overflow_encoded: Vec<EncodedCellRow> = overflow_materialized
        .iter()
        .map(|r| r.encoded.clone())
        .collect();

    let (sub0, sub1) = maint_pool()
        .install(|| opann::plan_sq8_split(&overflow_encoded, &clusters, split_cell, metric));
    let mut sub_centroids = sub0;
    sub_centroids.extend_from_slice(&sub1);

    let old_n_cent = clusters.n_cent;
    let (mut updated_clusters, new_cell_id) =
        opann::insert_split_centroid(&clusters, split_cell, &sub_centroids);
    let neighborhood = opann::reassign_neighborhood(split_cell, old_n_cent, new_cell_id);
    let neighborhood_slice: Vec<u32> = neighborhood
        .iter()
        .copied()
        .filter(|&c| c != super::handle::INCOMING_VECTOR_CELL)
        .collect();

    // Select files that actually contain a neighborhood cell (cell directory
    // for packed; partition_hint == cell_id for legacy).
    let mut to_remove: Vec<Arc<SuperfileEntry>> = Vec::new();
    let mut keep_cells_by_entry: Vec<(Arc<SuperfileEntry>, Vec<u32>)> = Vec::new();
    for entry in manifest.superfiles.iter() {
        if entry.vector_layout == VectorLayout::MultiCellIvf {
            let counts = cell_doc_counts_for_entry(&inner, entry).await?;
            let has_neighborhood = counts
                .iter()
                .any(|(cell, _)| neighborhood_slice.contains(cell));
            if !has_neighborhood {
                continue;
            }
            let keep: Vec<u32> = counts
                .into_iter()
                .map(|(cell, _)| cell)
                .filter(|cell| !neighborhood_slice.contains(cell))
                .collect();
            to_remove.push(Arc::clone(entry));
            if !keep.is_empty() {
                keep_cells_by_entry.push((Arc::clone(entry), keep));
            }
        } else {
            let Some(hint) = entry.partition_hint else {
                continue;
            };
            if neighborhood_slice.contains(&hint) {
                to_remove.push(Arc::clone(entry));
            }
        }
    }

    let mut all_materialized: Vec<MaterializedIvfRow> = Vec::new();
    for entry in &to_remove {
        let mut rows = load_materialized_rows_from_ivf_superfile(
            &inner,
            entry,
            &column,
            now,
            Some(&neighborhood_slice),
        )
        .await?;
        all_materialized.append(&mut rows);
    }
    if all_materialized.is_empty() {
        return Ok(());
    }

    // Rows leave the neighborhood cells; counts reset until routing lands them.
    opann::zero_cell_counts(&mut updated_clusters, &neighborhood);

    // Republish non-neighborhood cells that shared a packed shard so they are
    // not deleted with the neighborhood extract. Use the tombstone-aware
    // loader so deleted locals are not resurrected.
    let mut prepared_keep: Vec<PreparedSuperfile> = Vec::new();
    for (entry, keep_ids) in &keep_cells_by_entry {
        let mut packed: Vec<(u32, MergedIvfSubsection, Vec<i128>)> = Vec::new();
        for &cell_id in keep_ids {
            let mut rows = load_materialized_rows_from_ivf_superfile(
                &inner,
                entry,
                &column,
                now,
                Some(&[cell_id]),
            )
            .await?;
            if rows.is_empty() {
                continue;
            }
            for (i, row) in rows.iter_mut().enumerate() {
                row.local_doc_id = i as u32;
            }
            let stable_ids: Vec<i128> = rows.iter().map(|r| r.stable_id).collect();
            let mut cfg = inner
                .options
                .vector_columns
                .first()
                .cloned()
                .ok_or_else(|| BuildError::Store("missing vector column".into()))?;
            let n_cent = rows
                .iter()
                .map(|r| r.cluster as usize + 1)
                .max()
                .unwrap_or(1)
                .max(1);
            cfg.n_cent = n_cent;
            let sub = crate::superfile::vector::builder::build_merged_subsection_from_materialized(
                cfg, rows,
            )?;
            packed.push((cell_id, sub, stable_ids));
        }
        if packed.is_empty() {
            continue;
        }
        let shard_id = entry.partition_hint.unwrap_or_else(|| {
            packed_cell_shard(packed[0].0, packed_cell_shard_count(&inner.options)) as u32
        });
        prepared_keep.push(build_prepared_from_packed_cells(&inner, shard_id, packed)?);
    }

    let incoming_prepared = maint_pool().install(|| -> Result<PreparedSuperfile, BuildError> {
        let mut rows = all_materialized;
        rows.sort_by_key(|r| r.stable_id);
        for (local, row) in rows.iter_mut().enumerate() {
            row.local_doc_id = local as u32;
        }
        build_prepared_ivf_from_materialized(&inner, super::handle::INCOMING_VECTOR_CELL, rows)
    })?;

    let mut all_prepared = prepared_keep;
    all_prepared.push(incoming_prepared);
    let batch = collect_prepared_superfiles(&inner, all_prepared)?;

    inner
        .manifest
        .store(Arc::new(manifest.with_partition_strategy(
            PartitionStrategy::VectorCell {
                column: column.clone(),
                clusters: updated_clusters.clone(),
                routing,
            },
        )));

    let new_manifest = persist_commit_async(
        &inner,
        Arc::clone(&storage),
        batch.new_entries,
        &to_remove,
        batch.pending_storage_writes,
        Vec::new(),
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    inner.manifest.store(Arc::new(new_manifest));

    schedule_background_storage_reclaim(Arc::clone(&inner));

    Ok(())
}

// OCC retry budget — read from
// `SupertableOptions::max_commit_retries` (default 10) so
// callers with high contention can raise it. The
// `attempt + 1 < retries` check + the final
// `WriteContentionExhausted` return keep the loop bounded
// regardless of the configured value.

/// Jittered exponential backoff between OCC retries.
///
/// Base 10 ms, doubling per attempt, capped at 1 s, with ±30%
/// jitter to break up lockstep retries from racing writers.
/// Jitter source is the low bits of the system's nanosecond
/// clock — no `rand` dep needed.
pub(super) fn backoff_delay(attempt: u32) -> time::Duration {
    const BASE_MS: u64 = 10;
    const CAP_MS: u64 = 1000;
    // Cap the doubling exponent so the pre-cap delay plateaus instead
    // of overflowing the shift on a high attempt count.
    const MAX_SHIFT: u32 = 6;
    // Jitter is a uniform percentage in `-JITTER_RANGE_PCT..=+JITTER_RANGE_PCT`,
    // drawn from the clock's low nanosecond bits. `JITTER_MODULUS`
    // is `2 × JITTER_RANGE_PCT + 1` so the modulo spans the full range.
    const JITTER_RANGE_PCT: i64 = 30;
    const JITTER_MODULUS: u64 = 61;
    const PERCENT_DIVISOR: i64 = 100;
    let exp = BASE_MS.saturating_mul(1u64 << attempt.min(MAX_SHIFT));
    let capped = exp.min(CAP_MS);
    let nanos = time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jitter_pct = (nanos % JITTER_MODULUS) as i64 - JITTER_RANGE_PCT;
    let adjusted = ((capped as i64) + (capped as i64 * jitter_pct / PERCENT_DIVISOR)).max(1) as u64;
    time::Duration::from_millis(adjusted)
}

/// Storage write-through with OCC retry. Persist the new
/// superfiles + manifest to storage, returning the new
/// in-memory `ManifestSnapshot` with the fresh persisted Manifest +
/// loader installed.
///
/// **OCC retry semantics.** On each iteration:
///  1. Reload `inner.manifest` to incorporate any commit a
///     racing writer published since our last attempt.
///  2. Derive `new_superfile_list = old.superfile_list.with_appended(new_entries.clone())`.
///  3. Try `try_commit_attempt` (write superfiles → write part +
///     list → conditional pointer PUT).
///  4. On `WriteContentionExhausted` with retries left: refresh
///     `inner.manifest` from storage (inheriting unchanged
///     parts via content-addressed Arc::clone), sleep with
///     jittered backoff, loop.
///  5. After `opts.max_commit_retries` exhausted: surface
///     `CommitError::WriteContentionExhausted` to the caller.
///
/// **Idempotency across retries.** Superfile URIs are UUID v4 —
/// statically random, so a retry uses the same URIs as the
/// prior attempt. The superfile-bytes PUT swallows
/// `PreconditionFailed` (URI already exists with bit-identical
/// content from our prior attempt). ManifestSnapshot parts are
/// content-addressed; identical content yields identical URIs
/// and the part-write path already swallows
/// `PreconditionFailed`. Only the pointer PUT must win the
/// CAS; everything below it is idempotent.
///
/// When no real partitioning is configured, all post-commit
/// superfiles go into one `ManifestPart` with a fresh `PartId`.
/// With a real `PartitionStrategy`, `try_commit_attempt` runs
/// the per-partition part-reuse path described on that fn.
/// Publish the slow-CAS vector-state blob for `inner`'s CURRENT membership
/// and stamp its ref on the manifest list. Called after a maintenance
/// sequence settles hidden vector membership (end of drain; end of the
/// hidden compaction pass, after merges + finalize + any cell splits) —
/// scoped by call site, never by a table-kind test. `ManifestSnapshot::update`
/// cleared the ref when membership changed; this restamps it so consumers'
/// resident centroid state is invalidated exactly once, by maintenance.
///
/// Writes the content-addressed blob idempotently (`PreconditionFailed` =
/// already durable), then a list+pointer etag-CAS stamp with refresh-and-retry
/// on contention — so a lost race rebuilds the blob from the winning
/// membership, never stamping stale state.
pub(in crate::supertable) async fn refresh_slow_vector_state(
    inner: &SupertableInner,
) -> Result<(), BuildError> {
    let Some(storage) = inner.options.storage.clone() else {
        return Ok(());
    };
    let max_retries = inner.options.max_commit_retries.max(1);
    for attempt in 0..max_retries {
        let old = inner.manifest.load_full();
        let entries = old.get_all_superfiles();
        if entries.is_empty() {
            // Nothing to describe (pre-drain / empty table); the ref is
            // already absent because `update` never carries it forward.
            return Ok(());
        }
        let (uri, hash) = slow_vector_state::write_state(storage.as_ref(), entries)
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
        if let Some((cur_uri, cur_hash)) = old.slow_vector_state_blob()
            && cur_uri == uri
            && cur_hash == hash
        {
            // Same membership already stamped — republish is a no-op.
            return Ok(());
        }
        let new_manifest = old.with_slow_vector_state(uri, hash);
        let prev_etag = get_current_manifest_etag(&storage, Arc::clone(&old))
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
        match new_manifest
            .write(storage.as_ref(), prev_etag.as_deref(), &[])
            .await
        {
            Ok(()) => {
                inner.manifest.store(Arc::new(new_manifest));
                return Ok(());
            }
            Err(SupertableCommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                refresh_inner_state_async(inner, &storage)
                    .await
                    .map_err(|e| BuildError::Store(e.to_string()))?;
                sleep(backoff_delay(attempt)).await;
            }
            Err(e) => return Err(BuildError::Store(e.to_string())),
        }
    }
    Err(BuildError::Store(
        "slow vector-state refresh: write contention exhausted".into(),
    ))
}

async fn record_hidden_deleted_ids(
    inner: &SupertableInner,
    new_deleted: &[i128],
) -> Result<(), BuildError> {
    if new_deleted.is_empty() {
        return Ok(());
    }
    let Some(storage) = inner.options.storage.clone() else {
        return Ok(());
    };
    let max_retries = inner.options.max_commit_retries.max(1);
    for attempt in 0..max_retries {
        let old = inner.manifest.load_full();
        let mut ids = hidden_deleted::deleted_user_ids(&old)
            .map_err(|e| BuildError::Store(e.to_string()))?
            .as_ref()
            .clone();
        let before = ids.len();
        ids.extend_from_slice(new_deleted);
        ids.sort_unstable();
        ids.dedup();
        if ids.len() == before {
            return Ok(());
        }
        let bytes = encode_deleted_ids(&ids);
        let new_manifest = old.with_deleted_user_ids(bytes);
        let prev_etag = get_current_manifest_etag(&storage, Arc::clone(&old))
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
        match new_manifest
            .write(storage.as_ref(), prev_etag.as_deref(), &[])
            .await
        {
            Ok(()) => {
                inner.manifest.store(Arc::new(new_manifest));
                return Ok(());
            }
            Err(SupertableCommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                refresh_inner_state_async(inner, &storage)
                    .await
                    .map_err(|e| BuildError::Store(e.to_string()))?;
                sleep(backoff_delay(attempt)).await;
            }
            Err(e) => return Err(BuildError::Store(e.to_string())),
        }
    }
    Err(BuildError::Store(
        "deleted-set record: write contention exhausted".into(),
    ))
}

pub(in crate::supertable) async fn persist_commit_async(
    inner: &SupertableInner,
    storage: Arc<dyn StorageProvider>,
    new_entries: Vec<Arc<SuperfileEntry>>,
    entries_to_remove: &[Arc<SuperfileEntry>],
    mut pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    mut pending_storage_replaces: Vec<(SuperfileUri, Bytes)>,
) -> Result<ManifestSnapshot, SupertableCommitError> {
    let storage_async = Arc::clone(&storage);
    let opts = Arc::clone(&inner.options);
    let max_retries = opts.max_commit_retries.max(1);
    let drive = async move {
        let mut last_err: Option<SupertableCommitError> = None;
        for attempt in 0..max_retries {
            let old = inner.manifest.load_full();
            let pending_writes = &mut pending_storage_writes;
            let pending_replaces = &mut pending_storage_replaces;
            match try_commit_attempt(
                Arc::clone(&storage_async),
                Arc::clone(&opts),
                Arc::clone(&old),
                &new_entries,
                entries_to_remove,
                NewEntryBirthVersions::StampCommit,
                pending_writes,
                pending_replaces,
            )
            .await
            {
                Ok(new_manifest) => return Ok(new_manifest),
                Err(SupertableCommitError::WriteContentionExhausted)
                    if attempt + 1 < max_retries =>
                {
                    refresh_inner_state_async(inner, &storage_async).await?;
                    last_err = Some(SupertableCommitError::WriteContentionExhausted);
                    sleep(backoff_delay(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or(SupertableCommitError::WriteContentionExhausted))
    };
    // Genuinely async: callers `.await` this from async contexts already driven
    // on `query_runtime`. Driving it to completion here with a nested `block_on`
    // would serialize the `tokio::join!` in `commit` (the user + hidden publishes
    // are meant to overlap) and risk a nested-block_on panic. The sync→async
    // bridge lives only in the `persist_commit` wrapper below.
    drive.await
}

pub(in crate::supertable) fn persist_commit(
    inner: &SupertableInner,
    storage: Arc<dyn StorageProvider>,
    new_entries: Vec<Arc<SuperfileEntry>>,
    entries_to_remove: &[Arc<SuperfileEntry>],
    pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    let drive = persist_commit_async(
        inner,
        storage,
        new_entries,
        entries_to_remove,
        pending_storage_writes,
        pending_storage_replaces,
    );
    let new_manifest = bridge_on_runtime(drive, &inner.query_runtime())?;
    inner.manifest.store(Arc::new(new_manifest));
    Ok(())
}

// Writes the superfile list to storage. Performs the side-effect of modifying pending_storage_writes
// to remove successfully written entries.
// Swallow `PreconditionFailed` per-PUT: on a retry after a
// lost pointer-CAS, the same URI was already written by
// our prior attempt with bit-identical bytes (superfile URIs
// are UUID v4 — collision rate 2^-122). A "URI exists"
// hit here means our own prior attempt; treat as success
// so the retry path is fully idempotent.
//
// Size-gated dispatch: superfiles ≥
// `put_multipart_threshold_bytes` route through
// `put_multipart` (S3 multipart upload, in-place
// streaming on LocalFS) instead of a single `put_atomic`
// PUT. Smaller superfiles stay on the single-PUT path —
// multipart has per-request overhead that isn't worth
// the parallelism below the threshold. The default
// threshold (100 MiB) matches the S3 SDK's standard
// cutoff.
async fn put_superfile_replace(
    storage: &Arc<dyn StorageProvider>,
    path: &str,
    bytes: Bytes,
) -> Result<(), StorageError> {
    match storage.head(path).await {
        Ok(meta) => storage
            .put_if_match(path, bytes, meta.etag.as_deref())
            .await
            .map(|_| ()),
        Err(StorageError::NotFound { .. }) => storage.put_atomic(path, bytes).await.map(|_| ()),
        Err(e) => Err(e),
    }
}

/// Commit-time object-store write fanout width: half the machine's CPU
/// parallelism, floored at 1. A single commit and a concurrent background
/// maintenance compaction each fan out their PUTs at this width, so keeping
/// each at ~50% of cores bounds the combined in-flight PUTs to roughly the
/// core count rather than a multiple of it.
fn commit_write_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(1)
        .max(1)
}

/// Upper bound on the drain's auto-sized read fan-out — keeps a very large box
/// from stampeding a single S3 prefix. An explicit env override is not clamped.
const DRAIN_READ_CONCURRENCY_CAP: usize = 64;

/// Drain upload threshold: disable multipart so every drain superfile is
/// written through a single create-only `put_atomic`.
const DRAIN_PUT_MULTIPART_THRESHOLD_BYTES: u64 = u64::MAX;

/// Read fan-out for the drain's superfile opens — bulk S3 reads off the
/// query-critical path. Ideal sizing tracks network bandwidth; vCPU count is the
/// portable runtime proxy for it (a cloud instance's NIC scales with its size).
/// The auto default is one in-flight read per hardware thread, floored at the
/// read layer's background-fill default (`prefetch_concurrency`) so small boxes
/// still fan out, and capped at [`DRAIN_READ_CONCURRENCY_CAP`]. Sourced from
/// `vector.drain_read_concurrency`; an explicit integer there is used verbatim
/// (unclamped), while `auto` applies the vCPU-derived default.
fn drain_read_concurrency() -> usize {
    if let ThreadCount::Fixed(n) = config::global().vector.drain_read_concurrency
        && n > 0
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(
            crate::config::DEFAULT_PREFETCH_CONCURRENCY,
            DRAIN_READ_CONCURRENCY_CAP,
        )
}

pub async fn write_superfile_list(
    storage: &Arc<dyn StorageProvider>,
    opts: &Arc<SupertableOptions>,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    write_superfile_list_with_threshold(
        storage,
        opts,
        opts.put_multipart_threshold_bytes,
        pending_storage_writes,
        pending_storage_replaces,
    )
    .await
}

async fn write_superfile_list_with_threshold(
    storage: &Arc<dyn StorageProvider>,
    _opts: &Arc<SupertableOptions>,
    put_multipart_threshold_bytes: u64,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    // Bound object-store fanout to half the machine's CPU parallelism. A vector
    // commit can stage one hidden delta per touched cell plus user shards;
    // driving all PUTs at once opens dozens of sockets and can stall the commit
    // path. Crucially, bulk ingest commits overlap background hidden-index
    // OPANN maintenance (its own compaction PUT/GET waves), so a full-width
    // fanout from each stacks and starves the connection pool until requests
    // hit the per-request timeout. Capping each operation at ~50% of cores
    // leaves headroom for a concurrent maintenance pass without saturation.
    let write_concurrency = commit_write_concurrency();

    let replace_futs = pending_storage_replaces
        .iter()
        .enumerate()
        .map(|(i, (uri, bytes))| {
            let storage = Arc::clone(storage);
            let uri = *uri;
            let bytes = bytes.clone();
            async move {
                let path = superfile_storage_path(&uri);
                put_superfile_replace(&storage, &path, bytes)
                    .await
                    .map(|()| i)
                    .map_err(SupertableCommitError::from)
            }
        });
    let mut err = None;
    let mut successful_replace_idx = Vec::with_capacity(pending_storage_replaces.len());
    for r in stream::iter(replace_futs)
        .buffer_unordered(write_concurrency)
        .collect::<Vec<_>>()
        .await
    {
        match r {
            Ok(i) => successful_replace_idx.push(i),
            Err(e) => err = Some(e),
        }
    }
    successful_replace_idx.sort_unstable_by(|a, b| b.cmp(a));
    for idx in successful_replace_idx {
        pending_storage_replaces.remove(idx);
    }
    if let Some(e) = err {
        return Err(e);
    }

    let multipart_threshold = put_multipart_threshold_bytes;
    let put_futs = pending_storage_writes
        .iter()
        .enumerate()
        .map(|(i, (uri, bytes))| {
            let storage = Arc::clone(storage);
            let uri = *uri;
            let bytes = bytes.clone();
            async move {
                let path = superfile_storage_path(&uri);
                let result = if (bytes.len() as u64) >= multipart_threshold {
                    put_superfile_multipart(storage.as_ref(), &path, bytes.clone()).await
                } else {
                    storage.put_atomic(&path, bytes.clone()).await.map(|_| ())
                };
                match result {
                    Ok(()) => Ok(i),
                    Err(StorageError::PreconditionFailed { .. }) => Ok(i),
                    Err(e) => Err(SupertableCommitError::from(e)),
                }
            }
        });

    let mut err = None;
    let mut successful_writes_idx = Vec::with_capacity(pending_storage_writes.len());

    for r in stream::iter(put_futs)
        .buffer_unordered(write_concurrency)
        .collect::<Vec<_>>()
        .await
    {
        match r {
            Ok(i) => successful_writes_idx.push(i),
            Err(e) => err = Some(e),
        }
    }

    successful_writes_idx.sort_unstable_by(|a, b| b.cmp(a));
    for idx in successful_writes_idx {
        pending_storage_writes.remove(idx);
    }

    if let Some(e) = err {
        return Err(e);
    }

    Ok(())
}

/// One attempt at the commit sequence: write superfile bytes
/// → group new entries by partition → rewrite the latest part
/// per touched partition (preserving untouched parts' URIs)
/// → conditional pointer PUT. The retry loop in
/// `persist_commit` wraps this to handle contention.
///
/// **Partition-aware path.** Each commit's new superfiles are
/// routed by `assign_partition` into per-partition groups.
/// For each touched partition, the writer finds the latest
/// existing part (if any), rebuilds it with the union of its
/// existing superfiles + the new ones, and emits a new
/// `ManifestPartEntry` that replaces the prior one (same
/// `partition_key`, new `part_id` + content hash). Untouched
/// partitions' list entries carry over verbatim — no
/// re-encode, no PUT. A cold partition (no prior entry) gets
/// a fresh part with just the new superfiles. The result: a
/// single-partition commit rewrites exactly one part
/// regardless of how many other partitions exist — the
/// load-bearing property the part-reuse optimization relies
/// on.
#[derive(Clone, Copy)]
pub(crate) enum NewEntryBirthVersions {
    /// User append/update data is born in the manifest commit publishing it.
    StampCommit,
    /// Compaction changes physical residency but preserves logical lineage.
    Preserve,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn try_commit_attempt(
    storage: Arc<dyn StorageProvider>,
    opts: Arc<SupertableOptions>,
    current_manifest: Arc<ManifestSnapshot>,
    new_entries: &[Arc<SuperfileEntry>],
    entries_to_remove: &[Arc<SuperfileEntry>],
    birth_versions: NewEntryBirthVersions,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<ManifestSnapshot, SupertableCommitError> {
    // 1. Write each new superfile's bytes to storage in parallel.
    write_superfile_list(
        &storage,
        &opts,
        pending_storage_writes,
        pending_storage_replaces,
    )
    .await?;

    // 2. update the manifest for the commit.
    let (new_manifest, parts_to_write) = match birth_versions {
        NewEntryBirthVersions::StampCommit => {
            current_manifest
                .update(new_entries, entries_to_remove)
                .await?
        }
        NewEntryBirthVersions::Preserve => {
            current_manifest
                .update_preserving_birth_versions(new_entries, entries_to_remove)
                .await?
        }
    };

    // 3. Read the prior pointer's etag for the CAS. Fresh
    //    supertable → no pointer yet → None etag (initial
    //    commit).
    let prev_etag = get_current_manifest_etag(&storage, current_manifest).await?;

    // 4. Parallel-issue (touched parts) + list PUTs, then
    //    conditional pointer PUT (the visibility barrier).
    //    Untouched parts are NOT re-PUT — their URIs (and
    //    content-hashes) are unchanged in the new list.
    let encoded_refs: Vec<&[u8]> = parts_to_write
        .iter()
        .map(|ep| ep.encoded.as_slice())
        .collect();
    new_manifest
        .write(storage.as_ref(), prev_etag.as_deref(), &encoded_refs)
        .await?;
    // Silence the unused-import warning when no path uses
    // `PartId` / `part_mod` directly (helpers consume them
    // from inside `build_part_and_entry`).
    let _ = PhantomData::<(PartId, part_mod::ContentHash)>;

    Ok(new_manifest)
}

/// Re-read the manifest pointer from storage, load any newer
/// manifest list, inherit unchanged parts from the current
/// in-memory `ManifestSnapshot` via content-addressed `Arc::clone`,
/// eager-fetch newly-referenced parts, and `ArcSwap` the
/// refreshed `ManifestSnapshot` into `inner.manifest`.
///
/// Called from the OCC retry loop between attempts so the next
/// iteration's `inner.manifest.load_full()` sees the winning
/// writer's state — `with_appended` then chains our pending
/// superfiles onto theirs at the new monotonic `manifest_id`.
///
/// Mirrors the logic in [`Supertable::refresh`] but operates
/// on `&SupertableInner` so it can be called from inside the
/// writer's commit path without holding a `Supertable` handle.
pub(in crate::supertable) async fn refresh_inner_state_async(
    inner: &SupertableInner,
    storage: &Arc<dyn StorageProvider>,
) -> Result<(), SupertableCommitError> {
    let current = inner.manifest.load_full();
    let manifest = match ManifestSnapshot::load(Some(current), storage.clone(), None).await {
        Ok(manifest) => manifest,
        Err(ManifestLoadError::PointerNotFound) => return Ok(()),
        Err(ManifestLoadError::AlreadyLoaded) => return Ok(()),
        Err(err) => {
            return Err(SupertableCommitError::ManifestError(
                ManifestError::ManifestLoadError(err),
            ));
        }
    };
    inner.manifest.store(manifest);
    Ok(())
}

/// Storage path for a superfile's bytes. Lives under `data/`
/// alongside the `_supertable/` manifest hierarchy.
/// IPC-encode a `RecordBatch` to a byte buffer. Mirrors the
/// shape the WAL's arrow sidecar carries: an
/// `arrow_ipc::writer::StreamWriter` writes one batch followed
/// by a finish marker. The recovery / append-phase reader
/// decodes the same way.
fn encode_record_batch_ipc(batch: &RecordBatch) -> Result<Bytes, String> {
    let mut out: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, &batch.schema())
            .map_err(|e| format!("ipc writer init: {e}"))?;
        writer.write(batch).map_err(|e| format!("ipc write: {e}"))?;
        writer.finish().map_err(|e| format!("ipc finish: {e}"))?;
    }
    Ok(Bytes::from(out))
}

fn superfile_storage_path(uri: &SuperfileUri) -> String {
    uri.storage_path()
}

/// Multipart-upload variant of the writer's per-superfile put.
/// Routes through [`crate::storage::StorageProvider::put_multipart`]
/// for superfiles large enough that a single PUT is wasteful
/// (slow on a backend stall, high RSS during the put).
///
/// Idempotency: superfile URIs are UUID v4, so the only "URI
/// exists" hit on retry comes from our own prior attempt
/// with bit-identical bytes. Head-first lets us short-circuit
/// that case before re-running the multipart dance. The
/// single-PUT path achieves the same effect by returning
/// `PreconditionFailed`, which the call-site swallows;
/// multipart's `complete()` doesn't carry a precondition, so
/// we need to detect "already there" explicitly.
///
/// Part size: 8 MiB — comfortably above S3's 5-MiB minimum
/// and a clean fit for the cold-fetch coordinator's default
/// 16-MiB chunk reads on the way back out. Parts are pushed
/// in declaration order; the parts run concurrently inside
/// `object_store` after their futures are polled.
async fn put_superfile_multipart(
    storage: &dyn StorageProvider,
    path: &str,
    bytes: Bytes,
) -> Result<(), StorageError> {
    const PART_BYTES: usize = 8 * (1 << 20);

    // Same-bytes retry skip. Failures other than NotFound
    // propagate so we don't paper over a degraded backend.
    match storage.head(path).await {
        Ok(_) => return Err(StorageError::PreconditionFailed { uri: path.into() }),
        Err(StorageError::NotFound { .. }) => {}
        Err(e) => return Err(e),
    }

    let mut upload = storage.put_multipart(path).await?;
    let total = bytes.len();
    let mut parts: Vec<UploadPart> = Vec::with_capacity(total / PART_BYTES + 1);
    let mut offset = 0;
    while offset < total {
        let end = cmp::min(offset + PART_BYTES, total);
        let chunk = bytes.slice(offset..end);
        parts.push(upload.put_part(PutPayload::from_bytes(chunk)));
        offset = end;
    }
    // Drive part-uploads concurrently. `try_join_all` cancels
    // remaining parts if one fails — semantically equivalent to
    // abandoning the upload, with `abort()` below as cleanup.
    if let Err(e) = try_join_all(parts).await {
        // Best-effort abort; ignore failure (the upload may
        // already be in a terminal state, or the backend may
        // have lost the multipart-upload ID).
        let _ = upload.abort().await;
        return Err(StorageError::Permanent {
            uri: path.into(),
            source: Box::new(e),
        });
    }
    if let Err(e) = upload.complete().await {
        let _ = upload.abort().await;
        return Err(StorageError::Permanent {
            uri: path.into(),
            source: Box::new(e),
        });
    }
    Ok(())
}

/// After a successful compaction manifest commit: warm-insert the merged
/// output into the disk cache and schedule deferred reclaim of superseded
/// superfiles. Superseded cache entries are left to the LRU — they are no
/// longer manifest-visible and will age out.
pub(in crate::supertable) async fn finalize_compaction_commit(
    inner: Arc<SupertableInner>,
    _storage: &Arc<dyn crate::storage::StorageProvider>,
    _new_entries: &[Arc<SuperfileEntry>],
    _entries_to_remove: &[Arc<SuperfileEntry>],
    pending_cache_inserts: Vec<(SuperfileUri, Bytes)>,
) {
    schedule_background_storage_reclaim(Arc::clone(&inner));
    if !pending_cache_inserts.is_empty()
        && let Some(cache) = inner.options.disk_cache.as_ref().cloned()
    {
        warm_cache_after_commit(&inner, &cache, pending_cache_inserts);
    }
    if let (Some(cache), Some(budget)) = (
        inner.options.disk_cache.as_ref(),
        inner.options.memory_budget_bytes,
    ) {
        cache.sweep_for_budget(budget);
    }
}

/// Pre-populate the warm cache with each just-published superfile's bytes.
///
/// Best-effort: each failure is swallowed with a tracing warning — the
/// superfiles are already durable in storage and the manifest commit has
/// succeeded, so a cache miss becomes a cold-fetch on first read, not a
/// correctness break. Shared by every commit/route finalize path so the
/// loop + warning text live in one place.
async fn warm_cache_inserts(cache: &Arc<DiskCacheStore>, inserts: Vec<(SuperfileUri, Bytes)>) {
    for (uri, bytes) in inserts {
        if let Err(e) = cache.insert_warm(&uri, bytes).await {
            tracing::warn!(
                "supertable: warm cache pre-population failed for {}: {} \
                 (superfile is durable in storage; first query will cold-fetch)",
                uri.0,
                e
            );
        }
    }
}

/// Sync entry point for [`warm_cache_inserts`]: drives it on `query_runtime`
/// via the shared [`bridge_on_runtime`] bridge (the disk cache's async
/// coordination is bound to that runtime).
fn warm_cache_after_commit(
    inner: &SupertableInner,
    cache: &Arc<DiskCacheStore>,
    pending: Vec<(SuperfileUri, Bytes)>,
) {
    let cache = Arc::clone(cache);
    bridge_on_runtime(warm_cache_inserts(&cache, pending), &inner.query_runtime());
}

pub(crate) fn read_vector_layout_from_bytes(bytes: &Bytes) -> VectorLayout {
    match read_kv_metadata(bytes.as_ref()) {
        Ok(kvs) => vector_layout_from_kv(&kvs),
        Err(_) => VectorLayout::Ivf,
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use arrow_array::{
        Array, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};
    use figment::{
        Figment,
        providers::{Format, Yaml},
    };
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        config::Config,
        superfile::{
            builder::{FtsConfig, VectorConfig},
            fts::reader::BoolMode,
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{SupertableOptions, handle::Supertable, storage::LocalFsStorageProvider},
        test_helpers::default_tokenizer as tok,
    };

    /// Small fixed vector dimension accepted by the vector builder.
    const COMMIT_AS_DRAIN_TEST_DIM: usize = 16;
    /// Small row count that still exercises multiple global cells.
    const COMMIT_AS_DRAIN_TEST_ROWS: usize = 8;
    /// Boundary test target that permits one extra posting per input row.
    const BOUNDARY_STUB_TARGET_FACTOR: f32 = 2.0;

    fn schema_id_title() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    fn schema_id_title_emb(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]))
    }

    fn options_id_title() -> SupertableOptions {
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
    }

    /// Force a single-threaded writer pool for deterministic
    /// shard counts in tests.
    fn options_id_title_serial() -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("build pool"),
        );
        options_id_title().with_writer_pool(pool)
    }

    /// Build a writer pool with N threads.
    fn writer_pool_with(n: usize) -> Arc<rayon::ThreadPool> {
        Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("build pool"),
        )
    }

    fn build_simple_batch(_start: u64, n: usize) -> RecordBatch {
        // The supertable injects `_id` at append time; the
        // user-facing batch carries only the user columns.
        let titles =
            LargeStringArray::from((0..n).map(|i| format!("doc {i} alpha")).collect::<Vec<_>>());
        RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles)]).expect("build batch")
    }

    fn options_title_emb_serial(dim: usize, n_cent: usize) -> SupertableOptions {
        SupertableOptions::new(
            schema_id_title_emb(dim),
            vec![],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(writer_pool_with(1))
    }

    fn build_axis_vector_batch(n: usize, dim: usize) -> RecordBatch {
        let titles =
            LargeStringArray::from((0..n).map(|i| format!("doc {i} beta")).collect::<Vec<_>>());
        let mut flat = Vec::with_capacity(n * dim);
        for row in 0..n {
            for d in 0..dim {
                flat.push(if d == row % dim { 1.0 } else { 0.0 });
            }
        }
        let values = Arc::new(Float32Array::from(flat));
        let list = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            values,
            None,
        )
        .expect("fixed-size list");
        RecordBatch::try_new(
            schema_id_title_emb(dim),
            vec![Arc::new(titles), Arc::new(list)],
        )
        .expect("vector batch")
    }

    fn committed_reader(st: &Supertable) -> (Arc<SuperfileEntry>, Arc<SuperfileReader>) {
        let entry = Arc::clone(&st.reader().manifest().superfiles[0]);
        let reader = st
            .options()
            .store
            .reader(&entry.uri)
            .expect("committed superfile reader");
        (entry, reader)
    }

    // ---- writer slot exclusion ---------------------------------------

    #[test]
    fn writer_slot_is_exclusive() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let _w = st.writer().expect("first writer");
        let err = st.writer().expect_err("second writer should fail");
        assert!(matches!(err, BuildError::SupertableInUse));
    }

    #[test]
    fn writer_slot_releases_on_drop() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        {
            let _w = st.writer().expect("first writer");
            // dropped at scope end
        }
        // Slot now free.
        let _w2 = st.writer().expect("second writer after drop");
    }

    // ---- single-writer end-to-end (serial pool) ----------------------

    #[test]
    fn append_then_commit_publishes_one_superfile() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 4)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.manifest_id(), 1);
        assert_eq!(r.n_superfiles(), 1);
        assert_eq!(r.n_docs_total(), 4);
    }

    #[test]
    fn commit_with_empty_buffer_is_noop() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.commit().expect("commit-empty");
        assert_eq!(st.manifest_id(), 0, "no manifest swap on empty commit");
        assert_eq!(st.reader().n_superfiles(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn superfile_is_queryable_via_store() {
        // The published superfile's bytes are in the store; we
        // can fetch a SuperfileReader and run bm25_search on it.

        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 4)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let superfile = &r.manifest().superfiles[0];
        let store = &st.options().store;
        let sf_reader = store.reader(&superfile.uri).expect("reader");
        let hits = sf_reader
            .bm25_hits_async("title", "alpha", 10, BoolMode::Or)
            .await
            .expect("bm25");
        // All 4 docs contain "alpha"; should all be returned.
        assert_eq!(hits.len(), 4);
    }

    // ---- id_min / id_max + n_docs ------------------------------------

    #[test]
    fn superfile_entry_records_id_range_and_n_docs() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(100, 3)).expect("a");
        w.append(&build_simple_batch(50, 2)).expect("b");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        assert_eq!(seg.n_docs, 5);
        // _id values are auto-injected via the supertable's
        // monotonic generator. We don't know the exact values
        // (timestamp-prefixed); we just assert that min < max
        // and both are positive (high bit 0).
        assert!(seg.id_min > 0);
        assert!(seg.id_max > seg.id_min, "id_max should exceed id_min");
    }

    // ---- FTS summary --------------------------------------------------

    #[test]
    fn superfile_entry_carries_fts_summary() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 4)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        let fts = seg
            .fts_summary
            .get("title")
            .expect("title FTS summary present");

        // Each doc's title is "doc <i> alpha"; tokenized with
        // ASCII-lower, distinct terms include "doc", "alpha",
        // and digits 0-3. The FST will dedupe; n_terms_distinct
        // is at least 3 (doc, alpha, plus some digit tokens).
        assert!(
            fts.n_terms_distinct >= 3,
            "expected ≥ 3 distinct terms, got {}",
            fts.n_terms_distinct,
        );
        // Bloom should report present for inserted terms.
        assert!(fts.may_contain(b"alpha"));
        assert!(fts.may_contain(b"doc"));
        // Lex range should be present and consistent.
        let (min_term, max_term) = fts.term_range.as_ref().expect("non-empty FST has a range");
        assert!(!min_term.is_empty());
        assert!(!max_term.is_empty());
        assert!(min_term <= max_term, "min_term <= max_term invariant");
    }

    // ---- vector summary ----------------------------------------------

    fn build_vector_batch(_start: u64, n: usize, dim: usize) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                flat.push(((i + j) as f32) / 100.0);
            }
        }
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), None)
            .expect("FSL");
        RecordBatch::try_new(
            schema_id_title_emb(dim),
            vec![Arc::new(titles), Arc::new(fsl)],
        )
        .expect("batch")
    }

    fn options_with_vector(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("build pool"),
        );
        SupertableOptions::new(
            schema_id_title_emb(dim),
            vec![],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
            None,
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    #[test]
    fn superfile_entry_carries_vector_summary() {
        let dim = 16;
        let st = Supertable::create(options_with_vector(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        // Need at least n_cent docs so kmeans has data to cluster.
        w.append(&build_vector_batch(0, 8, dim)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        let vs = seg
            .vector_summary
            .get("emb")
            .expect("emb vector summary present");
        assert_eq!(vs.centroid.len(), dim);
        // Per-cluster centroids are staged into the manifest for
        // cross-superfile global cluster selection.
        assert!(
            vs.cells.iter().any(|cell| !cell.clusters.is_empty()),
            "cluster centroids must be populated"
        );
        assert!(vs.cells.iter().all(|cell| {
            cell.clusters.dim as usize == dim
                && cell.clusters.n_cent >= 1
                && cell.clusters.counts.len() == cell.clusters.n_cent as usize
                && cell.clusters.centroids.len() == cell.clusters.n_cent as usize * dim
        }));
        // Every indexed doc lands in exactly one cluster, so the
        // per-cluster counts sum to the superfile's doc count.
        let total: u64 = vs
            .cells
            .iter()
            .flat_map(|cell| cell.clusters.counts.iter())
            .map(|&count| count as u64)
            .sum();
        assert_eq!(total, seg.n_docs);
    }

    #[test]
    fn grid_commit_writes_multicell_parquet_in_vector_order() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            options_title_emb_serial(COMMIT_AS_DRAIN_TEST_DIM, COMMIT_AS_DRAIN_TEST_ROWS)
                .with_storage(storage),
        )
        .expect("create");
        assert!(
            !st.reader().options().vector_columns.is_empty(),
            "fixture must declare vector columns so commit takes the assign-pack path"
        );
        let mut w = st.writer().expect("writer");
        w.append(&build_axis_vector_batch(
            COMMIT_AS_DRAIN_TEST_ROWS,
            COMMIT_AS_DRAIN_TEST_DIM,
        ))
        .expect("append");
        w.commit().expect("commit");

        let (entry, reader) = committed_reader(&st);
        assert_eq!(entry.vector_layout, VectorLayout::MultiCellIvf);
        assert_eq!(entry.n_docs, COMMIT_AS_DRAIN_TEST_ROWS as u64);

        let vec_reader = reader.vec().expect("vector reader");
        assert!(vec_reader.is_multi_cell());
        let vector_locals: Vec<u32> = (0..vec_reader.n_docs() as u32).collect();
        let vector_ids = vec_reader
            .inline_stable_ids_for_locals(&vector_locals)
            .expect("inline stable ids");

        let parquet_locals: Vec<u32> = (0..entry.n_docs as u32).collect();
        let parquet_batch = reader
            .take_by_local_doc_ids(&parquet_locals, &["_id"])
            .expect("read parquet ids");
        let parquet_ids = parquet_batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids")
            .values()
            .to_vec();
        assert_eq!(parquet_ids, vector_ids);
    }

    #[test]
    fn assign_pack_boundary_replicas_are_vector_only_stubs() {
        let dim = COMMIT_AS_DRAIN_TEST_DIM;
        let mut centroids = vec![0.0f32; dim * 2];
        centroids[dim] = 1.0;
        let clusters = ClusterCentroids::from_fp32(2, dim as u32, &centroids, vec![0, 0]);
        let vectors = [
            vec![0.49; dim],
            vec![0.51; dim],
            vec![0.48; dim],
            vec![0.52; dim],
        ];
        let stable_ids = [10_i128, 11, 12, 13];
        let rows: Vec<PackRow<'_>> = vectors
            .iter()
            .zip(stable_ids)
            .map(|(vector, stable_id)| PackRow::Fp32 { stable_id, vector })
            .collect();
        let assigned = assign_cells(
            &rows,
            &clusters,
            Metric::L2Sq,
            false,
            BOUNDARY_STUB_TARGET_FACTOR,
        )
        .expect("assign");

        let postings: usize = assigned.iter().map(|group| group.members.len()).sum();
        let primaries: usize = assigned
            .iter()
            .flat_map(|group| group.members.iter())
            .filter(|(_, is_primary, _)| *is_primary)
            .count();
        assert_eq!(primaries, rows.len());
        assert!(
            postings > primaries,
            "boundary replicas add vector postings, not primary rows"
        );

        let cfg = VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: 2,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        for group in assigned {
            let n_members = group.members.len();
            let packed = drain_pack_assigned_cell(group, &cfg).expect("drain pack");
            assert_eq!(packed.stable_ids.len(), n_members);
            assert_eq!(packed.subsection.n_docs as usize, n_members);
        }
    }

    #[test]
    fn drain_fine_centroids_follow_two_mib_run_target() {
        const DIM: usize = 1024;
        const COMMIT_CELL_ROWS: usize = 98;
        const DRAINED_CELL_ROWS: usize = 1_562;

        let cfg = VectorConfig {
            column: "emb".into(),
            dim: DIM,
            n_cent: 64,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        assert_eq!(
            drain_cell_vector_config(&cfg, COMMIT_CELL_ROWS).n_cent,
            1,
            "a small commit delta fits one ~2 MiB fine run"
        );
        assert_eq!(
            drain_cell_vector_config(&cfg, DRAINED_CELL_ROWS).n_cent,
            2,
            "a fully drained cell needs two ~2 MiB fine runs"
        );
    }

    // ---- rayon-shard parallelism -------------------------------------

    #[test]
    fn commit_produces_one_superfile_per_writer_pool_thread() {
        // With N writer-pool threads and a buffer of M >= N
        // batches, commit should emit N superfiles (one per
        // shard).
        for n_threads in [1usize, 2, 4] {
            let opts = options_id_title().with_writer_pool(writer_pool_with(n_threads));
            let st = Supertable::create(opts).expect("create");
            let mut w = st.writer().expect("writer");
            // Push enough batches to fill every shard.
            for i in 0..n_threads * 2 {
                w.append(&build_simple_batch(i as u64 * 10, 3))
                    .expect("append");
            }
            w.commit().expect("commit");

            let r = st.reader();
            assert_eq!(
                r.n_superfiles(),
                n_threads,
                "expected {n_threads} superfiles for {n_threads}-thread pool",
            );
            assert_eq!(r.n_docs_total(), (n_threads * 2 * 3) as u64);
        }
    }

    #[test]
    fn commit_with_fewer_batches_than_threads_skips_empty_shards() {
        // 4 threads, only 2 batches — chunk_size = 1, two chunks
        // get one batch each, the other two get nothing.
        // Should produce 2 superfiles, not 4.
        let opts = options_id_title().with_writer_pool(writer_pool_with(4));
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 1)).expect("a");
        w.append(&build_simple_batch(1, 1)).expect("b");
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.n_superfiles(), 2);
        assert_eq!(r.n_docs_total(), 2);
    }

    #[test]
    fn apply_config_with_fixed_writer_threads_emits_that_many_superfiles() {
        let yaml = r#"
commit_threshold_size_mb: 1024
supertable:
  reader_threads: 1
  writer_threads: 4
"#;
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");

        // End-to-end: build options, route them through apply_config,
        // and verify the writer pool actually sized to the config's
        // 4 threads (one superfile per shard).
        let opts = options_id_title().apply_config(&cfg).expect("apply_config");
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        for i in 0..8u64 {
            w.append(&build_simple_batch(i * 10, 3)).expect("append");
        }
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(
            r.n_superfiles(),
            4,
            "writer_threads=4 should yield 4 shards"
        );
        assert_eq!(r.n_docs_total(), 24);
    }

    // ---- threshold auto-flush ----------------------------------------

    #[test]
    fn append_auto_flushes_when_buffer_crosses_threshold() {
        // 1 MiB threshold; one append > 1 MiB should auto-commit.
        let opts = options_id_title_serial().with_commit_threshold_size_mb(1);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");

        // Build a large batch: 50K docs × ~50-byte titles ≈ 2.5 MiB.
        let batch = build_simple_batch(0, 50_000);
        w.append(&batch).expect("append");

        // Threshold should have tripped; manifest_id has advanced.
        assert_eq!(st.manifest_id(), 1, "auto-flush should fire");
        assert_eq!(w.buffered_batches(), 0, "buffer drained on auto-flush");

        // No further commit should land an empty superfile.
        w.commit().expect("commit-empty");
        assert_eq!(st.manifest_id(), 1);
    }

    #[test]
    fn append_does_not_auto_flush_when_threshold_zero() {
        let opts = options_id_title_serial().with_commit_threshold_size_mb(0);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 50_000)).expect("append");
        assert_eq!(st.manifest_id(), 0, "no auto-flush at threshold=0");
        assert!(w.buffered_batches() > 0);
    }

    // commit latency O(n) regression with localfs storage provider

    /// Each `Supertable::append` call rewrites the entire manifest part
    /// (Avro-encode + zstd-compress all N accumulated superfile entries,
    /// then PUT to storage). Commit K is O(K), so 100 sequential commits
    /// are O(n²) total and latency grows linearly with superfile count.
    #[ignore = "known O(n) regression: manifest part rewrite on every commit"]
    #[test]
    fn commit_latency_is_constant_with_localfs() {
        const N: usize = 100;
        const DOCS_PER_COMMIT: usize = 64;
        const MAX_GROWTH_FACTOR: f64 = 2.0;

        let dir = TempDir::new().expect("tempdir");
        let storage = Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let opts = options_id_title_serial().with_storage(storage);
        let st = Supertable::create(opts).expect("create");

        let mut latencies_ms: Vec<u128> = Vec::with_capacity(N);
        for i in 0..N {
            let batch = build_simple_batch(i as u64, DOCS_PER_COMMIT);
            let t0 = Instant::now();
            st.append(&batch).expect("append");
            latencies_ms.push(t0.elapsed().as_millis());
        }

        let avg = |slice: &[u128]| slice.iter().sum::<u128>() as f64 / slice.len() as f64;
        let first5_avg = avg(&latencies_ms[..5]);
        let last5_avg = avg(&latencies_ms[N - 5..]);
        let ratio = last5_avg / first5_avg.max(1.0);

        println!(
            "first-5 avg: {first5_avg:.1}ms  last-5 avg: {last5_avg:.1}ms  ratio: {ratio:.1}x"
        );
        assert!(
            ratio <= MAX_GROWTH_FACTOR,
            "commit latency grew {ratio:.1}x from first-5 ({first5_avg:.1}ms) to \
             last-5 ({last5_avg:.1}ms) — O(n) growth in manifest rewrite path"
        );
    }

    // ---- manifest copy-on-write across multiple commits -------------

    #[test]
    fn each_commit_appends_to_existing_superfiles() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 2)).expect("a1");
        w.commit().expect("c1");
        w.append(&build_simple_batch(10, 3)).expect("a2");
        w.commit().expect("c2");
        w.append(&build_simple_batch(20, 1)).expect("a3");
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.manifest_id(), 3);
        assert_eq!(r.n_superfiles(), 3);
        assert_eq!(r.n_docs_total(), 6);
    }

    // ---- merge_ranges helper -----------------------------------------

    #[test]
    fn merge_ranges_coalesces_overlapping_and_adjacent_drops_empty() {
        // (off, len) inputs: an empty range (dropped), two
        // overlapping ranges (coalesced), one adjacent range
        // (coalesced, since `off <= last_end`), and one disjoint
        // range (kept separate). Unsorted on input.
        let input = vec![
            (100u64, 10u64), // disjoint, far away
            (0, 0),          // empty — dropped
            (10, 10),        // [10,20)
            (15, 10),        // [15,25) overlaps prior → [10,25)
            (25, 5),         // [25,30) adjacent → [10,30)
        ];
        let merged = merge_ranges(input);
        assert_eq!(merged, vec![(10, 20), (100, 10)]);
    }

    #[test]
    fn merge_ranges_empty_input_is_empty() {
        assert!(merge_ranges(Vec::new()).is_empty());
    }

    // ---- build_subsection_offsets on real superfile bytes ------------

    #[test]
    fn build_subsection_offsets_captures_total_size_and_fts_range() {
        // A freshly-built FTS superfile should produce subsection
        // offsets: total_size matches the byte length and the FTS
        // open ranges are non-empty (there's an FTS index).
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 8)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        let store = &st.options().store;
        // Fetch the bytes back from the in-memory store.
        let reader = store.reader(&seg.uri).expect("reader");
        // Confirm the manifest already carries subsection offsets and
        // that total_size is plausible (> 0).
        let offsets = seg
            .subsection_offsets
            .as_ref()
            .expect("offsets captured at commit");
        assert!(offsets.total_size > 0);
        assert!(
            offsets.fts.is_some(),
            "an FTS superfile must record an FTS subsection"
        );
        assert!(
            !offsets.fts_open_ranges.is_empty(),
            "FTS open ranges should be populated for the cold-open fast path"
        );
        // n_docs sanity via the reader, ensuring the bytes parse.
        assert_eq!(reader.n_docs(), 8);
    }

    #[test]
    fn build_subsection_offsets_on_garbage_returns_none() {
        // Bytes that aren't a valid superfile (no parquet footer)
        // must fall back to None rather than panic.
        let garbage = Bytes::from_static(b"not a parquet file at all");
        assert!(build_subsection_offsets(&garbage).is_none());
    }

    // ---- vector append path ------------------------------------------

    #[test]
    fn append_with_vector_column_publishes_superfile() {
        // Drive the vector branch of `append` (the FixedSizeList
        // downcast + Arc<Float32Array> buffering).
        let dim = 16;
        let st = Supertable::create(options_with_vector(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 8, dim)).expect("append");
        assert!(
            w.buffered_bytes() > 0,
            "buffered_bytes must account for the vector payload"
        );
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.n_superfiles(), 1);
        assert_eq!(r.n_docs_total(), 8);
    }

    // ---- end-to-end update / delete through Supertable ----------------

    /// A storage-backed supertable, required for the WAL-driven
    /// update/delete pipeline.
    fn storage_backed_st(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        Supertable::create(options_id_title_serial().with_storage(storage)).expect("create")
    }

    fn row(title: &str) -> RecordBatch {
        RecordBatch::try_new(
            schema_id_title(),
            vec![Arc::new(LargeStringArray::from(vec![title]))],
        )
        .expect("row batch")
    }

    #[test]
    fn delete_tombstones_matching_row() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&build_simple_batch(0, 1)).expect("append");
        // build_simple_batch titles are "doc 0 alpha".
        let stats = st
            .delete(col("title").eq(lit("doc 0 alpha")))
            .expect("delete");
        assert_eq!(stats.matched(), 1);
        assert_eq!(stats.n_tombstoned(), 1);
    }

    #[test]
    fn delete_unmatched_predicate_is_noop() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&build_simple_batch(0, 1)).expect("append");
        let stats = st
            .delete(col("title").eq(lit("no such title")))
            .expect("delete");
        assert_eq!(stats.matched(), 0);
        assert_eq!(stats.n_tombstoned(), 0);
    }

    #[test]
    fn update_replaces_matching_row() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&row("draft")).expect("append");
        let stats = st
            .update(col("title").eq(lit("draft")), &row("published"))
            .expect("update");
        assert_eq!(stats.matched(), 1);
        assert_eq!(stats.n_tombstoned(), 1);
    }

    #[test]
    fn update_cardinality_mismatch_is_rejected() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&row("draft")).expect("append");
        // Predicate matches one row but new_rows has two — cardinality
        // mismatch surfaces as a typed writer error.
        let two = RecordBatch::try_new(
            schema_id_title(),
            vec![Arc::new(LargeStringArray::from(vec!["a", "b"]))],
        )
        .expect("two-row batch");
        let mut w = st.writer().expect("writer");
        let err = w
            .update(col("title").eq(lit("draft")), two)
            .expect_err("cardinality mismatch");
        assert!(
            matches!(
                err,
                MutationError::CardinalityMismatch {
                    matched: 1,
                    new_rows: 2
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn update_without_storage_is_rejected() {
        use datafusion::prelude::{col, lit};
        // No storage attached → the update pre-flight rejects.
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        let err = w
            .update(col("title").eq(lit("x")), row("y"))
            .expect_err("no storage");
        assert!(matches!(err, MutationError::NoStorageAttached), "{err:?}");
    }

    #[test]
    fn delete_without_storage_is_rejected() {
        use datafusion::prelude::{col, lit};
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        let err = w.delete(col("title").eq(lit("x"))).expect_err("no storage");
        assert!(matches!(err, MutationError::NoStorageAttached), "{err:?}");
    }

    #[test]
    fn buffered_bytes_grows_then_resets_on_commit() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        assert_eq!(w.buffered_bytes(), 0);
        w.append(&build_simple_batch(0, 4)).expect("append");
        assert!(w.buffered_bytes() > 0, "buffer cost recorded");
        assert_eq!(w.buffered_batches(), 1);
        w.commit().expect("commit");
        assert_eq!(w.buffered_bytes(), 0, "buffer drained on commit");
        assert_eq!(w.buffered_batches(), 0);
    }
}
