// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! MVCC OPANN maintenance for the hidden global vector cell index.
//!
//! The user table stays time-ordered and immutable. The hidden index is a
//! derived, cell-ordered acceleration layer maintained with OPANN-style
//! logical updates expressed as append/MVCC physical swaps:
//!
//!   1. Assign incoming vectors to nearest manifest centroids with zero GETs.
//!   2. For each touched cell only: append one delta superfile (no GETs).
//!   3. Compaction merges multiple small IVF superfiles per cell toward one packed
//!      base via the standard `merge_superfiles` path.
//!   4. Locally refresh touched cell centroids and counts.
//!   5. Split overflow cells (Sq8+ε k-means, N→N+1 centroids).
//!   6. Reassign vectors in the split neighborhood (P−1, P, P₂, P+1).
//!   7. Redrive reassigned rows through the incoming staging region; route
//!      them into per-cell IVF superfiles (same path as commit ingest).
//!
//! Split/reassign stays on stored Sq8+ε bytes. Row assignment dequantizes
//! manifest centroids and rows to fp32 before [`distance`]; rows are
//! re-spliced with [`encode_encoded_rows`], never decoded to full fp32 corpora.

use std::{
    cmp::Ordering,
    collections::HashMap,
    sync::Mutex,
};

use crate::{
    config,
    superfile::vector::{
        cell_posting::{
            EncodedCellRow, MaterializedIvfRow, dequantize_sq8_residual_into,
            manifest_centroid_components_from_row, medoid_index_by,
        },
        distance::{
            Metric, distance, nearest_k_centroids_transposed, normalize,
            relative_score_window,
        },
        reservoir::Reservoir,
        spill::SpilledCellRows,
    },
    supertable::{
        error::BuildError,
        manifest::{
            ClusterCentroids, RABITQ_ADMIT_CELL_SHORTLIST_FRACTION,
            RABITQ_ADMIT_CELL_SHORTLIST_MIN, RabitqAdmitContext, list::WIDTH_LAW_KS,
        },
    },
};

/// Overflow threshold for cell split (OPANN step 7). Sourced from
/// `vector.cell_split_doc_cap`.
pub(crate) fn cell_split_doc_cap() -> u64 {
    config::global().vector.cell_split_doc_cap
}

/// True when a merged cell superfile should be split into two sub-cells.
pub(crate) fn split_overflow_needed(n_docs: u64) -> bool {
    n_docs > cell_split_doc_cap()
}

/// Append-only count bookkeeping for touched cells.
pub(crate) fn apply_cell_count_updates(
    base: &ClusterCentroids,
    count_updates: &HashMap<u32, u32>,
) -> ClusterCentroids {
    let mut updated = base.clone();
    for (&cell, &count) in count_updates {
        if let Some(slot) = updated.counts.get_mut(cell as usize) {
            *slot = count;
        }
    }
    updated
}

/// Apply count updates from maintenance (incoming routing / compaction).
pub(crate) fn apply_cell_updates(
    base: &ClusterCentroids,
    count_updates: &HashMap<u32, u32>,
) -> ClusterCentroids {
    apply_cell_count_updates(base, count_updates)
}

/// Replica candidates considered per row beyond its primary cell — the
/// SPANN-style closure depth. Together with the closure distance ratio this
/// bounds the candidate pool; the configured replica budget
/// (`drain_replica_target_factor`) still decides how many candidates are
/// actually materialized, thinnest margins first.
pub(crate) const REPLICA_CLOSURE_MAX_REPLICAS: usize = 3;

/// A cell qualifies as a replica candidate when the row's distance to it is
/// within this multiple of the row's primary-cell distance. Rows deep inside
/// their cell (small primary distance) get a proportionally tight window and
/// therefore no replicas; genuine boundary rows qualify toward every nearby
/// cell, not only the single second-nearest.
pub(crate) const REPLICA_CLOSURE_DISTANCE_RATIO: f32 = 1.2;

/// Primary cell assignment plus the row's replica-candidate cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundaryAssignment {
    pub primary: u32,
    /// Up to [`REPLICA_CLOSURE_MAX_REPLICAS`] cells within the closure
    /// distance ratio of the primary, each with the row's margin to the
    /// primary/candidate Voronoi boundary. Smaller margin means closer to
    /// the boundary and therefore a better replication candidate. Fixed-size
    /// (`None`-padded) so the per-row hot assign path stays allocation-free.
    pub replicas: [Option<(u32, f32)>; REPLICA_CLOSURE_MAX_REPLICAS],
}

fn score_row_against_cell(
    clusters: &ClusterCentroids,
    metric: Metric,
    cell: usize,
    row: &EncodedCellRow,
) -> f32 {
    let dim = clusters.dim as usize;
    let mut row_fp = vec![0f32; dim];
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        &mut row_fp,
    );
    distance(metric, &row_fp, clusters.centroid(cell))
}

fn boundary_margin(
    clusters: &ClusterCentroids,
    metric: Metric,
    primary: u32,
    neighbor: u32,
    primary_score: f32,
    neighbor_score: f32,
) -> f32 {
    let gap = (neighbor_score - primary_score).max(0.0);
    let c1 = clusters.centroid(primary as usize);
    let c2 = clusters.centroid(neighbor as usize);
    match metric {
        Metric::L2Sq => {
            let separation = distance(metric, c1, c2).sqrt();
            if separation > 0.0 {
                gap / (2.0 * separation)
            } else {
                f32::INFINITY
            }
        }
        Metric::Cosine | Metric::NegDot => {
            let separation = distance(metric, c1, c2).abs();
            if separation > 0.0 {
                gap / separation
            } else {
                f32::INFINITY
            }
        }
    }
}

/// Assignment shortlist width for `n_cells` grid cells: the shared 1-bit
/// admit fraction of the grid with the shared meaningful-window floor,
/// capped at the grid. Below the floor the window covers every cell and
/// [`boundary_assignment_fp32`] takes its exact-scan arm — small grids
/// (and small-dim tests, where a short sign sketch is noise) keep the
/// exact assignment they always had; the prefilter engages only where it
/// pays (measured shapes: 103 of 512, 205 of 1024).
pub(crate) fn assignment_shortlist_window(n_cells: usize) -> usize {
    let scaled = (n_cells as f64 * RABITQ_ADMIT_CELL_SHORTLIST_FRACTION).ceil() as usize;
    scaled
        .max(RABITQ_ADMIT_CELL_SHORTLIST_MIN)
        .min(n_cells.max(1))
}

/// Drain-side boundary assignment: decode the Sq8+ε row once, then assign
/// through the shared 1-bit shortlist + exact rescore. Same assignment
/// semantics as `nearest-two by score then Voronoi margin`.
pub(crate) fn boundary_assignment_encoded(
    clusters: &ClusterCentroids,
    metric: Metric,
    row: &EncodedCellRow,
    admit_ctx: &RabitqAdmitContext,
    window: usize,
) -> BoundaryAssignment {
    let dim = clusters.dim as usize;
    let mut row_fp = vec![0f32; dim];
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        &mut row_fp,
    );
    boundary_assignment_fp32(clusters, metric, &row_fp, admit_ctx, window)
}

/// Boundary assignment for an fp32 row (commit buffer path and the drain's
/// decoded rows): 1-bit admit shortlist over the grid (XOR+popcount, the
/// same estimator the query-side prefilter uses), exact fp32 rescore of
/// the shortlisted cells only, then the nearest-two + Voronoi-margin
/// closure on the exact scores. Placement is exact within the window;
/// per-row cost scales with `window` (20% of cells) instead of the grid.
pub(crate) fn boundary_assignment_fp32(
    clusters: &ClusterCentroids,
    metric: Metric,
    row_fp: &[f32],
    admit_ctx: &RabitqAdmitContext,
    window: usize,
) -> BoundaryAssignment {
    let n_cent = clusters.n_cent as usize;
    let top_k = REPLICA_CLOSURE_MAX_REPLICAS + 1;
    let ranked: Vec<(u32, f32)> = if window >= n_cent {
        // Window covers the grid: the exact blocked-SIMD scan is cheaper
        // than encode + estimate + rescore.
        nearest_k_centroids_transposed(
            metric,
            row_fp,
            clusters.transposed(),
            n_cent,
            clusters.dim as usize,
            None,
            top_k,
        )
    } else {
        let admit = admit_ctx.encode(row_fp);
        let mut exact: Vec<(u32, f32)> = clusters
            .admit_shortlist(metric, &admit, window)
            .into_iter()
            .map(|(cell, _)| (cell, clusters.score_one(metric, cell as usize, row_fp)))
            .collect();
        exact.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        exact.truncate(top_k);
        exact
    };
    boundary_from_ranked(clusters, metric, &ranked)
}

/// Shared closure tail: primary = best-ranked cell; replicas = ranked
/// cells within the closure distance ratio, carrying their margin to the
/// shared Voronoi boundary.
fn boundary_from_ranked(
    clusters: &ClusterCentroids,
    metric: Metric,
    ranked: &[(u32, f32)],
) -> BoundaryAssignment {
    let mut replicas = [None; REPLICA_CLOSURE_MAX_REPLICAS];
    let Some(&(primary, primary_score)) = ranked.first() else {
        return BoundaryAssignment {
            primary: 0,
            replicas,
        };
    };
    // Closure pool: every ranked cell whose distance sits within the ratio
    // window of the primary. The margin (distance to the shared Voronoi
    // boundary) orders candidates globally at the budget cut. Same window
    // definition as the routing cutoff (`relative_score_window`), so
    // replication and probing agree on what "near the boundary" means.
    let closure_threshold =
        relative_score_window(primary_score, REPLICA_CLOSURE_DISTANCE_RATIO - 1.0);
    for (slot, &(cell, score)) in ranked.iter().skip(1).enumerate() {
        if score > closure_threshold {
            break;
        }
        replicas[slot] = Some((
            cell,
            boundary_margin(clusters, metric, primary, cell, primary_score, score),
        ));
    }
    BoundaryAssignment { primary, replicas }
}

/// One-cluster [`ClusterCentroids`] prototype from a Sq8+ε row (split k-means seeds).
fn centroid_prototype_from_row(
    template: &ClusterCentroids,
    row: &EncodedCellRow,
) -> ClusterCentroids {
    let dim = template.dim as usize;
    let fp32 = manifest_centroid_components_from_row(row, dim);
    ClusterCentroids::from_fp32(1, template.dim, &fp32, vec![1])
}

fn fp32_distance_between_rows(metric: Metric, a: &EncodedCellRow, b: &EncodedCellRow) -> f32 {
    debug_assert_eq!(a.rerank_codec, b.rerank_codec);
    let dim = a.scale.len();
    let mut af = vec![0f32; dim];
    let mut bf = vec![0f32; dim];
    let divisor = a
        .rerank_codec
        .residual_divisor()
        .expect("encoded row uses residual-family codec");
    dequantize_sq8_residual_into(
        &a.scale,
        &a.offset,
        &a.codes,
        &a.residuals,
        divisor,
        &mut af,
    );
    dequantize_sq8_residual_into(
        &b.scale,
        &b.offset,
        &b.codes,
        &b.residuals,
        divisor,
        &mut bf,
    );
    distance(metric, &af, &bf)
}

/// Medoid index under fp32 dequant + [`distance`] row↔row (discrete k-means
/// centroid update).
fn medoid_index(metric: Metric, shard: &[&EncodedCellRow]) -> usize {
    medoid_index_by(shard, |a, b| fp32_distance_between_rows(metric, a, b))
}

/// 2-way Lloyd k-means on Sq8+ε overflow rows. Returns manifest centroid
/// components (dim each) for the two sub-cells.
/// Plan a binary split of `split_cell`: returns the two sub-cell centroids and,
/// aligned to `rows`, a `0/1` assignment of each row to sub-cell 0 / sub-cell 1
/// (reconciled with the empty-shard fixups, so it exactly matches the two
/// shards the centroids were derived from). The caller routes the cell's
/// materialized rows into the two sub-cells by this assignment.
/// The two diameter endpoints of `split_cell`'s rows: `seed1` is the row
/// farthest from the cell's existing centroid (an extreme edge point); `seed0`
/// is the row farthest from `seed1` (the opposite edge). The `seed0 → seed1`
/// line is the axis the split bisects along. Picking closest-to-centroid vs
/// farthest instead would seed a radius (center + edge), peeling a thin shell
/// off the dense core.
fn pick_split_seeds(
    rows: &[&EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: usize,
    metric: Metric,
) -> (usize, usize) {
    let seed1 = rows
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            score_row_against_cell(clusters, metric, split_cell, a)
                .partial_cmp(&score_row_against_cell(clusters, metric, split_cell, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let cent1 = centroid_prototype_from_row(clusters, rows[seed1]);
    let seed0 = rows
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            score_row_against_cell(&cent1, metric, 0, a)
                .partial_cmp(&score_row_against_cell(&cent1, metric, 0, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    (seed0, seed1)
}

/// Dequantize one Sq8+ε residual row to fp32.
fn dequantize_row(row: &EncodedCellRow, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        &mut out,
    );
    out
}

// ---------- Drain-time probe-width calibration ----------

/// Rows reservoir-sampled as stand-in queries for probe-width calibration.
/// Corpus rows are the right calibration distribution — on Cohere-1M/768d,
/// stored-row queries and the dataset's held-out test queries measured the
/// same top-k cell spread.
const WIDTH_LAW_QUERY_SAMPLE: usize = 256;
/// Mean top-k coverage a probe width must reach to be recorded in the law —
/// the engine's recall@k acceptance bar.
const WIDTH_LAW_TARGET_COVERAGE: f64 = 0.99;
/// Fixed seed for the calibration reservoir, so a re-drained identical
/// corpus stamps an identical law.
const WIDTH_LAW_SAMPLE_SEED: u64 = 0x51ED_CA1B;
/// Rows decoded per chunk while scoring a spilled cell.
const WIDTH_LAW_SCORE_CHUNK: usize = 1024;

/// [`dequantize_sq8_residual_into`] over one encoded row into caller-owned
/// scratch (hot loops reuse one allocation of the row's `dim`).
fn dequantize_row_into(row: &EncodedCellRow, out: &mut [f32]) {
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        out,
    );
}

/// Frozen query sample: dequantized fp32 vectors + their stable ids
/// (self-hit exclusion while scoring).
struct WidthLawQueries {
    queries: Vec<f32>,
    ids: Vec<i128>,
}

/// Drain-time probe-width calibration: measures, on this table's own data,
/// how many grid cells (in routing order) cover the exact top-k, and stamps
/// the result into the manifest's [`CellRoutingParams::width_for_k`] law.
///
/// How far the true top-k spreads over cells is a property of the corpus —
/// synthetic clustered data concentrates (1 cell at k = 10), real text
/// embeddings spray (Cohere-1M/768d measured ~30 of 256 cells at k = 100,
/// identical under a converged reference clustering, so it is the data and
/// not grid quality) — which is why the default probe width must be
/// measured per table rather than hardcoded.
///
/// Lifecycle inside one clean (non-resumed) drain:
///   1. [`Self::offer`] on every spilled row — [`Reservoir`]-samples the
///      stand-in queries (sequential; the spill loop holds `&mut`).
///   2. [`Self::freeze`] once all batches have spilled.
///   3. [`Self::score_cell`] per cell during the pack fan-out — re-reads
///      the cell's spill (the pack pass reads it anyway), scores every row
///      with the shared [`distance`] kernel (rows unit-normalized for
///      cosine, matching the rerank kernels' norm division), and merges
///      that cell's per-query top-k under one lock per cell.
///   4. [`Self::finish`] before the drain's manifest stamp — ranks cells
///      with the same [`ClusterCentroids::rank_cells`] routing order
///      queries use and extracts the width law.
///
/// Resumed drains skip calibration entirely (checkpointed batches never
/// re-stream), keeping whatever law the manifest already carries.
pub(crate) struct WidthLawCalibration {
    dim: usize,
    metric: Metric,
    reservoir: Reservoir,
    /// Stable id of each reservoir slot, kept in lockstep through
    /// [`Reservoir::update_traced`].
    slot_ids: Vec<i128>,
    dequant_scratch: Vec<f32>,
    frozen: Option<WidthLawQueries>,
    /// Per-query `(score, cell)` candidates, truncated to the largest law
    /// `k` as cells merge in.
    tops: Mutex<Vec<Vec<(f32, u32)>>>,
}

impl WidthLawCalibration {
    pub(crate) fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            reservoir: Reservoir::new(WIDTH_LAW_QUERY_SAMPLE, dim, WIDTH_LAW_SAMPLE_SEED),
            slot_ids: Vec::with_capacity(WIDTH_LAW_QUERY_SAMPLE),
            dequant_scratch: vec![0f32; dim],
            frozen: None,
            tops: Mutex::new(Vec::new()),
        }
    }

    /// Offer one spilled row as a calibration-query candidate.
    pub(crate) fn offer(&mut self, row: &MaterializedIvfRow) {
        debug_assert!(self.frozen.is_none(), "offer after freeze");
        dequantize_row_into(&row.encoded, &mut self.dequant_scratch);
        if let Some(slot) = self.reservoir.update_traced(&self.dequant_scratch) {
            if slot == self.slot_ids.len() {
                self.slot_ids.push(row.stable_id);
            } else {
                self.slot_ids[slot] = row.stable_id;
            }
        }
    }

    /// Freeze the sampled queries. Called once, after the last batch
    /// spilled and before cell packing scores.
    pub(crate) fn freeze(&mut self) {
        let queries = self.reservoir.sample().to_vec();
        let ids = self.slot_ids.clone();
        *self.tops.lock().expect("width-law tops lock") = vec![Vec::new(); ids.len()];
        self.frozen = Some(WidthLawQueries { queries, ids });
    }

    /// Score every row of one spilled cell against the frozen queries and
    /// merge into the per-query candidate lists. Safe to call from the
    /// pack fan-out workers; the merge takes one lock per cell.
    pub(crate) fn score_cell(
        &self,
        cell: u32,
        spill: &SpilledCellRows,
    ) -> Result<(), BuildError> {
        let Some(frozen) = self.frozen.as_ref() else {
            return Err(BuildError::Store(
                "width-law score_cell before freeze".into(),
            ));
        };
        let n_queries = frozen.ids.len();
        if n_queries == 0 {
            return Ok(());
        }
        let k_max = *WIDTH_LAW_KS.last().expect("law has k points");
        let mut partial: Vec<Vec<(f32, u32)>> = vec![Vec::new(); n_queries];
        let mut reader = spill.reader()?;
        let mut remaining = spill.n_rows();
        let mut scratch = vec![0f32; self.dim];
        while remaining > 0 {
            let chunk = reader.next_chunk(WIDTH_LAW_SCORE_CHUNK.min(remaining))?;
            remaining -= chunk.len();
            for row in &chunk {
                dequantize_row_into(&row.encoded, &mut scratch);
                if self.metric == Metric::Cosine {
                    // The rerank kernels divide by the stored row norm;
                    // unit-normalizing the row lets the shared [`distance`]
                    // kernel (which assumes unit inputs for cosine) score
                    // with the same ranking. Query scaling is per-query
                    // monotone and cannot reorder its candidates.
                    normalize(&mut scratch);
                }
                for (qi, q) in frozen.queries.chunks_exact(self.dim).enumerate() {
                    // Self-hit: a sampled query trivially covers itself.
                    if row.stable_id == frozen.ids[qi] {
                        continue;
                    }
                    partial[qi].push((distance(self.metric, q, &scratch), cell));
                }
            }
            // Bound the per-cell partials the same way the merge does.
            for cand in &mut partial {
                truncate_ascending(cand, k_max);
            }
        }
        let mut tops = self.tops.lock().expect("width-law tops lock");
        for (qi, mut cand) in partial.into_iter().enumerate() {
            tops[qi].append(&mut cand);
            truncate_ascending(&mut tops[qi], k_max);
        }
        Ok(())
    }

    /// Extract the width law: cells (in the grid's routing order) needed
    /// for mean [`WIDTH_LAW_TARGET_COVERAGE`] coverage of the exact top-k
    /// at each [`WIDTH_LAW_KS`] point. Points the sample cannot support
    /// (k exceeding any query's candidate count) stay `0` (uncalibrated).
    /// `None` when nothing was sampled.
    pub(crate) fn finish(
        self,
        grid: &ClusterCentroids,
    ) -> Option<[u32; WIDTH_LAW_KS.len()]> {
        let frozen = self.frozen?;
        let n_queries = frozen.ids.len();
        if n_queries == 0 || grid.n_cent == 0 {
            return None;
        }
        let n_cells = grid.n_cent as usize;
        let tops = self.tops.into_inner().expect("width-law tops lock");
        // Superseded or empty cells inflate ranks slightly (routing skips
        // them, this count does not) — an over-probe, never an under-probe;
        // fresh drains, the normal calibration moment, have neither.
        let mut law = [0u32; WIDTH_LAW_KS.len()];
        let mut coverage_sums: Vec<Vec<f64>> =
            vec![vec![0f64; n_cells]; WIDTH_LAW_KS.len()];
        let mut supported = [true; WIDTH_LAW_KS.len()];
        let mut rank_of_cell = vec![0u32; n_cells];
        for (qi, cand) in tops.iter().enumerate() {
            let q = &frozen.queries[qi * self.dim..(qi + 1) * self.dim];
            let ranked = grid.rank_cells(self.metric, q);
            for (rank, (cell, _)) in ranked.iter().enumerate() {
                if let Some(slot) = rank_of_cell.get_mut(*cell as usize) {
                    *slot = rank as u32;
                }
            }
            let mut sorted = cand.clone();
            sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for (ki, &k) in WIDTH_LAW_KS.iter().enumerate() {
                if sorted.len() < k {
                    supported[ki] = false;
                    continue;
                }
                // Per-rank counts of this query's top-k, then a prefix walk
                // accumulates the mean coverage curve.
                let mut per_rank = vec![0u32; n_cells];
                for (_, cell) in &sorted[..k] {
                    per_rank[rank_of_cell[*cell as usize] as usize] += 1;
                }
                let mut covered = 0u32;
                for (rank, count) in per_rank.iter().enumerate() {
                    covered += count;
                    coverage_sums[ki][rank] += f64::from(covered) / k as f64;
                }
            }
        }
        for (ki, sums) in coverage_sums.iter().enumerate() {
            if !supported[ki] {
                continue;
            }
            let target = WIDTH_LAW_TARGET_COVERAGE * n_queries as f64;
            if let Some(rank) = sums.iter().position(|&s| s >= target) {
                law[ki] = (rank + 1) as u32;
            }
        }
        (law.iter().any(|&w| w > 0)).then_some(law)
    }
}

/// Keep the ascending-best `cap` candidates in place.
fn truncate_ascending(cand: &mut Vec<(f32, u32)>, cap: usize) {
    if cand.len() > cap {
        cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        cand.truncate(cap);
    }
}

pub(crate) fn plan_sq8_split(
    rows: &[&EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: u32,
    metric: Metric,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let dim = clusters.dim as usize;
    let p = split_cell as usize;
    let mut assign = vec![0u8; rows.len()];
    if rows.len() < 2 {
        // Caller guards on MIN_ROWS_TO_SPLIT_CELL; stay defensive so a degenerate
        // input can't panic in medoid_index on an empty shard.
        let c = manifest_centroid_components_from_row(rows[0], dim);
        return (c.clone(), c, assign);
    }

    // Bisect along a DIAMETER of the cell: project every row onto the `seed0 →
    // seed1` axis and split at the MEDIAN. This makes the two sub-cells equal-
    // sized (±1) regardless of density, so any cell up to 2× the cap converges
    // in a single pass. A nearest-seed (k-means) assignment instead lets the
    // dense bulk fall to whichever seed it is closer to — lopsided in high
    // dimensions, where the two farthest points are outliers and the bulk favors
    // one — leaving sub-cells over-cap that re-split for several passes. The flat
    // median cut costs no recall: per-sub-cell fine re-clustering downstream
    // restores intra-cell routing.
    let (seed0, seed1) = pick_split_seeds(rows, clusters, p, metric);
    let v0 = dequantize_row(rows[seed0], dim);
    let v1 = dequantize_row(rows[seed1], dim);
    let axis: Vec<f32> = (0..dim).map(|d| v1[d] - v0[d]).collect();

    let mut proj: Vec<(usize, f32)> = rows
        .iter()
        .copied()
        .enumerate()
        .map(|(i, row)| {
            let rv = dequantize_row(row, dim);
            let s: f32 = (0..dim).map(|d| rv[d] * axis[d]).sum();
            (i, s)
        })
        .collect();
    proj.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let mid = rows.len() / 2;
    for (rank, (i, _)) in proj.iter().enumerate() {
        assign[*i] = u8::from(rank >= mid);
    }

    // Shards borrow the input rows (no payload clone) — the split extracts up
    // to a full over-cap cell (≈2× the ~500K cap), so cloning every row's
    // Sq8+ε bytes here would double the biggest cell's resident footprint.
    let mut shard0: Vec<&EncodedCellRow> = Vec::new();
    let mut shard1: Vec<&EncodedCellRow> = Vec::new();
    for (i, row) in rows.iter().copied().enumerate() {
        if assign[i] == 0 {
            shard0.push(row);
        } else {
            shard1.push(row);
        }
    }

    let m0 = medoid_index(metric, &shard0);
    let m1 = medoid_index(metric, &shard1);
    (
        manifest_centroid_components_from_row(shard0[m0], dim),
        manifest_centroid_components_from_row(shard1[m1], dim),
        assign,
    )
}

/// Replace cell `cell_id`'s centroid and append a second sub-cell at `n_cent`.
pub(crate) fn insert_split_centroid(
    base: &ClusterCentroids,
    cell_id: u32,
    sub_centroids: &[f32],
) -> (ClusterCentroids, u32) {
    let dim = base.dim as usize;
    let p = cell_id as usize;
    let old_n = base.n_cent as usize;
    let new_cell_id = base.n_cent;
    let new_n = old_n + 1;

    let mut fp32 = vec![0f32; new_n * dim];
    for c in 0..old_n {
        fp32[c * dim..(c + 1) * dim].copy_from_slice(base.centroid(c));
    }
    fp32[p * dim..(p + 1) * dim].copy_from_slice(&sub_centroids[..dim]);
    fp32[old_n * dim..new_n * dim].copy_from_slice(&sub_centroids[dim..2 * dim]);

    // Counts must have one entry per cell: grow to `new_n` so the split cell and
    // the new sub-cell both have a slot. Cloning `base.counts` alone leaves it at
    // `old_n`, which silently passes in-memory but truncates the wire encoding
    // (counts and centroids are adjacent) → the grid fails to reopen from storage.
    let mut counts = base.counts.clone();
    counts.resize(new_n, 0);
    let updated = ClusterCentroids::from_fp32(new_n as u32, base.dim, &fp32, counts);
    (updated, new_cell_id)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::superfile::vector::{
        cell_posting::{encode_blob, load_encoded_rows_from_blob},
        rerank_codec::{RerankCodec, SQ8_FIXED_OFFSET, SQ8_FIXED_SCALE},
    };

    fn synth_centroids(n_cent: u32, dim: u32) -> ClusterCentroids {
        let nc = n_cent as usize;
        let d = dim as usize;
        let mut fp32 = vec![0f32; nc * d];
        for c in 0..nc {
            for j in 0..d {
                fp32[c * d + j] = c as f32 * 0.5 + j as f32 * 0.01;
            }
        }
        let counts = vec![100; nc];
        ClusterCentroids::from_fp32(n_cent, dim, &fp32, counts)
    }

    fn synth_rows(dim: usize, n: usize, offset: f32) -> Vec<EncodedCellRow> {
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..n as u32 {
            ids.push(i);
            for d in 0..dim {
                vecs.push(offset + i as f32 * 0.01 + d as f32 * 0.001);
            }
        }
        let blob =
            encode_blob(Metric::L2Sq, dim, &ids, &vecs, RerankCodec::Sq8Residual).expect("encode");
        let stable_ids: Vec<i128> = (0..n).map(|i| i as i128).collect();
        load_encoded_rows_from_blob(&blob, &stable_ids, None).expect("load")
    }

    fn synth_fixed_rows(dim: usize, n: usize, code: u8) -> Vec<EncodedCellRow> {
        let scale: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_SCALE; dim]);
        let offset: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_OFFSET; dim]);
        (0..n)
            .map(|id| EncodedCellRow {
                stable_id: id as i128,
                rerank_codec: RerankCodec::Sq8FixedResidual,
                scale: Arc::clone(&scale),
                offset: Arc::clone(&offset),
                codes: vec![code; dim],
                residuals: vec![0; dim],
                norm_sq: None,
            })
            .collect()
    }

    /// Rotation seed for the assignment-test admit contexts.
    const TEST_ROT_SEED: u64 = 7;

    /// Closure replication: a row equidistant-ish to several cells collects a
    /// replica candidate for every cell inside the distance-ratio window
    /// (ordered nearest-first), and a row deep inside its cell collects none.
    /// (4 cells ⇒ the shortlist window covers the grid, so this exercises the
    /// exact-scan arm.)
    #[test]
    fn boundary_assignment_closure_matches_distance_ratio() {
        let dim = 4usize;
        // Four centroids at 0, 1, 2, 30 on every axis.
        let mut fp32 = Vec::new();
        for base in [0.0f32, 1.0, 2.0, 30.0] {
            fp32.extend(std::iter::repeat_n(base, dim));
        }
        let clusters = ClusterCentroids::from_fp32(4, dim as u32, &fp32, vec![1; 4]);
        let ctx = RabitqAdmitContext::new(dim, TEST_ROT_SEED);
        let window = assignment_shortlist_window(4);

        // Row at 0.9: distances (L2Sq per dim) to cells 0/1/2 are 0.81, 0.01,
        // 1.21 (per-dim) — cell 1 is primary; cell 0 and 2 are far outside a
        // 1.2 ratio window of 0.01. No replicas.
        let deep = vec![0.9f32; dim];
        let assignment = boundary_assignment_fp32(&clusters, Metric::L2Sq, &deep, &ctx, window);
        assert_eq!(assignment.primary, 1);
        assert_eq!(assignment.replicas, [None; REPLICA_CLOSURE_MAX_REPLICAS]);

        // Row at 1.01 — just past the exact midpoint region between cells 0.98
        // and 1.02... use 1.5: exactly between cells 1 and 2 (distances equal),
        // both inside each other's ratio window; cell 0 at 1.5 distance 2.25
        // per dim is outside 1.2 × 0.25. Expect primary = 1 (tie broken by
        // lower id) and exactly one replica: cell 2.
        let boundary = vec![1.5f32; dim];
        let assignment = boundary_assignment_fp32(&clusters, Metric::L2Sq, &boundary, &ctx, window);
        assert_eq!(assignment.primary, 1);
        assert_eq!(assignment.replicas[0].map(|(cell, _)| cell), Some(2));
        assert_eq!(assignment.replicas[1], None);
        let margin = assignment.replicas[0].expect("replica").1;
        assert!(
            margin.is_finite() && margin >= 0.0,
            "boundary margin must be a finite non-negative distance, got {margin}"
        );
    }

    /// The shortlist window is the shared 20% fraction with the shared 48
    /// floor, capped at the grid: at or under the floor the window covers
    /// every cell (exact assignment), past it the 20% slice scales.
    #[test]
    fn assignment_shortlist_window_scales_with_grid() {
        // At or under the floor: the whole grid (exact-scan arm).
        assert_eq!(assignment_shortlist_window(1), 1);
        assert_eq!(assignment_shortlist_window(16), 16);
        assert_eq!(assignment_shortlist_window(48), 48);
        // Floor binds until 20% overtakes it at 240 cells.
        assert_eq!(
            assignment_shortlist_window(64),
            RABITQ_ADMIT_CELL_SHORTLIST_MIN
        );
        assert_eq!(
            assignment_shortlist_window(240),
            RABITQ_ADMIT_CELL_SHORTLIST_MIN
        );
        // Plain 20% past the floor.
        assert_eq!(assignment_shortlist_window(256), 52);
        assert_eq!(assignment_shortlist_window(512), 103);
        assert_eq!(assignment_shortlist_window(1024), 205);
    }

    /// The 1-bit shortlisted assignment must agree with the exact scan on
    /// rows that clearly belong to a cell — the regime every committed row
    /// is in. Planted well-separated centroids, rows jittered around them;
    /// primaries must match the exact path cell-for-cell. The grid sits
    /// past the shared floor so the shortlist arm actually engages.
    #[test]
    fn shortlisted_assignment_matches_exact_on_planted_cells() {
        let dim = 64usize;
        let n_cells = 300usize;
        let mut fp32 = vec![0.0f32; n_cells * dim];
        for (c, chunk) in fp32.chunks_mut(dim).enumerate() {
            // Distinct direction per cell: two active axes with distinct
            // magnitudes keep centroids well separated.
            chunk[c % dim] = 4.0 + (c / dim) as f32;
            chunk[(c * 7 + 3) % dim] = 2.0;
        }
        let clusters =
            ClusterCentroids::from_fp32(n_cells as u32, dim as u32, &fp32, vec![1; n_cells]);
        let ctx = RabitqAdmitContext::new(dim, TEST_ROT_SEED);
        let window = assignment_shortlist_window(n_cells);
        assert!(window < n_cells, "test must exercise the shortlist arm");

        let mut state = 0x9e37_79b9_97f4_a7c5u64;
        let mut jitter = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) % 1000) as f32 / 1000.0 * 0.2 - 0.1
        };
        for c in 0..n_cells {
            let mut row = fp32[c * dim..(c + 1) * dim].to_vec();
            for v in row.iter_mut() {
                *v += jitter();
            }
            let shortlisted = boundary_assignment_fp32(&clusters, Metric::L2Sq, &row, &ctx, window);
            let exact = boundary_assignment_fp32(&clusters, Metric::L2Sq, &row, &ctx, n_cells);
            assert_eq!(
                shortlisted.primary, exact.primary,
                "cell {c}: shortlisted primary diverged from exact"
            );
            assert_eq!(shortlisted.primary, c as u32, "cell {c}: wrong placement");
        }
    }

    #[test]
    fn insert_split_centroid_extends_n_cent() {
        let base = synth_centroids(4, 8);
        let sub = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        ];
        let (updated, new_id) = insert_split_centroid(&base, 2, &sub);
        assert_eq!(new_id, 4);
        assert_eq!(updated.n_cent, 5);
        // Counts and centroids must both match n_cent, or the wire encoding
        // (counts adjacent to centroids) truncates and the grid fails to reopen.
        assert_eq!(updated.counts.len(), 5);
        assert_eq!(updated.centroids.len(), 5 * base.dim as usize);
        // Round-trips through the manifest wire format cleanly.
        let bytes = crate::supertable::manifest::encoding::encode_cluster_centroids(&updated);
        let decoded = crate::supertable::manifest::encoding::decode_cluster_centroids(&bytes)
            .expect("split grid must reopen from wire bytes");
        assert_eq!(decoded.n_cent, 5);
        assert_eq!(decoded.centroids.len(), 5 * base.dim as usize);
    }

    #[test]
    fn plan_sq8_split_separates_two_blobs() {
        let dim = 4usize;
        let mut rows = synth_rows(dim, 10, 0.0);
        rows.extend(synth_rows(dim, 10, 10.0));
        let clusters = synth_centroids(4, dim as u32);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (c0, c1, assign) = plan_sq8_split(&refs, &clusters, 1, Metric::L2Sq);
        assert_eq!(c0.len(), dim);
        assert_eq!(c1.len(), dim);
        let dist: f32 = (0..dim).map(|d| (c0[d] - c1[d]).abs()).sum();
        assert!(dist > 1.0, "split centroids should separate, got {dist}");
        // Assignment is aligned to `rows` and routes each row to one sub-cell;
        // the two well-separated blobs land on opposite sides.
        assert_eq!(assign.len(), rows.len());
        assert_ne!(
            assign[0],
            assign[rows.len() - 1],
            "the two separated blobs should split across sub-cells"
        );
    }

    #[test]
    fn plan_fixed_residual_split_preserves_payloads() {
        let dim = 4usize;
        let mut rows = synth_fixed_rows(dim, 10, 64);
        rows.extend(synth_fixed_rows(dim, 10, 192));
        let before: Vec<(Vec<u8>, Vec<u8>)> = rows
            .iter()
            .map(|row| (row.codes.clone(), row.residuals.clone()))
            .collect();
        let clusters = synth_centroids(4, dim as u32);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (left, right, _assign) = plan_sq8_split(&refs, &clusters, 1, Metric::Cosine);
        let separation: f32 = left.iter().zip(&right).map(|(a, b)| (a - b).abs()).sum();
        assert!(separation > 1.0);
        let after: Vec<(Vec<u8>, Vec<u8>)> = rows
            .iter()
            .map(|row| (row.codes.clone(), row.residuals.clone()))
            .collect();
        assert_eq!(after, before);
    }

    #[test]
    fn pick_split_seeds_returns_diameter_endpoints() {
        let dim = 4usize;
        // 20 rows evenly spaced along a line (row i ≈ 0.01·i per dim), centroid
        // pinned at the line's middle. The two farthest-apart rows are the
        // endpoints (0 and 19) — NOT the middle row a closest-to-centroid seed
        // would pick — so the returned pair must be {0, 19}.
        let rows = synth_rows(dim, 20, 0.0);
        let mid = vec![0.095f32, 0.096, 0.097, 0.098];
        let clusters = ClusterCentroids::from_fp32(1, dim as u32, &mid, vec![rows.len() as u32]);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (seed0, seed1) = pick_split_seeds(&refs, &clusters, 0, Metric::L2Sq);
        let mut ends = [seed0, seed1];
        ends.sort_unstable();
        assert_eq!(
            ends,
            [0, rows.len() - 1],
            "seeds must be the diameter endpoints, got {ends:?}"
        );
    }

    #[test]
    fn plan_sq8_split_median_cut_balances_skewed_cell() {
        let dim = 4usize;
        // A dense core (16 rows near origin) plus a far sparse tail (4 rows). A
        // nearest-seed split would peel the 4 tail rows off (16/4); the median cut
        // splits by count regardless of density, so the halves come out ~10/10.
        let mut rows = synth_rows(dim, 16, 0.0);
        rows.extend(synth_rows(dim, 4, 50.0));
        let clusters =
            ClusterCentroids::from_fp32(1, dim as u32, &vec![0.0f32; dim], vec![rows.len() as u32]);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (_c0, _c1, assign) = plan_sq8_split(&refs, &clusters, 0, Metric::L2Sq);
        let ones = assign.iter().filter(|&&a| a == 1).count();
        let zeros = assign.len() - ones;
        assert!(
            (ones as i64 - zeros as i64).abs() <= 1,
            "median cut must balance the split, got {zeros} vs {ones}"
        );
    }
}
