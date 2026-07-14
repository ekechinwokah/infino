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

use std::{cmp::Ordering, collections::HashMap};

use crate::{
    config,
    superfile::vector::{
        cell_posting::{
            EncodedCellRow, dequantize_sq8_residual_into, manifest_centroid_components_from_row,
            medoid_index_by,
        },
        distance::{Metric, distance, nearest_k_centroids_transposed},
    },
    supertable::manifest::ClusterCentroids,
};

/// Lloyd iterations for 2-way Sq8+ε k-means at split time.
const CELL_SPLIT_KMEANS_ITERS: usize = 5;

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

/// Drain-only boundary assignment: decode each row once, then rank centroids
/// with a prebuilt transposed centroid cache.
///
/// Same assignment semantics as `nearest-two by score then Voronoi margin`,
/// but without changing ingest/manifest structs.
///
/// Centroid ranking uses a prebuilt
/// transposed centroid cache. The cache is derived from manifest centroids by
/// the drain caller and is not stored on ingest/manifest structs.
pub(crate) fn boundary_assignment_encoded_with_transposed(
    clusters: &ClusterCentroids,
    transposed_centroids: &[f32],
    metric: Metric,
    row: &EncodedCellRow,
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
    boundary_assignment_decoded(clusters, Some(transposed_centroids), metric, &row_fp)
}

/// Boundary assignment for an already-decoded fp32 row (commit buffer path).
/// Uses the same nearest-two + margin logic as the encoded drain wrapper.
pub(crate) fn boundary_assignment_fp32(
    clusters: &ClusterCentroids,
    transposed_centroids: Option<&[f32]>,
    metric: Metric,
    row_fp: &[f32],
) -> BoundaryAssignment {
    boundary_assignment_decoded(clusters, transposed_centroids, metric, row_fp)
}

fn boundary_assignment_decoded(
    clusters: &ClusterCentroids,
    transposed_centroids: Option<&[f32]>,
    metric: Metric,
    row_fp: &[f32],
) -> BoundaryAssignment {
    let n_cent = clusters.n_cent as usize;
    let top_k = REPLICA_CLOSURE_MAX_REPLICAS + 1;
    let ranked: Vec<(u32, f32)> = match transposed_centroids {
        Some(transposed) => nearest_k_centroids_transposed(
            metric,
            row_fp,
            transposed,
            n_cent,
            clusters.dim as usize,
            None,
            top_k,
        ),
        None => {
            // No transposed cache (small callers / tests): scalar-score every
            // cell into the same ascending top-k shape.
            let mut all: Vec<(u32, f32)> = (0..n_cent)
                .map(|cell| (cell as u32, clusters.score_one(metric, cell, row_fp)))
                .collect();
            all.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            all.truncate(top_k);
            all
        }
    };
    let mut replicas = [None; REPLICA_CLOSURE_MAX_REPLICAS];
    let Some(&(primary, primary_score)) = ranked.first() else {
        return BoundaryAssignment {
            primary: 0,
            replicas,
        };
    };
    // Closure pool: every ranked cell whose distance sits within the ratio
    // window of the primary. The margin (distance to the shared Voronoi
    // boundary) orders candidates globally at the budget cut.
    let closure_threshold = primary_score
        + primary_score.abs().max(f32::EPSILON) * (REPLICA_CLOSURE_DISTANCE_RATIO - 1.0);
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
fn medoid_index(metric: Metric, shard: &[EncodedCellRow]) -> usize {
    medoid_index_by(shard, |a, b| fp32_distance_between_rows(metric, a, b))
}

/// 2-way Lloyd k-means on Sq8+ε overflow rows. Returns manifest centroid
/// components (dim each) for the two sub-cells.
pub(crate) fn plan_sq8_split(
    rows: &[EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: u32,
    metric: Metric,
) -> (Vec<f32>, Vec<f32>) {
    let dim = clusters.dim as usize;
    let p = split_cell as usize;

    let seed0 = rows
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            score_row_against_cell(clusters, metric, p, a)
                .partial_cmp(&score_row_against_cell(clusters, metric, p, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let seed1 = rows
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            score_row_against_cell(clusters, metric, p, a)
                .partial_cmp(&score_row_against_cell(clusters, metric, p, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut cent0 = centroid_prototype_from_row(clusters, &rows[seed0]);
    let mut cent1 = centroid_prototype_from_row(clusters, &rows[seed1]);

    let mut assign = vec![0u8; rows.len()];
    for _ in 0..CELL_SPLIT_KMEANS_ITERS {
        for (i, row) in rows.iter().enumerate() {
            let d0 = score_row_against_cell(&cent0, metric, 0, row);
            let d1 = score_row_against_cell(&cent1, metric, 0, row);
            assign[i] = u8::from(d1 < d0);
        }
        let mut shard0 = Vec::new();
        let mut shard1 = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if assign[i] == 0 {
                shard0.push(row.clone());
            } else {
                shard1.push(row.clone());
            }
        }
        if shard0.is_empty() || shard1.is_empty() {
            break;
        }
        let m0 = medoid_index(metric, &shard0);
        let m1 = medoid_index(metric, &shard1);
        cent0 = centroid_prototype_from_row(clusters, &shard0[m0]);
        cent1 = centroid_prototype_from_row(clusters, &shard1[m1]);
    }

    // Re-assign against the converged centroids: the loop's last `assign` pass
    // ran against the *previous* iteration's centroids (cent0/cent1 are updated
    // after it), so the final shards must reflect one more assignment pass.
    for (i, row) in rows.iter().enumerate() {
        let d0 = score_row_against_cell(&cent0, metric, 0, row);
        let d1 = score_row_against_cell(&cent1, metric, 0, row);
        assign[i] = u8::from(d1 < d0);
    }
    let mut shard0 = Vec::new();
    let mut shard1 = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if assign[i] == 0 {
            shard0.push(row.clone());
        } else {
            shard1.push(row.clone());
        }
    }
    if shard1.is_empty() {
        shard1.push(rows[seed1].clone());
        shard0.retain(|r| r.stable_id != rows[seed1].stable_id);
    }
    if shard0.is_empty() {
        shard0.push(rows[seed0].clone());
        shard1.retain(|r| r.stable_id != rows[seed0].stable_id);
    }

    let m0 = medoid_index(metric, &shard0);
    let m1 = medoid_index(metric, &shard1);
    (
        manifest_centroid_components_from_row(&shard0[m0], dim),
        manifest_centroid_components_from_row(&shard1[m1], dim),
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

    let counts = base.counts.clone();
    let updated = ClusterCentroids::from_fp32(new_n as u32, base.dim, &fp32, counts);
    (updated, new_cell_id)
}

/// Neighbor cells touched by a split of `split_cell`: P−1, P, the new sub-cell, P+1.
pub(crate) fn reassign_neighborhood(
    split_cell: u32,
    old_n_cent: u32,
    new_cell_id: u32,
) -> Vec<u32> {
    let mut ids = Vec::new();
    if split_cell > 0 {
        ids.push(split_cell - 1);
    }
    ids.push(split_cell);
    ids.push(new_cell_id);
    if split_cell + 1 < old_n_cent {
        ids.push(split_cell + 1);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Clear per-cell counts when superfiles for those cells are removed and
/// rows are redriven through the incoming staging region.
pub(crate) fn zero_cell_counts(clusters: &mut ClusterCentroids, cells: &[u32]) {
    for &cell in cells {
        let c = cell as usize;
        if c < clusters.counts.len() {
            clusters.counts[c] = 0;
        }
    }
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

    /// Closure replication: a row equidistant-ish to several cells collects a
    /// replica candidate for every cell inside the distance-ratio window
    /// (ordered nearest-first), and a row deep inside its cell collects none.
    #[test]
    fn boundary_assignment_closure_matches_distance_ratio() {
        let dim = 4usize;
        // Four centroids at 0, 1, 2, 30 on every axis.
        let mut fp32 = Vec::new();
        for base in [0.0f32, 1.0, 2.0, 30.0] {
            fp32.extend(std::iter::repeat_n(base, dim));
        }
        let clusters = ClusterCentroids::from_fp32(4, dim as u32, &fp32, vec![1; 4]);

        // Row at 0.9: distances (L2Sq per dim) to cells 0/1/2 are 0.81, 0.01,
        // 1.21 (per-dim) — cell 1 is primary; cell 0 and 2 are far outside a
        // 1.2 ratio window of 0.01. No replicas.
        let deep = vec![0.9f32; dim];
        let assignment = boundary_assignment_fp32(&clusters, None, Metric::L2Sq, &deep);
        assert_eq!(assignment.primary, 1);
        assert_eq!(assignment.replicas, [None; REPLICA_CLOSURE_MAX_REPLICAS]);

        // Row at 1.01 — just past the exact midpoint region between cells 0.98
        // and 1.02... use 1.5: exactly between cells 1 and 2 (distances equal),
        // both inside each other's ratio window; cell 0 at 1.5 distance 2.25
        // per dim is outside 1.2 × 0.25. Expect primary = 1 (tie broken by
        // lower id) and exactly one replica: cell 2.
        let boundary = vec![1.5f32; dim];
        let assignment = boundary_assignment_fp32(&clusters, None, Metric::L2Sq, &boundary);
        assert_eq!(assignment.primary, 1);
        assert_eq!(assignment.replicas[0].map(|(cell, _)| cell), Some(2));
        assert_eq!(assignment.replicas[1], None);
        let margin = assignment.replicas[0].expect("replica").1;
        assert!(
            margin.is_finite() && margin >= 0.0,
            "boundary margin must be a finite non-negative distance, got {margin}"
        );
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
    }

    #[test]
    fn reassign_neighborhood_includes_neighbors_and_new_cell() {
        let ids = reassign_neighborhood(3, 8, 8);
        assert_eq!(ids, vec![2, 3, 4, 8]);
    }

    #[test]
    fn plan_sq8_split_separates_two_blobs() {
        let dim = 4usize;
        let mut rows = synth_rows(dim, 10, 0.0);
        rows.extend(synth_rows(dim, 10, 10.0));
        let clusters = synth_centroids(4, dim as u32);
        let (c0, c1) = plan_sq8_split(&rows, &clusters, 1, Metric::L2Sq);
        assert_eq!(c0.len(), dim);
        assert_eq!(c1.len(), dim);
        let dist: f32 = (0..dim).map(|d| (c0[d] - c1[d]).abs()).sum();
        assert!(dist > 1.0, "split centroids should separate, got {dist}");
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
        let (left, right) = plan_sq8_split(&rows, &clusters, 1, Metric::Cosine);
        let separation: f32 = left.iter().zip(&right).map(|(a, b)| (a - b).abs()).sum();
        assert!(separation > 1.0);
        let after: Vec<(Vec<u8>, Vec<u8>)> = rows
            .iter()
            .map(|row| (row.codes.clone(), row.residuals.clone()))
            .collect();
        assert_eq!(after, before);
    }
}
