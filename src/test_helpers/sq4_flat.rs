// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Flat-scan probe over the 4-bit resident plane.
//!
//! A measurement seam, not a serving path. It answers one question the
//! engine cannot currently be configured to answer: **what does our
//! 4-bit codec score, and how fast, when it ranks terminally instead of
//! navigating a graph?**
//!
//! The distinction matters because the graph walk and a flat scan fail
//! differently. A walk can miss a neighbourhood outright — its recall
//! mixes codec error with routing error. A scan visits every vector, so
//! whatever it loses is quantization error alone. That is the regime a
//! compressed flat index competes in, so it is the only honest way to
//! compare our codec against one.
//!
//! Nothing here is new arithmetic: the encoder ([`Sq4Scorer`]) and the
//! SIMD nibble kernel it scores with are the shipping ones, reached
//! through the same [`NodeScorer`] interface the walk uses. The only
//! addition is the loop that visits every node instead of a beam.
//!
//! Byte accounting note: the plane rotates through the *blocked*
//! transform, which keeps the rotated space at exactly `dim`, so stored
//! bytes are `dim/2` per plane per row with no power-of-two padding.
//! That matters because a flat scan's per-query cost is bytes-read ÷
//! bandwidth, so stored padding would be paid twice — once in residency
//! and once in latency. [`Sq4FlatIndex::minimum_bytes`] recomputes the
//! floor independently and should equal
//! [`Sq4FlatIndex::resident_bytes`]; a divergence means padding crept
//! back into the plane.

use std::{cmp::Ordering, collections::BinaryHeap};

use rayon::prelude::*;

use crate::superfile::vector::{
    distance::encode_sq16_row,
    hnsw::{NodeScorer, Sq4Scorer},
};

/// Bytes a 4-bit code occupies per dimension, expressed as its
/// reciprocal: two coordinates share one byte.
const COORDS_PER_BYTE: usize = 2;
/// Bytes per `f32` ruler entry (offset and step each store one per
/// rotated coordinate).
const RULER_ENTRY_BYTES: usize = 4;
/// Minimum rows a rayon task claims. Large enough that per-task setup
/// and the fold's heap allocation stay negligible against the scan, small
/// enough that the tail does not idle threads at the corpus sizes a flat
/// index is used at.
const SCAN_BLOCK_ROWS: usize = 4_096;

/// One scored candidate, ordered so a [`BinaryHeap`] of bounded size
/// evicts the current worst (largest score, since lower is nearer).
#[derive(PartialEq)]
struct Candidate {
    score: f32,
    node: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Scores are finite by construction (fitted ruler, finite
        // codes); `total_cmp` keeps the ordering total regardless.
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A resident 4-bit plane scanned exhaustively per query.
pub struct Sq4FlatIndex {
    scorer: Sq4Scorer,
    dim: usize,
    len: usize,
}

impl Sq4FlatIndex {
    /// Encode `vectors` (row-major, `len × dim` fp32) into the 4-bit
    /// plane.
    ///
    /// The plane's encoder consumes the stored Sq16 representation, so
    /// the rows go through the same [`encode_sq16_row`] the builder
    /// writes with before being fitted and packed to nibbles — i.e. this
    /// reproduces the exact bytes a drain would produce, rather than a
    /// parallel encode path that could drift from it.
    ///
    /// `with_residual` selects the 1 byte/dim construction (coarse plane
    /// plus a residual nibble) over the bare 0.5 byte/dim one.
    pub fn build(vectors: &[f32], dim: usize, rot_seed: u64, with_residual: bool) -> Self {
        assert!(dim > 0, "dim must be non-zero");
        assert!(
            vectors.len().is_multiple_of(dim),
            "vector buffer must be a whole number of rows"
        );
        let len = vectors.len() / dim;
        let mut sq16 = vec![0u8; len * dim * 2];
        for (row, codes) in vectors
            .chunks_exact(dim)
            .zip(sq16.chunks_exact_mut(dim * 2))
        {
            encode_sq16_row(row, codes);
        }
        let scorer = Sq4Scorer::from_sq16_plane(&sq16, dim, len, with_residual, rot_seed, None);
        Self { scorer, dim, len }
    }

    /// Stored rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the index holds no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes held resident to serve: the code plane, the residual plane
    /// when present, and the two ruler vectors. This is the number to
    /// set against a competing index's resident footprint.
    pub fn resident_bytes(&self) -> usize {
        let (codes, residual, offset, step) = self.scorer.parts();
        codes.len()
            + residual.map_or(0, <[u8]>::len)
            + (offset.len() + step.len()) * RULER_ENTRY_BYTES
    }

    /// The byte floor for these codes, recomputed from `dim` and the
    /// plane count rather than read off the buffers. Equal to
    /// [`Self::resident_bytes`] while the rotation stays unpadded; a
    /// divergence is padding creeping back in.
    pub fn minimum_bytes(&self) -> usize {
        let (_, residual, _, _) = self.scorer.parts();
        let planes = if residual.is_some() { 2 } else { 1 };
        let per_row = self.dim.div_ceil(COORDS_PER_BYTE) * planes;
        per_row * self.len + self.dim * COORDS_PER_BYTE * RULER_ENTRY_BYTES
    }

    /// Exhaustive top-`k` for one query. Returns `(node, score)` with
    /// **lower score nearer**, matching the engine's `NegDot` convention,
    /// sorted nearest-first.
    ///
    /// Single query per call by design: a batched form would amortize
    /// per-query setup and layout reuse across a batch, which is exactly
    /// the accounting that makes a published batch figure incomparable
    /// to a served one.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimensionality mismatch");
        if k == 0 || self.len == 0 {
            return Vec::new();
        }
        let prepared = self.scorer.prepare(query);
        // Blocked and parallel, matching how the engine's own scan path
        // runs (rayon for the CPU wave). A single-threaded per-node loop
        // measures the loop, not the codec: on this hardware it sustained
        // ~0.76 GB/s against the ~34 GB/s the engine's all-cells Sq16 scan
        // reaches, so the harness rather than the kernel set the number.
        let heap = (0..self.len)
            .into_par_iter()
            .with_min_len(SCAN_BLOCK_ROWS)
            .fold(
                || BinaryHeap::<Candidate>::with_capacity(k + 1),
                |mut heap, node| {
                    let node = node as u32;
                    let score = self.scorer.score(&prepared, node);
                    // The root is the worst kept candidate, so a bounded
                    // push/pop keeps the k nearest without sorting N.
                    if heap.len() < k {
                        heap.push(Candidate { score, node });
                    } else if heap.peek().is_some_and(|worst| score < worst.score) {
                        heap.pop();
                        heap.push(Candidate { score, node });
                    }
                    heap
                },
            )
            .reduce(
                || BinaryHeap::<Candidate>::with_capacity(k + 1),
                |mut a, b| {
                    for c in b {
                        if a.len() < k {
                            a.push(c);
                        } else if a.peek().is_some_and(|worst| c.score < worst.score) {
                            a.pop();
                            a.push(c);
                        }
                    }
                    a
                },
            );
        let mut out: Vec<(u32, f32)> = heap.into_iter().map(|c| (c.node, c.score)).collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out
    }
}
