// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Resident FTS block-max routing state — the text-superfile analog of
//! the vector path's 1-bit admit slab.
//!
//! For every text superfile (merged inverted-index shard) the drain
//! publishes, this blob carries each **heavy** term's per-block BM25
//! upper bounds, 1-byte quantized, plus the term's postings-region
//! offset. Hydrated resident, it lets a query select which posting
//! blocks to visit — best-first by bound, stopping once the running
//! kth-best floor exceeds every remaining bound — and fetch **only the
//! selected blocks' byte ranges**, exactly as the vector path ranks
//! cells on the resident slab and fetches only admitted cells' bytes.
//! Deeper scans (larger `k`) admit more blocks.
//!
//! The exact 16-byte skip entries stay inside the superfile next to
//! the postings (the fetch source for selected blocks' offsets); this
//! blob is a pure routing accelerator: lose it and queries fall back
//! to whole-term posting fetches, wrong it can never be (quantization
//! only ever rounds bounds UP).
//!
//! One content-addressed object per generation, referenced from the
//! hidden manifest list ([`ManifestSnapshot::slow_fts_state_blob`]),
//! stamped by the drain in the same commit that publishes the text
//! shards it describes, and kept alive by GC via that ref.

use std::sync::Arc;

use bytes::Bytes;
use uuid::Uuid;

use crate::{
    storage::{StorageError, StorageProvider},
    superfile::{
        error::FtsError,
        fts::reader::{FtsReader, RoutedTermRow},
    },
    supertable::manifest::{RoutingRef, part::ContentHash},
};

/// Storage prefix for published generations (content-addressed —
/// superseded generations fall out of the live set and get swept).
pub(crate) const STORAGE_PREFIX: &str = "slow-fts-state/";

/// 8-byte magic at the start of the blob.
const MAGIC: &[u8; 8] = b"INFFBM01";

/// Blob format version.
const VERSION: u32 = 2;

/// Quantization steps for the 1-byte per-block bound (`u8::MAX`).
const QUANT_STEPS: f32 = 255.0;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SlowFtsStateError {
    #[error("storage: {0}")]
    Storage(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("fts: {0}")]
    Fts(String),
}

/// One heavy term's resident routing row.
#[derive(Debug, Clone)]
pub(crate) struct TermBlockMax {
    /// Term bytes (tokenizer output), sorted within the column.
    pub term: Vec<u8>,
    /// The term's postings-region `metadata_offset` — saves the FST
    /// lookup on the block-selected path.
    pub metadata_offset: u64,
    /// Document frequency — the idf input, resident so kernels never
    /// parse the term header.
    pub df: u64,
    /// Per-term dequantization scale: the largest block bound.
    pub scale: f32,
    /// One byte per posting block: `ceil(bound / scale × 255)` — an
    /// UPPER bound after dequantization, so selection can skip a block
    /// only when it truly cannot beat the floor.
    pub quantized: Vec<u8>,
    /// Per-block last doc id — the covering-block binary search runs
    /// on resident data instead of a fetched skip table.
    pub last_docs: Vec<u32>,
    /// Fence-post block byte offsets relative to `metadata_offset`
    /// (`len == quantized.len() + 1`; the last entry is the region
    /// end), sizing any block fetch without the term header. With
    /// `last_docs` this makes the term's skip table fully resident:
    /// a cold consumer's first query was dominated by per-shard ×
    /// per-term head fetches re-reading these exact bytes (150 KB -
    /// 1 MB each at 1M-10M docs), on every query until the
    /// background fill landed.
    pub offsets: Vec<u32>,
}

impl TermBlockMax {
    /// The row borrowed in kernel-call shape — the one construction
    /// point for [`RoutedTermRow`] from resident state.
    pub(crate) fn as_row(&self) -> RoutedTermRow<'_> {
        RoutedTermRow {
            metadata_offset: self.metadata_offset,
            df: self.df,
            quantized: &self.quantized,
            scale: self.scale,
            last_docs: &self.last_docs,
            offsets: &self.offsets,
        }
    }

    /// Dequantized upper bound for block `i` (test-only convenience;
    /// the query kernel dequantizes inline over the raw fields).
    #[cfg(test)]
    fn block_bound(&self, i: usize) -> f32 {
        self.quantized[i] as f32 / QUANT_STEPS * self.scale
    }

    #[cfg(test)]
    fn n_blocks(&self) -> usize {
        self.quantized.len()
    }
}

/// One column's heavy terms, sorted by term bytes.
#[derive(Debug, Clone)]
pub(crate) struct ColumnBlockMax {
    pub column: String,
    pub terms: Vec<TermBlockMax>,
}

/// One text superfile's routing rows.
#[derive(Debug, Clone)]
pub(crate) struct FileBlockMax {
    pub superfile_id: Uuid,
    pub columns: Vec<ColumnBlockMax>,
}

/// The hydrated resident state for one published generation.
#[derive(Debug, Clone, Default)]
pub(crate) struct SlowFtsState {
    /// Sorted by `superfile_id` for binary-search lookup.
    pub files: Vec<FileBlockMax>,
}

impl SlowFtsState {
    /// The resident routing row for `(superfile, column, term)`, if the
    /// term was heavy enough to carry one.
    pub(crate) fn term_block_max(
        &self,
        superfile_id: Uuid,
        column: &str,
        term: &str,
    ) -> Option<&TermBlockMax> {
        let file = self
            .files
            .binary_search_by(|f| f.superfile_id.cmp(&superfile_id))
            .ok()
            .map(|i| &self.files[i])?;
        let col = file.columns.iter().find(|c| c.column == column)?;
        col.terms
            .binary_search_by(|t| t.term.as_slice().cmp(term.as_bytes()))
            .ok()
            .map(|i| &col.terms[i])
    }
}

/// Extract one text superfile's routing rows from its (resident)
/// FTS reader: every PFOR term at or above `df_floor`, with its exact
/// skip-table bounds ceil-quantized to one byte.
pub(crate) async fn build_file_block_max(
    superfile_id: Uuid,
    reader: &FtsReader,
    df_floor: u32,
) -> Result<FileBlockMax, SlowFtsStateError> {
    let columns: Vec<String> = reader.fts_columns().map(str::to_string).collect();
    let mut out_columns = Vec::with_capacity(columns.len());
    for column in columns {
        let rows = reader
            .column_block_maxes(&column, df_floor)
            .await
            .map_err(|e: FtsError| SlowFtsStateError::Fts(e.to_string()))?;
        if rows.is_empty() {
            continue;
        }
        let terms = rows
            .into_iter()
            .map(|row| {
                let scale = row.maxes.iter().copied().fold(0.0f32, f32::max);
                let quantized = row
                    .maxes
                    .iter()
                    .map(|&m| match scale > 0.0 {
                        // ceil() keeps the dequantized value a true
                        // upper bound; the +epsilon-free form is safe
                        // because ceil of the exact ratio can only
                        // round up.
                        true => (m / scale * QUANT_STEPS).ceil().min(QUANT_STEPS) as u8,
                        false => 0,
                    })
                    .collect();
                TermBlockMax {
                    term: row.term,
                    metadata_offset: row.metadata_offset,
                    df: row.df,
                    scale,
                    quantized,
                    last_docs: row.last_docs,
                    offsets: row.offsets,
                }
            })
            .collect();
        out_columns.push(ColumnBlockMax { column, terms });
    }
    Ok(FileBlockMax {
        superfile_id,
        columns: out_columns,
    })
}

/// Encode the state to its wire form (see the module docs for the
/// role; layout below is length-prefixed little-endian).
pub(crate) fn encode_state(state: &SlowFtsState) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(state.files.len() as u32).to_le_bytes());
    for file in &state.files {
        out.extend_from_slice(file.superfile_id.as_bytes());
        out.extend_from_slice(&(file.columns.len() as u32).to_le_bytes());
        for col in &file.columns {
            out.extend_from_slice(&(col.column.len() as u16).to_le_bytes());
            out.extend_from_slice(col.column.as_bytes());
            out.extend_from_slice(&(col.terms.len() as u32).to_le_bytes());
            for t in &col.terms {
                out.extend_from_slice(&(t.term.len() as u16).to_le_bytes());
                out.extend_from_slice(&t.term);
                out.extend_from_slice(&t.metadata_offset.to_le_bytes());
                out.extend_from_slice(&t.df.to_le_bytes());
                out.extend_from_slice(&t.scale.to_le_bytes());
                out.extend_from_slice(&(t.quantized.len() as u32).to_le_bytes());
                out.extend_from_slice(&t.quantized);
                for &d in &t.last_docs {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                for &o in &t.offsets {
                    out.extend_from_slice(&o.to_le_bytes());
                }
            }
        }
    }
    out
}

/// Decode a blob written by [`encode_state`]. Corrupt input yields
/// `Err`, never a panic — consumers fall back to whole-term fetches.
pub(crate) fn decode_state(bytes: &[u8]) -> Result<SlowFtsState, SlowFtsStateError> {
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Result<&[u8], SlowFtsStateError> {
        let end = at
            .checked_add(n)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| SlowFtsStateError::Parse("truncated slow-fts-state blob".into()))?;
        let s = &bytes[*at..end];
        *at = end;
        Ok(s)
    };
    if take(&mut at, MAGIC.len())? != MAGIC {
        return Err(SlowFtsStateError::Parse("bad slow-fts-state magic".into()));
    }
    let version = u32::from_le_bytes(take(&mut at, 4)?.try_into().expect("4 bytes"));
    if version != VERSION {
        return Err(SlowFtsStateError::Parse(format!(
            "unsupported slow-fts-state version {version}"
        )));
    }
    // Length prefixes are untrusted until their bytes are consumed:
    // clamp every pre-allocation by the bytes remaining (each element
    // is at least one byte), so a corrupt count returns `Err` at the
    // first truncated element instead of aborting on allocation.
    let bounded_cap = |n: usize, at: usize| n.min(bytes.len().saturating_sub(at));
    let n_files = u32::from_le_bytes(take(&mut at, 4)?.try_into().expect("4 bytes")) as usize;
    let mut files = Vec::with_capacity(bounded_cap(n_files, at));
    for _ in 0..n_files {
        let superfile_id = Uuid::from_slice(take(&mut at, 16)?)
            .map_err(|e| SlowFtsStateError::Parse(e.to_string()))?;
        let n_columns = u32::from_le_bytes(take(&mut at, 4)?.try_into().expect("4 bytes")) as usize;
        let mut columns = Vec::with_capacity(bounded_cap(n_columns, at));
        for _ in 0..n_columns {
            let name_len =
                u16::from_le_bytes(take(&mut at, 2)?.try_into().expect("2 bytes")) as usize;
            let column = String::from_utf8(take(&mut at, name_len)?.to_vec())
                .map_err(|e| SlowFtsStateError::Parse(e.to_string()))?;
            let n_terms =
                u32::from_le_bytes(take(&mut at, 4)?.try_into().expect("4 bytes")) as usize;
            let mut terms = Vec::with_capacity(bounded_cap(n_terms, at));
            for _ in 0..n_terms {
                let term_len =
                    u16::from_le_bytes(take(&mut at, 2)?.try_into().expect("2 bytes")) as usize;
                let term = take(&mut at, term_len)?.to_vec();
                let metadata_offset =
                    u64::from_le_bytes(take(&mut at, 8)?.try_into().expect("8 bytes"));
                let df = u64::from_le_bytes(take(&mut at, 8)?.try_into().expect("8 bytes"));
                let scale = f32::from_le_bytes(take(&mut at, 4)?.try_into().expect("4 bytes"));
                let n_blocks =
                    u32::from_le_bytes(take(&mut at, 4)?.try_into().expect("4 bytes")) as usize;
                let quantized = take(&mut at, n_blocks)?.to_vec();
                let last_docs: Vec<u32> = take(&mut at, n_blocks * 4)?
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
                    .collect();
                let offsets: Vec<u32> = take(&mut at, (n_blocks + 1) * 4)?
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
                    .collect();
                terms.push(TermBlockMax {
                    term,
                    metadata_offset,
                    df,
                    scale,
                    quantized,
                    last_docs,
                    offsets,
                });
            }
            columns.push(ColumnBlockMax { column, terms });
        }
        files.push(FileBlockMax {
            superfile_id,
            columns,
        });
    }
    Ok(SlowFtsState { files })
}

/// Content-addressed URI for one published generation.
pub(crate) fn storage_path(hash: &ContentHash) -> String {
    format!("{STORAGE_PREFIX}state-{}.bin", hash.to_hex())
}

/// Publish one generation: encode, content-address, idempotent PUT.
pub(crate) async fn write_state(
    storage: &dyn StorageProvider,
    state: &SlowFtsState,
) -> Result<RoutingRef, SlowFtsStateError> {
    let bytes = encode_state(state);
    let content_hash = ContentHash::of(&bytes);
    let uri = storage_path(&content_hash);
    match storage.put_atomic(&uri, Bytes::from(bytes)).await {
        Ok(_) | Err(StorageError::PreconditionFailed { .. }) => {}
        Err(error) => return Err(SlowFtsStateError::Storage(error.to_string())),
    }
    Ok(RoutingRef { uri, content_hash })
}

/// Fetch + verify + decode one generation. Any failure surfaces as
/// `Err`; consumers treat it as "no resident routing" and fall back.
pub(crate) async fn fetch_state(
    storage: &dyn StorageProvider,
    reference: &RoutingRef,
) -> Result<Arc<SlowFtsState>, SlowFtsStateError> {
    let (bytes, _) = storage
        .get(&reference.uri)
        .await
        .map_err(|e| SlowFtsStateError::Storage(e.to_string()))?;
    if ContentHash::of(&bytes) != reference.content_hash {
        return Err(SlowFtsStateError::Parse(format!(
            "slow-fts-state content hash mismatch at {}",
            reference.uri
        )));
    }
    Ok(Arc::new(decode_state(&bytes)?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc as StdArc;

    use bytes::Bytes;

    use super::*;
    use crate::superfile::fts::{
        builder::FtsBuilder, reader::BoolMode, tokenize::AsciiLowerTokenizer,
    };

    fn sample_state() -> SlowFtsState {
        SlowFtsState {
            files: vec![FileBlockMax {
                superfile_id: Uuid::from_u128(7),
                columns: vec![ColumnBlockMax {
                    column: "title".into(),
                    terms: vec![
                        TermBlockMax {
                            term: b"common".to_vec(),
                            metadata_offset: 1234,
                            df: 4096,
                            scale: 8.5,
                            quantized: vec![255, 30, 254, 1],
                            last_docs: vec![100, 250, 900, 4095],
                            offsets: vec![0, 64, 130, 220, 300],
                        },
                        TermBlockMax {
                            term: b"heavy".to_vec(),
                            metadata_offset: 99,
                            df: 300 * 128,
                            scale: 3.25,
                            quantized: vec![128; 300],
                            last_docs: (0..300).map(|b| (b + 1) * 128 - 1).collect(),
                            offsets: (0..=300).map(|b| b * 96).collect(),
                        },
                    ],
                }],
            }],
        }
    }

    #[test]
    fn state_round_trips() {
        let state = sample_state();
        let bytes = encode_state(&state);
        let decoded = decode_state(&bytes).expect("decode");
        assert_eq!(decoded.files.len(), 1);
        let t = decoded
            .term_block_max(Uuid::from_u128(7), "title", "common")
            .expect("row present");
        assert_eq!(t.metadata_offset, 1234);
        assert_eq!(t.n_blocks(), 4);
        assert_eq!(t.block_bound(0), 8.5);
        assert!(
            decoded
                .term_block_max(Uuid::from_u128(7), "title", "absent")
                .is_none()
        );
    }

    #[test]
    fn decode_rejects_truncation_and_bad_magic() {
        let bytes = encode_state(&sample_state());
        assert!(decode_state(&bytes[..bytes.len() - 3]).is_err());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(decode_state(&bad).is_err());
    }

    /// The block-selected walk must return the same top-k SCORES as
    /// the exhaustive kernel (tie-broken doc ids may differ at the k
    /// boundary; the score multiset may not).
    #[tokio::test]
    async fn block_selected_matches_full_walk() {
        let tok = StdArc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        // 2000 docs, one common term with tf variance (score variance
        // across blocks is what best-first selection exploits) plus a
        // unique filler per doc (doc-length variance).
        for d in 0..2000u32 {
            let tf = (d % 5) + 1;
            let text = format!("{}filler{}", "common ".repeat(tf as usize), d);
            b.add_doc(0, d, &text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let r = crate::superfile::fts::reader::FtsReader::open(
            blob,
            r#"[{"name":"body","tokenizer":"ascii_lower"}]"#,
        )
        .expect("open");

        let file = build_file_block_max(Uuid::from_u128(1), &r, 2)
            .await
            .expect("rows");
        let state = SlowFtsState { files: vec![file] };
        let row = state
            .term_block_max(Uuid::from_u128(1), "body", "common")
            .expect("'common' is heavy enough at floor 2");
        assert!(row.n_blocks() > 4, "corpus spans multiple blocks");

        for k in [1usize, 3, 10, 100, 500] {
            let expected = r
                .search_with_floor("body", &["common"], k, BoolMode::Or, f32::NEG_INFINITY)
                .await
                .expect("full walk");
            let got = r
                .bm25_single_term_block_selected("body", k, f32::NEG_INFINITY, &row.as_row())
                .await
                .expect("block-selected walk");
            // Full (doc, score) equality: the tie contract (kth-score
            // ties resolve to ascending doc id) must hold under
            // bound-ordered visits, not just the score profile.
            assert_eq!(got, expected, "k={k}");
        }
    }

    /// kth-score ties must resolve to ascending doc id even when the
    /// tied docs live in blocks with unequal bounds (visited out of
    /// doc order). Planted shape: doc 500 is the clear top hit, docs
    /// 0 and 501 tie exactly for the kth slot (same tf, same doc
    /// length) — doc 501 rides the high-bound block that is visited
    /// first, doc 0 sits alone in a lower-bound block. The kernel
    /// must still visit doc 0's block (bar excludes the kth tie) and
    /// keep doc 0 over doc 501.
    #[tokio::test]
    async fn block_selected_keeps_smallest_ids_on_kth_ties() {
        let tok = StdArc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for d in 0..2000u32 {
            // `common` lives only in docs 0..600 (healthy idf; an
            // every-doc term's block maxes collapse to the fixed-point
            // floor and the bounds go flat, which skips the pruning
            // path this test exists to exercise).
            let tf = match d {
                500 => 5,
                0 | 501 => 3,
                _ if d < 600 => 1,
                _ => 0,
            };
            let text = format!("{}filler{}", "common ".repeat(tf as usize), d);
            b.add_doc(0, d, &text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let r =
            FtsReader::open(blob, r#"[{"name":"body","tokenizer":"ascii_lower"}]"#).expect("open");

        let file = build_file_block_max(Uuid::from_u128(1), &r, 2)
            .await
            .expect("rows");
        let state = SlowFtsState { files: vec![file] };
        let row = state
            .term_block_max(Uuid::from_u128(1), "body", "common")
            .expect("'common' carries a row");

        let expected = r
            .search_with_floor("body", &["common"], 2, BoolMode::Or, f32::NEG_INFINITY)
            .await
            .expect("full walk");
        assert_eq!(
            expected.iter().map(|&(d, _)| d).collect::<Vec<_>>(),
            vec![500, 0],
            "planted corpus: doc 0 wins the kth tie in the plain walk"
        );
        let got = r
            .bm25_single_term_block_selected("body", 2, f32::NEG_INFINITY, &row.as_row())
            .await
            .expect("block-selected walk");
        assert_eq!(got, expected);
    }

    /// The quantized bound must never drop below the exact bound —
    /// selection may only ever visit MORE blocks than strictly needed.
    #[test]
    fn quantization_is_an_upper_bound() {
        let maxes = [7.3f32, 0.01, 3.999, 8.0, 0.0, 5.5551];
        let scale = maxes.iter().copied().fold(0.0f32, f32::max);
        for &m in &maxes {
            let q = (m / scale * QUANT_STEPS).ceil().min(QUANT_STEPS) as u8;
            let dequant = q as f32 / QUANT_STEPS * scale;
            assert!(
                dequant >= m,
                "dequantized {dequant} must bound exact {m} from above"
            );
        }
    }
}
