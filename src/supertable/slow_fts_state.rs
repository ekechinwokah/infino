// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Tiered FTS block-max routing state — the text-superfile analog of
//! the vector path's resident 1-bit admit slab + spilled fp32
//! centroid section.
//!
//! For every text superfile (merged inverted-index shard) the drain
//! publishes, one content-addressed blob carries each **heavy** term's
//! routing data in two tiers:
//!
//! * **Resident directory (L0)** — per term: postings-region offset,
//!   df, dequantization scale, block count, and COARSE bounds (one
//!   ceil-quantized byte per [`COARSE_GROUP_BLOCKS`]-block group).
//!   Hydration keeps ONLY this tier in memory: ~1 MB at 10M docs where
//!   full per-block residency was ~70 MB (and ~7 GB at 1B — the
//!   VERSION-2 lesson). The coarse bounds drive the dispatch gates:
//!   which files engage block selection at all.
//! * **Spilled per-term slices (L1)** — per block: the 1-byte bound,
//!   last doc id, and fence-post offset. Fetched on demand per
//!   (file, term) as one ranged read of this same blob (CRC-checked,
//!   single-flight, cached for the generation's lifetime), exactly as
//!   the vector path preads its once-per-generation centroid section.
//!   Engaged queries pay at most one extra GET per routed term cold
//!   and nothing warm.
//!
//! Kernels then select posting blocks best-first by bound, stopping
//! once the running kth-best floor exceeds every remaining bound, and
//! fetch **only the selected blocks' byte ranges**. Deeper scans
//! (larger `k`) admit more blocks.
//!
//! The exact 16-byte skip entries stay inside the superfile next to
//! the postings; this blob is a pure routing accelerator: lose it and
//! queries fall back to whole-term posting fetches, wrong it can never
//! be (quantization only ever rounds bounds UP).
//!
//! One content-addressed object per generation, referenced from the
//! hidden manifest list ([`ManifestSnapshot::slow_fts_state_blob`]),
//! stamped by the drain in the same commit that publishes the text
//! shards it describes, and kept alive by GC via that ref. Hydration
//! currently downloads the whole blob once to verify the content hash
//! and drops everything past the directory; moving to a ranged
//! header-only read (blob size in [`RoutingRef`]) is a scoped
//! follow-up.

use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::{
    storage::{StorageError, StorageProvider},
    superfile::{
        error::FtsError,
        format::checksum::crc32c,
        fts::reader::{FtsReader, RoutedTermRow},
    },
    supertable::manifest::{RoutingRef, part::ContentHash},
};

/// Storage prefix for published generations (content-addressed —
/// superseded generations fall out of the live set and get swept).
pub(crate) const STORAGE_PREFIX: &str = "slow-fts-state/";

/// 8-byte magic at the start of the blob.
const MAGIC: &[u8; 8] = b"INFFBM01";

/// Blob format version. 3 = tiered layout: resident directory with
/// coarse bounds, per-term block data spilled behind `spill_base`.
const VERSION: u32 = 3;

/// Quantization steps for the 1-byte per-block bound (`u8::MAX`).
pub(crate) const QUANT_STEPS: f32 = 255.0;

/// Blocks per coarse-bound group in the resident directory: each
/// group byte is the max of its blocks' quantized bounds, so the
/// resident tier is 1/8 the per-block data and still a true upper
/// bound for every block it covers.
pub(crate) const COARSE_GROUP_BLOCKS: usize = 8;

/// Fixed-width prefix before the directory: magic + version +
/// `spill_base` (u64).
const HEADER_PREFIX_BYTES: usize = 8 + 4 + 8;

/// Trailing crc32c width, used by both the directory (last 4 bytes
/// before `spill_base`) and every spilled term slice.
const CRC_BYTES: usize = 4;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SlowFtsStateError {
    #[error("storage: {0}")]
    Storage(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("fts: {0}")]
    Fts(String),
}

/// One heavy term's FULL routing row: the build-side shape the drain
/// produces, and the fetched shape a query hydrates from one spilled
/// slice (L1) when the term engages block selection.
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

/// One column's heavy terms, sorted by term bytes (build side).
#[derive(Debug, Clone)]
pub(crate) struct ColumnBlockMax {
    pub column: String,
    pub terms: Vec<TermBlockMax>,
}

/// One text superfile's routing rows (build side).
#[derive(Debug, Clone)]
pub(crate) struct FileBlockMax {
    pub superfile_id: Uuid,
    pub columns: Vec<ColumnBlockMax>,
}

/// The build-side state: every file's FULL rows, as the drain computes
/// them. [`encode_state`] is its only consumer — the encoder derives
/// the resident directory (coarse bounds, spill offsets) from it.
#[derive(Debug, Clone, Default)]
pub(crate) struct SlowFtsStateFull {
    /// Sorted by `superfile_id`.
    pub files: Vec<FileBlockMax>,
}

/// One heavy term's RESIDENT directory row (L0): everything the
/// dispatch gates need to decide engagement, plus the location of the
/// term's spilled per-block slice.
#[derive(Debug)]
pub(crate) struct TermDirRow {
    /// Term bytes (tokenizer output), sorted within the column.
    pub term: Vec<u8>,
    /// The term's postings-region `metadata_offset`.
    pub metadata_offset: u64,
    /// Document frequency — the idf input.
    pub df: u64,
    /// Per-term dequantization scale: the largest block bound.
    pub scale: f32,
    /// Posting block count (sizes the spilled slice).
    pub n_blocks: u32,
    /// Spilled-slice offset relative to the blob's `spill_base`.
    spill_rel: u64,
    /// Coarse bounds: one byte per [`COARSE_GROUP_BLOCKS`]-block
    /// group, the max of the group's quantized bounds — a true upper
    /// bound for every covered block, at 1/8 the residency.
    pub coarse: Vec<u8>,
    /// Single-flight cache of the fetched full row, per generation.
    cell: OnceCell<Arc<TermBlockMax>>,
}

impl TermDirRow {
    /// Largest coarse bound — by ceil-quantization construction this
    /// is always the term max (255) for a non-empty row.
    #[cfg(test)]
    fn max_coarse(&self) -> u8 {
        self.coarse.iter().copied().max().unwrap_or(0)
    }
}

/// One column's resident directory rows, sorted by term bytes.
#[derive(Debug)]
pub(crate) struct ColumnDir {
    pub column: String,
    pub terms: Vec<TermDirRow>,
}

/// One text superfile's resident directory.
#[derive(Debug)]
pub(crate) struct FileDir {
    pub superfile_id: Uuid,
    pub columns: Vec<ColumnDir>,
}

/// Where a hydrated state's spilled slices live.
#[derive(Debug)]
enum Spill {
    /// The whole blob is in memory (tests). Slices decode from it
    /// directly, through the same directory-parse + slice-decode
    /// path as production.
    #[cfg(test)]
    Inline(Bytes),
    /// Slices are ranged reads of the published generation object.
    Remote {
        storage: Arc<dyn StorageProvider>,
        uri: String,
    },
}

/// The hydrated QUERY-side state for one published generation:
/// resident directory + on-demand spilled slices.
#[derive(Debug, Default)]
pub(crate) struct SlowFtsState {
    /// Sorted by `superfile_id` for binary-search lookup.
    pub files: Vec<FileDir>,
    /// Absolute blob offset where the spilled section starts.
    spill_base: u64,
    /// `None` only for the `Default` empty state (no files ⇒ no
    /// lookups ⇒ no fetches).
    spill: Option<Spill>,
}

impl SlowFtsState {
    /// The resident directory row for `(superfile, column, term)`, if
    /// the term was heavy enough to carry one.
    pub(crate) fn term_dir(
        &self,
        superfile_id: Uuid,
        column: &str,
        term: &str,
    ) -> Option<&TermDirRow> {
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

    /// The term's FULL row, fetching its spilled slice on first use
    /// (single-flight per term per generation; one ranged read, CRC
    /// checked). Warm calls return the cached `Arc` untouched.
    pub(crate) async fn fetch_row(
        &self,
        dir: &TermDirRow,
    ) -> Result<Arc<TermBlockMax>, SlowFtsStateError> {
        dir.cell
            .get_or_try_init(|| async {
                let n = dir.n_blocks as usize;
                let len = slice_len(n) as u64;
                let start = self
                    .spill_base
                    .checked_add(dir.spill_rel)
                    .ok_or_else(|| SlowFtsStateError::Parse("spill offset overflow".into()))?;
                let bytes = match self.spill.as_ref() {
                    #[cfg(test)]
                    Some(Spill::Inline(blob)) => {
                        let s = start as usize;
                        let e = s
                            .checked_add(len as usize)
                            .filter(|&e| e <= blob.len())
                            .ok_or_else(|| {
                                SlowFtsStateError::Parse("spill slice out of bounds".into())
                            })?;
                        blob.slice(s..e)
                    }
                    Some(Spill::Remote { storage, uri }) => storage
                        .get_range(uri, start..start + len)
                        .await
                        .map_err(|e| SlowFtsStateError::Storage(e.to_string()))?,
                    None => {
                        return Err(SlowFtsStateError::Parse(
                            "slow-fts-state has no spill source".into(),
                        ));
                    }
                };
                decode_term_slice(&bytes, dir).map(Arc::new)
            })
            .await
            .map(Arc::clone)
    }
}

/// Spilled-slice byte length for `n` blocks: bounds + last docs +
/// fence-post offsets + trailing crc32c.
fn slice_len(n: usize) -> usize {
    n + n * 4 + (n + 1) * 4 + CRC_BYTES
}

/// Coarse bounds for one full row: max of each
/// [`COARSE_GROUP_BLOCKS`]-block group's quantized bounds.
fn coarse_bounds(quantized: &[u8]) -> Vec<u8> {
    quantized
        .chunks(COARSE_GROUP_BLOCKS)
        .map(|g| g.iter().copied().max().unwrap_or(0))
        .collect()
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

/// Encode the build-side state to its V3 wire form: a resident
/// directory (crc-terminated) at the front, every term's per-block
/// slice (crc-terminated) behind `spill_base`. Spill offsets in the
/// directory are RELATIVE to `spill_base`, so the directory's length
/// never depends on the absolute positions it describes.
pub(crate) fn encode_state(state: &SlowFtsStateFull) -> Vec<u8> {
    // Pass 1: the spilled section, recording each term's relative
    // offset in build order (the same order the directory serializes).
    let mut spill: Vec<u8> = Vec::new();
    let mut spill_rels: Vec<u64> = Vec::new();
    for file in &state.files {
        for col in &file.columns {
            for t in &col.terms {
                spill_rels.push(spill.len() as u64);
                let payload_start = spill.len();
                spill.extend_from_slice(&t.quantized);
                for &d in &t.last_docs {
                    spill.extend_from_slice(&d.to_le_bytes());
                }
                for &o in &t.offsets {
                    spill.extend_from_slice(&o.to_le_bytes());
                }
                let crc = crc32c(&spill[payload_start..]);
                spill.extend_from_slice(&crc.to_le_bytes());
            }
        }
    }

    // Pass 2: the directory, with coarse bounds derived per term.
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    // `spill_base` back-patched once the directory length is known.
    let spill_base_at = out.len();
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(state.files.len() as u32).to_le_bytes());
    let mut next_rel = spill_rels.iter().copied();
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
                let rel = next_rel.next().expect("one spill slice per term");
                out.extend_from_slice(&rel.to_le_bytes());
                out.extend_from_slice(&coarse_bounds(&t.quantized));
            }
        }
    }
    let dir_crc = crc32c(&out);
    out.extend_from_slice(&dir_crc.to_le_bytes());
    let spill_base = out.len() as u64;
    out[spill_base_at..spill_base_at + 8].copy_from_slice(&spill_base.to_le_bytes());
    // The directory crc covers the back-patched `spill_base`, so
    // recompute it now that the field is final.
    let crc_at = out.len() - CRC_BYTES;
    let dir_crc = crc32c(&out[..crc_at]);
    let crc_bytes = dir_crc.to_le_bytes();
    out[crc_at..].copy_from_slice(&crc_bytes);
    out.extend_from_slice(&spill);
    out
}

/// Decode a V3 directory written by [`encode_state`] from the blob's
/// prefix (`bytes` must reach `spill_base`; extra bytes are ignored).
/// Corrupt input yields `Err`, never a panic — consumers fall back to
/// whole-term fetches. The returned state carries no spill source;
/// [`fetch_state`] (or a test's inline attach) supplies it.
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
    let spill_base = u64::from_le_bytes(take(&mut at, 8)?.try_into().expect("8 bytes"));
    let dir_len = usize::try_from(spill_base)
        .ok()
        .filter(|&d| d >= HEADER_PREFIX_BYTES + CRC_BYTES && d <= bytes.len())
        .ok_or_else(|| SlowFtsStateError::Parse("bad slow-fts-state spill base".into()))?;
    // Directory integrity: the trailing crc32c covers everything
    // before it, `spill_base` included (it was back-patched before the
    // crc was computed).
    let crc_at = dir_len - CRC_BYTES;
    let want = u32::from_le_bytes(bytes[crc_at..dir_len].try_into().expect("4 bytes"));
    if crc32c(&bytes[..crc_at]) != want {
        return Err(SlowFtsStateError::Parse(
            "slow-fts-state directory crc mismatch".into(),
        ));
    }
    // Length prefixes are untrusted until their bytes are consumed:
    // clamp every pre-allocation by the bytes remaining (each element
    // is at least one byte), so a corrupt count returns `Err` at the
    // first truncated element instead of aborting on allocation.
    let bounded_cap = |n: usize, at: usize| n.min(dir_len.saturating_sub(at));
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
                let n_blocks = u32::from_le_bytes(take(&mut at, 4)?.try_into().expect("4 bytes"));
                let spill_rel = u64::from_le_bytes(take(&mut at, 8)?.try_into().expect("8 bytes"));
                let coarse =
                    take(&mut at, (n_blocks as usize).div_ceil(COARSE_GROUP_BLOCKS))?.to_vec();
                terms.push(TermDirRow {
                    term,
                    metadata_offset,
                    df,
                    scale,
                    n_blocks,
                    spill_rel,
                    coarse,
                    cell: OnceCell::new(),
                });
            }
            columns.push(ColumnDir { column, terms });
        }
        files.push(FileDir {
            superfile_id,
            columns,
        });
    }
    if at != crc_at {
        return Err(SlowFtsStateError::Parse(
            "slow-fts-state directory has trailing bytes".into(),
        ));
    }
    Ok(SlowFtsState {
        files,
        spill_base,
        spill: None,
    })
}

/// Decode one spilled term slice (crc-terminated) into the FULL row.
fn decode_term_slice(bytes: &[u8], dir: &TermDirRow) -> Result<TermBlockMax, SlowFtsStateError> {
    let n = dir.n_blocks as usize;
    if bytes.len() != slice_len(n) {
        return Err(SlowFtsStateError::Parse(
            "slow-fts-state slice length mismatch".into(),
        ));
    }
    let crc_at = bytes.len() - CRC_BYTES;
    let want = u32::from_le_bytes(bytes[crc_at..].try_into().expect("4 bytes"));
    if crc32c(&bytes[..crc_at]) != want {
        return Err(SlowFtsStateError::Parse(
            "slow-fts-state slice crc mismatch".into(),
        ));
    }
    let quantized = bytes[..n].to_vec();
    let last_docs: Vec<u32> = bytes[n..n + n * 4]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    let offsets: Vec<u32> = bytes[n + n * 4..crc_at]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    Ok(TermBlockMax {
        term: dir.term.clone(),
        metadata_offset: dir.metadata_offset,
        df: dir.df,
        scale: dir.scale,
        quantized,
        last_docs,
        offsets,
    })
}

/// Content-addressed URI for one published generation.
pub(crate) fn storage_path(hash: &ContentHash) -> String {
    format!("{STORAGE_PREFIX}state-{}.bin", hash.to_hex())
}

/// Publish one generation: encode, content-address, idempotent PUT.
pub(crate) async fn write_state(
    storage: &dyn StorageProvider,
    state: &SlowFtsStateFull,
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

/// Fetch + verify one generation, keeping ONLY the resident directory
/// in memory; per-term slices come back later as ranged reads of the
/// same (immutable, content-addressed) object. Any failure surfaces
/// as `Err`; consumers treat it as "no resident routing" and fall
/// back. The whole blob is downloaded once here so the content hash
/// can be verified end-to-end, then dropped past the directory;
/// slices re-fetched later are guarded by their own crc32c.
pub(crate) async fn fetch_state(
    storage: Arc<dyn StorageProvider>,
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
    let mut state = decode_state(&bytes)?;
    state.spill = Some(Spill::Remote {
        storage,
        uri: reference.uri.clone(),
    });
    Ok(Arc::new(state))
}

/// Hydrate a build-side state through the real wire path with the
/// whole blob retained inline — the query-side state tests use this,
/// exercising the same directory parse and slice decode as production
/// without a storage provider.
#[cfg(test)]
pub(crate) fn hydrate_inline(full: &SlowFtsStateFull) -> Result<SlowFtsState, SlowFtsStateError> {
    let bytes = encode_state(full);
    let mut state = decode_state(&bytes)?;
    state.spill = Some(Spill::Inline(Bytes::from(bytes)));
    Ok(state)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc as StdArc;

    use bytes::Bytes;

    use super::*;
    use crate::superfile::fts::{
        builder::FtsBuilder, reader::BoolMode, tokenize::AsciiLowerTokenizer,
    };

    fn sample_state() -> SlowFtsStateFull {
        SlowFtsStateFull {
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

    #[tokio::test]
    async fn state_round_trips() {
        let full = sample_state();
        let state = hydrate_inline(&full).expect("hydrate");
        assert_eq!(state.files.len(), 1);
        let dir = state
            .term_dir(Uuid::from_u128(7), "title", "common")
            .expect("dir row present");
        assert_eq!(dir.metadata_offset, 1234);
        assert_eq!(dir.n_blocks, 4);
        assert_eq!(dir.coarse.len(), 1, "4 blocks fit one coarse group");
        assert_eq!(dir.max_coarse(), 255, "ceil-quant max is always full");
        // The fetched full row equals the encoded input exactly.
        let row = state.fetch_row(dir).await.expect("slice fetch");
        let want = &full.files[0].columns[0].terms[0];
        assert_eq!(row.quantized, want.quantized);
        assert_eq!(row.last_docs, want.last_docs);
        assert_eq!(row.offsets, want.offsets);
        assert_eq!(row.n_blocks(), 4);
        assert_eq!(row.block_bound(0), 8.5);
        // Second fetch returns the cached Arc (single-flight cell).
        let again = state.fetch_row(dir).await.expect("cached fetch");
        assert!(StdArc::ptr_eq(&row, &again));
        // The 300-block term spans multiple coarse groups.
        let heavy = state
            .term_dir(Uuid::from_u128(7), "title", "heavy")
            .expect("heavy dir row");
        assert_eq!(heavy.coarse.len(), 300usize.div_ceil(COARSE_GROUP_BLOCKS));
        assert!(
            state
                .term_dir(Uuid::from_u128(7), "title", "absent")
                .is_none()
        );
    }

    #[tokio::test]
    async fn decode_rejects_truncation_and_bad_magic() {
        let bytes = encode_state(&sample_state());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(decode_state(&bad).is_err());
        // A cut inside the directory fails decode outright.
        let spill_base = u64::from_le_bytes(bytes[12..20].try_into().expect("8")) as usize;
        assert!(decode_state(&bytes[..spill_base - 3]).is_err());
        // A cut inside the SPILL leaves the directory decodable — the
        // damage surfaces as a failed (crc/length-checked) slice fetch.
        let cut = Bytes::from(bytes[..bytes.len() - 3].to_vec());
        let mut state = decode_state(&cut).expect("directory intact");
        state.spill = Some(Spill::Inline(cut));
        let dir = state
            .term_dir(Uuid::from_u128(7), "title", "heavy")
            .expect("dir row");
        assert!(state.fetch_row(dir).await.is_err());
    }

    /// `write_state` / `fetch_state` round-trip through LocalFs, and a
    /// wrong content hash fails closed before decode.
    #[tokio::test]
    async fn write_and_fetch_state_round_trip_and_rejects_hash_mismatch() {
        use std::sync::Arc;

        use tempfile::TempDir;

        use crate::storage::LocalFsStorageProvider;

        let dir = TempDir::new().expect("tmpdir");
        let storage = Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let state = sample_state();
        let storage_dyn: Arc<dyn StorageProvider> = storage.clone();
        let reference = write_state(storage.as_ref(), &state)
            .await
            .expect("write_state");
        let fetched = fetch_state(Arc::clone(&storage_dyn), &reference)
            .await
            .expect("fetch_state");
        assert_eq!(fetched.files.len(), state.files.len());
        assert_eq!(fetched.files[0].superfile_id, state.files[0].superfile_id);
        // Remote spill path: a slice comes back as a ranged read of
        // the published object and matches the encoded input.
        let dir = fetched
            .term_dir(Uuid::from_u128(7), "title", "common")
            .expect("dir row");
        let row = fetched.fetch_row(dir).await.expect("remote slice fetch");
        assert_eq!(row.quantized, state.files[0].columns[0].terms[0].quantized);

        // Tamper with the expected hash — fetch must refuse the bytes.
        let mut bad = reference.clone();
        bad.content_hash.0[0] ^= 0xff;
        let err = fetch_state(Arc::clone(&storage_dyn), &bad)
            .await
            .expect_err("hash mismatch");
        assert!(
            matches!(err, SlowFtsStateError::Parse(ref msg) if msg.contains("hash mismatch")),
            "{err:?}"
        );

        // Missing object → Storage error (not a parse/hash failure).
        let missing = RoutingRef {
            uri: "slow-fts-state/does-not-exist.bin".into(),
            content_hash: ContentHash([0u8; 32]),
        };
        let err = fetch_state(storage_dyn, &missing)
            .await
            .expect_err("missing uri");
        assert!(matches!(err, SlowFtsStateError::Storage(_)), "{err:?}");
    }

    /// Non-UTF-8 column names and unsupported versions must fail
    /// closed — corrupt routing blobs never produce a partial state.
    #[test]
    fn decode_rejects_bad_version_and_non_utf8_column() {
        let mut bytes = encode_state(&sample_state());
        // VERSION is the u32 immediately after MAGIC (checked before
        // the directory crc, so an unsupported version reports as
        // such rather than as corruption).
        let ver_at = MAGIC.len();
        bytes[ver_at..ver_at + 4].copy_from_slice(&999u32.to_le_bytes());
        let err = decode_state(&bytes).expect_err("bad version");
        assert!(
            matches!(err, SlowFtsStateError::Parse(ref msg) if msg.contains("unsupported")),
            "{err:?}"
        );

        // A tampered column name without a crc re-patch fails closed
        // at the directory crc.
        let mut bytes = encode_state(&sample_state());
        // Layout after MAGIC+version+spill_base+n_files+uuid+n_columns:
        // u16 name_len, name bytes.
        let name_len_at = MAGIC.len() + 4 + 8 + 4 + 16 + 4;
        let name_len =
            u16::from_le_bytes(bytes[name_len_at..name_len_at + 2].try_into().expect("2"));
        assert_eq!(name_len as usize, "title".len());
        let name_at = name_len_at + 2;
        bytes[name_at] = 0xff; // invalid UTF-8 lead byte
        let err = decode_state(&bytes).expect_err("crc catches the tamper");
        assert!(
            matches!(err, SlowFtsStateError::Parse(ref msg) if msg.contains("crc")),
            "{err:?}"
        );

        // Re-patch the crc so the parse reaches the name itself — the
        // UTF-8 validation must still fail closed.
        let spill_base = u64::from_le_bytes(bytes[12..20].try_into().expect("8")) as usize;
        let crc_at = spill_base - CRC_BYTES;
        let crc = crc32c(&bytes[..crc_at]).to_le_bytes();
        bytes[crc_at..spill_base].copy_from_slice(&crc);
        let err = decode_state(&bytes).expect_err("non-utf8 column");
        assert!(matches!(err, SlowFtsStateError::Parse(_)), "{err:?}");
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
        let state = hydrate_inline(&SlowFtsStateFull { files: vec![file] }).expect("hydrate");
        let dir = state
            .term_dir(Uuid::from_u128(1), "body", "common")
            .expect("'common' is heavy enough at floor 2");
        let row = state.fetch_row(dir).await.expect("slice fetch");
        assert!(row.n_blocks() > 4, "corpus spans multiple blocks");

        for k in [1usize, 3, 10, 100, 500] {
            let expected = r
                .search_with_floor("body", &["common"], k, BoolMode::Or, f32::NEG_INFINITY)
                .await
                .expect("full walk");
            let got = r
                .bm25_single_term_block_selected(
                    "body",
                    k,
                    f32::NEG_INFINITY,
                    &row.as_row(),
                    None,
                    None,
                )
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
        let state = hydrate_inline(&SlowFtsStateFull { files: vec![file] }).expect("hydrate");
        let dir = state
            .term_dir(Uuid::from_u128(1), "body", "common")
            .expect("'common' carries a row");
        let row = state.fetch_row(dir).await.expect("slice fetch");

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
            .bm25_single_term_block_selected(
                "body",
                2,
                f32::NEG_INFINITY,
                &row.as_row(),
                None,
                None,
            )
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
