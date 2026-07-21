// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Merge N source FTS blobs into one — the write side of the hidden
//! index's text superfiles.
//!
//! A text superfile holds a lex term-range slice of the drained
//! corpus's merged inverted index. This module produces its FTS blob
//! by walking every source's FST in lex order (k-way union), decoding
//! each term's postings, remapping source-local doc ids into the
//! merged doc space (dropping tombstoned docs), and re-encoding
//! through the same [`encode_and_emit_term`] + [`assemble_and_write_blob`]
//! pipeline the [`FtsBuilder`](super::builder::FtsBuilder) finish
//! paths use — every byte the reader observes still passes through
//! the one shared encode/assembly path. No re-tokenization: sources
//! are the postings themselves.

// TODO(hidden-fts drain): staging-only — the drain's fts-merge
// consolidation (next commit in this series) is the non-test consumer;
// remove with it.
#![allow(dead_code)]

use std::{
    fs::File,
    io::{BufWriter, Error, ErrorKind, Write},
    str::from_utf8,
};

use super::{
    builder::{
        BlobAssemblyInputs, FinishProfile, FstSinkFinish, PositionsSink, TermScratch,
        assemble_and_write_blob, encode_and_emit_term, map_fst_err,
    },
    dict::StreamingDictBuilder,
    positions::skip_run,
    reader::FtsReader,
};
use crate::superfile::{
    error::{BuildError, FtsError},
    format::FST_SEPARATOR,
};

/// One merge input: a source blob and its doc-id remap into the
/// merged doc space.
pub(crate) struct MergeSource<'a> {
    pub reader: &'a FtsReader,
    /// `doc_id_remap[source_local_doc_id]` = merged doc id, or `None`
    /// for a dropped (tombstoned) doc. Length must equal the source's
    /// `n_docs`. The caller assigns merged ids **monotonically within
    /// each source and in disjoint ascending ranges across sources in
    /// slice order** — that makes a term's merged posting list the
    /// in-order concatenation of its sources' remapped lists, which
    /// the encoder requires (postings are sorted by doc id).
    pub doc_id_remap: &'a [Option<u32>],
}

/// A column of the merged blob, in declaration order — must match the
/// `inf.fts.columns` JSON the enclosing superfile will carry.
pub(crate) struct MergeColumn {
    pub name: String,
    /// Whether the column records token positions. Must equal the
    /// sources' positional flag for this column (the format never
    /// mixes strides within a column).
    pub positions: bool,
}

/// Merge the sources' postings for `columns` into one FTS blob
/// written to `w`, keeping only terms whose full FST key
/// (`<column>\x1F<term>`) falls in the half-open `key_bounds` range
/// (`None` = every term). `n_docs_merged` is the merged doc-space
/// size (the blob header's `n_docs`, which also scales idf).
///
/// Terms whose postings are empty after tombstone filtering are
/// dropped entirely — the merged FST only holds live terms.
pub(crate) async fn merge_fts_blobs<W: Write>(
    sources: &[MergeSource<'_>],
    columns: &[MergeColumn],
    n_docs_merged: u32,
    key_bounds: Option<(&[u8], &[u8])>,
    mut w: W,
) -> Result<(), BuildError> {
    for s in sources {
        debug_assert_eq!(
            s.doc_id_remap.len(),
            s.reader.n_docs() as usize,
            "doc_id_remap must cover every source doc"
        );
    }

    let scratch_dir = tempfile::tempdir()?;
    let scratch_path = scratch_dir.path().to_path_buf();
    let postings_path = scratch_path.join("infino_fts_postings.bin");
    let mut postings_writer = BufWriter::new(File::create(&postings_path)?);
    let mut postings_len: u64 = 0;
    let mut postings_crc_acc: u32 = 0;
    let mut positions_sink = PositionsSink::create(&scratch_path)?;
    let fst_streaming_path = scratch_path.join("infino_fts_dict.bin");
    let mut fst_streaming =
        StreamingDictBuilder::new(BufWriter::new(File::create(&fst_streaming_path)?))
            .map_err(map_fst_err)?;
    let mut key_buf: Vec<u8> = Vec::with_capacity(64);
    let mut term_scratch = TermScratch::default();
    let mut finish_profile = FinishProfile::from_config();

    let n_columns = columns.len() as u32;
    let mut n_terms_total_usize: usize = 0;
    let mut avgdl_per_col: Vec<f32> = vec![0.0; columns.len()];
    let mut doc_lengths_by_orig_col: Vec<Option<Vec<u32>>> =
        (0..columns.len()).map(|_| None).collect();

    // FST keys are `<column>\x1F<term>`, so emission must follow lex
    // column order — mirror the builder finish paths' sort-by-name.
    let mut work: Vec<usize> = (0..columns.len()).collect();
    work.sort_unstable_by(|&a, &b| columns[a].name.cmp(&columns[b].name));

    // Reused per-term buffers (mirrors the builder's TermScratch
    // discipline: allocations amortize across every merged term).
    let mut src_pairs: Vec<(u32, u32)> = Vec::new();
    let mut merged_pairs: Vec<(u32, u32)> = Vec::new();
    let mut merged_runs: Vec<u8> = Vec::new();

    for &orig_idx in &work {
        let MergeColumn { name, positions } = &columns[orig_idx];
        let col_name_bytes = name.as_bytes();

        // Merged per-doc lengths: scatter each source's raw doc-length
        // array through its remap. Docs absent from the column keep 0.
        let mut merged_dl: Vec<u32> = vec![0; n_docs_merged as usize];
        for s in sources {
            if !s.reader.fts_columns().any(|c| c == name) {
                continue;
            }
            let src_dl = s
                .reader
                .column_doc_lengths_raw(name)
                .await
                .map_err(map_source_err)?;
            debug_assert_eq!(src_dl.len(), s.doc_id_remap.len());
            for (local, &dl) in src_dl.iter().enumerate() {
                if let Some(merged) = s.doc_id_remap[local] {
                    merged_dl[merged as usize] = dl;
                }
            }
        }
        // Same semantics as the builder's `total_tokens / n_docs`:
        // zero-length docs stay in the denominator.
        let avgdl = match n_docs_merged {
            0 => 0.0,
            n => merged_dl.iter().map(|&d| d as u64).sum::<u64>() as f32 / n as f32,
        };
        avgdl_per_col[orig_idx] = avgdl;

        // Per-source lex-ordered term entries + a cursor each; the
        // k-way union scans cursors linearly (source counts are drain
        // batch sizes — small — so a heap would be overhead, not
        // simplification).
        let mut entries: Vec<Vec<(Vec<u8>, u64)>> = Vec::with_capacity(sources.len());
        for s in sources {
            entries.push(s.reader.column_term_entries(name).map_err(map_source_err)?);
        }
        let mut cursors: Vec<usize> = vec![0; sources.len()];

        loop {
            // Smallest un-consumed term across sources.
            let mut min_term: Option<&[u8]> = None;
            for (i, e) in entries.iter().enumerate() {
                if let Some((term, _)) = e.get(cursors[i])
                    && min_term.map(|m| term.as_slice() < m).unwrap_or(true)
                {
                    min_term = Some(term);
                }
            }
            let Some(term) = min_term else { break };
            let term = term.to_vec();

            // Full-key bounds check (shard slicing).
            let in_bounds = key_bounds
                .map(|(lo, hi)| {
                    key_buf.clear();
                    key_buf.extend_from_slice(col_name_bytes);
                    key_buf.push(FST_SEPARATOR);
                    key_buf.extend_from_slice(&term);
                    key_buf.as_slice() >= lo && key_buf.as_slice() < hi
                })
                .unwrap_or(true);

            merged_pairs.clear();
            merged_runs.clear();
            for (i, s) in sources.iter().enumerate() {
                let Some((src_term, packed)) = entries[i].get(cursors[i]) else {
                    continue;
                };
                if src_term != &term {
                    continue;
                }
                cursors[i] += 1;
                if !in_bounds {
                    continue;
                }
                let runs = s
                    .reader
                    .decode_term_postings(*positions, *packed, &mut src_pairs)
                    .await
                    .map_err(map_source_err)?;
                match runs {
                    Some(runs) => {
                        // Copy each surviving doc's run alongside its
                        // remapped pair; a dropped doc's bytes are
                        // skipped, keeping runs aligned pair-for-pair.
                        let mut at = 0usize;
                        for &(local, tf) in &src_pairs {
                            let run_start = at;
                            skip_run(&runs, &mut at, tf).ok_or_else(|| {
                                BuildError::Io(Error::new(
                                    ErrorKind::InvalidData,
                                    "source position runs truncated",
                                ))
                            })?;
                            if let Some(merged) = s.doc_id_remap[local as usize] {
                                merged_pairs.push((merged, tf));
                                merged_runs.extend_from_slice(&runs[run_start..at]);
                            }
                        }
                    }
                    None => {
                        for &(local, tf) in &src_pairs {
                            if let Some(merged) = s.doc_id_remap[local as usize] {
                                merged_pairs.push((merged, tf));
                            }
                        }
                    }
                }
            }
            if merged_pairs.is_empty() {
                continue;
            }

            let term_str = from_utf8(&term).map_err(|_| {
                BuildError::Io(Error::new(
                    ErrorKind::InvalidData,
                    "source FST term is not valid UTF-8",
                ))
            })?;
            let term_positions = positions.then_some((&mut positions_sink, merged_runs.as_slice()));
            encode_and_emit_term(
                term_str,
                &merged_pairs,
                col_name_bytes,
                &merged_dl,
                avgdl,
                n_docs_merged,
                &mut key_buf,
                &mut postings_writer,
                &mut postings_crc_acc,
                &mut postings_len,
                None,
                Some(&mut fst_streaming),
                term_positions,
                &mut finish_profile,
                &mut term_scratch,
            )?;
            n_terms_total_usize += 1;
        }

        doc_lengths_by_orig_col[orig_idx] = Some(merged_dl);
    }

    assemble_and_write_blob(
        BlobAssemblyInputs {
            postings_writer,
            postings_path,
            postings_crc_acc,
            postings_len,
            positions_sink,
            fst_sink: FstSinkFinish::Streaming {
                builder: fst_streaming,
                path: fst_streaming_path,
            },
            n_columns,
            n_docs: n_docs_merged,
            n_terms_total_usize,
            avgdl_per_col,
            doc_lengths_by_orig_col,
            scratch_dir,
            finish_profile,
        },
        &mut w,
    )
}

/// Source blobs were validated at their own build + open time, so a
/// decode failure here means a corrupt input, surfaced as an
/// `InvalidData` build error rather than a panic.
fn map_source_err(e: FtsError) -> BuildError {
    BuildError::Io(Error::new(ErrorKind::InvalidData, e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, sync::Arc};

    use arrow_array::{ArrayRef, Decimal128Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;

    use super::*;
    use crate::superfile::{
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::{
            builder::FtsBuilder, positions::decode_run, reader::BoolMode,
            tokenize::AsciiLowerTokenizer,
        },
        reader::SuperfileReader,
    };

    /// Two positionless sources with an overlapping vocabulary and a
    /// tombstoned doc. Source-local → merged: src0 = [0, dropped, 1],
    /// src1 = [2, 3].
    async fn merged_positionless_blob() -> Bytes {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b0 = FtsBuilder::new(tok.clone());
        b0.register_column("body".into(), false).expect("register");
        b0.add_doc(0, 0, "rust engine").expect("add");
        b0.add_doc(0, 1, "zombie doomed rust").expect("add"); // tombstoned
        b0.add_doc(0, 2, "engine parquet").expect("add");
        let src0 = FtsReader::open(
            Bytes::from(b0.finish().expect("finish")),
            r#"[{"name":"body","tokenizer":"ascii_lower"}]"#,
        )
        .expect("open src0");

        let mut b1 = FtsBuilder::new(tok);
        b1.register_column("body".into(), false).expect("register");
        b1.add_doc(0, 0, "rust rust storage").expect("add");
        b1.add_doc(0, 1, "parquet").expect("add");
        let src1 = FtsReader::open(
            Bytes::from(b1.finish().expect("finish")),
            r#"[{"name":"body","tokenizer":"ascii_lower"}]"#,
        )
        .expect("open src1");

        let mut out = Vec::new();
        merge_fts_blobs(
            &[
                MergeSource {
                    reader: &src0,
                    doc_id_remap: &[Some(0), None, Some(1)],
                },
                MergeSource {
                    reader: &src1,
                    doc_id_remap: &[Some(2), Some(3)],
                },
            ],
            &[MergeColumn {
                name: "body".into(),
                positions: false,
            }],
            4,
            None,
            &mut out,
        )
        .await
        .expect("merge");
        Bytes::from(out)
    }

    #[tokio::test]
    async fn merge_round_trips_remapped_postings() {
        let blob = merged_positionless_blob().await;
        // `open` validates magic, header, and every region CRC.
        let r = FtsReader::open(blob, r#"[{"name":"body","tokenizer":"ascii_lower"}]"#)
            .expect("open merged blob");
        assert_eq!(r.n_docs(), 4);

        let positional = r.column_positional("body").expect("column");
        let mut got: Vec<(String, Vec<(u32, u32)>)> = Vec::new();
        let mut pairs = Vec::new();
        for (term, packed) in r.column_term_entries("body").expect("entries") {
            r.decode_term_postings(positional, packed, &mut pairs)
                .await
                .expect("decode");
            got.push((String::from_utf8(term).expect("utf8"), pairs.clone()));
        }
        // The tombstoned doc's exclusive terms ("zombie", "doomed")
        // must vanish; shared terms lose only its posting; tf survives
        // the remap ("rust rust storage" keeps tf=2).
        let expect: Vec<(&str, Vec<(u32, u32)>)> = vec![
            ("engine", vec![(0, 1), (1, 1)]),
            ("parquet", vec![(1, 1), (3, 1)]),
            ("rust", vec![(0, 1), (2, 2)]),
            ("storage", vec![(2, 1)]),
        ];
        assert_eq!(
            got,
            expect
                .into_iter()
                .map(|(t, p)| (t.to_string(), p))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn merged_blob_is_searchable_with_remapped_ids() {
        let blob = merged_positionless_blob().await;
        let r = FtsReader::open(blob, r#"[{"name":"body","tokenizer":"ascii_lower"}]"#)
            .expect("open merged blob");
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("search");
        let mut ids: Vec<u32> = hits.iter().map(|&(d, _)| d).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 2], "remapped ids; tombstoned doc gone");
    }

    #[tokio::test]
    async fn merge_respects_key_bounds() {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "alpha beta zeta").expect("add");
        let src = FtsReader::open(
            Bytes::from(b.finish().expect("finish")),
            r#"[{"name":"body","tokenizer":"ascii_lower"}]"#,
        )
        .expect("open src");

        // Keys are `body\x1F<term>`; keep only ["body\x1Falpha", "body\x1Fbeta"] —
        // "beta" itself is excluded (half-open), as is "zeta".
        let mut lo = b"body".to_vec();
        lo.push(FST_SEPARATOR);
        lo.extend_from_slice(b"alpha");
        let mut hi = b"body".to_vec();
        hi.push(FST_SEPARATOR);
        hi.extend_from_slice(b"beta");

        let mut out = Vec::new();
        merge_fts_blobs(
            &[MergeSource {
                reader: &src,
                doc_id_remap: &[Some(0)],
            }],
            &[MergeColumn {
                name: "body".into(),
                positions: false,
            }],
            1,
            Some((&lo, &hi)),
            &mut out,
        )
        .await
        .expect("merge");
        let r = FtsReader::open(
            Bytes::from(out),
            r#"[{"name":"body","tokenizer":"ascii_lower"}]"#,
        )
        .expect("open merged blob");
        let terms: Vec<Vec<u8>> = r
            .column_term_entries("body")
            .expect("entries")
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(terms, vec![b"alpha".to_vec()]);
    }

    #[tokio::test]
    async fn merge_carries_positions_through_remap() {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b0 = FtsBuilder::new(tok.clone());
        b0.register_column("body".into(), true).expect("register");
        b0.add_doc(0, 0, "alpha beta alpha").expect("add");
        b0.add_doc(0, 1, "alpha dropped").expect("add"); // tombstoned
        let src0 = FtsReader::open(
            Bytes::from(b0.finish().expect("finish")),
            r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#,
        )
        .expect("open src0");
        let mut b1 = FtsBuilder::new(tok);
        b1.register_column("body".into(), true).expect("register");
        b1.add_doc(0, 0, "beta alpha").expect("add");
        let src1 = FtsReader::open(
            Bytes::from(b1.finish().expect("finish")),
            r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#,
        )
        .expect("open src1");

        let mut out = Vec::new();
        merge_fts_blobs(
            &[
                MergeSource {
                    reader: &src0,
                    doc_id_remap: &[Some(0), None],
                },
                MergeSource {
                    reader: &src1,
                    doc_id_remap: &[Some(1)],
                },
            ],
            &[MergeColumn {
                name: "body".into(),
                positions: true,
            }],
            2,
            None,
            &mut out,
        )
        .await
        .expect("merge");
        let r = FtsReader::open(
            Bytes::from(out),
            r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#,
        )
        .expect("open merged blob");

        let mut pairs = Vec::new();
        let mut by_term = std::collections::HashMap::new();
        for (term, packed) in r.column_term_entries("body").expect("entries") {
            let runs = r
                .decode_term_postings(true, packed, &mut pairs)
                .await
                .expect("decode")
                .expect("positional runs");
            let mut at = 0;
            let decoded: Vec<(u32, Vec<u32>)> = pairs
                .iter()
                .map(|&(d, tf)| {
                    let mut positions = Vec::new();
                    decode_run(&runs, &mut at, tf, &mut positions).expect("run");
                    (d, positions)
                })
                .collect();
            assert_eq!(at, runs.len(), "runs cover exactly the kept pairs");
            by_term.insert(String::from_utf8(term).expect("utf8"), decoded);
        }
        // "alpha": src0 doc0 @[0,2] → merged 0; src1 doc0 @[1] → merged 1.
        // The tombstoned src0 doc1's run is dropped.
        assert_eq!(by_term["alpha"], vec![(0, vec![0, 2]), (1, vec![1])],);
        // "beta": src0 doc0 @[1] → merged 0; src1 doc0 @[0] → merged 1.
        assert_eq!(by_term["beta"], vec![(0, vec![1]), (1, vec![0])]);
        // "dropped" lived only in the tombstoned doc.
        assert!(!by_term.contains_key("dropped"));
    }

    /// End-to-end text superfile: merged FTS blob + `_id`-stub Parquet
    /// body spliced by `finish_with_prebuilt_fts_to`, opened through
    /// the real `SuperfileReader` (footer, KV, CRCs), searched with
    /// BM25, and its stub resolved back to stable ids.
    #[tokio::test]
    async fn text_superfile_round_trips_through_superfile_reader() {
        let blob = merged_positionless_blob().await;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "_id",
            DataType::Decimal128(38, 0),
            false,
        )]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "_id",
            vec![FtsConfig {
                column: "body".into(),
                positions: false,
            }],
            vec![],
            None,
        )
        .with_prebuilt_fts();
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        // Merged doc order = merged local id order: stable ids for the
        // 4 live docs of `merged_positionless_blob`.
        let stable_ids: Vec<i128> = vec![101, 202, 303, 404];
        let id_array = Decimal128Array::from_iter_values(stable_ids.clone())
            .with_precision_and_scale(38, 0)
            .expect("precision 38 scale 0 holds any i128");
        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_array) as ArrayRef])
            .expect("ids-only batch");
        b.add_batch_ids_only(&batch).expect("add ids");
        let mut superfile = Vec::new();
        b.finish_with_prebuilt_fts_to(
            Cursor::new(blob.to_vec()),
            blob.len() as u64,
            &mut superfile,
        )
        .expect("finish text superfile");

        let reader = SuperfileReader::open(Bytes::from(superfile)).expect("open text superfile");
        // BM25 over the spliced merged blob: "rust" lives in merged
        // docs 0 and 2 (see `merged_positionless_blob`).
        let hits = reader
            .bm25_search_pretokenized("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("bm25 over text superfile");
        let mut local_ids: Vec<u32> = hits.iter().map(|&(d, _)| d).collect();
        local_ids.sort_unstable();
        assert_eq!(local_ids, vec![0, 2]);

        // The `_id` stub column maps merged local ids to stable ids.
        let stub = reader.get_record_batch(None).expect("stub batch");
        assert_eq!(stub.num_rows(), stable_ids.len());
        let col = stub
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        let got: Vec<i128> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, stable_ids);
    }
}
