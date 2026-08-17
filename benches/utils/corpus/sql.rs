// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Schema-driven SQL corpus: derives a queryable Arrow schema from a
//! parquet dataset's own columns (no fixed [`crate::harness::SqlRow`]
//! fixture) and streams the shards through it, converting `Binary` /
//! `LargeBinary` columns to `LargeUtf8` since the SQL engines under test
//! don't index raw bytes.

use std::{fs::File, str::from_utf8, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, RecordBatch,
    RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use infino::superfile::vector::distance::Metric;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{
    corpus::{CorpusSource, PARQUET_VECTOR_COLUMNS, parquet_shards_for},
    harness::{SqlCorpusSpec, SqlVectorSpec},
};

/// Rows per streamed Parquet batch. Matches the corpus module's reader
/// chunking so memory behaviour is the same across corpus paths.
const SQL_BATCH_ROWS: usize = 1_024;

/// A schema-driven SQL corpus: a dataset's own Arrow schema (binary
/// columns rewritten to text) plus the batches read up to `max_rows`.
pub struct ParquetSqlCorpus {
    spec: SqlCorpusSpec,
    batches: Vec<RecordBatch>,
    lossy_rows: usize,
}

impl ParquetSqlCorpus {
    pub fn spec(&self) -> &SqlCorpusSpec {
        &self.spec
    }

    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    pub fn n_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// Rows across the whole corpus that needed lossy UTF-8 replacement.
    pub fn lossy_rows(&self) -> usize {
        self.lossy_rows
    }
}

/// Map each field's type to what the SQL engines under test can index:
/// `Binary` / `LargeBinary` become `LargeUtf8`, everything else passes
/// through unchanged. Nullability is preserved.
pub(crate) fn cast_schema_for_sql(schema: &Schema) -> SchemaRef {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Binary | DataType::LargeBinary => {
                Field::new(f.name(), DataType::LargeUtf8, f.is_nullable())
            }
            _ => f.as_ref().clone(),
        })
        .collect();
    Arc::new(Schema::new(fields))
}

/// Convert one `Binary`/`LargeBinary` column to `LargeUtf8`. Null stays
/// null; valid UTF-8 passes through; invalid bytes are replaced lossily
/// and counted. Not the arrow `cast` kernel: `cast` errors on invalid
/// UTF-8, and real datasets (ClickBench `hits` included) carry some.
pub(crate) fn binary_to_large_utf8(array: &ArrayRef) -> (ArrayRef, usize) {
    let mut lossy_rows = 0;
    let mut convert = |bytes: Option<&[u8]>| -> Option<String> {
        let bytes = bytes?;
        Some(match from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => {
                lossy_rows += 1;
                String::from_utf8_lossy(bytes).into_owned()
            }
        })
    };
    let values: Vec<Option<String>> = if let Some(a) = array.as_any().downcast_ref::<BinaryArray>()
    {
        a.iter().map(&mut convert).collect()
    } else if let Some(a) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        a.iter().map(&mut convert).collect()
    } else {
        panic!(
            "binary_to_large_utf8 expects Binary or LargeBinary, got {:?}",
            array.data_type()
        )
    };
    (Arc::new(LargeStringArray::from(values)), lossy_rows)
}

/// Convert every `Binary`/`LargeBinary` column of one batch to `LargeUtf8`,
/// against the already-derived `schema`. Returns the converted batch and
/// the number of rows across the batch that needed lossy replacement.
fn convert_batch(schema: &SchemaRef, batch: &RecordBatch) -> (RecordBatch, usize) {
    let mut lossy_rows = 0;
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, column)| match field.data_type() {
            DataType::LargeUtf8
                if matches!(column.data_type(), DataType::Binary | DataType::LargeBinary) =>
            {
                let (converted, rows) = binary_to_large_utf8(column);
                lossy_rows += rows;
                converted
            }
            _ => Arc::clone(column),
        })
        .collect();
    let converted = RecordBatch::try_new(Arc::clone(schema), columns)
        .expect("converted columns match the derived schema");
    (converted, lossy_rows)
}

/// The `dim` of an embedding column, when `data_type` is a
/// `FixedSizeList<Float32>` — the only vector encoding this corpus reads.
fn fixed_size_float32_dim(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::FixedSizeList(item, size) if *item.data_type() == DataType::Float32 => {
            Some(*size as usize)
        }
        _ => None,
    }
}

/// Derive a [`SqlCorpusSpec`] from a dataset's own schema: no FTS columns
/// (see [`SqlCorpusSpec::fts_columns`]), and a vector spec only when one of
/// [`PARQUET_VECTOR_COLUMNS`] is present as `FixedSizeList<Float32>`.
pub(crate) fn spec_from_schema(schema: SchemaRef) -> SqlCorpusSpec {
    let vector = PARQUET_VECTOR_COLUMNS.iter().find_map(|name| {
        let field = schema.column_with_name(name)?.1;
        let dim = fixed_size_float32_dim(field.data_type())?;
        Some(SqlVectorSpec {
            column: (*name).to_string(),
            dim,
            metric: Metric::Cosine,
        })
    });
    SqlCorpusSpec {
        schema,
        fts_columns: Vec::new(),
        vector,
    }
}

/// Read a parquet dataset's shards in order, deriving the SQL-facing
/// schema from the first shard and streaming batches (converted to that
/// schema) until `max_rows` rows are collected or the shards are
/// exhausted. Prints one lossy-UTF-8 summary line for the whole corpus.
pub fn open(source: &CorpusSource, max_rows: usize) -> ParquetSqlCorpus {
    let shards = parquet_shards_for(source);
    let mut spec: Option<SqlCorpusSpec> = None;
    let mut batches = Vec::new();
    let mut lossy_rows = 0;
    let mut n_rows = 0;
    'shards: for shard in &shards {
        let file = File::open(shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("read {}: {e}", shard.display()))
            .with_batch_size(SQL_BATCH_ROWS)
            .build()
            .unwrap_or_else(|e| panic!("build reader {}: {e}", shard.display()));
        let schema = spec
            .get_or_insert_with(|| spec_from_schema(cast_schema_for_sql(&reader.schema())))
            .schema
            .clone();
        for batch in reader {
            let batch = batch.unwrap_or_else(|e| panic!("read batch {}: {e}", shard.display()));
            // Slice to the retained prefix BEFORE converting: `lossy_rows` must count
            // only rows that end up in `batches`, never a discarded straddling tail.
            let keep = batch.num_rows().min(max_rows - n_rows);
            let batch = if keep < batch.num_rows() {
                batch.slice(0, keep)
            } else {
                batch
            };
            let (converted, rows) = convert_batch(&schema, &batch);
            lossy_rows += rows;
            n_rows += converted.num_rows();
            batches.push(converted);
            if n_rows >= max_rows {
                break 'shards;
            }
        }
    }
    if lossy_rows > 0 {
        eprintln!(
            "[corpus/sql] {lossy_rows} rows contained invalid UTF-8 and were replaced lossily"
        );
    }
    ParquetSqlCorpus {
        spec: spec.unwrap_or_else(|| panic!("parquet source has no shards")),
        batches,
        lossy_rows,
    }
}

#[cfg(test)]
mod tests {
    use parquet::arrow::ArrowWriter;
    use tempfile::TempDir;

    use super::*;

    /// Rows in the straddling-batch regression fixture's first (full) batch.
    const STRADDLE_FIRST_BATCH_ROWS: usize = SQL_BATCH_ROWS;
    /// Rows in the fixture's second batch, which straddles `max_rows`.
    const STRADDLE_SECOND_BATCH_ROWS: usize = 300;
    /// Rows kept from the second batch — the rest is discarded as over quota.
    const STRADDLE_RETAINED_FROM_SECOND_BATCH: usize = 100;
    /// `max_rows` for the fixture: exactly the retained-row count.
    const STRADDLE_MAX_ROWS: usize =
        STRADDLE_FIRST_BATCH_ROWS + STRADDLE_RETAINED_FROM_SECOND_BATCH;

    #[test]
    fn binary_columns_become_large_utf8_in_the_derived_schema() {
        let schema = Schema::new(vec![
            Field::new("Title", DataType::Binary, true),
            Field::new("URL", DataType::LargeBinary, true),
            Field::new("UserID", DataType::Int64, false),
        ]);
        let out = cast_schema_for_sql(&schema);
        assert_eq!(out.field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(out.field(1).data_type(), &DataType::LargeUtf8);
        assert_eq!(
            out.field(2).data_type(),
            &DataType::Int64,
            "non-binary columns must pass through untouched"
        );
    }

    #[test]
    fn invalid_utf8_is_replaced_and_counted_once_per_row() {
        // Second value is invalid UTF-8 (0xff is never a valid lead byte).
        let array: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"ok".as_slice()),
            Some(&[0xffu8, 0xfe][..]),
            None,
        ]));
        let (converted, replaced) = binary_to_large_utf8(&array);
        assert_eq!(replaced, 1, "exactly one row had invalid bytes");
        let strings = converted
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("large utf8");
        assert_eq!(strings.value(0), "ok");
        assert!(strings.is_null(2), "nulls stay null");
    }

    #[test]
    fn spec_has_no_fts_columns_and_no_vector_without_an_embedding_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("Title", DataType::LargeUtf8, true),
            Field::new("UserID", DataType::Int64, false),
        ]));
        let spec = spec_from_schema(schema);
        assert!(
            spec.fts_columns.is_empty(),
            "schema-driven corpora index no FTS columns"
        );
        assert!(spec.vector.is_none());
    }

    #[test]
    fn lossy_rows_excludes_rows_discarded_by_the_max_rows_truncation() {
        // First batch fills SQL_BATCH_ROWS with valid ASCII. Second batch
        // straddles `max_rows`: its retained prefix is valid UTF-8, its
        // discarded tail is invalid UTF-8 that must never be counted.
        let mut values: Vec<Vec<u8>> = (0..STRADDLE_FIRST_BATCH_ROWS)
            .map(|i| format!("row{i}").into_bytes())
            .collect();
        for i in 0..STRADDLE_SECOND_BATCH_ROWS {
            let global = STRADDLE_FIRST_BATCH_ROWS + i;
            values.push(if i < STRADDLE_RETAINED_FROM_SECOND_BATCH {
                format!("row{global}").into_bytes()
            } else {
                vec![0xffu8, 0xfe]
            });
        }
        let array: ArrayRef = Arc::new(BinaryArray::from(
            values.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "junk",
            DataType::Binary,
            true,
        )]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).expect("batch");

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("shard-0.parquet");
        let file = File::create(&path).expect("create parquet shard");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("arrow writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");

        let source = CorpusSource::LocalParquet {
            dir: dir.path().to_path_buf(),
        };
        let corpus = open(&source, STRADDLE_MAX_ROWS);

        assert_eq!(
            corpus.n_rows(),
            STRADDLE_MAX_ROWS,
            "truncates exactly at max_rows"
        );
        assert_eq!(
            corpus.lossy_rows(),
            0,
            "invalid UTF-8 in the discarded tail must not be counted"
        );
    }
}
