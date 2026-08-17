// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-generic SQL driver.
//!
//! Builds one canonical 1-writer queryable artifact, optionally measures
//! an N-writer build-throughput row, and times SQL queries against the
//! canonical artifact. `run_sql_with_index` returns the artifact so
//! in-tree benches can run additional correctness/warm/cold checks before
//! calling `close`/`delete`.

use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
    time::{Duration, Instant},
};

use arrow_array::RecordBatch;

use super::{SchemaDrivenSqlEngine, SqlCorpusSpec, SqlEngine, SqlRow};
use crate::{
    cpu,
    markdown::fmt_count,
    rss::{PeakSampler, RssStats},
};

#[derive(Clone, Copy, Debug)]
pub struct SqlQuery {
    pub name: &'static str,
    pub sql: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct SqlRunConfig {
    pub iters: usize,
    pub parallel: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SqlBuildStat {
    pub writers: usize,
    pub wall: Duration,
    pub rss: RssStats,
    /// Measured on-CPU seconds of the build (all-thread schedstat delta),
    /// when sampled — prices the build compute instead of a NOT-METERED gap.
    pub cpu_s: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SqlQueryStats {
    pub name: &'static str,
    pub p50: Duration,
    pub rss: RssStats,
    pub rows: usize,
}

#[derive(Clone, Debug)]
pub struct EngineSqlResult {
    pub engine: &'static str,
    pub builds: Vec<SqlBuildStat>,
    pub queries: Vec<SqlQueryStats>,
}

pub fn run_sql<E: SqlEngine>(
    cfg: SqlRunConfig,
    rows: &[SqlRow<'_>],
    queries: &[SqlQuery],
) -> EngineSqlResult {
    let (result, mut index) = run_sql_with_index::<E>(cfg, rows, queries);
    E::close(&mut index);
    E::delete(index);
    result
}

/// Shared measurement skeleton: 1-writer ingest, optional N-writer probe,
/// then the query battery. Ingest is supplied as closures so the row path
/// and the schema-driven batch path share one copy of the timing logic.
fn measure_sql<E: SqlEngine>(
    cfg: SqlRunConfig,
    index: E::Index,
    n_rows: usize,
    ingest_1w: impl FnOnce(&mut E::Index),
    ingest_nw: impl FnOnce(usize),
    queries: &[SqlQuery],
) -> (EngineSqlResult, E::Index) {
    eprintln!(
        "[harness/sql] {}: building 1-writer table over {} rows...",
        E::name(),
        fmt_count(n_rows),
    );
    let mut index = index;
    let sampler = PeakSampler::start_default();
    let ((), wall, cpu_s) = cpu::timed(|| ingest_1w(&mut index));
    let rss = sampler.stop_stats();
    let mut builds = vec![SqlBuildStat {
        writers: 1,
        wall,
        rss,
        cpu_s,
    }];

    if cfg.parallel > 1 {
        eprintln!(
            "[harness/sql] {}: parallel build probe ({} writers)...",
            E::name(),
            cfg.parallel,
        );
        let sampler = PeakSampler::start_default();
        let ((), wall, cpu_s) = cpu::timed(|| ingest_nw(cfg.parallel));
        let rss = sampler.stop_stats();
        builds.push(SqlBuildStat {
            writers: cfg.parallel,
            wall,
            rss,
            cpu_s,
        });
    }

    // One battery-level progress line; per-query results land in the
    // report table, so per-query progress lines are just noise.
    if !queries.is_empty() {
        eprintln!(
            "[harness/sql] {}: warm query battery ({} queries × {} timed iters)...",
            E::name(),
            queries.len(),
            cfg.iters,
        );
    }
    let mut queries_out = Vec::with_capacity(queries.len());
    let mut unplannable = Vec::new();
    // A query a real dataset's schema can't satisfy (e.g. a type DataFusion
    // won't implicitly cast) must not take down every other query's timing
    // and the report that depends on it — catch, count, and keep going.
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    for q in queries {
        let sampler = PeakSampler::start_default();
        let warm = match panic::catch_unwind(AssertUnwindSafe(|| E::read(&index, q.sql))) {
            Ok(out) => out,
            Err(payload) => {
                sampler.stop_stats();
                unplannable.push((q.name, panic_payload_message(&payload)));
                continue;
            }
        };
        let mut samples = Vec::with_capacity(cfg.iters.max(1));
        for _ in 0..cfg.iters.max(1) {
            let t0 = Instant::now();
            let out = E::read(&index, q.sql);
            samples.push(t0.elapsed());
            std::hint::black_box(out);
        }
        let rss = sampler.stop_stats();
        queries_out.push(SqlQueryStats {
            name: q.name,
            p50: percentile_duration(&mut samples, 50),
            rss,
            rows: warm.rows,
        });
    }
    panic::set_hook(prev_hook);
    if !unplannable.is_empty() {
        eprintln!(
            "[harness/sql] {}: {} of {} queries could not be planned or executed \
             against this schema and were skipped:",
            E::name(),
            unplannable.len(),
            queries.len(),
        );
        for (name, message) in &unplannable {
            eprintln!("[harness/sql]   {name}: {message}");
        }
    }

    (
        EngineSqlResult {
            engine: E::name(),
            builds,
            queries: queries_out,
        },
        index,
    )
}

pub fn run_sql_with_index<E: SqlEngine>(
    cfg: SqlRunConfig,
    rows: &[SqlRow<'_>],
    queries: &[SqlQuery],
) -> (EngineSqlResult, E::Index) {
    measure_sql::<E>(
        cfg,
        E::open(),
        rows.len(),
        |index| E::write(index, rows),
        |writers| E::parallel_write(rows, writers),
        queries,
    )
}

/// Schema-driven counterpart to [`run_sql_with_index`]: same measurement
/// skeleton, batches instead of the fixed row fixture.
pub fn run_sql_batches_with_index<E: SchemaDrivenSqlEngine>(
    cfg: SqlRunConfig,
    spec: &SqlCorpusSpec,
    batches: &[RecordBatch],
    queries: &[SqlQuery],
) -> (EngineSqlResult, E::Index) {
    let n_rows = batches.iter().map(RecordBatch::num_rows).sum();
    measure_sql::<E>(
        cfg,
        E::create_with_spec(spec),
        n_rows,
        |index| E::write_batches(index, batches),
        |writers| E::parallel_write_batches(spec, batches, writers),
        queries,
    )
}

/// Best-effort text for a caught panic payload — `catch_unwind` only
/// guarantees `Any`, and the two payload shapes `panic!`/`.expect()` use
/// cover it in practice.
fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

fn percentile_duration(samples: &mut [Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    let rank = ((percentile as f64 / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::harness::{
        Capabilities, SchemaDrivenSqlEngine, SqlCorpusSpec, SqlEngine, SqlOutput, SqlRow,
    };

    struct StubEngine;
    #[derive(Default)]
    struct StubIndex {
        rows_written: usize,
    }

    impl SqlEngine for StubEngine {
        type Index = StubIndex;
        fn name() -> &'static str {
            "stub"
        }
        fn capabilities() -> Capabilities {
            Capabilities {
                sql: true,
                ..Default::default()
            }
        }
        fn create() -> Self::Index {
            StubIndex::default()
        }
        fn write(index: &mut Self::Index, rows: &[SqlRow<'_>]) {
            index.rows_written = rows.len();
        }
        fn parallel_write(_rows: &[SqlRow<'_>], _writers: usize) {}
        fn read(_index: &Self::Index, _sql: &str) -> SqlOutput {
            SqlOutput { rows: 7 }
        }
        fn close(_index: &mut Self::Index) {}
        fn delete(_index: Self::Index) {}
    }

    impl SchemaDrivenSqlEngine for StubEngine {
        fn create_with_spec(_spec: &SqlCorpusSpec) -> Self::Index {
            StubIndex::default()
        }
        fn write_batches(index: &mut Self::Index, batches: &[RecordBatch]) {
            index.rows_written = batches.iter().map(RecordBatch::num_rows).sum();
        }
        fn parallel_write_batches(
            _spec: &SqlCorpusSpec,
            _batches: &[RecordBatch],
            _writers: usize,
        ) {
        }
    }

    fn stub_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let col = Int64Array::from((0..rows as i64).collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(col) as ArrayRef]).expect("stub batch")
    }

    /// The batch path reports the summed row count of its batches and
    /// reuses the same skeleton as the row path.
    #[test]
    fn batch_path_counts_rows_across_batches() {
        let batches = [stub_batch(3), stub_batch(5)];
        let spec = SqlCorpusSpec {
            schema: batches[0].schema(),
            fts_columns: Vec::new(),
            vector: None,
        };
        let (result, index) = run_sql_batches_with_index::<StubEngine>(
            SqlRunConfig {
                iters: 1,
                parallel: 1,
            },
            &spec,
            &batches,
            &[],
        );
        assert_eq!(index.rows_written, 8);
        assert_eq!(result.builds.len(), 1);
    }

    /// Compile-time proof that the public row-path signature is unchanged:
    /// this is exactly how an out-of-repo engine calls it.
    #[test]
    fn run_sql_with_index_keeps_its_row_signature() {
        let rows = [SqlRow {
            doc_id: 0,
            title: "a",
            category: "c",
            score: 1,
        }];
        let queries = [SqlQuery {
            name: "q",
            sql: "SELECT 1",
        }];
        let (result, index) = run_sql_with_index::<StubEngine>(
            SqlRunConfig {
                iters: 1,
                parallel: 1,
            },
            &rows,
            &queries,
        );
        assert_eq!(index.rows_written, 1);
        assert_eq!(result.engine, "stub");
        assert_eq!(
            result.builds.len(),
            1,
            "parallel=1 must not add a build row"
        );
        assert_eq!(result.queries.len(), 1);
        assert_eq!(result.queries[0].rows, 7);
    }

    struct FlakyEngine;

    impl SqlEngine for FlakyEngine {
        type Index = StubIndex;
        fn name() -> &'static str {
            "flaky"
        }
        fn capabilities() -> Capabilities {
            Capabilities {
                sql: true,
                ..Default::default()
            }
        }
        fn create() -> Self::Index {
            StubIndex::default()
        }
        fn write(_index: &mut Self::Index, _rows: &[SqlRow<'_>]) {}
        fn parallel_write(_rows: &[SqlRow<'_>], _writers: usize) {}
        fn read(_index: &Self::Index, sql: &str) -> SqlOutput {
            assert_ne!(
                sql, "BAD",
                "a query DataFusion can't plan panics, like a real engine"
            );
            SqlOutput { rows: 1 }
        }
        fn close(_index: &mut Self::Index) {}
        fn delete(_index: Self::Index) {}
    }

    /// One query in the battery panicking (the real-world case: DataFusion
    /// can't plan it against the dataset's schema) must not take down the
    /// rest of the battery or lose the report the caller builds from it.
    #[test]
    fn one_unplannable_query_is_skipped_not_fatal() {
        let queries = [
            SqlQuery {
                name: "ok",
                sql: "GOOD",
            },
            SqlQuery {
                name: "unplannable",
                sql: "BAD",
            },
        ];
        let (result, _index) = run_sql_with_index::<FlakyEngine>(
            SqlRunConfig {
                iters: 1,
                parallel: 1,
            },
            &[],
            &queries,
        );
        assert_eq!(
            result.queries.len(),
            1,
            "the unplannable query is dropped, not fabricated a timing"
        );
        assert_eq!(result.queries[0].name, "ok");
    }

    /// The N-writer probe runs only above parallel=1, and receives the
    /// configured writer count.
    #[test]
    fn parallel_probe_runs_only_above_one_writer() {
        let rows = [SqlRow {
            doc_id: 0,
            title: "a",
            category: "c",
            score: 1,
        }];
        let (result, _index) = run_sql_with_index::<StubEngine>(
            SqlRunConfig {
                iters: 1,
                parallel: 4,
            },
            &rows,
            &[],
        );
        assert_eq!(result.builds.len(), 2);
        assert_eq!(result.builds[1].writers, 4);
    }
}
