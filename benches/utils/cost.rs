// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cost model for the bench — turns measured latency, footprint, and
//! object-store request counts into dollars, per the rule "a resource
//! costs money only to the extent that holding it blocks the next
//! tenant."
//!
//! Four blocks, kept separate:
//!
//!   1. **Rate card** — the headline dollars, every figure in one of two
//!      units: **$/1M docs** (write path; storage over the stated
//!      retention) and **$/1M queries** (serving; per-query costs are
//!      sub-cent, so a per-query dollar figure would round to $0 and hide
//!      the real number). RAM appears as an instance-sizing fact, not a
//!      dollar line.
//!   2. **Object-store I/O ledger** — measured HEAD/GET/PUT counts and
//!      byte volumes per lifecycle phase, with per-unit normalization
//!      (PUT/commit, GET/query). Counts come from the
//!      [`crate::storage_meter`] wrapper; phases that did not run metered
//!      are omitted, never guessed.
//!   3. **Compute ledger** — one-time phases (ingest/drain/compaction)
//!      priced from measured on-CPU seconds (wall × vCPU-share fallback);
//!      per-query phases from p50 latency. One-time phases in absolute
//!      dollars, per-query phases per 1M queries.
//!   4. **Serving** — latency per dollar; cold rows include request cost.
//!
//! Local NVMe (file-backed disk-cache mmap) is treated as free.

use std::{collections::HashMap, sync::OnceLock};

use crate::{
    executors::{ColdTiming, fts::FtsQueryStat, sql::QuerySets, vector::RecallRow},
    markdown::{fmt_count, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
    rss::fmt_bytes,
    storage_meter::ObjectStoreMeter,
};

/// S3 Standard capacity, USD per GB-month (decimal GB).
const USD_PER_GB_MONTH: f64 = 0.023;
/// USD per PUT request ($5 per 1M).
const USD_PER_PUT: f64 = 5.0e-6;
/// USD per GET or HEAD request ($0.40 per 1M).
const USD_PER_GET: f64 = 4.0e-7;

/// Default assumed retention when turning stored bytes into GB-months.
const DEFAULT_STORAGE_MONTHS: f64 = 1.0;

/// Bytes per GiB (RAM is reasoned about in GiB).
const BYTES_PER_GIB: f64 = (1u64 << 30) as f64;
/// Bytes per GB (object storage is priced per decimal GB).
const BYTES_PER_GB: f64 = 1.0e9;
/// Seconds per hour.
const SECS_PER_HOUR: f64 = 3600.0;
/// Queries per "per-million" pricing unit.
const PER_MILLION: f64 = 1.0e6;

/// The instance the model prices against. Default is a portable cloud SKU
/// with local NVMe; override via `INFINO_BENCH_COST_*` env vars.
#[derive(Clone, Debug)]
pub struct Instance {
    pub name: String,
    pub vcpu: u32,
    pub ram_gib: f64,
    pub nvme_gb: f64,
    pub usd_per_hour: f64,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            name: "c7gd.2xlarge".into(),
            vcpu: 8,
            ram_gib: 16.0,
            nvme_gb: 237.0,
            usd_per_hour: 0.3629,
        }
    }
}

impl Instance {
    pub fn current() -> &'static Instance {
        static INSTANCE: OnceLock<Instance> = OnceLock::new();
        INSTANCE.get_or_init(Instance::from_env)
    }

    fn from_env() -> Self {
        let d = Instance::default();
        let s = |k: &str, v: String| std::env::var(k).unwrap_or(v);
        let f = |k: &str, v: f64| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(v)
        };
        let u = |k: &str, v: u32| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(v)
        };
        Instance {
            name: s("INFINO_BENCH_COST_INSTANCE", d.name),
            vcpu: u("INFINO_BENCH_COST_VCPU", d.vcpu),
            ram_gib: f("INFINO_BENCH_COST_RAM_GIB", d.ram_gib),
            nvme_gb: f("INFINO_BENCH_COST_NVME_GB", d.nvme_gb),
            usd_per_hour: f("INFINO_BENCH_COST_USD_PER_HOUR", d.usd_per_hour),
        }
    }

    fn usd_per_sec(&self) -> f64 {
        self.usd_per_hour / SECS_PER_HOUR
    }

    /// Fraction of the instance's RAM a resident set occupies.
    fn ram_share(&self, resident_bytes: u64) -> f64 {
        resident_bytes as f64 / BYTES_PER_GIB / self.ram_gib
    }

    fn parallel_vcpu_seconds(&self, wall_s: f64, writers: u32) -> f64 {
        let cpu_share = f64::from(writers.min(self.vcpu)) / f64::from(self.vcpu.max(1));
        wall_s * cpu_share
    }

    /// Binding vCPU·s for a one-time parallel phase, preferring measured
    /// on-CPU time. The CPU leg is the measured `cpu_s` when present
    /// (schedstat excludes I/O wait — the correct compute basis), else the
    /// modeled `wall × pool-share`. The RAM leg is `wall × peak-RSS share`
    /// (RAM is held for the whole wall). The phase bills on whichever binds.
    fn phase_vcpu_seconds(
        &self,
        wall_s: f64,
        writers: u32,
        peak_rss_bytes: Option<u64>,
        cpu_s: Option<f64>,
    ) -> f64 {
        let cpu_leg = cpu_s.unwrap_or_else(|| self.parallel_vcpu_seconds(wall_s, writers));
        let ram_leg = wall_s * peak_rss_bytes.map(|b| self.ram_share(b)).unwrap_or(0.0);
        cpu_leg.max(ram_leg)
    }

    /// Whether the measured/modeled CPU leg (not RAM) binds a one-time phase.
    fn phase_cpu_binds(
        &self,
        wall_s: f64,
        writers: u32,
        peak_rss_bytes: Option<u64>,
        cpu_s: Option<f64>,
    ) -> bool {
        let cpu_leg = cpu_s.unwrap_or_else(|| self.parallel_vcpu_seconds(wall_s, writers));
        let ram_leg = wall_s * peak_rss_bytes.map(|b| self.ram_share(b)).unwrap_or(0.0);
        cpu_leg >= ram_leg
    }

    fn per_query_usd(&self, p50_s: f64, resident_anon_bytes: u64) -> f64 {
        self.per_query_vcpu_seconds(p50_s, resident_anon_bytes) * self.usd_per_sec()
    }

    fn per_query_vcpu_seconds(&self, p50_s: f64, resident_anon_bytes: u64) -> f64 {
        let cpu_share = 1.0 / f64::from(self.vcpu.max(1));
        let ram_share = resident_anon_bytes as f64 / BYTES_PER_GIB / self.ram_gib;
        p50_s * cpu_share.max(ram_share)
    }

    fn ram_binds(&self, resident_anon_bytes: u64) -> bool {
        let cpu_share = 1.0 / f64::from(self.vcpu.max(1));
        let ram_share = resident_anon_bytes as f64 / BYTES_PER_GIB / self.ram_gib;
        ram_share > cpu_share
    }
}

/// Cold open + search latency for one query shape.
pub struct ColdQuery {
    pub name: String,
    pub open_s: f64,
    pub search_s: f64,
}

/// Metered object-store I/O for the lifecycle phases of one bench cell.
/// Every field is optional: a phase that wasn't metered is reported as
/// such — the model never substitutes an estimate for a measurement.
#[derive(Default, Clone, Copy)]
pub struct StorePhases {
    /// The ingest window (all commits): superfile uploads (multipart
    /// parts included), manifest parts/lists, pointer CAS writes.
    pub ingest: Option<ObjectStoreMeter>,
    /// The hidden vector-index drain: reads user vector blobs, writes
    /// per-cell superfiles + routing/manifest updates.
    pub drain: Option<ObjectStoreMeter>,
    /// Wall-clock seconds of the drain window, when it ran.
    pub drain_wall_s: Option<f64>,
    /// Measured on-CPU seconds (all-thread schedstat delta) over the drain
    /// window. `Some` ⇒ price compute from this instead of `wall × share`;
    /// `None` ⇒ fall back to the wall-clock model.
    pub drain_cpu_s: Option<f64>,
    /// Peak RSS sampled over the drain window — the drain is billed at
    /// `max(pool CPU share, peak-RSS share)` for its wall duration.
    pub drain_peak_rss_bytes: Option<u64>,
    /// Maintenance compaction (`optimize()`: user + hidden tables) —
    /// reads the small superfiles, writes merged replacements.
    pub compaction: Option<ObjectStoreMeter>,
    /// Wall-clock seconds of the compaction window, when it ran.
    pub compaction_wall_s: Option<f64>,
    /// Measured on-CPU seconds over the compaction window (same semantics as
    /// [`Self::drain_cpu_s`]).
    pub compaction_cpu_s: Option<f64>,
    /// Peak RSS sampled over the compaction window (same billing rule).
    pub compaction_peak_rss_bytes: Option<u64>,
    /// One cold table open on a fresh cache (manifest + pointer + open
    /// blobs) — one-time, amortized across queries on a supertable.
    pub cold_open: Option<ObjectStoreMeter>,
    /// The first query on the cold cache — the per-query cold fetch.
    /// This is the "GETs per query" number.
    pub cold_query: Option<ObjectStoreMeter>,
    /// Pre-drain counterparts of `cold_open` / `cold_query`: the transient
    /// shape a fresh table serves (hidden IVF still in INCOMING) until
    /// maintenance drains it. Priced so the cost of querying *before*
    /// maintenance catches up is visible next to the steady state.
    pub cold_open_pre: Option<ObjectStoreMeter>,
    pub cold_query_pre: Option<ObjectStoreMeter>,
    /// The same query repeated on the same *fresh* consumer. Probes
    /// cache fill lag: if the disk cache absorbed the first query this
    /// is ~0 GETs; a repeat of the full fan means foreground reads are
    /// not retained (or background fill has not landed yet).
    pub cold_repeat_query: Option<ObjectStoreMeter>,
    /// Steady-state warm window: [`Self::warm_query_iters`] queries on
    /// the shared, cache-hot consumer — the same consumer the warm
    /// latency battery timed, so I/O and CPU describe the same path.
    pub warm_query: Option<ObjectStoreMeter>,
    pub warm_query_iters: u64,
    /// Filtered-search window ([`Self::filtered_query_iters`] queries)
    /// on the same shared consumer — filtered vs unfiltered GET/query.
    pub filtered_query: Option<ObjectStoreMeter>,
    pub filtered_query_iters: u64,
}

/// Everything one cell (one tier × modality) needs to be priced.
pub struct CellCost<'a> {
    pub ingest_wall_s: f64,
    pub writers: u32,
    /// Peak RSS during the ingest window, when sampled. Ingest is billed
    /// on the *binding* resource — `max(writer-pool CPU share, peak-RSS
    /// share of RAM)` — same rule queries use; `None` bills CPU share.
    pub ingest_peak_rss_bytes: Option<u64>,
    /// Measured on-CPU seconds over the ingest window. `Some` ⇒ price the
    /// CPU leg from this instead of `wall × pool-share`; `None` ⇒ wall model.
    pub ingest_cpu_s: Option<f64>,
    /// Commits in the ingest window, for PUT-per-commit normalization.
    pub n_commits: u64,
    /// Exact PUT count for write paths that are known without metering
    /// (the superfile tier's single `put_atomic`). `None` + no metered
    /// ingest ⇒ the write-request line reports "not metered".
    pub unmetered_put_count: Option<u64>,
    pub stored_bytes: u64,
    pub corpus_bytes: u64,
    pub n_docs: usize,
    pub resident_anon_bytes: u64,
    /// Steady-state (post-drain, on a vector cell) warm latency battery.
    pub warm: &'a [(String, f64)],
    /// Cold latency rows (open and search timed separately), steady state.
    pub cold: Option<&'a [ColdQuery]>,
    /// Pre-drain warm battery — the transient shape before maintenance.
    pub warm_pre: Option<&'a [(String, f64)]>,
    /// Pre-drain cold latency rows.
    pub cold_pre: Option<&'a [ColdQuery]>,
    /// Measured object-store I/O per phase.
    pub store: StorePhases,
    /// Whether this cell has the vector maintenance lifecycle (drain,
    /// compaction, filtered search, pre/post-drain split). Those ledger
    /// rows always render on such a cell — as "NOT METERED" when the
    /// harness failed to measure them — and never render elsewhere
    /// (an FTS cell has no drain to meter).
    pub vector_cell: bool,
    /// Assumed retention for the capacity line (GB-months). Default 1 month.
    pub storage_months: Option<f64>,
    /// Whether a cold `open` is a one-time table/namespace open that is
    /// amortized across every query (supertable: manifest load + consumer
    /// setup, paid once), rather than per-query latency. For a single
    /// superfile the open is part of each cold read, so this is `false`.
    pub cold_open_amortized: bool,
}

/// `$X` with adaptive precision: two decimals at or above one cent,
/// otherwise two significant digits — sub-cent values never collapse to
/// a meaningless "$0.0000".
fn usd(v: f64) -> String {
    if v == 0.0 {
        return "$0".into();
    }
    if v >= 0.01 {
        return format!("${v:.2}");
    }
    let decimals = ((-v.log10()).ceil() as usize + 1).min(9);
    format!("${v:.decimals$}")
}

/// Per-query dollars expressed at the meaningful scale: `$X/1M`.
fn usd_per_million(per_unit: f64) -> String {
    format!("{}/1M", usd(per_unit * PER_MILLION))
}

fn usd_per_gb(v: f64) -> String {
    if v < 0.01 {
        format!("${v:.4}/GB")
    } else {
        format!("${v:.2}/GB")
    }
}

fn storage_months() -> f64 {
    static MONTHS: OnceLock<f64> = OnceLock::new();
    *MONTHS.get_or_init(|| {
        std::env::var("INFINO_BENCH_COST_STORAGE_MONTHS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(DEFAULT_STORAGE_MONTHS)
    })
}

fn fmt_vcpu_seconds(s: f64) -> String {
    if s >= 10.0 {
        format!("{s:.1} vCPU·s")
    } else {
        format!("{s:.2} vCPU·s")
    }
}

fn fmt_wall_seconds(s: f64) -> String {
    if s >= 10.0 {
        format!("{s:.1}s wall")
    } else {
        format!("{s:.2}s wall")
    }
}

/// Request dollars for one metered window (PUTs at the PUT rate,
/// HEAD + GET at the GET rate).
fn request_usd(io: &ObjectStoreMeter) -> f64 {
    io.put_count as f64 * USD_PER_PUT + io.read_requests() as f64 * USD_PER_GET
}

/// "N PUT + M GET (+ K HEAD)" — the request-count cell of an I/O row.
fn fmt_requests(io: &ObjectStoreMeter) -> String {
    let mut parts = Vec::new();
    if io.put_count > 0 {
        parts.push(format!("{} PUT", io.put_count));
    }
    if io.get_count > 0 {
        parts.push(format!("{} GET", io.get_count));
    }
    if io.head_count > 0 {
        parts.push(format!("{} HEAD", io.head_count));
    }
    if parts.is_empty() {
        "0".into()
    } else {
        parts.join(" + ")
    }
}

/// "X up · Y down" byte-volume cell (only the directions that moved).
fn fmt_io_bytes(io: &ObjectStoreMeter) -> String {
    let mut parts = Vec::new();
    if io.put_bytes > 0 {
        parts.push(format!("{} up", fmt_bytes(io.put_bytes)));
    }
    if io.get_bytes > 0 {
        parts.push(format!("{} down", fmt_bytes(io.get_bytes)));
    }
    if parts.is_empty() {
        "—".into()
    } else {
        parts.join(" · ")
    }
}

pub fn emit(report: &mut Report, anchor: &str, title: String, c: &CellCost) {
    let inst = Instance::current();
    let retention_months = c.storage_months.unwrap_or_else(storage_months);

    // ---- Write path: ingest + drain + compaction (compute and requests).
    // Each phase is billed at its binding share — pool CPU or peak-RSS
    // share of RAM, whichever is larger — for its full wall duration.
    // Compute is priced from measured on-CPU seconds when the harness
    // captured them (schedstat, I/O wait excluded); otherwise it falls back
    // to the `wall × pool-share` model. vCPU·s and $ share one basis so the
    // ledger's two columns can't disagree.
    let ingest_vcpu_s = inst.phase_vcpu_seconds(
        c.ingest_wall_s,
        c.writers,
        c.ingest_peak_rss_bytes,
        c.ingest_cpu_s,
    );
    let ingest_compute = ingest_vcpu_s * inst.usd_per_sec();
    let drain_wall_s = c.store.drain_wall_s.unwrap_or(0.0);
    let drain_vcpu_s = inst.phase_vcpu_seconds(
        drain_wall_s,
        c.writers,
        c.store.drain_peak_rss_bytes,
        c.store.drain_cpu_s,
    );
    let drain_compute = drain_vcpu_s * inst.usd_per_sec();

    let compaction_wall_s = c.store.compaction_wall_s.unwrap_or(0.0);
    let compaction_vcpu_s = inst.phase_vcpu_seconds(
        compaction_wall_s,
        c.writers,
        c.store.compaction_peak_rss_bytes,
        c.store.compaction_cpu_s,
    );
    let compaction_compute = compaction_vcpu_s * inst.usd_per_sec();

    let ingest_req_usd = match (c.store.ingest, c.unmetered_put_count) {
        (Some(io), _) => request_usd(&io),
        (None, Some(puts)) => puts as f64 * USD_PER_PUT,
        (None, None) => 0.0,
    };
    let drain_req_usd = c.store.drain.map(|io| request_usd(&io)).unwrap_or(0.0);
    let compaction_req_usd = c.store.compaction.map(|io| request_usd(&io)).unwrap_or(0.0);

    let write_compute = ingest_compute + drain_compute + compaction_compute;
    let write_requests = ingest_req_usd + drain_req_usd + compaction_req_usd;
    let write_total = write_compute + write_requests;
    let write_per_million_docs = if c.n_docs > 0 {
        write_total / (c.n_docs as f64 / PER_MILLION)
    } else {
        0.0
    };
    // "$X per 1M docs" for a one-time maintenance phase's requests.
    let per_million_docs = |usd_total: f64| {
        if c.n_docs > 0 {
            usd_total / (c.n_docs as f64 / PER_MILLION)
        } else {
            0.0
        }
    };

    // ---- Storage capacity ----
    let stored_gb = c.stored_bytes as f64 / BYTES_PER_GB;
    let gb_months = stored_gb * retention_months;
    let storage_month = gb_months * USD_PER_GB_MONTH;

    // ---- Warm query battery (CPU-priced) ----
    let warm_costs: Vec<(f64, f64, String)> = c
        .warm
        .iter()
        .map(|(name, p50_s)| {
            let per_q = inst.per_query_usd(*p50_s, c.resident_anon_bytes);
            (per_q, *p50_s, name.clone())
        })
        .collect();
    let (min_q_cost, max_q_cost, fastest_name, fastest_p50) = if warm_costs.is_empty() {
        (0.0, 0.0, String::new(), 0.0)
    } else {
        warm_costs.iter().fold(
            (f64::INFINITY, 0.0_f64, String::new(), f64::INFINITY),
            |(min_c, max_c, fast_name, fast_p50), (cost, p50, name)| {
                let (fast_name, fast_p50) = if *p50 < fast_p50 {
                    (name.clone(), *p50)
                } else {
                    (fast_name, fast_p50)
                };
                (min_c.min(*cost), max_c.max(*cost), fast_name, fast_p50)
            },
        )
    };

    // Anchor cold row: the shape whose open/search latency and metered I/O
    // represent "one cold query" in the rate card and ledgers.
    let anchor_cold = c.cold.and_then(|rows| {
        rows.iter()
            .find(|q| q.name == "ten_term_or")
            .or_else(|| rows.first())
    });

    // Per-query cold dollars = marginal CPU for the search + measured
    // object-store requests for the first-query fetch window.
    let cold_query_req_usd = c.store.cold_query.map(|io| request_usd(&io));
    let cold_query_usd = anchor_cold.map(|q| {
        inst.per_query_usd(q.search_s, c.resident_anon_bytes) + cold_query_req_usd.unwrap_or(0.0)
    });

    // ---- Block 1: rate card ----
    let warm_query_cell = if warm_costs.is_empty() {
        "—".into()
    } else if (max_q_cost - min_q_cost).abs() < f64::EPSILON {
        format!(
            "{} queries @ {} p50 ({})",
            usd_per_million(min_q_cost),
            fmt_time(fastest_p50 * 1e9),
            fastest_name,
        )
    } else {
        format!(
            "{}–{} queries ({}–{} p50 battery)",
            usd(min_q_cost * PER_MILLION),
            usd_per_million(max_q_cost),
            fmt_time(fastest_p50 * 1e9),
            fmt_time(
                warm_costs
                    .iter()
                    .map(|(_, p50, _)| *p50)
                    .fold(0.0_f64, f64::max)
                    * 1e9,
            ),
        )
    };

    let has_drain = c.store.drain.is_some() || c.store.drain_wall_s.is_some();
    let has_compaction = c.store.compaction.is_some() || c.store.compaction_wall_s.is_some();
    let write_label = match (has_drain, has_compaction) {
        (true, true) => "Write path (ingest + drain + compaction)",
        (true, false) => "Write path (ingest + hidden-index drain)",
        (false, true) => "Write path (ingest + compaction)",
        (false, false) => "Write path (ingest)",
    };
    let mut rate_rows = vec![
        vec![
            text("Storage"),
            text(format!(
                "{}/1M docs ({} × {retention_months:.0} mo retention)",
                usd(per_million_docs(storage_month)),
                usd_per_gb(USD_PER_GB_MONTH),
            )),
        ],
        vec![
            text(write_label),
            text(format!(
                "{} compute + {} requests → {} total ({}/1M docs)",
                usd(write_compute),
                usd(write_requests),
                usd(write_total),
                usd(write_per_million_docs),
            )),
        ],
        vec![
            text("Serving RAM (instance sizing)"),
            text(format!(
                "{} resident = {:.0}% of {:.0} GiB RAM — sizing fact, not a dollar line: \
                 RAM held during a query is already inside $/1M queries via the \
                 binding-resource rule",
                fmt_bytes(c.resident_anon_bytes),
                inst.ram_share(c.resident_anon_bytes) * 100.0,
                inst.ram_gib,
            )),
        ],
        vec![
            text("Warm query (marginal, binding resource)"),
            text(warm_query_cell),
        ],
    ];

    if let Some(q) = anchor_cold {
        if let Some(per_q) = cold_query_usd.filter(|_| cold_query_req_usd.is_some()) {
            let io = c.store.cold_query.expect("guarded by cold_query_req_usd");
            rate_rows.push(vec![
                text("Cold query (CPU + requests)"),
                text(format!(
                    "{} queries — {} GET/query, {}/query fetched ({} search, {})",
                    usd_per_million(per_q),
                    io.get_count,
                    fmt_bytes(io.get_bytes),
                    fmt_time(q.search_s * 1e9),
                    q.name,
                )),
            ]);
        } else {
            rate_rows.push(vec![
                text("Cold query (latency only — requests not metered)"),
                text(format!(
                    "{} open + {} search ({})",
                    fmt_time(q.open_s * 1e9),
                    fmt_time(q.search_s * 1e9),
                    q.name,
                )),
            ]);
        }
        if c.cold_open_amortized {
            let open_io = c
                .store
                .cold_open
                .map(|io| {
                    format!(
                        " · {} GET, {} fetched",
                        io.read_requests(),
                        fmt_bytes(io.get_bytes)
                    )
                })
                .unwrap_or_default();
            rate_rows.push(vec![
                text("Table open (one-time, amortized)"),
                text(format!(
                    "{}{open_io} — manifest + consumer, paid once per open",
                    fmt_time(q.open_s * 1e9),
                )),
            ]);
        }
    }

    let rate_card = Block {
        subtitle: format!(
            "Rate card — {} docs, {} stored (Infino measured; latency lives in the \
             search table — warm vs cold are not interchangeable)",
            fmt_count(c.n_docs),
            fmt_bytes(c.stored_bytes),
        ),
        headers: vec!["Line".into(), "Infino (measured)".into()],
        rows: rate_rows,
    };

    // ---- Block 2: object-store I/O ledger ----
    let mut io_rows: Vec<Vec<Cell>> = Vec::new();
    // A lifecycle phase this cell *has* but the harness failed to measure
    // renders as a loud placeholder — a phase must never silently vanish.
    let not_metered_row = |label: &str| -> Vec<Cell> {
        vec![
            text(label),
            text("NOT METERED"),
            text("—"),
            text("—"),
            text("—"),
        ]
    };
    match (c.store.ingest, c.unmetered_put_count) {
        (Some(io), _) => {
            io_rows.push(vec![
                text(format!("Ingest ({} commits)", c.n_commits)),
                text(fmt_requests(&io)),
                text(fmt_io_bytes(&io)),
                text(format!(
                    "{}/1M docs",
                    usd(per_million_docs(request_usd(&io)))
                )),
                metric(request_usd(&io), usd(request_usd(&io)), Better::Lower),
            ]);
        }
        (None, Some(puts)) => {
            let req = puts as f64 * USD_PER_PUT;
            io_rows.push(vec![
                text(format!("Ingest ({} commits)", c.n_commits)),
                text(format!("{puts} PUT (exact, unmetered)")),
                text(format!("{} up", fmt_bytes(c.stored_bytes))),
                text(format!("{}/1M docs", usd(per_million_docs(req)))),
                metric(req, usd(req), Better::Lower),
            ]);
        }
        (None, None) => io_rows.push(not_metered_row("Ingest (opened pre-built)")),
    }
    let one_time_row =
        |rows: &mut Vec<Vec<Cell>>, label: &str, io: Option<ObjectStoreMeter>, per_unit: &str| {
            match io {
                Some(io) => {
                    let per_unit = if per_unit.is_empty() {
                        format!("{}/1M docs", usd(per_million_docs(request_usd(&io))))
                    } else {
                        per_unit.to_string()
                    };
                    rows.push(vec![
                        text(label),
                        text(fmt_requests(&io)),
                        text(fmt_io_bytes(&io)),
                        text(per_unit),
                        metric(request_usd(&io), usd(request_usd(&io)), Better::Lower),
                    ]);
                }
                None if c.vector_cell => rows.push(not_metered_row(label)),
                None => {}
            }
        };
    let per_query_row =
        |rows: &mut Vec<Vec<Cell>>, label: &str, io: Option<ObjectStoreMeter>| match io {
            Some(io) => {
                let per_million = request_usd(&io) * PER_MILLION;
                rows.push(vec![
                    text(label),
                    text(fmt_requests(&io)),
                    text(fmt_io_bytes(&io)),
                    metric(
                        io.get_count as f64,
                        format!("{} GET/query", io.get_count),
                        Better::Lower,
                    ),
                    metric(
                        per_million,
                        format!("{}/1M queries", usd(per_million)),
                        Better::Lower,
                    ),
                ]);
            }
            None if c.vector_cell => rows.push(not_metered_row(label)),
            None => {}
        };
    one_time_row(
        &mut io_rows,
        "Drain → hidden index (one-time)",
        c.store.drain,
        "",
    );
    one_time_row(
        &mut io_rows,
        "Compaction (optimize, one-time)",
        c.store.compaction,
        "",
    );
    if c.vector_cell {
        one_time_row(
            &mut io_rows,
            "Cold table open (pre-drain)",
            c.store.cold_open_pre,
            "transient — before maintenance",
        );
        per_query_row(
            &mut io_rows,
            "Cold query (pre-drain, transient)",
            c.store.cold_query_pre,
        );
    }
    one_time_row(
        &mut io_rows,
        "Cold table open",
        c.store.cold_open,
        "once per open, amortized",
    );
    per_query_row(
        &mut io_rows,
        "Cold query (first on cold cache)",
        c.store.cold_query,
    );
    if let Some(io) = c.store.cold_repeat_query {
        // Diagnostic, not a steady-state price: a repeat of the full GET
        // fan here means the first query's foreground reads were not
        // retained by the cache (fill lag); ~0 means the cache absorbed it.
        io_rows.push(vec![
            text("Repeat query on cold consumer (fill-lag probe)"),
            text(fmt_requests(&io)),
            text(fmt_io_bytes(&io)),
            metric(
                io.get_count as f64,
                format!("{} GET/query", io.get_count),
                Better::Lower,
            ),
            text("diagnostic — not steady-state"),
        ]);
    } else if c.vector_cell {
        io_rows.push(not_metered_row(
            "Repeat query on cold consumer (fill-lag probe)",
        ));
    }
    // Averaged multi-query windows on the shared cache-hot consumer: the
    // same consumer the warm latency battery timed, so the ledger's warm
    // I/O and the compute ledger's warm CPU describe one path.
    let averaged_row =
        |rows: &mut Vec<Vec<Cell>>, label: &str, io: Option<ObjectStoreMeter>, iters: u64| match io
        {
            Some(io) => {
                let iters = iters.max(1);
                let per_query_get = io.get_count as f64 / iters as f64;
                let per_query_usd = request_usd(&io) / iters as f64;
                let per_million = per_query_usd * PER_MILLION;
                rows.push(vec![
                    text(label),
                    text(format!("{} over {iters} queries", fmt_requests(&io))),
                    text(fmt_io_bytes(&io)),
                    metric(
                        per_query_get,
                        format!("{per_query_get:.1} GET/query"),
                        Better::Lower,
                    ),
                    metric(
                        per_million,
                        format!("{}/1M queries", usd(per_million)),
                        Better::Lower,
                    ),
                ]);
            }
            None if c.vector_cell => rows.push(not_metered_row(label)),
            None => {}
        };
    averaged_row(
        &mut io_rows,
        "Warm query (shared consumer, cache hot)",
        c.store.warm_query,
        c.store.warm_query_iters,
    );
    averaged_row(
        &mut io_rows,
        "Filtered query (warm, ~10% selectivity)",
        c.store.filtered_query,
        c.store.filtered_query_iters,
    );
    let io_ledger = (!io_rows.is_empty()).then(|| Block {
        subtitle: "Object-store I/O — measured requests + bytes per phase (PUT $5/1M; \
                   GET + HEAD $0.40/1M). A phase this cell has but the harness failed to \
                   measure says NOT METERED; phases a cell does not have are omitted."
            .into(),
        headers: vec![
            "Phase".into(),
            "Requests".into(),
            "Bytes".into(),
            "Per-unit".into(),
            "Request $".into(),
        ],
        rows: io_rows,
    });

    // ---- Block 3: compute ledger ----
    let writers_used = c.writers.min(inst.vcpu);
    let vcpu_share = format!("{writers_used}/{}/vCPU share", inst.vcpu);
    // A parallel phase's label carries its binding resource: peak RSS when
    // the phase's RAM share exceeds the (measured or modeled) CPU leg, else
    // CPU.
    let phase_binding_tag = |wall_s: f64, peak_rss: Option<u64>, cpu_s: Option<f64>| -> String {
        match peak_rss {
            Some(rss) if !inst.phase_cpu_binds(wall_s, c.writers, Some(rss), cpu_s) => {
                format!(" — RAM-bound: {} peak held for the window", fmt_bytes(rss))
            }
            Some(rss) => format!(" — {} peak, CPU binds", fmt_bytes(rss)),
            None => String::new(),
        }
    };
    // Wall/basis cell: measured on-CPU time when captured, else the
    // `wall × pool-share` model.
    let phase_wall_cell = |wall_s: f64, cpu_s: Option<f64>| -> String {
        if cpu_s.is_some() {
            format!("{} · measured CPU", fmt_wall_seconds(wall_s))
        } else {
            format!("{} × {vcpu_share}", fmt_wall_seconds(wall_s))
        }
    };
    let mut compute_rows = vec![vec![
        text(format!(
            "Ingest ({}w on {} vCPU{})",
            c.writers,
            inst.vcpu,
            phase_binding_tag(c.ingest_wall_s, c.ingest_peak_rss_bytes, c.ingest_cpu_s),
        )),
        text(phase_wall_cell(c.ingest_wall_s, c.ingest_cpu_s)),
        text(fmt_vcpu_seconds(ingest_vcpu_s)),
        metric(ingest_compute, usd(ingest_compute), Better::Lower),
    ]];
    if c.store.drain_wall_s.is_some() {
        compute_rows.push(vec![
            text(format!(
                "Drain (hidden index, one-time{})",
                phase_binding_tag(drain_wall_s, c.store.drain_peak_rss_bytes, c.store.drain_cpu_s),
            )),
            text(phase_wall_cell(drain_wall_s, c.store.drain_cpu_s)),
            text(fmt_vcpu_seconds(drain_vcpu_s)),
            metric(drain_compute, usd(drain_compute), Better::Lower),
        ]);
    } else if c.vector_cell {
        compute_rows.push(vec![
            text("Drain (hidden index, one-time)"),
            text("NOT METERED"),
            text("—"),
            text("—"),
        ]);
    }
    if c.store.compaction_wall_s.is_some() {
        compute_rows.push(vec![
            text(format!(
                "Compaction (optimize, one-time{})",
                phase_binding_tag(
                    compaction_wall_s,
                    c.store.compaction_peak_rss_bytes,
                    c.store.compaction_cpu_s,
                ),
            )),
            text(phase_wall_cell(compaction_wall_s, c.store.compaction_cpu_s)),
            text(fmt_vcpu_seconds(compaction_vcpu_s)),
            metric(compaction_compute, usd(compaction_compute), Better::Lower),
        ]);
    } else if c.vector_cell {
        compute_rows.push(vec![
            text("Compaction (optimize, one-time)"),
            text("NOT METERED"),
            text("—"),
            text("—"),
        ]);
    }
    if let Some(q) = anchor_cold {
        let open_vcpu = inst.per_query_vcpu_seconds(q.open_s, c.resident_anon_bytes);
        let open_usd = open_vcpu * inst.usd_per_sec();
        let open_label = if c.cold_open_amortized {
            format!("Table open CPU (one-time, {})", q.name)
        } else {
            format!("Cold open CPU ({})", q.name)
        };
        compute_rows.push(vec![
            text(open_label),
            text(fmt_wall_seconds(q.open_s)),
            text(fmt_vcpu_seconds(open_vcpu)),
            metric(open_usd, usd(open_usd), Better::Lower),
        ]);
        let search_per_q = inst.per_query_usd(q.search_s, c.resident_anon_bytes);
        compute_rows.push(vec![
            text(format!("Cold query CPU ({})", q.name)),
            text(format!("{} p50", fmt_time(q.search_s * 1e9))),
            text(fmt_vcpu_seconds(
                inst.per_query_vcpu_seconds(q.search_s, c.resident_anon_bytes),
            )),
            metric(
                search_per_q * PER_MILLION,
                format!("{} queries", usd_per_million(search_per_q)),
                Better::Lower,
            ),
        ]);
    }
    if let Some((name, p50_s)) = c
        .warm
        .iter()
        .find(|(n, _)| n == "ten_term_or")
        .or_else(|| c.warm.first())
    {
        let per_q = inst.per_query_usd(*p50_s, c.resident_anon_bytes);
        compute_rows.push(vec![
            text(format!("Warm query CPU ({name})")),
            text(format!("{} p50", fmt_time(*p50_s * 1e9))),
            text(fmt_vcpu_seconds(
                inst.per_query_vcpu_seconds(*p50_s, c.resident_anon_bytes),
            )),
            metric(
                per_q * PER_MILLION,
                format!("{} queries", usd_per_million(per_q)),
                Better::Lower,
            ),
        ]);
    }
    let compute_ledger = Block {
        subtitle: format!(
            "Compute — one-time phases (ingest/drain/compaction) priced from measured on-CPU \
             seconds (I/O wait excluded; wall × pool-share fallback when /proc is unavailable), \
             per-query phases from p50 latency. Priced on {} ({} vCPU / {:.0} GiB / {:.0} GB \
             NVMe @ ${:.4}/hr); one-time phases in absolute $, per-query phases per 1M queries",
            inst.name, inst.vcpu, inst.ram_gib, inst.nvme_gb, inst.usd_per_hour,
        ),
        headers: vec![
            "Phase".into(),
            "Wall / p50".into(),
            "vCPU·s".into(),
            "Cost".into(),
        ],
        rows: compute_rows,
    };

    // ---- Block 4: serving ----
    let binding = if inst.ram_binds(c.resident_anon_bytes) {
        "DRAM"
    } else {
        "CPU"
    };
    let mut serving_rows: Vec<Vec<Cell>> = c
        .warm
        .iter()
        .map(|(name, p50_s)| {
            let per_q = inst.per_query_usd(*p50_s, c.resident_anon_bytes);
            let per_q_usd = per_q.max(f64::MIN_POSITIVE);
            let queries_per_usd = 1.0 / per_q_usd;
            vec![
                text(format!("{name} — warm")),
                text(fmt_time(p50_s * 1e9)),
                metric(
                    queries_per_usd,
                    format!("{queries_per_usd:.0}"),
                    Better::Higher,
                ),
                text(usd(per_q * PER_MILLION)),
            ]
        })
        .collect();
    if let (Some(q), Some(per_q)) = (anchor_cold, cold_query_usd) {
        let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
        let requests_note = c
            .store
            .cold_query
            .map(|io| format!(" (incl. {} GET/query)", io.get_count))
            .unwrap_or_default();
        serving_rows.push(vec![
            text(format!("{} — cold{requests_note}", q.name)),
            text(fmt_time(q.search_s * 1e9)),
            metric(
                queries_per_usd,
                format!("{queries_per_usd:.0}"),
                Better::Higher,
            ),
            text(usd(per_q * PER_MILLION)),
        ]);
    }
    // Pre-drain (transient) serving rows: what a query costs on a fresh
    // table before maintenance drains the hidden index.
    if let Some((name, p50_s)) = c.warm_pre.and_then(|rows| rows.first()) {
        let per_q = inst.per_query_usd(*p50_s, c.resident_anon_bytes);
        let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
        serving_rows.push(vec![
            text(format!("{name} — warm, pre-drain (transient)")),
            text(fmt_time(p50_s * 1e9)),
            metric(
                queries_per_usd,
                format!("{queries_per_usd:.0}"),
                Better::Higher,
            ),
            text(usd(per_q * PER_MILLION)),
        ]);
    }
    if let Some(q) = c.cold_pre.and_then(|rows| rows.first()) {
        let per_q = inst.per_query_usd(q.search_s, c.resident_anon_bytes)
            + c.store
                .cold_query_pre
                .map(|io| request_usd(&io))
                .unwrap_or(0.0);
        let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
        let requests_note = c
            .store
            .cold_query_pre
            .map(|io| format!(" (incl. {} GET/query)", io.get_count))
            .unwrap_or_default();
        serving_rows.push(vec![
            text(format!(
                "{} — cold, pre-drain (transient){requests_note}",
                q.name
            )),
            text(fmt_time(q.search_s * 1e9)),
            metric(
                queries_per_usd,
                format!("{queries_per_usd:.0}"),
                Better::Higher,
            ),
            text(usd(per_q * PER_MILLION)),
        ]);
    }
    let serving = Block {
        subtitle: format!(
            "Serving — latency per dollar (binding: {binding}; resident heap {}, file-backed \
             cache free on NVMe; cold row includes measured request cost)",
            fmt_bytes(c.resident_anon_bytes),
        ),
        headers: vec![
            "Query".into(),
            "p50".into(),
            "queries/$".into(),
            "$/1M queries".into(),
        ],
        rows: serving_rows,
    };

    let mut blocks = vec![rate_card];
    if let Some(io_ledger) = io_ledger {
        blocks.push(io_ledger);
    }
    blocks.push(compute_ledger);
    blocks.push(serving);

    report.emit(&Section {
        anchor: anchor.into(),
        title,
        note: "Cost model on measured bench rows. **Object-store I/O** counts come from a \
               metering wrapper around the storage provider — requests and bytes are measured, \
               never estimated, and multipart uploads count one PUT per part plus the create and \
               complete calls. A phase this cell has but the harness failed to measure says \
               **NOT METERED**. **RAM billing:** per-query and per-phase marginals bill the \
               *binding* resource — `max(CPU share, resident-RSS share)` — because at packing \
               density the non-binding resource is stranded capacity that cannot be sold twice \
               (dominant-resource pricing, so CPU and RAM are not summed); the **Serving \
               RAM** line is an instance-sizing fact, not a dollar line — query-time RAM is \
               already inside the per-query marginal. Every dollar figure is normalized per \
               **1M docs** (write path; storage over the stated retention) or per **1M \
               queries** (serving). Per-query costs are per **1M queries** (warm = marginal \
               binding resource; cold = the same + measured GET requests). Warm/filtered I/O \
               rows average a multi-query window on the **same cache-hot consumer the warm \
               latency battery timed**; the fill-lag probe row is a diagnostic (repeat query \
               on a fresh consumer), not a steady-state price. Pre-drain rows show the \
               transient shape before hidden-index maintenance. The supertable's cold `open` \
               is one-time and amortized. Δ is vs the previous run."
            .into(),
        blocks,
    });
}

/// Flatten cold `(open, search)` timings keyed by query name into cost
/// rows. Shared by the FTS and SQL runners (both measure per-query
/// `ColdTiming` maps).
pub fn cold_from_timings(cold: &HashMap<&'static str, ColdTiming>) -> Vec<ColdQuery> {
    cold.iter()
        .map(|(name, t)| ColdQuery {
            name: (*name).to_string(),
            open_s: t.open.as_secs_f64(),
            search_s: t.search.as_secs_f64(),
        })
        .collect()
}

/// Flatten warm FTS stats into `(name, p50_seconds)` for the cost model.
pub fn warm_from_fts(stats: &[FtsQueryStat]) -> Vec<(String, f64)> {
    stats
        .iter()
        .map(|s| (s.name.to_string(), s.p50.as_secs_f64()))
        .collect()
}

/// Flatten warm SQL query sets into `(name, p50_seconds)`.
pub fn warm_from_sql(sets: &QuerySets) -> Vec<(String, f64)> {
    sets.scalar
        .iter()
        .chain(&sets.tvf)
        .chain(&sets.fts_pushdown)
        .chain(&sets.agg_idx)
        .map(|s| (s.name.to_string(), s.p50.as_secs_f64()))
        .collect()
}

/// Flatten warm vector recall rows into `(label, p50_seconds)`.
pub fn warm_from_vector(rows: &[RecallRow]) -> Vec<(String, f64)> {
    rows.iter()
        .filter_map(|r| {
            r.warm.as_ref().map(|w| {
                let label = if r.params.is_empty() || r.params == "—" {
                    r.target.clone()
                } else {
                    format!("{} ({})", r.target, r.params)
                };
                (label, w.p50_ns / 1e9)
            })
        })
        .collect()
}

/// Flatten cold vector recall rows into `(label, open, search)` for the cost model.
pub fn cold_from_vector(rows: &[RecallRow]) -> Vec<ColdQuery> {
    rows.iter()
        .filter_map(|r| {
            r.cold.map(|t| {
                let label = if r.params.is_empty() || r.params == "—" {
                    r.target.clone()
                } else {
                    format!("{} ({})", r.target, r.params)
                };
                ColdQuery {
                    name: label,
                    open_s: t.open.as_secs_f64(),
                    search_s: t.search.as_secs_f64(),
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_instance() -> Instance {
        Instance {
            name: "test".into(),
            vcpu: 8,
            ram_gib: 16.0,
            nvme_gb: 237.0,
            usd_per_hour: 0.3629,
        }
    }

    #[test]
    fn parallel_ingest_costs_more_per_second_than_single_writer() {
        let inst = test_instance();
        let single = inst.phase_vcpu_seconds(10.0, 1, None, None) * inst.usd_per_sec();
        let full = inst.phase_vcpu_seconds(10.0, 8, None, None) * inst.usd_per_sec();
        assert!((full / single - 8.0).abs() < 1e-9);
    }

    #[test]
    fn ram_bound_phase_bills_rss_share_for_full_wall() {
        let inst = test_instance();
        // 1 writer on 8 vCPU = 12.5% CPU share; 8 GiB peak on 16 GiB = 50%
        // RAM share → RAM binds and the phase bills 4× the CPU-only price.
        let eight_gib = 8u64 << 30;
        let cpu_only = inst.phase_vcpu_seconds(10.0, 1, None, None) * inst.usd_per_sec();
        let ram_bound = inst.phase_vcpu_seconds(10.0, 1, Some(eight_gib), None) * inst.usd_per_sec();
        assert!(!inst.phase_cpu_binds(10.0, 1, Some(eight_gib), None));
        assert!((ram_bound / cpu_only - 4.0).abs() < 1e-9);
        // Full-pool CPU (100%) dominates the same 50% RAM share.
        assert!(inst.phase_cpu_binds(10.0, 8, Some(eight_gib), None));
    }

    #[test]
    fn measured_cpu_overrides_wall_model_for_phase() {
        let inst = test_instance();
        // No measurement → wall model: 10s × (8/8 pool share) = 10 vCPU·s.
        assert!((inst.phase_vcpu_seconds(10.0, 8, None, None) - 10.0).abs() < 1e-9);
        // Measured on-CPU 3.5s (I/O wait excluded) is billed verbatim when it
        // exceeds the RAM leg — the whole point of measuring instead of wall.
        assert!((inst.phase_vcpu_seconds(10.0, 8, None, Some(3.5)) - 3.5).abs() < 1e-9);
        // RAM leg (50% of 16 GiB over a 10s wall = 5 vCPU·s) still binds when
        // it exceeds the measured CPU.
        let eight_gib = 8u64 << 30;
        assert!((inst.phase_vcpu_seconds(10.0, 8, Some(eight_gib), Some(3.5)) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn lower_latency_yields_more_queries_per_dollar() {
        let inst = test_instance();
        let fast = inst.per_query_usd(0.001, 1 << 20);
        let slow = inst.per_query_usd(0.010, 1 << 20);
        assert!(slow > fast);
        assert!((slow / fast - 10.0).abs() < 1e-6);
    }

    #[test]
    fn ram_binds_only_when_heap_exceeds_per_core_budget() {
        let inst = test_instance();
        assert!(!inst.ram_binds(1 << 30));
        assert!(inst.ram_binds(3 * (1 << 30)));
    }

    #[test]
    fn usd_never_collapses_sub_cent_values_to_zero() {
        assert_eq!(usd(0.0), "$0");
        assert_eq!(usd(1.014), "$1.01");
        assert_eq!(usd(0.02), "$0.02");
        // Two significant digits below one cent instead of "$0.0000".
        assert_eq!(usd(2.8e-5), "$0.000028");
        assert_eq!(usd(7.0e-5), "$0.000070");
        assert_eq!(usd(0.0028), "$0.0028");
    }

    #[test]
    fn per_million_scales_per_query_dollars() {
        // 175 GET/query at $0.40/1M requests = $70 per 1M queries.
        let per_query = 175.0 * USD_PER_GET;
        assert_eq!(usd_per_million(per_query), "$70.00/1M");
    }

    #[test]
    fn request_usd_prices_puts_and_reads() {
        let io = ObjectStoreMeter {
            head_count: 10,
            get_count: 90,
            get_bytes: 0,
            put_count: 1000,
            put_bytes: 0,
            ..Default::default()
        };
        // 1000 PUT × $5e-6 + 100 reads × $4e-7.
        let expected = 1000.0 * 5.0e-6 + 100.0 * 4.0e-7;
        assert!((request_usd(&io) - expected).abs() < 1e-12);
    }
}
