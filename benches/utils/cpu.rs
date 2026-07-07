// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Process CPU-time sampling for benchmark cost accounting.
//!
//! Source: the sum of on-CPU nanoseconds across **every thread** of this
//! process, read from `/proc/self/task/<tid>/schedstat` (field 0 = time spent
//! on the CPU, in ns). Properties that make this the right billing source:
//!
//!   - **All-thread aggregate.** Includes the rayon compute-pool workers and
//!     tokio I/O workers, so it captures the whole process's compute — the
//!     foreground+background total we want.
//!   - **Excludes I/O wait.** A thread awaiting an object-store GET is *not*
//!     on-CPU, so its wait does not accrue here. This is exactly why wall time
//!     must not be used for compute cost: wall includes the wait, on-CPU time
//!     does not.
//!   - **ns resolution, no dependency.** Two `/proc` reads per measured
//!     window; nothing added to the dependency graph.
//!
//! Latency stays wall-clock (`Instant`); only *compute cost* uses these
//! on-CPU deltas.
//!
//! ## Measurement regime
//!
//! Warm queries finish in hundreds of ns. You cannot time a single such op —
//! any clock read is the same order as the work. So warm CPU is **amortized**:
//! run the op in a loop long enough that the two boundary reads are negligible,
//! then divide (see [`amortized_cpu_per_iter`]). Cold opens / drain /
//! compaction run in ms–s, where a per-window delta is already accurate.

use std::{
    fs,
    time::{Duration, Instant},
};

/// Directory holding one entry per live thread of this process.
const PROC_SELF_TASK: &str = "/proc/self/task";
/// `schedstat` file name under each task dir.
const SCHEDSTAT: &str = "schedstat";
/// Nanoseconds per second.
const NS_PER_SEC: f64 = 1e9;

/// Sum of on-CPU nanoseconds across every thread of this process, or `None`
/// on a platform without `/proc/self/task/*/schedstat`.
pub fn process_cpu_ns() -> Option<u128> {
    let mut total: u128 = 0;
    let mut any = false;
    for entry in fs::read_dir(PROC_SELF_TASK).ok()? {
        let Ok(entry) = entry else { continue };
        let Ok(raw) = fs::read_to_string(entry.path().join(SCHEDSTAT)) else {
            // A thread can exit between the readdir and the open — skip it.
            continue;
        };
        if let Some(ns) = raw.split_whitespace().next().and_then(|f| f.parse::<u128>().ok()) {
            total += ns;
            any = true;
        }
    }
    any.then_some(total)
}

/// Non-negative CPU seconds elapsed since a `process_cpu_ns()` snapshot.
/// `None` if either snapshot is unavailable.
pub fn cpu_seconds_since(start_ns: Option<u128>) -> Option<f64> {
    let start = start_ns?;
    let end = process_cpu_ns()?;
    Some(end.saturating_sub(start) as f64 / NS_PER_SEC)
}

/// Run `f`, returning `(result, wall_duration, measured_on_cpu_seconds)`.
///
/// The single primitive every phase / cold-query measurement uses, so wall
/// and on-CPU time are always bracketed identically around the same work.
/// `cpu` is `None` only when `/proc/self/task` sampling is unavailable
/// (never on the Linux bench hosts); callers treat that as "not measured",
/// never as an excuse to substitute a wall-clock approximation.
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration, Option<f64>) {
    let cpu0 = process_cpu_ns();
    let t0 = Instant::now();
    let out = f();
    let wall = t0.elapsed();
    (out, wall, cpu_seconds_since(cpu0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The calling thread's own on-CPU nanoseconds — the exact per-thread
    /// `schedstat` field [`process_cpu_ns`] sums. Reading *this thread* makes
    /// the exclusion check below deterministic regardless of other test
    /// threads the parallel harness runs concurrently (the process-wide
    /// aggregate would include their on-CPU time).
    fn thread_cpu_ns() -> Option<u128> {
        let raw = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
        raw.split_whitespace().next()?.parse().ok()
    }

    /// The metric is on-CPU time: busy work accrues it; a sleep (off-CPU I/O
    /// wait) accrues far less. This is the property that makes it correct to
    /// price compute from CPU and never from wall — async object-store wait is
    /// off-CPU and is not billed.
    #[test]
    fn on_cpu_time_excludes_sleep() {
        let Some(start_busy) = thread_cpu_ns() else {
            return; // non-Linux / no procfs: skip
        };
        let t = Instant::now();
        let mut acc = 0u64;
        while t.elapsed() < Duration::from_millis(50) {
            for i in 0..4096u64 {
                acc = acc.wrapping_add(i.wrapping_mul(2654435761));
            }
        }
        std::hint::black_box(acc);
        let busy = (thread_cpu_ns().expect("cpu") - start_busy) as f64 / NS_PER_SEC;

        let start_sleep = thread_cpu_ns().expect("cpu");
        std::thread::sleep(Duration::from_millis(50));
        let slept = (thread_cpu_ns().expect("cpu") - start_sleep) as f64 / NS_PER_SEC;

        assert!(busy > 0.010, "50ms of busy work must accrue on-CPU time, got {busy}s");
        assert!(
            slept < busy / 5.0,
            "a sleep ({slept}s) must accrue far less on-CPU time than busy work ({busy}s)"
        );
    }

    /// `timed` returns the closure's value, its wall duration, and — on a
    /// procfs host — a measured on-CPU figure. (The "on-CPU excludes I/O
    /// wait" property is asserted by [`on_cpu_time_excludes_sleep`] using a
    /// thread-local read; `timed`'s process-wide aggregate can't assert it
    /// under the parallel test harness, where sibling test threads run.)
    #[test]
    fn timed_returns_value_wall_and_cpu() {
        let (value, wall, cpu) = timed(|| {
            std::thread::sleep(Duration::from_millis(20));
            42u64
        });
        assert_eq!(value, 42);
        assert!(wall >= Duration::from_millis(20));
        if process_cpu_ns().is_some() {
            assert!(cpu.expect("procfs host measures cpu") >= 0.0);
        }
    }
}
