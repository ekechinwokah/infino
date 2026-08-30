# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Numbers and provenance for the README performance charts.

Two measurement windows feed these charts and they are *not* directly
comparable — different commits, different hardware:

  1M  — CI Benchmark (Azure Blob, 4 cores pinned), run 33245831329 on
        commit 3aaffb64, 2026-08-29.
  10M — the default supertable scale, measured separately (commit 339e621
        window). See benches/README.md.

Every chart therefore labels its scale groups and carries both provenances in
the footnote. Footnote text is drawn straight into SVG, so it must stay plain:
no markdown, no angle brackets.
"""

from __future__ import annotations

from dataclasses import dataclass

# Milliseconds is the single internal unit; the renderer picks µs/ms/s per value.
US = 1 / 1000


@dataclass(frozen=True)
class Bar:
    """One horizontal bar. `group` heads a run of consecutive bars."""

    group: str
    label: str
    ms: float
    cold: bool = False


CI_RUN = "33245831329"
CI_RUN_URL = f"https://github.com/infino-ai/infino/actions/runs/{CI_RUN}"
CI_COMMIT = "3aaffb64"

# Both windows in one line, plain text, for the chart footnotes.
PROVENANCE = (
    f"1M: Azure Blob, 4 cores pinned, CI run {CI_RUN} ({CI_COMMIT}) "
    "· 10M: supertable default scale, separate run"
)

# ── Internal latency ────────────────────────────────────────────────────────

VECTOR = {
    "title": "Vector search",
    "subtitle": "1024-d cosine · top-10 · post-drain · recall@10 0.992",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "warm p50", 591 * US),
        Bar("1M", "warm p99", 687 * US),
        Bar("1M", "cold p50", 114, cold=True),
        Bar("10M", "warm p50", 5),
        Bar("10M", "warm p99", 12),
        Bar("10M", "cold p50", 314, cold=True),
    ],
}

FTS = {
    "title": "Full-text search (BM25)",
    "subtitle": "top-10 including row fetch · 1M: single_rare · 10M: median shape",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "warm p50", 125 * US),
        Bar("1M", "warm p99", 143 * US),
        Bar("1M", "cold p50", 16.4, cold=True),
        Bar("10M", "warm p50", 2),
        Bar("10M", "warm p99", 7),
        Bar("10M", "cold p50", 275, cold=True),
    ],
}

SQL = {
    "title": "SQL query shapes",
    "subtitle": "warm p50 · supertable on object storage",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "metadata", 186 * US),
        Bar("1M", "lookup", 4.41),
        Bar("1M", "scan", 5.16),
        Bar("1M", "crosstab", 7.63),
        Bar("10M", "metadata", 260 * US),
        Bar("10M", "lookup", 2.74),
        Bar("10M", "scan", 41.14),
        Bar("10M", "crosstab", 75.14),
    ],
}
