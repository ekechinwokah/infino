#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Generate README performance chart SVGs from scripts/readme_charts/bench_data.py."""

from __future__ import annotations

import math
from pathlib import Path

from bench_data import (
    FTS_1M,
    FTS_10M,
    SBG_META,
    SBG_ROWS,
    SQL_1M,
    SQL_10M,
    SQL_EXT_META,
    SQL_EXT_ROWS,
    VDB_META,
    VDB_ROWS,
    VECTOR_1M,
    VECTOR_10M,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "readme"

WARM = "#D6400F"
COLD = "#B7B4AB"
BG = "#FBF9F3"
INK = "#17130F"
MUTED = "#5E5749"
GRID = "#DCD6C7"
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _ms(row) -> tuple[float, float]:
    warm = row.warm_ms if row.warm_ms is not None else (row.warm_us or 0) / 1000
    cold = row.cold_ms if row.cold_ms is not None else (row.cold_us or 0) / 1000
    return warm, cold


def _fmt_ms(v: float) -> str:
    if v >= 1:
        return f"{v:.0f} ms" if v >= 10 else f"{v:.1f} ms"
    if v >= 0.1:
        return f"{v:.1f} ms"
    return f"{v * 1000:.0f} µs"


def latency_chart(spec: dict, filename: str) -> None:
    rows = spec["rows"]
    values: list[tuple[str, float, float]] = []
    for row in rows:
        warm, cold = _ms(row)
        values.append((row.label, warm, cold))
    max_v = max(max(w, c) for _, w, c in values) or 1

    height = 56 + len(values) * 52 + 72
    width = 880
    track_x = 164
    track_w = 520
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{BG}" stroke="{GRID}"/>',
        f'<text x="26" y="32" font-family="{FONT}" font-size="13" font-weight="600" fill="{INK}">{spec["title"]}</text>',
        f'<text x="{width - 26}" y="32" font-family="{FONT}" font-size="11" fill="{MUTED}" text-anchor="end">{spec["subtitle"]}</text>',
    ]

    y = 58
    for label, warm, cold in values:
        lines.append(
            f'<text x="150" y="{y + 14}" font-family="{FONT}" font-size="12" fill="{MUTED}" text-anchor="end">{label}</text>'
        )
        for val, color in ((warm, WARM), (cold, COLD)):
            bar_w = max(track_w * (val / max_v), 4)
            yy = y + (0 if color == WARM else 18)
            lines.append(
                f'<rect x="{track_x}" y="{yy}" width="{bar_w:.1f}" height="9" rx="1" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{width - 26}" y="{yy + 8}" font-family="{FONT}" font-size="11" '
                f'font-weight="600" fill="{INK if color == WARM else MUTED}" text-anchor="end">{_fmt_ms(val)}</text>'
            )
        y += 52

    foot_y = height - 22
    lines.append(f'<line x1="26" y1="{foot_y - 14}" x2="{width - 26}" y2="{foot_y - 14}" stroke="{GRID}"/>')
    lines.append(f'<rect x="26" y="{foot_y - 6}" width="9" height="9" rx="1" fill="{WARM}"/>')
    lines.append(f'<text x="42" y="{foot_y + 1}" font-family="{FONT}" font-size="11" fill="{MUTED}">warm cache</text>')
    lines.append(f'<rect x="130" y="{foot_y - 6}" width="9" height="9" rx="1" fill="{COLD}"/>')
    lines.append(
        f'<text x="146" y="{foot_y + 1}" font-family="{FONT}" font-size="11" fill="{MUTED}">cold · first query on idle table</text>'
    )
    lines.append(
        f'<text x="{width - 26}" y="{foot_y + 1}" font-family="{FONT}" font-size="10" fill="{MUTED}" text-anchor="end">{spec["footnote"]}</text>'
    )
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def sql_shapes_chart(spec: dict, filename: str) -> None:
    rows = spec["rows"]
    values = [(r.label, _ms(r)[0]) for r in rows]
    max_v = max(v for _, v in values) or 1
    height = 56 + len(values) * 36 + 48
    width = 880
    track_x = 220
    track_w = 480
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{BG}" stroke="{GRID}"/>',
        f'<text x="26" y="32" font-family="{FONT}" font-size="13" font-weight="600" fill="{INK}">{spec["title"]}</text>',
        f'<text x="{width - 26}" y="32" font-family="{FONT}" font-size="11" fill="{MUTED}" text-anchor="end">{spec["subtitle"]}</text>',
    ]
    y = 58
    for label, val in values:
        bar_w = max(track_w * (val / max_v), 4)
        lines.append(
            f'<text x="210" y="{y + 12}" font-family="{FONT}" font-size="12" fill="{MUTED}" text-anchor="end">{label}</text>'
        )
        lines.append(f'<rect x="{track_x}" y="{y + 3}" width="{bar_w:.1f}" height="9" rx="1" fill="{WARM}"/>')
        lines.append(
            f'<text x="{width - 26}" y="{y + 12}" font-family="{FONT}" font-size="11" font-weight="600" fill="{INK}" text-anchor="end">{_fmt_ms(val)}</text>'
        )
        y += 36
    foot_y = height - 18
    lines.append(
        f'<text x="{width - 26}" y="{foot_y}" font-family="{FONT}" font-size="10" fill="{MUTED}" text-anchor="end">{spec["footnote"]}</text>'
    )
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_chart(
    title: str,
    subtitle: str,
    rows: list,
    filename: str,
    *,
    log: bool = False,
    dual: bool = False,
) -> None:
    if dual:
        height = 56 + len(rows) * 44 + 56
    else:
        height = 56 + len(rows) * 34 + 48
    width = 880
    label_x = 250
    track_x = 264
    track_w = 420

    if dual:
        max_v = max(max(search, count) for _, _, search, count, _ in rows)

        def pct(v: float) -> float:
            return max(track_w * (v / max_v), 4)

    elif log:
        vals = [r.value for r in rows]
        min_v = min(vals)
        max_v = max(vals)
        lead = 0.5

        def pct(v: float) -> float:
            span = math.log10(max_v / min_v) + lead
            return max(((math.log10(v / min_v) + lead) / span) * track_w, 4)

    else:
        max_v = max(r.value for r in rows)

        def pct(v: float) -> float:
            return max(track_w * (v / max_v), 4)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{BG}" stroke="{GRID}"/>',
        f'<text x="26" y="32" font-family="{FONT}" font-size="13" font-weight="600" fill="{INK}">{title}</text>',
        f'<text x="{width - 26}" y="32" font-family="{FONT}" font-size="11" fill="{MUTED}" text-anchor="end">{subtitle}</text>',
    ]
    y = 56
    if dual:
        for name, cfg, search, count, self_row in rows:
            color = WARM if self_row else COLD
            lines.append(
                f'<text x="{label_x}" y="{y + 10}" font-family="{FONT}" font-size="12" '
                f'fill="{INK if self_row else MUTED}" text-anchor="end">{name}</text>'
            )
            lines.append(
                f'<text x="{label_x}" y="{y + 24}" font-family="{FONT}" font-size="10" fill="{MUTED}" text-anchor="end">{cfg}</text>'
            )
            for idx, val in enumerate((search, count)):
                bar_w = pct(val)
                yy = y + idx * 14
                fill = color if idx == 0 else ("#E8A090" if self_row else "#CFCBC2")
                lines.append(f'<rect x="{track_x}" y="{yy}" width="{bar_w:.1f}" height="8" rx="1" fill="{fill}"/>')
                lines.append(
                    f'<text x="{width - 26}" y="{yy + 7}" font-family="{FONT}" font-size="10" '
                    f'font-weight="600" fill="{INK if self_row else MUTED}" text-anchor="end">{val:.2f}×</text>'
                )
            y += 44
        lines.append(f'<rect x="26" y="{height - 28}" width="9" height="9" rx="1" fill="{COLD}"/>')
        lines.append(f'<text x="42" y="{height - 21}" font-family="{FONT}" font-size="11" fill="{MUTED}">search (top-k)</text>')
        lines.append(f'<rect x="150" y="{height - 28}" width="9" height="9" rx="1" fill="#CFCBC2"/>')
        lines.append(f'<text x="166" y="{height - 21}" font-family="{FONT}" font-size="11" fill="{MUTED}">count</text>')
    else:
        for row in rows:
            color = WARM if row.self_row else COLD
            bar_w = pct(row.value)
            lines.append(
                f'<text x="{label_x}" y="{y + 10}" font-family="{FONT}" font-size="12" '
                f'fill="{INK if row.self_row else MUTED}" text-anchor="end">{row.name}</text>'
            )
            if row.config:
                lines.append(
                    f'<text x="{label_x}" y="{y + 24}" font-family="{FONT}" font-size="10" fill="{MUTED}" text-anchor="end">{row.config}</text>'
                )
            lines.append(f'<rect x="{track_x}" y="{y + 4}" width="{bar_w:.1f}" height="9" rx="1" fill="{color}"/>')
            lines.append(
                f'<text x="{width - 26}" y="{y + 12}" font-family="{FONT}" font-size="11" '
                f'font-weight="600" fill="{INK if row.self_row else MUTED}" text-anchor="end">{row.label}</text>'
            )
            y += 34
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    latency_chart(VECTOR_1M, "vector-1m.svg")
    latency_chart(VECTOR_10M, "vector-10m.svg")
    latency_chart(FTS_1M, "fts-1m.svg")
    latency_chart(FTS_10M, "fts-10m.svg")
    sql_shapes_chart(SQL_1M, "sql-1m.svg")
    sql_shapes_chart(SQL_10M, "sql-10m.svg")
    compare_chart(VDB_META["title"], VDB_META["subtitle"], VDB_ROWS, "compare-vdb.svg")
    compare_chart(
        SBG_META["title"],
        SBG_META["subtitle"],
        SBG_ROWS,
        "compare-fts.svg",
        dual=True,
    )
    compare_chart(
        SQL_EXT_META["title"],
        SQL_EXT_META["subtitle"],
        SQL_EXT_ROWS,
        "compare-sql.svg",
        log=True,
    )
    print(f"wrote charts to {OUT}")


if __name__ == "__main__":
    main()
