#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Render the README performance charts from bench_data.py.

Latency spans three or more orders of magnitude between a warm hit and a cold
first query, so the charts use a base-10 log axis with a labelled gridline on
every decade. A linear axis collapses the warm bar to a stub and hides the
number the chart exists to show.
"""

from __future__ import annotations

import math
from pathlib import Path

from bench_data import FTS, SQL, VECTOR

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "readme"

WARM = "#D6400F"
COLD = "#B7B4AB"
BG = "#FBF9F3"
INK = "#17130F"
MUTED = "#5E5749"
GRID = "#DCD6C7"
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

WIDTH = 880
PAD = 26
LABEL_X = 170
TRACK_X = 184
TRACK_W = 600
VALUE_X = WIDTH - PAD

PLOT_TOP = 56
BAR_H = 9
ROW_PITCH = 22
GROUP_GAP = 14

# A bar whose value sits on the axis minimum would render as zero width.
MIN_BAR_W = 3
# Headroom above the largest value so its bar stops short of the frame.
AXIS_HEADROOM = 1.15
# Start the axis a decade lower when the smallest value sits near a decade
# boundary, otherwise its bar is a stub against the axis origin.
LOW_DECADE_MARGIN = 2


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_value(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f} µs"
    if ms < 10:
        return f"{ms:g} ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:g} s"


def log_axis(values: list[float]) -> tuple[float, float, list[float]]:
    """Return (axis min, axis max, decade tick values) for a log scale."""
    lo_v, hi_v = min(values), max(values)
    lo_dec = math.floor(math.log10(lo_v))
    if lo_v / 10.0**lo_dec < LOW_DECADE_MARGIN:
        lo_dec -= 1
    lo = 10.0**lo_dec
    hi = hi_v * AXIS_HEADROOM
    ticks = []
    dec = lo_dec
    while 10.0**dec <= hi:
        ticks.append(10.0**dec)
        dec += 1
    return lo, hi, ticks


def text(x, y, s, *, size=12, fill=MUTED, anchor="start", weight=None) -> str:
    w = f' font-weight="{weight}"' if weight else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}"{w}{a}>{esc(s)}</text>'
    )


def frame(height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img">',
        f'<rect width="{WIDTH}" height="{height}" rx="10" fill="{BG}" stroke="{GRID}"/>',
        text(PAD, 32, title, size=13, fill=INK, weight=600),
        text(VALUE_X, 32, subtitle, size=11, anchor="end"),
    ]


def latency_chart(spec: dict, filename: str) -> None:
    bars = spec["bars"]
    groups: list[tuple[str, list]] = []
    for bar in bars:
        if not groups or groups[-1][0] != bar.group:
            groups.append((bar.group, []))
        groups[-1][1].append(bar)

    lo, hi, ticks = log_axis([b.ms for b in bars])
    span = math.log10(hi / lo)

    def bar_w(v: float) -> float:
        return max((math.log10(v / lo) / span) * TRACK_W, MIN_BAR_W)

    plot_h = len(bars) * ROW_PITCH + (len(groups) - 1) * GROUP_GAP
    axis_y = PLOT_TOP + plot_h + 4
    has_cold = any(b.cold for b in bars)
    height = int(axis_y + 24 + (30 if has_cold else 26))

    lines = frame(height, spec["title"], spec["subtitle"])

    for tick in ticks:
        gx = TRACK_X + (math.log10(tick / lo) / span) * TRACK_W
        lines.append(
            f'<line x1="{gx:.1f}" y1="{PLOT_TOP - 6}" x2="{gx:.1f}" y2="{axis_y}" '
            f'stroke="{GRID}"/>'
        )
        lines.append(text(gx, axis_y + 16, fmt_value(tick), size=10, anchor="middle"))

    y = PLOT_TOP
    for name, members in groups:
        lines.append(text(PAD, y + BAR_H, name, size=12, fill=INK, weight=600))
        for bar in members:
            color = COLD if bar.cold else WARM
            lines.append(text(LABEL_X, y + BAR_H, bar.label, size=11, anchor="end"))
            lines.append(
                f'<rect x="{TRACK_X}" y="{y}" width="{bar_w(bar.ms):.1f}" '
                f'height="{BAR_H}" rx="1" fill="{color}"/>'
            )
            lines.append(
                text(
                    VALUE_X,
                    y + BAR_H,
                    fmt_value(bar.ms),
                    size=11,
                    fill=MUTED if bar.cold else INK,
                    anchor="end",
                    weight=600,
                )
            )
            y += ROW_PITCH
        y += GROUP_GAP

    foot_y = height - 12
    if has_cold:
        lines.append(
            f'<rect x="{PAD}" y="{foot_y - 21}" width="9" height="9" rx="1" fill="{WARM}"/>'
        )
        lines.append(text(PAD + 16, foot_y - 13, "warm cache", size=11))
        lines.append(
            f'<rect x="130" y="{foot_y - 21}" width="9" height="9" rx="1" fill="{COLD}"/>'
        )
        lines.append(text(146, foot_y - 13, "cold · first query on an idle table", size=11))
    lines.append(text(VALUE_X, foot_y, spec["footnote"], size=10, anchor="end"))
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    latency_chart(VECTOR, "vector.svg")
    latency_chart(FTS, "fts.svg")
    latency_chart(SQL, "sql.svg")
    print(f"wrote charts to {OUT}")


if __name__ == "__main__":
    main()
