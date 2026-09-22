#!/usr/bin/env python3
"""Hand-rolled SVG for the mu-threshold null result (matplotlib broken in env).

Reads artifacts/phase4_amr/mu_threshold_v1/stats.json, draws three panels
(v5 / v5cap / all): binned routing success vs displacement concentration C,
binomial-SE whiskers, base-rate dashed line, Youden-threshold shading.
Stdlib only. The figure honestly shows the NULL result (flat lines).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "artifacts" / "phase4_amr" / "mu_threshold_v1" / "stats.json"
OUT = ROOT / "artifacts" / "phase4_amr" / "mu_threshold_v1" / "figure.svg"

W, H = 1180, 360
ML, MR, MT, MB = 56, 16, 44, 44
PW = (W - ML - MR - 20 * 2) // 3
PH = H - MT - MB
X0, X1 = 0.30, 0.85
Y0, Y1 = 0.35, 1.0


def sx(x: float, i: int) -> float:
    return ML + i * (PW + 20) + (x - X0) / (X1 - X0) * PW


def sy(y: float) -> float:
    return MT + PH - (y - Y0) / (Y1 - Y0) * PH


def main() -> None:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    el: list[str] = []
    el.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
              'font-family="sans-serif" font-size="12">')
    el.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    el.append('<text x="20" y="24" font-size="15" font-weight="bold">'
              'Routing success vs displacement concentration C (null result: no monotone relation)</text>')
    for i, pool in enumerate(("v5", "v5cap", "all")):
        d = stats[pool]
        ox = ML + i * (PW + 20)
        el.append(f'<rect x="{ox}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#999"/>')
        el.append(f'<text x="{ox + 8}" y="{MT - 8}" font-weight="bold">'
                  f'{pool} (AUC={d["auc"]:.2f}, n={d["n"]})</text>')
        # y gridlines
        for yv in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            el.append(f'<line x1="{ox}" y1="{sy(yv):.1f}" x2="{ox + PW}" y2="{sy(yv):.1f}" '
                      'stroke="#ddd"/>')
            el.append(f'<text x="{ox - 6}" y="{sy(yv) + 4:.1f}" text-anchor="end" fill="#666">{yv:.1f}</text>')
        # sub-threshold shading
        thr = d["youden_threshold"]
        if math.isfinite(thr) and X0 < thr < X1:
            el.append(f'<rect x="{ox}" y="{MT}" width="{sx(thr, i) - ox:.1f}" height="{PH}" '
                      'fill="#e74c3c" opacity="0.10"/>')
            el.append(f'<line x1="{sx(thr, i):.1f}" y1="{MT}" x2="{sx(thr, i):.1f}" y2="{MT + PH}" '
                      'stroke="#e74c3c" stroke-dasharray="4,3"/>')
            el.append(f'<text x="{sx(thr, i) + 3:.1f}" y="{MT + 12}" fill="#e74c3c" font-size="10">'
                      f'thr={thr:.2f}</text>')
        # base-rate dashed line
        el.append(f'<line x1="{ox}" y1="{sy(d["base_rate"]):.1f}" x2="{ox + PW}" '
                  f'y2="{sy(d["base_rate"]):.1f}" stroke="#333" stroke-dasharray="6,3"/>')
        # bins with SE whiskers
        pts = []
        for b in d["bins"]:
            x = (b["lo"] + b["hi"]) / 2
            p, n = b["rate"], b["n"]
            se = math.sqrt(max(p * (1 - p), 0) / max(n, 1))
            pts.append((x, p))
            el.append(f'<line x1="{sx(x, i):.1f}" y1="{sy(max(p - se, Y0)):.1f}" '
                      f'x2="{sx(x, i):.1f}" y2="{sy(min(p + se, Y1)):.1f}" stroke="#2471a3"/>')
            el.append(f'<circle cx="{sx(x, i):.1f}" cy="{sy(p):.1f}" r="4.5" fill="#2471a3"/>')
            el.append(f'<text x="{sx(x, i):.1f}" y="{sy(min(p + se, Y1)) - 6:.1f}" text-anchor="middle" '
                      f'fill="#555" font-size="10">n={n}</text>')
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            el.append(f'<line x1="{sx(x1, i):.1f}" y1="{sy(y1):.1f}" x2="{sx(x2, i):.1f}" '
                      f'y2="{sy(y2):.1f}" stroke="#2471a3"/>')
        el.append(f'<text x="{ox + PW / 2}" y="{MT + PH + 30}" text-anchor="middle">concentration C</text>')
    el.append(f'<text x="{ML - 44}" y="{MT + PH / 2}" transform="rotate(-90 {ML - 44},{MT + PH / 2})" '
              'text-anchor="middle">routing success rate</text>')
    el.append('</svg>')
    OUT.write_text("\n".join(el) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
