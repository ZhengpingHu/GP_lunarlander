#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse GA/GP training logs and plot rewards vs. generation.

Supported log line example (emojis/extra text are fine):
    🏆 [GEN 001] Best=+29.76  AvgTop12=-84.87  Sigma=0.100

We extract:
  - generation number (1-based)
  - Best
  - AvgTopK (e.g., AvgTop12) or explicitly "AvgTop 10%"

Usage:
  python gp_log_plot.py --input path/to/train.log --out rewards.png
  # or read from stdin
  cat train.log | python gp_log_plot.py --out rewards.png

The chart shows two series: 'Best' and 'Avg Top 10%'.
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

GEN_PATTERN = re.compile(
    r"\[GEN\s*(?P<gen>\d+)\].*?"  # [GEN 001]
    r"Best\s*=\s*(?P<best>[+\-]?\d+(?:\.\d+)?)"  # Best=+29.76
    r".*?"
    r"Avg(?:Top\s*\d+|\s*Top\s*10%?)\s*=\s*(?P<avg>[+\-]?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

# Fallback patterns (more permissive) in case the primary one misses
FALLBACKS = [
    re.compile(r"\[GEN\s*(\d+)\].*?Best\s*=\s*([+\-]?\d+(?:\.\d+)?).*?AvgTop(\d+)\s*=\s*([+\-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\[GEN\s*(\d+)\].*?Best\s*=\s*([+\-]?\d+(?:\.\d+)?).*?Avg\s*Top\s*10%\s*=\s*([+\-]?\d+(?:\.\d+)?)", re.IGNORECASE),
]


def parse_lines(text: str) -> List[Tuple[int, float, float]]:
    """Return list of tuples (gen, best, avg_top10pct) sorted by gen.

    We treat any 'AvgTopK' field as the top-10% average if K roughly equals 10% of
    the population; if not known, we still parse it and label the curve 'Avg Top 10%'.
    """
    rows: List[Tuple[int, float, float]] = []

    for line in text.splitlines():
        m = GEN_PATTERN.search(line)
        if m:
            gen = int(m.group('gen'))
            best = float(m.group('best'))
            avg = float(m.group('avg'))
            rows.append((gen, best, avg))
            continue
        # Try fallbacks
        for fb in FALLBACKS:
            mm = fb.search(line)
            if mm:
                # Normalize groups across fallbacks
                if fb.pattern.startswith('\\[GEN') and 'AvgTop' in fb.pattern:
                    gen = int(mm.group(1))
                    best = float(mm.group(2))
                    avg = float(mm.group(4))
                else:
                    gen = int(mm.group(1))
                    best = float(mm.group(2))
                    avg = float(mm.group(3))
                rows.append((gen, best, avg))
                break

    rows.sort(key=lambda r: r[0])
    # Deduplicate by generation, keeping the last seen entry
    dedup = {}
    for gen, best, avg in rows:
        dedup[gen] = (gen, best, avg)
    return list(dedup.values())


def plot_rewards(rows: List[Tuple[int, float, float]], title: str | None = None, out: Path | None = None) -> None:
    if not rows:
        print("No matching '[GEN ...] Best=... AvgTop...' lines found.", file=sys.stderr)
        sys.exit(2)

    gens = [r[0] for r in rows]
    bests = [r[1] for r in rows]
    avgs = [r[2] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(gens, bests, marker='o', label='Best')
    plt.plot(gens, avgs, marker='s', label='Avg Top 10%')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(title or 'GP/GA Rewards over Generations')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved figure to: {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description='Parse GP/GA logs and plot Best & Avg Top 10% vs Generation')
    ap.add_argument('-i', '--input', type=str, default='-', help='Input log file path (default: stdin)')
    ap.add_argument('-o', '--out', type=str, default=None, help='Output image path (PNG/SVG/PDF). If omitted, show interactively')
    ap.add_argument('--title', type=str, default=None, help='Custom chart title')
    args = ap.parse_args()

    # Read text
    if args.input == '-' or args.input is None:
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding='utf-8', errors='ignore')

    rows = parse_lines(text)
    out_path = Path(args.out) if args.out else None
    plot_rewards(rows, title=args.title, out=out_path)


if __name__ == '__main__':
    main()
