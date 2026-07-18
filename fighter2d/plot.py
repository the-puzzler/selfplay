"""Render a training-metrics dashboard PNG from a run's metrics.jsonl.

Usage:
  uv run python -m fighter2d.plot --run runs/diverse-cpu-v4
Writes <run>/dashboard.png (and <run>/dashboard.html once, an auto-refreshing
viewer for watching a live run).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#333333"
INK_MUTED = "#666666"
GRID = "#e6e6e6"
SURFACE = "#ffffff"
# Categorical slots (dataviz reference palette, light mode), one per metric.
C_EP_LEN = "#2a78d6"  # blue
C_TIMEOUT = "#008300"  # green
C_DRAW = "#e87ba4"  # magenta
C_VLOSS = "#eb6834"  # orange
C_ENTROPY = "#4a3aa7"  # violet

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>fighter2d training</title>
<style>body{margin:0;background:#fff;display:grid;place-items:center;min-height:100vh}
img{max-width:100vw;height:auto}</style></head>
<body><img id="dash" src="dashboard.png">
<script>setInterval(()=>{document.getElementById('dash').src='dashboard.png?t='+Date.now()},15000)</script>
</body></html>
"""


def _panel(ax, x, series, title, ylabel=""):
    """series: list of (values, color, label_or_None)."""
    for vals, color, label in series:
        ax.plot(x, vals, color=color, linewidth=2, label=label)
        ax.plot(x[-1], vals[-1], "o", color=color, markersize=6)
        ax.annotate(
            f" {vals[-1]:.3g}",
            (x[-1], vals[-1]),
            color=INK,
            fontsize=9,
            va="center",
        )
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.margins(x=0.02)
    if any(label for _, _, label in series):
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)


def render(run: Path):
    lines = [json.loads(l) for l in (run / "metrics.jsonl").open()]
    if not lines:
        return 0
    steps = [m["env_steps"] for m in lines]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)

    _panel(
        axes[0, 0], steps,
        [([m["mean_ep_len"] for m in lines], C_EP_LEN, None)],
        "Mean episode length", "steps (300 = timeout)",
    )
    _panel(
        axes[0, 1], steps,
        [
            ([m["timeout_rate"] for m in lines], C_TIMEOUT, "timeout"),
            ([m["draw_rate"] for m in lines], C_DRAW, "draw (incl. timeouts)"),
        ],
        "Episode outcomes", "fraction of episodes",
    )
    _panel(
        axes[1, 0], steps,
        [([m["v_loss"] for m in lines], C_VLOSS, None)],
        "Value loss", "",
    )
    _panel(
        axes[1, 1], steps,
        [([m["entropy"] for m in lines], C_ENTROPY, None)],
        "Policy entropy", "nats",
    )
    for ax in axes[1]:
        ax.set_xlabel("env steps", color=INK_MUTED, fontsize=9)

    last = lines[-1]
    fig.suptitle(
        f"{run.name} — iter {last['iter']:.0f}, {last['env_steps']:.2e} steps, "
        f"{last['sps']:.0f} steps/s — updated {time.strftime('%H:%M:%S')}",
        color=INK, fontsize=11, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(run / "dashboard.png", dpi=110)
    plt.close(fig)
    return len(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    args = p.parse_args()
    run = Path(args.run)
    html = run / "dashboard.html"
    if not html.exists():
        html.write_text(HTML)
    n = render(run)
    print(f"dashboard rendered from {n} iterations -> {run}/dashboard.png")


if __name__ == "__main__":
    main()
