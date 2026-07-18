"""Render a training-metrics dashboard from a run's metrics.jsonl.

Writes two artifacts into the run directory:
  dashboard.png  — static matplotlib figure (for commits/blog drafts)
  dashboard.html — self-contained interactive SVG dashboard (hover crosshair
                   + tooltip, light/dark aware, data table), auto-reloads
                   while a run is live.

Usage:
  uv run python -m fighter2d.plot --run runs/diverse-cpu-v4
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
# Categorical slots from the dataviz reference palette (pre-validated),
# light-mode steps; dark-mode steps of the same hues live in the HTML CSS.
C_EP_LEN = "#2a78d6"  # slot 1 blue
C_TIMEOUT = "#008300"  # slot 2 green
C_DRAW = "#e87ba4"  # slot 3 magenta
C_VLOSS = "#eb6834"  # slot 6 orange
C_ENTROPY = "#4a3aa7"  # slot 7 violet

PANELS = [
    # (key, title, unit, [(series_key, css_var, label), ...])
    ("ep_len", "Mean episode length", "steps (300 = timeout)", [("mean_ep_len", "s1", None)]),
    ("outcomes", "Episode outcomes", "fraction of episodes",
     [("timeout_rate", "s2", "timeout"), ("draw_rate", "s3", "draw (incl. timeouts)")]),
    ("v_loss", "Value loss", "", [("v_loss", "s6", None)]),
    ("entropy", "Policy entropy", "nats", [("entropy", "s7", None)]),
]


# ------------------------------------------------------------------ PNG

def render_png(run: Path, lines):
    steps = [m["env_steps"] for m in lines]
    colors = {"s1": C_EP_LEN, "s2": C_TIMEOUT, "s3": C_DRAW, "s6": C_VLOSS, "s7": C_ENTROPY}

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), facecolor=SURFACE)
    for ax, (_, title, unit, series) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for key, cvar, label in series:
            vals = [m[key] for m in lines]
            ax.plot(steps, vals, color=colors[cvar], linewidth=2, label=label)
            ax.plot(steps[-1], vals[-1], "o", color=colors[cvar], markersize=6)
            ax.annotate(f" {vals[-1]:.3g}", (steps[-1], vals[-1]), color=INK, fontsize=9, va="center")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylabel(unit, color=INK_MUTED, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        ax.margins(x=0.02)
        if any(lbl for _, _, lbl in series):
            ax.legend(frameon=False, fontsize=9, labelcolor=INK)
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


# ----------------------------------------------------------------- HTML

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="20">
<title>__TITLE__</title>
<style>
:root { color-scheme: light dark;
  --surface:#ffffff; --ink:#333333; --ink-muted:#666666; --grid:#e6e6e6;
  --s1:#2a78d6; --s2:#008300; --s3:#e87ba4; --s6:#eb6834; --s7:#4a3aa7; }
@media (prefers-color-scheme: dark) { :root {
  --surface:#1a1a19; --ink:#ffffff; --ink-muted:#c3c2b7; --grid:#3a3a38;
  --s1:#3987e5; --s2:#008300; --s3:#d55181; --s6:#d95926; --s7:#9085e9; } }
body { margin:0; padding:16px; background:var(--surface); color:var(--ink);
  font:14px/1.4 -apple-system, system-ui, sans-serif; }
h1 { font-size:15px; font-weight:600; margin:0 0 2px; }
.sub { color:var(--ink-muted); font-size:12px; margin-bottom:12px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }
.panel h2 { font-size:13px; font-weight:600; margin:0 0 2px; }
.panel .unit { color:var(--ink-muted); font-size:11px; }
svg { width:100%; height:auto; display:block; }
.gridline { stroke:var(--grid); stroke-width:1; }
.tick { fill:var(--ink-muted); font-size:10px; }
.endlab { fill:var(--ink); font-size:11px; }
.series { fill:none; stroke-width:2; }
.crosshair { stroke:var(--ink-muted); stroke-width:1; stroke-dasharray:3 3; opacity:0; }
#tooltip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--grid); border-radius:6px; padding:6px 9px; font-size:12px;
  opacity:0; box-shadow:0 2px 8px rgba(0,0,0,.15); z-index:9; white-space:nowrap; }
#tooltip .row { display:flex; gap:6px; align-items:center; }
#tooltip .chip { width:8px; height:8px; border-radius:2px; display:inline-block; }
.legend { display:flex; gap:14px; font-size:11px; color:var(--ink); margin:2px 0 0 44px; }
.legend .chip { width:9px; height:9px; border-radius:2px; display:inline-block; margin-right:4px; }
details { margin-top:18px; color:var(--ink-muted); font-size:12px; }
table { border-collapse:collapse; margin-top:6px; }
td,th { padding:3px 10px; text-align:right; border-bottom:1px solid var(--grid); color:var(--ink); font-size:11px; }
th { color:var(--ink-muted); font-weight:500; }
</style></head><body>
<h1>__TITLE__</h1>
<div class="sub">__SUBTITLE__</div>
<div class="grid2">__PANELS__</div>
<div id="tooltip"></div>
<details><summary>Data table (last 20 iterations)</summary>__TABLE__</details>
<script>
const D = __DATA__;
const tip = document.getElementById('tooltip');
const fmt = v => Math.abs(v) >= 1000 ? v.toExponential(2) : (+v.toFixed(3)).toString();
document.querySelectorAll('svg[data-panel]').forEach(svg => {
  const meta = JSON.parse(svg.dataset.panel);
  const ch = svg.querySelector('.crosshair');
  svg.addEventListener('mousemove', e => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const fx = (pt.x - meta.x0) / (meta.x1 - meta.x0);
    if (fx < 0 || fx > 1) { ch.style.opacity = 0; tip.style.opacity = 0; return; }
    const i = Math.round(fx * (D.steps.length - 1));
    const px = meta.x0 + (D.steps[i] - D.steps[0]) / (D.steps[D.steps.length-1] - D.steps[0] || 1) * (meta.x1 - meta.x0);
    ch.setAttribute('x1', px); ch.setAttribute('x2', px); ch.style.opacity = 1;
    tip.innerHTML = `<div class="row"><b>iter ${D.iter[i]}</b>&nbsp;·&nbsp;${fmt(D.steps[i])} steps</div>` +
      meta.series.map(s => `<div class="row"><span class="chip" style="background:var(--${s.v})"></span>${s.n}: <b>${fmt(D[s.k][i])}</b></div>`).join('');
    tip.style.opacity = 1;
    tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 10) + 'px';
  });
  svg.addEventListener('mouseleave', () => { ch.style.opacity = 0; tip.style.opacity = 0; });
});
</script>
</body></html>
"""

W, H = 520, 260
ML, MR, MT, MB = 46, 84, 10, 24


def _nice_ticks(lo, hi, n=4):
    import math
    if hi <= lo:
        hi = lo + 1e-9
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min(s * mag for s in (1, 2, 5, 10) if s * mag >= raw)
    t0 = math.ceil(lo / step) * step
    ticks = []
    t = t0
    while t <= hi + 1e-12:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _fmt(v):
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.3g}"


def _svg_panel(lines, title, unit, series):
    steps = [m["env_steps"] for m in lines]
    x0s, x1s = steps[0], steps[-1] or 1
    allv = [m[k] for k, _, _ in series for m in lines]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1
    lo, hi = lo - pad, hi + pad
    X = lambda s: ML + (s - x0s) / (x1s - x0s or 1) * (W - ML - MR)
    Y = lambda v: MT + (hi - v) / (hi - lo) * (H - MT - MB)

    parts = []
    for t in _nice_ticks(lo, hi):
        y = Y(t)
        parts.append(f'<line class="gridline" x1="{ML}" y1="{y:.1f}" x2="{W-MR}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{ML-6}" y="{y+3:.1f}" text-anchor="end">{_fmt(t)}</text>')
    for t in _nice_ticks(x0s, x1s, 5):
        parts.append(f'<text class="tick" x="{X(t):.1f}" y="{H-8}" text-anchor="middle">{_fmt(t)}</text>')
    for k, cvar, label in series:
        pts = " ".join(f"{X(s):.1f},{Y(m[k]):.1f}" for s, m in zip(steps, lines))
        parts.append(f'<polyline class="series" style="stroke:var(--{cvar})" points="{pts}"/>')
        lx, ly = X(steps[-1]), Y(lines[-1][k])
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" style="fill:var(--{cvar})"/>')
        end = f"{label + ' ' if label else ''}{lines[-1][k]:.3g}"
        parts.append(f'<text class="endlab" x="{lx+7:.1f}" y="{ly+4:.1f}">{end}</text>')
    parts.append(f'<line class="crosshair" y1="{MT}" y2="{H-MB}" x1="0" x2="0"/>')

    meta = json.dumps({
        "x0": ML, "x1": W - MR,
        "series": [{"k": k, "v": cvar, "n": label or title} for k, cvar, label in series],
    })
    legend = ""
    if sum(1 for _ in series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span><span class="chip" style="background:var(--{cvar})"></span>{label}</span>'
            for _, cvar, label in series) + "</div>"
    return (
        f'<div class="panel"><h2>{title}</h2><div class="unit">{unit}&nbsp;</div>'
        f"<svg viewBox=\"0 0 {W} {H}\" data-panel='{meta}'>{''.join(parts)}</svg>{legend}</div>"
    )


def render_html(run: Path, lines, note=""):
    last = lines[-1]
    panels = "".join(_svg_panel(lines, t, u, s) for _, t, u, s in PANELS)
    cols = ["iter", "env_steps", "mean_ep_len", "timeout_rate", "draw_rate", "v_loss", "entropy"]
    table = "<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{m[c]:.4g}</td>" for c in cols) + "</tr>" for m in lines[-20:]
    ) + "</table>"
    data = json.dumps({
        "steps": [m["env_steps"] for m in lines],
        "iter": [m["iter"] for m in lines],
        **{k: [m[k] for m in lines] for k in
           ("mean_ep_len", "timeout_rate", "draw_rate", "v_loss", "entropy")},
    })
    html = (
        HTML_TEMPLATE
        .replace("__TITLE__", f"fighter2d — {run.name}")
        .replace("__SUBTITLE__",
                 f"iter {last['iter']:.0f} · {last['env_steps']:.2e} env steps · "
                 f"{last['sps']:.0f} steps/s · updated {time.strftime('%H:%M:%S')} · "
                 "auto-reloads every 20s while the run is live" + note)
        .replace("__PANELS__", panels)
        .replace("__TABLE__", table)
        .replace("__DATA__", data)
    )
    (run / "dashboard.html").write_text(html)


def adjust_for_spawn_dead(lines, f_either, f_both):
    """Correct aggregate metrics to exclude spawn-dead episodes (which last
    1 step and are decided at spawn). Only for runs whose reset distribution
    allowed them; newer runs exclude these at the source."""
    out = []
    for m in lines:
        m = dict(m)
        m["mean_ep_len"] = max((m["mean_ep_len"] - f_either) / (1 - f_either), 0.0)
        m["timeout_rate"] = m["timeout_rate"] / (1 - f_either)
        m["draw_rate"] = max((m["draw_rate"] - f_both) / (1 - f_either), 0.0)
        out.append(m)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--doa-either", type=float, default=0.0,
                   help="measured fraction of episodes with a spawn-dead fighter")
    p.add_argument("--doa-both", type=float, default=0.0)
    args = p.parse_args()
    run = Path(args.run)
    lines = [json.loads(l) for l in (run / "metrics.jsonl").open()]
    if not lines:
        print("no metrics yet")
        return
    note = ""
    if args.doa_either > 0:
        lines = adjust_for_spawn_dead(lines, args.doa_either, args.doa_both)
        note = (f" · ep_len/draw/timeout adjusted to exclude spawn-dead episodes "
                f"(measured {args.doa_either:.0%} of spawns)")
    render_png(run, lines)
    render_html(run, lines, note)
    print(f"dashboard ({len(lines)} iters) -> {run}/dashboard.png + dashboard.html")


if __name__ == "__main__":
    main()
