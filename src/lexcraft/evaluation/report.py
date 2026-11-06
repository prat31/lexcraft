"""Aggregate eval rows into a summary, a markdown report and an interactive HTML report."""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from lexcraft.evaluation.judge import PairVerdict
from lexcraft.evaluation.promptfoo import Row

DETERMINISTIC = [
    "facts_preserved",
    "placeholders",
    "negation_kept",
    "region_neutral",
    "no_preamble",
    "no_contractions",
    "legal_register",
]
SEMANTIC = ["similarity", "entail_fwd", "entail_bwd"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def summarize(
    rows: list[Row], verdicts: list[PairVerdict], primary: str, order: list[str] | None = None
) -> dict:
    """`order` fixes the system order in every table and chart (default: as rows arrive)."""
    rank = {name: i for i, name in enumerate(order or [])}

    def ordered(names) -> list[str]:
        return sorted(names, key=lambda n: rank.get(n, len(rank)))

    by_sys: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_sys[r.system].append(r)
    by_sys = {n: by_sys[n] for n in ordered(by_sys)}
    systems = {}
    for name, rs in by_sys.items():
        ok = [r for r in rs if not r.error]
        lat = [r.latency_ms for r in ok if r.latency_ms]
        metric_names = sorted({m for r in ok for m in r.metrics})
        systems[name] = {
            "n": len(rs),
            "errors": len(rs) - len(ok),
            "pass_rate": mean(r.passed for r in ok) if ok else 0.0,
            "latency_ms_median": median(lat) if lat else None,
            "metrics": {m: mean(r.metrics.get(m, 0.0) for r in ok) for m in metric_names},
        }
    pairwise: dict[str, dict] = {}
    by_other: dict[str, list[PairVerdict]] = defaultdict(list)
    for v in verdicts:
        by_other[v.other].append(v)
    by_other = {n: by_other[n] for n in ordered(by_other)}
    for other, vs in by_other.items():
        valid = [v for v in vs if v.winner != "error"]
        n = len(valid)
        win = sum(v.winner == "primary" for v in valid)
        tie = sum(v.winner == "tie" for v in valid)
        loss = sum(v.winner == "other" for v in valid)
        both_fail = sum(v.winner == "both_fail" for v in valid)
        lo, hi = wilson(win + tie, n)
        pairwise[other] = {
            "n": n,
            "errors": len(vs) - n,
            "win": win,
            "tie": tie,
            "loss": loss,
            "both_fail": both_fail,
            "win_rate": win / n if n else 0.0,
            "win_or_tie_rate": (win + tie) / n if n else 0.0,
            "win_or_tie_ci95": [lo, hi],
            "at_par_or_better": bool(n) and (win + tie) / n >= 0.5,
        }
    return {"primary": primary, "systems": systems, "pairwise": pairwise}


def write_markdown(summary: dict, meta: dict, path: Path) -> None:
    p = summary["primary"]
    lines = [
        "# lexcraft eval report",
        "",
        f"- Run: `{meta['run_id']}`  ",
        f"- Test items: {meta['n_tests']}  ",
        f"- Judge: `{meta['judge']}` ({meta['judge_family']})  ",
        f"- Primary system: `{p}`",
        "",
        "## Pairwise (judge, both orders)",
        "",
        "A rewrite that is not legal register, or changes the meaning, can never win;",
        "if both fail that gate the item counts as *both fail* (not a tie).",
        "",
        f"| {p} vs | n | win | tie | loss | both fail | win+tie | 95% CI | at par or better |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for other, s in summary["pairwise"].items():
        lo, hi = s["win_or_tie_ci95"]
        lines.append(
            f"| {other} | {s['n']} | {s['win']} | {s['tie']} | {s['loss']} | {s.get('both_fail', 0)} | "
            f"{s['win_or_tie_rate']:.0%} | {lo:.0%}–{hi:.0%} | {'yes' if s['at_par_or_better'] else 'no'} |"
        )
    metric_names = sorted({m for s in summary["systems"].values() for m in s["metrics"]})
    lines += [
        "",
        "## Metrics (mean, 0–1)",
        "",
        "| system | pass | " + " | ".join(metric_names) + " | median latency |",
        "|---|---|" + "---|" * len(metric_names) + "---|",
    ]
    for name, s in summary["systems"].items():
        lat = f"{s['latency_ms_median']:.0f} ms" if s["latency_ms_median"] else "–"
        vals = " | ".join(f"{s['metrics'].get(m, float('nan')):.2f}" for m in metric_names)
        lines.append(f"| {name} | {s['pass_rate']:.0%} | {vals} | {lat} |")
    path.write_text("\n".join(lines) + "\n")


def write_html(
    summary: dict,
    rows: list[Row],
    verdicts: list[PairVerdict],
    meta: dict,
    path: Path,
    train_metrics: list[dict] | None = None,
) -> bool:
    """Interactive Plotly report. Returns False if plotly is not installed."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    figs: list[str] = []
    # 1) pairwise stacked bars
    pw = summary["pairwise"]
    if pw:
        others = list(pw)
        fig = go.Figure()
        for key, label, color in (
            ("win", "win", "#2e7d32"),
            ("tie", "tie", "#9e9e9e"),
            ("loss", "loss", "#c62828"),
            ("both_fail", "both fail", "#6d4c41"),
        ):
            fig.add_bar(
                y=others,
                x=[pw[o][key] / max(pw[o]["n"], 1) for o in others],
                name=label,
                orientation="h",
                marker_color=color,
                text=[f"{pw[o][key]}" for o in others],
                textposition="inside",
            )
        fig.update_layout(
            barmode="stack",
            title=f"{summary['primary']} vs others (pairwise judge)",
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            height=120 + 60 * len(others),
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # 2) metric bars per system
    systems = list(summary["systems"])
    metric_names = [
        m
        for m in [*DETERMINISTIC, "judge_rubric", *SEMANTIC]
        if any(m in s["metrics"] for s in summary["systems"].values())
    ]
    if metric_names:
        fig = go.Figure()
        for sname in systems:
            fig.add_bar(
                x=metric_names,
                y=[summary["systems"][sname]["metrics"].get(m) for m in metric_names],
                name=sname,
            )
        fig.update_layout(
            barmode="group",
            title="Quality metrics (mean, higher is better)",
            yaxis=dict(range=[0, 1.05]),
            height=420,
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # 3) latency
    lat_sys = [s for s in systems if summary["systems"][s]["latency_ms_median"]]
    if lat_sys:
        fig = go.Figure()
        for s in lat_sys:
            fig.add_box(y=[r.latency_ms for r in rows if r.system == s and r.latency_ms], name=s)
        fig.update_layout(title="Latency per item (ms)", height=380, showlegend=False)
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # 4) training curve
    if train_metrics:
        tr = [m for m in train_metrics if m["kind"] == "train"]
        va = [m for m in train_metrics if m["kind"] == "val"]
        fig = make_subplots()
        fig.add_scatter(
            x=[m["iteration"] for m in tr], y=[m["train_loss"] for m in tr], name="train loss"
        )
        fig.add_scatter(
            x=[m["iteration"] for m in va],
            y=[m["val_loss"] for m in va],
            name="val loss",
            mode="lines+markers",
        )
        fig.update_layout(title="LoRA training loss", height=360, xaxis_title="iteration")
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # sample table
    by_test: dict[str, dict[str, str]] = defaultdict(dict)
    src: dict[str, str] = {}
    for r in rows:
        by_test[r.test_id][r.system] = r.output if not r.error else f"ERROR: {r.error}"
        src[r.test_id] = r.source
    verdict_map = {(v.test_id, v.other): v.winner for v in verdicts}
    head = "".join(f"<th>{html.escape(s)}</th>" for s in systems)
    body = []
    for tid, outs in list(by_test.items())[:200]:
        cells = []
        for s in systems:
            w = verdict_map.get((tid, s))
            badge = f" <span class='b {w}'>{w}</span>" if w else ""
            cells.append(f"<td>{html.escape(outs.get(s, ''))}{badge}</td>")
        body.append(
            f"<tr><td class='src'><b>{html.escape(tid)}</b><br>{html.escape(src[tid])}</td>{''.join(cells)}</tr>"
        )

    meta_html = "".join(
        f"<li><b>{html.escape(k)}</b>: {html.escape(str(v))}</li>" for k, v in meta.items()
    )
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>lexcraft eval · {html.escape(meta["run_id"])}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Inter,sans-serif;margin:32px auto;max-width:1200px;color:#1d1d1f;padding:0 16px}}
h1{{font-weight:650}} .muted{{color:#666}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #e3e3e3;padding:8px;vertical-align:top;text-align:left}} th{{background:#fafafa}}
td.src{{background:#fcfbf6;width:22%}} .b{{font-size:11px;padding:1px 6px;border-radius:8px;color:#fff}}
.b.primary{{background:#c62828}} .b.other{{background:#2e7d32}} .b.tie{{background:#9e9e9e}} .b.error{{background:#555}} .b.both_fail{{background:#6d4c41}}
</style></head><body>
<h1>⚖ lexcraft eval</h1><ul class="muted">{meta_html}</ul>
{"".join(f"<div>{f}</div>" for f in figs)}
<h2>Outputs</h2><p class="muted">Badges show the judge verdict for that system vs <b>{html.escape(summary["primary"])}</b>
(green = the other system won).</p>
<table><tr><th>input</th>{head}</tr>{"".join(body)}</table>
<script type="application/json" id="summary">{html.escape(json.dumps(summary))}</script>
</body></html>"""
    path.write_text(doc)
    return True
