"""Generate README images from a finished eval run.

    uv run python scripts/readme_assets.py eval/results/base-vs-lora
    uv run python scripts/readme_assets.py --demo   # CLI screenshot (runs the local model)

Writes SVGs to docs/assets/. Pure SVG: no browser or image libraries needed.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

INK, MUTED, GRID = "#1f2328", "#656d76", "#d0d7de"
WIN, TIE, LOSS = "#1a7f37", "#8c959f", "#cf222e"
SERIES = ["#c9a227", "#57606a", "#0969da", "#8250df"]
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"

METRICS = [
    ("judge_rubric", "Judge rubric"),
    ("legal_register", "Legal register"),
    ("facts_preserved", "Facts kept"),
    ("negation_kept", "Negations kept"),
    ("no_contractions", "No contractions"),
    ("entail_fwd", "Entailment →"),
    ("entail_bwd", "Entailment ←"),
    ("similarity", "Similarity"),
]


def _text(
    x: float,
    y: float,
    s: str,
    size: int = 13,
    color: str = INK,
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{color}' "
        f"text-anchor='{anchor}' font-weight='{weight}'>{html.escape(s)}</text>"
    )


def chart(summary: dict, meta: dict, out: Path) -> None:
    primary = summary["primary"]
    systems = list(summary["systems"])
    order = [primary] + [s for s in systems if s != primary]
    metrics = [
        (k, label)
        for k, label in METRICS
        if any(k in summary["systems"][s]["metrics"] for s in order)
    ]

    W, pad = 860, 32
    parts: list[str] = []
    y = 40
    parts.append(_text(pad, y, "lexcraft eval", 22, weight="650"))
    parts.append(
        _text(
            W - pad, y, f"{meta['n_tests']} test items · judge: {meta['judge']}", 13, MUTED, "end"
        )
    )
    y += 34

    # Pairwise bars
    for other, r in summary["pairwise"].items():
        parts.append(_text(pad, y, f"{primary} vs {other}", 14, weight="600"))
        y += 10
        bar_w, x = W - 2 * pad, pad
        n = max(r["n"], 1)
        for key, color, label in (("win", WIN, "win"), ("tie", TIE, "tie"), ("loss", LOSS, "loss")):
            w = bar_w * r[key] / n
            if w > 0:
                parts.append(
                    f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='28' fill='{color}' rx='3'/>"
                )
                if w > 60:
                    parts.append(
                        _text(x + w / 2, y + 19, f"{r[key]} {label}", 13, "#fff", "middle", "600")
                    )
            x += w
        lo, hi = r["win_or_tie_ci95"]
        y += 46
        parts.append(
            _text(
                pad,
                y,
                f"win+tie {r['win_or_tie_rate']:.0%}  (95% CI {lo:.0%}–{hi:.0%})"
                f"  ·  pairwise, both orders",
                12,
                MUTED,
            )
        )
        y += 30

    # Metric bars
    parts.append(_text(pad, y, "Quality metrics (mean, 0–1, higher is better)", 14, weight="600"))
    y += 18
    label_w, chart_w = 150, W - 2 * pad - 150
    row_h = 14 * len(order) + 10
    top = y
    for i in range(0, 11, 2):
        gx = pad + label_w + chart_w * i / 10
        parts.append(
            f"<line x1='{gx:.1f}' y1='{top}' x2='{gx:.1f}' y2='{top + row_h * len(metrics)}' "
            f"stroke='{GRID}' stroke-width='1'/>"
        )
        parts.append(
            _text(gx, top + row_h * len(metrics) + 16, f"{i / 10:.1f}", 11, MUTED, "middle")
        )
    for key, label in metrics:
        parts.append(_text(pad, y + row_h / 2 + 4, label, 12))
        for j, s in enumerate(order):
            v = summary["systems"][s]["metrics"].get(key)
            if v is None:
                continue
            by = y + 4 + j * 14
            parts.append(
                f"<rect x='{pad + label_w}' y='{by}' width='{chart_w * max(v, 0):.1f}' "
                f"height='11' fill='{SERIES[j % len(SERIES)]}' rx='2'/>"
            )
            parts.append(
                _text(pad + label_w + chart_w * max(v, 0) + 4, by + 10, f"{v:.2f}", 10, MUTED)
            )
        y += row_h
    y += 30
    lx = pad + label_w
    for j, s in enumerate(order):
        parts.append(
            f"<rect x='{lx}' y='{y - 10}' width='12' height='12' fill='{SERIES[j % len(SERIES)]}' rx='2'/>"
        )
        parts.append(_text(lx + 18, y, s, 12))
        lx += 30 + 8 * len(s)
    y += 24
    lat = "  ·  ".join(
        f"{s}: {summary['systems'][s]['latency_ms_median']:.0f} ms/item"
        for s in order
        if summary["systems"][s].get("latency_ms_median")
    )
    parts.append(_text(pad, y, f"Median latency  ·  {lat}", 12, MUTED))
    H = y + 24
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
        f"<rect width='100%' height='100%' fill='#ffffff' rx='8'/>{''.join(parts)}</svg>"
    )
    out.write_text(svg)


def demo(out: Path) -> None:
    """Record a real rewrite with Rich and export it as an SVG terminal screenshot."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text

    from lexcraft.backends import load_backend, plan_model
    from lexcraft.config import load_settings
    from lexcraft.engine import Rewriter

    sample = (ROOT / "docs" / "examples" / "house-rules.md").read_text()
    settings = load_settings(ROOT / "lexcraft.toml")
    choice = plan_model(settings)
    rw = Rewriter(load_backend(choice, settings), settings.generation)
    result = rw.rewrite(sample)

    con = Console(record=True, width=118, force_terminal=True, color_system="truecolor")
    con.print(
        Text.from_markup(
            "[bold #c9a227]⚖  lexcraft[/] [grey58]· rewrite  ·  "
            f"{choice.backend} · {choice.model} + LoRA[/]"
        )
    )
    con.print(
        Group(
            Panel(Text(sample.rstrip()), title="[grey58]input", border_style="grey35"),
            Panel(
                Text(result.text.rstrip()),
                title="[bold #c9a227]legal version",
                border_style="#c9a227",
            ),
        )
    )
    kept = sum(b.fell_back for b in result.blocks)
    con.print(
        Text.from_markup(
            f"[bold green]✓[/] {len(result.blocks)} blocks · {result.seconds:.1f}s · "
            f"formatting, links and emails preserved · {kept} block(s) kept original by the fact check"
        )
    )
    con.save_svg(str(out), title="lexcraft")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--demo", action="store_true", help="Render the CLI demo screenshot.")
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        data = json.loads((args.run_dir / "results.json").read_text())
        chart(data["summary"], data["meta"], ASSETS / "eval-base-vs-lora.svg")
        print("wrote", ASSETS / "eval-base-vs-lora.svg")
    if args.demo:
        demo(ASSETS / "cli-demo.svg")
        print("wrote", ASSETS / "cli-demo.svg")


if __name__ == "__main__":
    main()
