"""lexcraft command-line interface."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from lexcraft import __version__

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from lexcraft.config import Settings, load_settings
from lexcraft.ui import console, err_console, fail, header, kv_table, panel

app = typer.Typer(
    name="lexcraft",
    help="Rewrite plain text into region-agnostic legal language with a small local LLM.",
    no_args_is_help=False,
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
data_app = typer.Typer(help="Build and inspect the training dataset.", no_args_is_help=True)
eval_app = typer.Typer(help="Compare models with Promptfoo + an LLM judge.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to lexcraft.toml.", show_default=False)
]
BackendOpt = Annotated[
    str | None, typer.Option("--backend", "-b", help="auto | mlx | llamacpp | remote")
]
ModelOpt = Annotated[
    str | None, typer.Option("--model", "-m", help="Model path / HF id (overrides config).")
]
AdapterOpt = Annotated[str | None, typer.Option("--adapter", help="LoRA adapter directory.")]
NoAdapterOpt = Annotated[
    bool, typer.Option("--no-adapter", help="Use the base model without the LoRA.")
]


def _settings(config: Path | None) -> Settings:
    try:
        return load_settings(config)
    except Exception as e:
        fail(f"Could not load config: {e}")
        raise typer.Exit(2) from e


def _version(value: bool) -> None:
    if value:
        console.print(f"lexcraft {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """[bold]⚖ lexcraft[/bold] — plain text in, legal language out.

    Run with no command to paste text interactively.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(rewrite)


# ─────────────────────────────── rewrite ───────────────────────────────


def _read_input(file: Path | None) -> str:
    if file and str(file) != "-":
        if not file.exists():
            fail(f"File not found: {file}")
            raise typer.Exit(2)
        return file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    eof = "Ctrl-Z then Enter" if os.name == "nt" else "Ctrl-D"
    err_console.print(
        panel(
            Text.from_markup(
                f"Paste or type your text. Markdown is kept as-is.\n"
                f"Finish with [key]{eof}[/key] on an empty line."
            ),
            "input",
        )
    )
    try:
        return sys.stdin.read()
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


def _copy_to_clipboard(text: str) -> bool:
    for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["clip"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, input=text, text=True, check=False)
            return True
    return False


@app.command()
def rewrite(
    file: Annotated[
        Path | None, typer.Argument(help="Input file ('-' for stdin). Omit to paste.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write result to file.")
    ] = None,
    config: ConfigOpt = None,
    backend: BackendOpt = None,
    model: ModelOpt = None,
    adapter: AdapterOpt = None,
    no_adapter: NoAdapterOpt = False,
    no_verify: Annotated[
        bool, typer.Option("--no-verify", help="Skip fact checks / retries / fallback.")
    ] = False,
    copy: Annotated[bool, typer.Option("--copy", help="Copy the result to the clipboard.")] = False,
    show_blocks: Annotated[
        bool, typer.Option("--blocks", help="Show a per-block before/after table.")
    ] = False,
) -> None:
    """Rewrite text into legal language, keeping formatting, meaning and structure."""
    from lexcraft.backends import load_backend, plan_model
    from lexcraft.engine import BlockResult, Rewriter
    from lexcraft.hardware import detect

    settings = _settings(config)
    if no_verify:
        settings.generation.verify = False
    text = _read_input(file)
    if not text.strip():
        fail("No input text.")
        raise typer.Exit(1)

    hw = detect()
    try:
        choice = plan_model(settings, backend, model, adapter, no_adapter, hw)
    except RuntimeError as e:
        fail(str(e), "Run `lexcraft doctor` to see what is available on this machine.")
        raise typer.Exit(2) from e

    header("rewrite")
    err_console.print(
        kv_table(
            [
                ("backend", f"{choice.backend}  [muted]({hw.gpu_name or hw.cpu})[/muted]"),
                ("model", choice.model),
                ("adapter", choice.adapter or "—"),
                ("mode", choice.note),
            ]
        )
    )
    if "not trained" in choice.note:
        err_console.print(
            "[warn]![/warn] LoRA adapter not found — output quality will be lower. "
            "[muted]Train with `lexcraft train`.[/muted]"
        )

    with err_console.status("[brand]Loading model…[/brand]", spinner="dots"):
        t0 = time.perf_counter()
        try:
            be = load_backend(choice, settings)
        except Exception as e:
            fail(f"Could not load model: {e}")
            raise typer.Exit(1) from e
        load_s = time.perf_counter() - t0
    err_console.print(f"[ok]✓[/ok] model loaded [muted]in {load_s:.1f}s[/muted]")

    rw = Rewriter(be, settings.generation)
    total = rw.count_blocks(text)
    progress = Progress(
        SpinnerColumn(style="brand"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None, complete_style="brand"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[muted]{task.fields[preview]}"),
        console=err_console,
        transient=True,
    )

    def on_block(n: int, _total: int, b: BlockResult) -> None:
        status = "⚠" if b.fell_back or b.check.problems else "✓"
        preview = (b.output[:48] + "…") if len(b.output) > 48 else b.output
        progress.update(task, advance=1, preview=f"{status} {preview}")

    with progress:
        task = progress.add_task("Rewriting", total=max(total, 1), preview="")
        result = rw.rewrite(text, on_block)
    be.close()

    if output:
        output.write_text(result.text, encoding="utf-8")
    if not sys.stdout.isatty():
        sys.stdout.write(result.text)
        if not result.text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        console.print(panel(Text(result.text), "legal version"))

    if show_blocks:
        t = Table(title="blocks", show_lines=True, expand=True)
        t.add_column("#", style="muted", width=3)
        t.add_column("kind", style="muted", width=9)
        t.add_column("original")
        t.add_column("rewritten")
        t.add_column("checks", width=14)
        for b in result.blocks:
            mark = (
                "[warn]kept original[/warn]"
                if b.fell_back
                else (
                    "[warn]" + "; ".join(b.check.problems) + "[/warn]"
                    if b.check.problems
                    else "[ok]ok[/ok]"
                )
            )
            t.add_row(str(b.index + 1), b.kind, b.source, b.output, mark)
        err_console.print(t)

    for w in result.warnings:
        err_console.print(f"[warn]![/warn] {w}")
    words = len(result.text.split())
    err_console.print(
        f"[ok]✓[/ok] {len(result.blocks)} blocks · {words} words · "
        f"{result.seconds:.1f}s [muted]({words / max(result.seconds, 1e-6):.0f} words/s)[/muted]"
        + (f" · saved to [key]{output}[/key]" if output else "")
    )
    if copy:
        if _copy_to_clipboard(result.text):
            err_console.print("[ok]✓[/ok] copied to clipboard")
        else:
            err_console.print("[warn]![/warn] no clipboard tool found")


# ─────────────────────────────── doctor ────────────────────────────────


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Show hardware, available backends and which model files are ready."""
    from lexcraft.backends import plan_model
    from lexcraft.hardware import detect

    settings = _settings(config)
    hw = detect()
    header("doctor")
    rows = [
        ("os", f"{hw.os} ({hw.arch})"),
        ("cpu", hw.cpu),
        ("memory", f"{hw.ram_gb} GB"),
        ("gpu", (hw.gpu_name or "—") + (f" · {hw.vram_gb} GB" if hw.vram_gb else "")),
        ("backends", ", ".join(hw.available_backends)),
        ("recommended", hw.recommended_backend),
    ]
    console.print(panel(kv_table(rows), "hardware"))

    def state(p: Path | None, marker: str | None = None) -> str:
        if p is None:
            return "[muted]not configured[/muted]"
        ok = (p / marker).exists() if marker else p.exists()
        return f"[ok]ready[/ok]  [muted]{p}[/muted]" if ok else f"[muted]missing  {p}[/muted]"

    data_dir = settings.path(settings.train.data_dir)
    assert data_dir
    files = [
        ("config", str(settings.root / "lexcraft.toml")),
        ("base model", settings.model.base),
        ("MLX base", settings.model.mlx_base or "—"),
        ("dataset", state(data_dir, "train.jsonl")),
        ("LoRA adapter", state(settings.path(settings.model.adapter), "adapters.safetensors")),
        ("fused MLX", state(settings.path(settings.model.fused_mlx), "config.json")),
        ("GGUF", state(settings.path(settings.model.gguf))),
        (
            "node / npx",
            "[ok]found[/ok]" if shutil.which("npx") else "[warn]missing (needed for eval)[/warn]",
        ),
        (
            "OPENCODE_API_KEY",
            "[ok]set[/ok]" if os.environ.get("OPENCODE_API_KEY") else "[muted]not set[/muted]",
        ),
    ]
    console.print(panel(kv_table(files), "project"))
    try:
        choice = plan_model(settings, hw=hw)
        console.print(
            f"[ok]✓[/ok] `lexcraft rewrite` will use [key]{choice.backend}[/key] · "
            f"{choice.model} [muted]({choice.note})[/muted]"
        )
    except RuntimeError as e:
        console.print(f"[warn]![/warn] {e}")
    for n in hw.notes:
        console.print(f"[warn]![/warn] {n}")


# ──────────────────────────────── data ─────────────────────────────────


@data_app.command("build")
def data_build(
    seed_dir: Annotated[Path, typer.Option(help="Directory of seed *.jsonl pair files.")] = Path(
        "data/seed"
    ),
    out: Annotated[Path | None, typer.Option(help="Output dir (default: train.data_dir).")] = None,
    strict: Annotated[bool, typer.Option(help="Drop pairs that fail fact checks.")] = True,
    config: ConfigOpt = None,
) -> None:
    """Validate, dedupe and split seed pairs into train / valid / test."""
    from lexcraft.data import build

    settings = _settings(config)
    seed = seed_dir if seed_dir.is_absolute() else settings.root / seed_dir
    out_dir = out or settings.path(settings.train.data_dir)
    assert out_dir
    header("data build")
    with err_console.status("Validating and splitting…"):
        rep = build(seed, out_dir, seed=settings.train.seed, strict=strict)

    summary = kv_table(
        [
            ("loaded", str(rep.loaded)),
            ("schema errors", str(len(rep.invalid_schema))),
            ("failed checks", str(len(rep.failed_checks)) + ("" if strict else " (kept)")),
            ("duplicates", f"{rep.duplicates} exact · {rep.near_duplicates} near"),
            ("kept", f"[ok]{rep.kept}[/ok]"),
            ("splits", " · ".join(f"{k} {v}" for k, v in rep.splits.items())),
            ("legal markers", f"{rep.avg_marker_density:.1f} per 100 words"),
            ("contractions", str(rep.contractions_in_legal)),
        ]
    )
    console.print(panel(summary, "dataset"))
    t = Table(title="by category", header_style="key", box=None)
    t.add_column("category")
    for s in ("train", "valid", "test"):
        t.add_column(s, justify="right")
    for cat, counts in rep.by_category.items():
        t.add_row(cat, *(str(counts.get(s, 0)) for s in ("train", "valid", "test")))
    console.print(t)
    for msg in rep.invalid_schema[:10]:
        console.print(f"[err]schema[/err] {msg}")
    for pid, probs in rep.failed_checks[:10]:
        console.print(f"[warn]check[/warn] {pid}: {'; '.join(probs)}")
    console.print(f"[ok]✓[/ok] wrote {out_dir}")


@data_app.command("stats")
def data_stats(config: ConfigOpt = None) -> None:
    """Show stats of the built dataset."""
    settings = _settings(config)
    d = settings.path(settings.train.data_dir)
    assert d
    f = d / "stats.json"
    if not f.exists():
        fail("No dataset built yet.", "Run `lexcraft data build`.")
        raise typer.Exit(1)
    s = json.loads(f.read_text())
    console.print(
        panel(
            kv_table(
                [
                    (k, json.dumps(v))
                    for k, v in s.items()
                    if k not in ("failed_checks", "invalid_schema")
                ]
            ),
            "dataset stats",
        )
    )


@data_app.command("sample")
def data_sample(
    n: Annotated[int, typer.Argument(help="How many pairs to show.")] = 5,
    split: Annotated[str, typer.Option(help="train | valid | test")] = "train",
    config: ConfigOpt = None,
) -> None:
    """Print random pairs from a split."""
    import random

    settings = _settings(config)
    d = settings.path(settings.train.data_dir)
    assert d
    f = d / f"{split}_pairs.jsonl"
    if not f.exists():
        fail(f"{f} not found.", "Run `lexcraft data build`.")
        raise typer.Exit(1)
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    for r in random.sample(rows, min(n, len(rows))):
        console.print(
            panel(
                Text.assemble(
                    ("plain  ", "muted"), r["plain"], "\n", ("legal  ", "brand"), r["legal"]
                ),
                f"{r['id']} · {r['category']}",
            )
        )


# ──────────────────────────────── train ────────────────────────────────


@app.command()
def train(
    config: ConfigOpt = None,
    backend: Annotated[str, typer.Option(help="auto | mlx | cuda")] = "auto",
    iters: Annotated[int | None, typer.Option(help="Training iterations.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Batch size.")] = None,
    learning_rate: Annotated[float | None, typer.Option("--lr", help="Peak learning rate.")] = None,
    rank: Annotated[int | None, typer.Option(help="LoRA rank.")] = None,
    adapter_path: Annotated[Path | None, typer.Option(help="Where to save the adapter.")] = None,
) -> None:
    """Train the LoRA adapter on the built dataset (MLX on Apple Silicon, PEFT on CUDA)."""
    from lexcraft.hardware import detect
    from lexcraft.train import train_cuda, train_mlx

    settings = _settings(config)
    cfg = settings.train.model_copy()
    for k, v in (
        ("iters", iters),
        ("batch_size", batch_size),
        ("learning_rate", learning_rate),
        ("lora_rank", rank),
    ):
        if v is not None:
            setattr(cfg, k, v)
    adapter = adapter_path or settings.path(settings.model.adapter)
    assert adapter
    hw = detect()
    if backend == "auto":
        backend = "mlx" if "mlx" in hw.available_backends else ("cuda" if hw.cuda else "none")
    if backend not in ("mlx", "cuda"):
        fail(
            "No training backend: needs Apple Silicon (MLX) or an NVIDIA GPU (pip install 'lexcraft[cuda]')."
        )
        raise typer.Exit(2)

    header("train")
    console.print(
        panel(
            kv_table(
                [
                    (
                        "backend",
                        f"{backend}  [muted]({hw.gpu_name or hw.cpu}, {hw.ram_gb} GB)[/muted]",
                    ),
                    (
                        "base model",
                        (settings.model.mlx_base or settings.model.base)
                        if backend == "mlx"
                        else settings.model.base,
                    ),
                    ("data", str(settings.path(cfg.data_dir))),
                    ("adapter →", str(adapter)),
                    (
                        "iters",
                        f"{cfg.iters}  · batch {cfg.batch_size} · lr {cfg.learning_rate:g} (cosine, warmup)",
                    ),
                    (
                        "LoRA",
                        f"rank {cfg.lora_rank} · alpha {cfg.lora_alpha:g} · dropout {cfg.lora_dropout} · "
                        f"layers {'all' if cfg.num_layers < 0 else cfg.num_layers}",
                    ),
                ]
            ),
            "training run",
        )
    )

    def on_event(kind: str, row: dict) -> None:
        if kind == "val":
            err_console.print(
                f"  [brand]◆ val[/brand] iter {int(row['iteration'])}: "
                f"[bold]{row['val_loss']:.3f}[/bold]"
            )

    fn = train_mlx if backend == "mlx" else train_cuda
    try:
        summary = fn(settings, cfg, adapter, on_event)
    except FileNotFoundError as e:
        fail(str(e))
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        fail("Training interrupted. Last checkpoint is in the adapter directory.")
        raise typer.Exit(130) from None

    mins = summary.seconds / 60
    console.print(
        panel(
            kv_table(
                [
                    ("time", f"{mins:.1f} min"),
                    (
                        "final train loss",
                        f"{summary.final_train_loss:.3f}" if summary.final_train_loss else "—",
                    ),
                    (
                        "best val loss",
                        f"{summary.best_val_loss:.3f}" if summary.best_val_loss else "—",
                    ),
                    (
                        "peak memory",
                        f"{summary.peak_memory_gb:.1f} GB" if summary.peak_memory_gb else "—",
                    ),
                    ("best checkpoint", summary.best_checkpoint or "last"),
                    ("early stop", "yes" if summary.stopped_early else "no"),
                    ("adapter", summary.adapter_path),
                ]
            ),
            "done",
            "ok",
        )
    )
    console.print(
        "[muted]next:[/muted] `lexcraft rewrite`  ·  `lexcraft export`  ·  `lexcraft eval run`"
    )


# ──────────────────────────────── export ───────────────────────────────


@app.command()
def export(
    config: ConfigOpt = None,
    gguf: Annotated[
        bool, typer.Option(help="Also build a GGUF for llama.cpp (needs llama.cpp tools).")
    ] = True,
    q_bits: Annotated[
        int, typer.Option(help="MLX quantization bits. 8 keeps the LoRA effect; 4 can wash it out.")
    ] = 8,
) -> None:
    """Fuse the LoRA into the base model and export MLX (+ GGUF) builds."""
    from lexcraft.export import ExportStep, run_steps
    from lexcraft.export import plan as export_plan

    settings = _settings(config)
    adapter = settings.path(settings.model.adapter)
    mlx_out = settings.path(settings.model.fused_mlx)
    gguf_out = settings.path(settings.model.gguf) if gguf else None
    assert adapter and mlx_out
    if not (adapter / "adapters.safetensors").exists():
        fail(f"No adapter at {adapter}.", "Run `lexcraft train` first.")
        raise typer.Exit(1)
    fused = mlx_out.parent / (
        mlx_out.name.replace("-mlx-8bit", "").replace("-mlx-4bit", "") + "-fused-bf16"
    )
    steps, warnings = export_plan(
        settings.model.mlx_base or settings.model.base, adapter, fused, mlx_out, gguf_out, q_bits
    )
    header("export")
    for w in warnings:
        err_console.print(f"[warn]![/warn] {w}")
    progress = Progress(
        SpinnerColumn(style="brand"),
        TextColumn("{task.description}"),
        BarColumn(complete_style="brand"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err_console,
    )
    with progress:
        task = progress.add_task("export", total=len(steps))

        def on_step(i: int, step: ExportStep) -> None:
            progress.update(task, completed=i - 1, description=step.name)

        try:
            run_steps(steps, on_step, log=mlx_out.parent / "export.log")
        except RuntimeError as e:
            fail(str(e))
            raise typer.Exit(1) from e
        progress.update(task, completed=len(steps), description="done")
    for s in steps:
        console.print(f"[ok]✓[/ok] {s.name}: [key]{s.output}[/key]")
    if gguf_out and gguf_out.exists():
        from lexcraft.export import write_modelfile

        mf = write_modelfile(gguf_out, gguf_out.parent / "Modelfile")
        console.print(f"[ok]✓[/ok] Ollama Modelfile: [key]{mf}[/key]")
        console.print(
            f"[muted]next:[/muted] cd {gguf_out.parent} && ollama create lexcraft -f Modelfile"
        )


# ──────────────────────────────── serve ────────────────────────────────


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port.")] = 8765,
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    name: Annotated[str, typer.Option(help="Model name reported by the API.")] = "lexcraft",
    config: ConfigOpt = None,
    backend: BackendOpt = None,
    model: ModelOpt = None,
    adapter: AdapterOpt = None,
    no_adapter: NoAdapterOpt = False,
) -> None:
    """Serve the local model over an OpenAI-compatible API (used by eval)."""
    from lexcraft.backends import load_backend, plan_model
    from lexcraft.server import serve as run_server

    settings = _settings(config)
    choice = plan_model(settings, backend, model, adapter, no_adapter)
    header(f"serve · {name}")
    err_console.print(
        kv_table(
            [("model", choice.model), ("adapter", choice.adapter or "—"), ("mode", choice.note)]
        )
    )
    with err_console.status("Loading model…"):
        be = load_backend(choice, settings)
    err_console.print(
        f"[ok]✓[/ok] listening on http://{host}:{port}/v1  [muted](Ctrl-C to stop)[/muted]"
    )
    with contextlib.suppress(KeyboardInterrupt):
        run_server(be, name, host, port)


# ──────────────────────────────── eval ─────────────────────────────────


def _split(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


@eval_app.command("models")
def eval_models(
    models: Annotated[Path, typer.Option(help="Eval model roles file.")] = Path("eval/models.yaml"),
    config: ConfigOpt = None,
) -> None:
    """List configured systems and judges, and check the model-family guard."""
    from lexcraft.evaluation.models import ModelsConfig, family_conflicts

    settings = _settings(config)
    path = models if models.is_absolute() else settings.root / models
    cfg = ModelsConfig.load(path)
    t = Table(header_style="key", box=None)
    for c in ("role", "name", "family", "where"):
        t.add_column(c)
    for s in cfg.candidates:
        t.add_row(
            "candidate" + (" ★" if s.name == cfg.primary else ""),
            s.name,
            s.family,
            "local" + (" (no LoRA)" if s.no_adapter else ""),
        )
    for s in cfg.baselines:
        t.add_row(
            "baseline" + (" ★" if s.name == cfg.primary else ""),
            s.name,
            s.family,
            f"{s.endpoint}:{s.model}",
        )
    t.add_row(
        "[brand]judge[/brand]",
        cfg.judge.name,
        cfg.judge.family,
        f"{cfg.judge.endpoint}:{cfg.judge.model}",
    )
    for j in cfg.alt_judges:
        t.add_row("[muted]alt judge[/muted]", j.name, j.family, f"{j.endpoint}:{j.model}")
    console.print(panel(t, "eval models"))
    clash = family_conflicts(cfg.all_systems, cfg.judge)
    if clash:
        console.print(f"[warn]![/warn] judge family clashes with: {', '.join(clash)}")
    else:
        console.print("[ok]✓[/ok] judge family differs from every compared system")


@eval_app.command("run")
def eval_run(
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Number of test items (stratified).")
    ] = None,
    systems: Annotated[
        str | None, typer.Option(help="Comma-separated system names to compare.")
    ] = None,
    judge: Annotated[
        str | None, typer.Option(help="Judge name or model id (same endpoint).")
    ] = None,
    add: Annotated[
        list[str] | None, typer.Option("--add", help="Ad-hoc baseline endpoint:model (repeatable).")
    ] = None,
    models: Annotated[Path, typer.Option(help="Eval model roles file.")] = Path("eval/models.yaml"),
    test_file: Annotated[Path | None, typer.Option(help="Test pairs JSONL.")] = None,
    out: Annotated[Path, typer.Option(help="Runs directory.")] = Path("eval/runs"),
    rubric: Annotated[bool, typer.Option(help="Promptfoo llm-rubric scoring by the judge.")] = True,
    pairwise: Annotated[bool, typer.Option(help="Pairwise judge vs the primary system.")] = True,
    semantic: Annotated[bool, typer.Option(help="Embedding + NLI meaning checks.")] = True,
    allow_family_overlap: Annotated[
        bool, typer.Option(help="Allow judge to share a family with a system.")
    ] = False,
    open_report: Annotated[
        bool, typer.Option("--open", help="Open the HTML report when done.")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Run the eval matrix: our LoRA vs base vs frontier models, judged blind."""
    from lexcraft.evaluation.judge import PairVerdict
    from lexcraft.evaluation.runner import RunOptions, run_eval

    settings = _settings(config)
    root = settings.root
    data_dir = settings.path(settings.train.data_dir)
    assert data_dir
    adapter = settings.path(settings.model.adapter)
    opts = RunOptions(
        models_path=models if models.is_absolute() else root / models,
        test_path=test_file or data_dir / "test_pairs.jsonl",
        out_root=out if out.is_absolute() else root / out,
        assertion_file=root / "eval" / "assertions" / "lexcraft_checks.py",
        limit=limit,
        systems=_split(systems),
        judge=judge,
        extra=add,
        rubric=rubric,
        pairwise=pairwise,
        semantic=semantic,
        allow_family_overlap=allow_family_overlap,
        train_metrics=(adapter / "metrics.jsonl") if adapter else None,
        config_path=(root / "lexcraft.toml") if (root / "lexcraft.toml").exists() else None,
    )
    header("eval")

    def log(msg: str) -> None:
        err_console.print(f"[brand]›[/brand] {msg}")

    jprog = Progress(
        SpinnerColumn(style="brand"),
        TextColumn("[bold]judging"),
        BarColumn(bar_width=None, complete_style="brand"),
        MofNCompleteColumn(),
        TextColumn("{task.fields[tally]}"),
        TimeElapsedColumn(),
        console=err_console,
        transient=False,
    )
    tally = {"primary": 0, "tie": 0, "other": 0, "both_fail": 0, "error": 0}
    state: dict = {}

    def on_judge(v: PairVerdict, total: int) -> None:
        if "task" not in state:
            jprog.start()
            state["task"] = jprog.add_task("judge", total=total, tally="")
        tally[v.winner] = tally.get(v.winner, 0) + 1
        jprog.update(
            state["task"],
            advance=1,
            tally=f"[ok]{tally['primary']} win[/ok] · [muted]{tally['tie']} tie[/muted] · "
            f"[err]{tally['other']} loss[/err]"
            + (f" · [warn]{tally['both_fail']} both fail[/warn]" if tally["both_fail"] else "")
            + (f" · [warn]{tally['error']} err[/warn]" if tally["error"] else ""),
        )

    try:
        run_dir = run_eval(opts, log, on_judge)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        fail(str(e))
        raise typer.Exit(1) from e
    finally:
        if "task" in state:
            jprog.stop()

    _print_summary(run_dir)
    if open_report and (run_dir / "report.html").exists():
        typer.launch(str(run_dir / "report.html"))


def _print_summary(run_dir: Path) -> None:
    data = json.loads((run_dir / "results.json").read_text())
    s, meta = data["summary"], data["meta"]
    p = s["primary"]
    if s["pairwise"]:
        t = Table(title=f"{p} vs …  (judge: {meta['judge']}, both orders)", header_style="key")
        for c in ("system", "n", "win", "tie", "loss", "both fail", "win+tie", "95% CI", "verdict"):
            t.add_column(c, justify="right" if c not in ("system", "verdict") else "left")
        for other, r in s["pairwise"].items():
            lo, hi = r["win_or_tie_ci95"]
            verdict = "[ok]at par or better[/ok]" if r["at_par_or_better"] else "[err]behind[/err]"
            t.add_row(
                other,
                str(r["n"]),
                str(r["win"]),
                str(r["tie"]),
                str(r["loss"]),
                str(r.get("both_fail", 0)),
                f"{r['win_or_tie_rate']:.0%}",
                f"{lo:.0%}–{hi:.0%}",
                verdict,
            )
        console.print(t)
    metric_names = sorted({m for v in s["systems"].values() for m in v["metrics"]})
    t = Table(title="metrics (mean)", header_style="key")
    t.add_column("system")
    t.add_column("pass", justify="right")
    for m in metric_names:
        t.add_column(m.replace("_", " "), justify="right")
    t.add_column("latency", justify="right")
    for name, v in s["systems"].items():
        lat = f"{v['latency_ms_median']:.0f}ms" if v["latency_ms_median"] else "—"
        t.add_row(
            name + (" ★" if name == p else ""),
            f"{v['pass_rate']:.0%}",
            *(f"{v['metrics'].get(m, float('nan')):.2f}" for m in metric_names),
            lat,
        )
    console.print(t)
    console.print(
        f"[ok]✓[/ok] report: [key]{run_dir / 'report.html'}[/key]  ·  {run_dir / 'report.md'}"
    )


@eval_app.command("report")
def eval_report(
    run: Annotated[str, typer.Argument(help="Run id (default: latest).")] = "latest",
    out: Annotated[Path, typer.Option(help="Runs directory.")] = Path("eval/runs"),
    open_report: Annotated[bool, typer.Option("--open", help="Open the HTML report.")] = False,
    config: ConfigOpt = None,
) -> None:
    """Show the summary of a finished eval run."""
    settings = _settings(config)
    run_dir = (out if out.is_absolute() else settings.root / out) / run
    if not (run_dir / "results.json").exists():
        fail(f"No results at {run_dir}.")
        raise typer.Exit(1)
    header(f"eval report · {run_dir.resolve().name}")
    _print_summary(run_dir)
    if open_report and (run_dir / "report.html").exists():
        typer.launch(str(run_dir / "report.html"))


if __name__ == "__main__":  # pragma: no cover
    app()
