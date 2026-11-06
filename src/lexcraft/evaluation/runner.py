"""Eval orchestration: local servers -> Promptfoo -> pairwise judge -> semantic -> report."""

from __future__ import annotations

import contextlib
import json
import random
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from lexcraft.backends.remote import RemoteBackend
from lexcraft.evaluation import promptfoo, report
from lexcraft.evaluation.judge import PairVerdict, judge_all
from lexcraft.evaluation.models import ModelsConfig, System, family_conflicts


@dataclass
class RunOptions:
    models_path: Path
    test_path: Path
    out_root: Path
    assertion_file: Path
    limit: int | None = None
    systems: list[str] | None = None
    judge: str | None = None
    extra: list[str] | None = None
    seed: int = 42
    rubric: bool = True
    pairwise: bool = True
    semantic: bool = True
    allow_family_overlap: bool = False
    train_metrics: Path | None = None
    config_path: Path | None = None


Log = Callable[[str], None]


def load_tests(path: Path, limit: int | None, seed: int) -> list[dict]:
    """Stratified sample: round-robin over categories so small runs still cover every domain."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `lexcraft data build` first.")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tests = [
        {"id": r["id"], "text": r["plain"], "reference": r["legal"], "category": r["category"]}
        for r in rows
    ]
    if not limit or limit >= len(tests):
        return tests
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in tests:
        groups[t["category"]].append(t)
    for g in groups.values():
        rng.shuffle(g)
    out: list[dict] = []
    cats = sorted(groups)
    while len(out) < limit:
        for c in cats:
            if groups[c] and len(out) < limit:
                out.append(groups[c].pop())
    return out


def _free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port")


@contextlib.contextmanager
def local_servers(
    systems: list[System], log: Log, config_path: Path | None, timeout: float = 900
) -> Iterator[dict[str, int]]:
    procs: list[subprocess.Popen] = []
    ports: dict[str, int] = {}
    try:
        port = 8765
        for s in systems:
            if not s.is_local:
                continue
            port = _free_port(port)
            cmd = [sys.executable, "-m", "lexcraft", "serve", "--port", str(port), "--name", s.name]
            if config_path:
                cmd += ["--config", str(config_path)]
            if s.no_adapter:
                cmd.append("--no-adapter")
            if s.local_model:
                cmd += ["--model", s.local_model]
            if s.adapter:
                cmd += ["--adapter", s.adapter]
            log(f"starting local server [key]{s.name}[/key] on :{port}")
            procs.append(
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            )
            ports[s.name] = port
            port += 1
        deadline = time.time() + timeout
        for (name, p), proc in zip(ports.items(), procs, strict=True):
            while True:
                if proc.poll() is not None:
                    err = (proc.stderr.read() if proc.stderr else "")[-1500:]
                    raise RuntimeError(f"local server {name} exited:\n{err}")
                try:
                    if httpx.get(f"http://127.0.0.1:{p}/v1/models", timeout=2).status_code == 200:
                        log(f"[ok]✓[/ok] {name} ready")
                        break
                except httpx.HTTPError:
                    pass
                if time.time() > deadline:
                    raise RuntimeError(f"local server {name} did not start in {timeout:.0f}s")
                time.sleep(1)
        yield ports
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def run_eval(
    opts: RunOptions, log: Log, on_judge: Callable[[PairVerdict, int], None] | None = None
) -> Path:
    cfg = ModelsConfig.load(opts.models_path)
    systems, judge = cfg.select(opts.systems, opts.judge, opts.extra)
    conflicts = family_conflicts(systems, judge)
    if conflicts and not opts.allow_family_overlap:
        raise ValueError(
            f"judge '{judge.name}' ({judge.family}) shares a model family with: {', '.join(conflicts)}. "
            "Pick another judge (--judge) or pass --allow-family-overlap."
        )
    tests = load_tests(opts.test_path, opts.limit, opts.seed)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = opts.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"run [key]{run_id}[/key] · {len(tests)} tests · {len(systems)} systems · judge {judge.name}"
    )

    with local_servers(systems, log, opts.config_path) as ports:
        cfg_path = promptfoo.write_config(
            run_dir, tests, systems, judge, cfg, ports, opts.assertion_file, opts.rubric
        )
        log("running Promptfoo (generation + deterministic checks + rubric judge)")
        code = promptfoo.run(cfg_path, run_dir / "promptfoo_results.json", cfg.settings.concurrency)
    # promptfoo exits 100 when some assertions fail; that's a normal outcome.
    if code not in (0, 100) or not (run_dir / "promptfoo_results.json").exists():
        raise RuntimeError(f"promptfoo failed with exit code {code}")

    tests_by_id = {t["id"]: t for t in tests}
    rows = promptfoo.parse_results(run_dir / "promptfoo_results.json", tests_by_id)

    verdicts: list[PairVerdict] = []
    primary = cfg.primary
    if opts.pairwise:
        outs: dict[tuple[str, str], str] = {
            (r.system, r.test_id): r.output for r in rows if not r.error
        }
        jobs = [
            (t["id"], t["text"], primary, outs[(primary, t["id"])], s.name, outs[(s.name, t["id"])])
            for t in tests
            for s in systems
            if s.name != primary and (primary, t["id"]) in outs and (s.name, t["id"]) in outs
        ]
        if jobs:
            ep = cfg.endpoint(judge)
            jb = RemoteBackend(
                ep.base_url,
                judge.model or judge.name,
                ep.api_key_env,
                headers={"User-Agent": "lexcraft-eval", **ep.headers},
            )
            log(
                f"pairwise judging {len(jobs)} pairs × {2 if cfg.settings.pairwise_both_orders else 1} orders"
            )
            verdicts = judge_all(
                jb,
                jobs,
                cfg.settings.pairwise_both_orders,
                cfg.settings.concurrency,
                (lambda v: on_judge(v, len(jobs))) if on_judge else None,
                temperature=judge.temperature if judge.temperature is not None else 0.0,
            )
            jb.close()

    if opts.semantic:
        from lexcraft.evaluation import semantic

        if semantic.available():
            log("semantic checks (embedding similarity + bidirectional NLI)")
            scorer = semantic.SemanticScorer()
            ok = [r for r in rows if not r.error and r.output]
            sc = scorer.score([r.source for r in ok], [r.output for r in ok])
            for r, a, b, c in zip(ok, sc.similarity, sc.entail_fwd, sc.entail_bwd, strict=True):
                r.metrics.update(similarity=float(a), entail_fwd=float(b), entail_bwd=float(c))
        else:
            log("[muted]semantic checks skipped (pip install 'lexcraft[eval]')[/muted]")

    summary = report.summarize(rows, verdicts, primary, order=[s.name for s in systems])
    meta = {
        "run_id": run_id,
        "n_tests": len(tests),
        "systems": ", ".join(s.name for s in systems),
        "judge": judge.model or judge.name,
        "judge_family": judge.family,
        "test_file": str(opts.test_path),
    }
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "summary": summary,
                "rows": [asdict(r) for r in rows],
                "verdicts": [asdict(v) for v in verdicts],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    report.write_markdown(summary, meta, run_dir / "report.md")
    tm = None
    if opts.train_metrics and opts.train_metrics.exists():
        tm = [json.loads(x) for x in opts.train_metrics.read_text().splitlines() if x.strip()]
    if not report.write_html(summary, rows, verdicts, meta, run_dir / "report.html", tm):
        log("[muted]HTML report skipped (pip install 'lexcraft[eval]' for plotly)[/muted]")
    latest = opts.out_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_id)
    return run_dir
