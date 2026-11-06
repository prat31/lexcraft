"""Generate a Promptfoo config from models.yaml, run it, and parse the results."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lexcraft import __version__
from lexcraft.evaluation.models import ModelsConfig, System
from lexcraft.prompts import JUDGE_RUBRIC, SYSTEM_PROMPT

PROMPTFOO_VERSION = os.environ.get("LEXCRAFT_PROMPTFOO_VERSION", "0.125.1")


@dataclass
class Row:
    system: str
    test_id: str
    category: str
    source: str
    reference: str
    output: str
    latency_ms: float | None
    passed: bool
    error: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


def provider_entry(s: System, cfg: ModelsConfig, local_ports: dict[str, int], session: str) -> dict:
    temp = s.temperature if s.temperature is not None else cfg.settings.temperature
    common = {
        "temperature": temp,
        "max_tokens": cfg.settings.max_tokens,
        # reasoning models return reasoning_content; keep it out of the scored output
        "showThinking": False,
    }
    if s.is_local:
        return {
            "id": f"openai:chat:{s.name}",
            "label": s.name,
            "config": {
                "apiBaseUrl": f"http://127.0.0.1:{local_ports[s.name]}/v1",
                "apiKey": "local",
                **common,
            },
        }
    ep = cfg.endpoint(s)
    return {
        "id": f"openai:chat:{s.model}",
        "label": s.name,
        "config": {
            "apiBaseUrl": ep.base_url,
            **({"apiKeyEnvar": ep.api_key_env} if ep.api_key_env else {}),
            "headers": {
                "User-Agent": f"lexcraft-eval/{__version__}",
                "x-opencode-session": f"{session}-{s.name}",
                **ep.headers,
            },
            **common,
        },
    }


def write_config(
    run_dir: Path,
    tests: list[dict],
    systems: list[System],
    judge: System,
    cfg: ModelsConfig,
    local_ports: dict[str, int],
    assertion_file: Path,
    use_rubric: bool = True,
) -> Path:
    session = f"lexcraft-{uuid.uuid4().hex[:10]}"
    prompt = (
        "[\n"
        f'  {{"role": "system", "content": {json.dumps(SYSTEM_PROMPT, ensure_ascii=False)}}},\n'
        '  {"role": "user", "content": {{ text | dump }}}\n'
        "]\n"
    )
    (run_dir / "prompt.json").write_text(prompt)
    (run_dir / "tests.yaml").write_text(
        yaml.safe_dump(
            [{"description": t["id"], "vars": t} for t in tests],
            allow_unicode=True,
            sort_keys=False,
        )
    )
    asserts: list[dict] = [{"type": "python", "value": f"file://{assertion_file}"}]
    if use_rubric:
        asserts.append({"type": "llm-rubric", "value": JUDGE_RUBRIC, "metric": "judge_rubric"})
    judge_provider = provider_entry(judge, cfg, local_ports, session)
    judge_provider["config"]["max_tokens"] = 4096
    config = {
        "description": "lexcraft: plain text -> legal language",
        "prompts": ["file://prompt.json"],
        "providers": [provider_entry(s, cfg, local_ports, session) for s in systems],
        "defaultTest": {"options": {"provider": judge_provider}, "assert": asserts},
        "tests": "file://tests.yaml",
        "evaluateOptions": {"maxConcurrency": cfg.settings.concurrency, "cache": False},
        "sharing": False,
    }
    path = run_dir / "promptfooconfig.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return path


def promptfoo_cmd() -> list[str]:
    if shutil.which("promptfoo"):
        return ["promptfoo"]
    if shutil.which("npx"):
        return ["npx", "--yes", f"promptfoo@{PROMPTFOO_VERSION}"]
    raise RuntimeError("Promptfoo needs Node.js. Install Node 20+ (npx) or `npm i -g promptfoo`.")


def run(config_path: Path, out_path: Path, concurrency: int) -> int:
    env = {
        **os.environ,
        "PROMPTFOO_PYTHON": sys.executable,
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
        "PROMPTFOO_DISABLE_SHARING": "1",
        "PROMPTFOO_CONFIG_DIR": str(config_path.parent / ".promptfoo"),
    }
    cmd = [
        *promptfoo_cmd(),
        "eval",
        "-c",
        str(config_path),
        "-o",
        str(out_path),
        "-j",
        str(concurrency),
        "--no-cache",
        "--no-table",
        "--no-share",
    ]
    # Inherit stdio: promptfoo renders its own live progress bar.
    return subprocess.call(cmd, cwd=config_path.parent, env=env)


def _real_error(r: dict) -> str | None:
    """Provider/runtime errors only. Promptfoo also puts assertion failures in `error`."""
    resp = r.get("response") or {}
    if resp.get("error"):
        return str(resp["error"])
    # failureReason: 0 = none, 1 = assertion failed, 2 = error
    if r.get("failureReason") == 2:
        return str(r.get("error") or "error")
    return None


def parse_results(path: Path, tests_by_id: dict[str, dict]) -> list[Row]:
    data = json.loads(path.read_text())
    results = data.get("results", {})
    items = results.get("results", results) if isinstance(results, dict) else results
    rows: list[Row] = []
    for r in items:
        prov = r.get("provider") or {}
        system = prov.get("label") or prov.get("id", "?")
        vars_ = r.get("vars") or {}
        tid = vars_.get("id") or r.get("testCase", {}).get("description", "?")
        t = tests_by_id.get(tid, vars_)
        resp = r.get("response") or {}
        output = resp.get("output")
        if not isinstance(output, str):
            output = json.dumps(output) if output is not None else ""
        grading = r.get("gradingResult") or {}
        metrics: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for comp in grading.get("componentResults") or []:
            sub = comp.get("componentResults")
            for c in sub or [comp]:
                metric = (c.get("assertion") or {}).get("metric") or (c.get("assertion") or {}).get(
                    "type"
                )
                if metric:
                    metrics[metric] = float(c.get("score") or 0.0)
                    reasons[metric] = str(c.get("reason", ""))[:300]
        named = r.get("namedScores") or {}
        for k, v in named.items():
            metrics.setdefault(k, float(v))
        rows.append(
            Row(
                system=system,
                test_id=tid,
                category=t.get("category", ""),
                source=t.get("text", ""),
                reference=t.get("reference", ""),
                output=output.strip(),
                latency_ms=r.get("latencyMs"),
                passed=bool(r.get("success")),
                error=_real_error(r),
                metrics=metrics,
                reasons=reasons,
            )
        )
    return rows
