"""Dataset pipeline: seed pairs -> validate -> dedupe -> stratified split -> chat JSONL."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from lexcraft.prompts import SYSTEM_PROMPT
from lexcraft.verify import check, contraction_count, legal_marker_density

Format = Literal["sentence", "paragraph", "heading", "list_item"]


class Pair(BaseModel):
    id: str = ""
    category: str
    format: Format = "sentence"
    plain: str
    legal: str
    source: str = "claude-synthetic"

    def key(self) -> str:
        return re.sub(r"[^a-z0-9]+", " ", self.plain.lower()).strip()


@dataclass
class BuildReport:
    loaded: int = 0
    invalid_schema: list[str] = field(default_factory=list)
    failed_checks: list[tuple[str, list[str]]] = field(default_factory=list)
    duplicates: int = 0
    near_duplicates: int = 0
    kept: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    by_format: dict[str, int] = field(default_factory=dict)
    avg_marker_density: float = 0.0
    contractions_in_legal: int = 0


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_seed(seed_dir: Path, report: BuildReport) -> list[Pair]:
    pairs: list[Pair] = []
    for path in sorted(seed_dir.glob("*.jsonl")):
        for n, row in enumerate(read_jsonl(path), start=1):
            report.loaded += 1
            try:
                p = Pair(**row)
            except ValidationError as e:
                report.invalid_schema.append(f"{path.name}:{n}: {e.errors()[0]['msg']}")
                continue
            if not p.id:
                p.id = f"{path.stem}-{n:04d}"
            pairs.append(p)
    return pairs


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def dedupe(pairs: list[Pair], report: BuildReport, threshold: float = 0.85) -> list[Pair]:
    seen: set[str] = set()
    exact: list[Pair] = []
    for p in pairs:
        if p.key() in seen:
            report.duplicates += 1
            continue
        seen.add(p.key())
        exact.append(p)
    kept: list[Pair] = []
    kept_tokens: list[set[str]] = []
    for p in exact:
        t = _tokens(p.plain)
        if any(len(t & k) / max(len(t | k), 1) >= threshold for k in kept_tokens):
            report.near_duplicates += 1
            continue
        kept.append(p)
        kept_tokens.append(t)
    return kept


def stratified_split(
    pairs: list[Pair], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[Pair]]:
    """Deterministic split, stratified by category so every split covers every domain."""
    groups: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        groups[p.category].append(p)
    out: dict[str, list[Pair]] = {"train": [], "valid": [], "test": []}
    rng = random.Random(seed)
    for cat in sorted(groups):
        items = sorted(groups[cat], key=lambda p: hashlib.sha256(p.id.encode()).hexdigest())
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * ratios[2])) if n >= 5 else 0
        n_valid = max(1, round(n * ratios[1])) if n >= 5 else 0
        out["test"] += items[:n_test]
        out["valid"] += items[n_test : n_test + n_valid]
        out["train"] += items[n_test + n_valid :]
    for split in out.values():
        rng.shuffle(split)
    return out


def to_chat(p: Pair) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p.plain},
            {"role": "assistant", "content": p.legal},
        ]
    }


def build(
    seed_dir: Path,
    out_dir: Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    strict: bool = True,
) -> BuildReport:
    report = BuildReport()
    pairs = load_seed(seed_dir, report)

    valid: list[Pair] = []
    for p in pairs:
        res = check(p.plain, p.legal)
        if not res.ok or res.added_region_terms:
            report.failed_checks.append((p.id, res.problems))
            if strict:
                continue
        valid.append(p)

    kept = dedupe(valid, report)
    report.kept = len(kept)
    splits = stratified_split(kept, ratios, seed)

    for name, items in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [to_chat(p) for p in items])
        write_jsonl(out_dir / f"{name}_pairs.jsonl", [p.model_dump() for p in items])
        report.splits[name] = len(items)

    cat_counts: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for name, items in splits.items():
        for p in items:
            cat_counts[p.category][name] += 1
    report.by_category = {k: dict(v) for k, v in sorted(cat_counts.items())}
    report.by_format = dict(Counter(p.format for p in kept).most_common())
    if kept:
        report.avg_marker_density = sum(legal_marker_density(p.legal) for p in kept) / len(kept)
    report.contractions_in_legal = sum(contraction_count(p.legal) for p in kept)

    stats = {k: v for k, v in report.__dict__.items()}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=list))
    return report
