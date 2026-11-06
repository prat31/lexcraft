"""Pairwise LLM judge with position-bias control (each pair judged in both orders)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from lexcraft.backends.remote import RemoteBackend, RemoteError
from lexcraft.prompts import JUDGE_PAIRWISE_PROMPT

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class PairVerdict:
    test_id: str
    primary: str
    other: str
    winner: str  # "primary" | "other" | "tie" | "both_fail" | "error"
    reasons: list[str]
    raw: list[str]
    gates: list[dict] = field(default_factory=list)  # per order: {"primary": {...}, "other": {...}}


def _parse(text: str) -> tuple[str, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = _JSON.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            w = str(obj.get("winner", "")).strip().upper()
            if w in ("A", "B", "TIE"):
                return w, str(obj.get("reason", ""))
        except json.JSONDecodeError:
            pass
    m2 = re.search(r'"?winner"?\s*[:=]\s*"?(A|B|tie)\b', text, re.I)
    if m2:
        return m2.group(1).upper(), ""
    return "ERROR", text[:200]


def _gates(text: str) -> dict[str, dict[str, bool]] | None:
    """Per-rewrite pass/fail gates {"A": {"legal", "meaning"}, "B": {...}}, or None if absent."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = _JSON.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for side in ("a", "b"):
        legal, meaning = obj.get(f"{side}_legal"), obj.get(f"{side}_meaning")
        if not isinstance(legal, bool) or not isinstance(meaning, bool):
            return None
        out[side.upper()] = {"legal": legal, "meaning": meaning}
    return out


def resolve(winner: str, gates: dict[str, dict[str, bool]] | None) -> str:
    """Apply the hard gates: a rewrite that is not legal register, or changes the meaning,
    can never win. Returns "A" | "B" | "TIE" | "BOTH_FAIL" | "ERROR"."""
    if winner == "ERROR" or gates is None:
        return "ERROR"
    ok = {k: g["legal"] and g["meaning"] for k, g in gates.items()}
    if ok["A"] and ok["B"]:
        return winner
    if ok["A"]:
        return "A"
    if ok["B"]:
        return "B"
    return "BOTH_FAIL"


def judge_pair(
    judge: RemoteBackend,
    test_id: str,
    original: str,
    primary_name: str,
    primary_out: str,
    other_name: str,
    other_out: str,
    both_orders: bool = True,
    temperature: float = 0.0,
) -> PairVerdict:
    orders = [(primary_out, other_out, "primary"), (other_out, primary_out, "other")]
    if not both_orders:
        orders = orders[:1]
    votes, reasons, raws, gates_log = [], [], [], []
    for a, b, a_is in orders:
        prompt = JUDGE_PAIRWISE_PROMPT.format(original=original, a=a, b=b)
        try:
            raw = judge.generate([{"role": "user", "content": prompt}], 2048, temperature)
        except RemoteError as e:
            raw = f"ERROR: {e}"
        raws.append(raw)
        w, reason = _parse(raw)
        g = _gates(raw)
        w = resolve(w, g)
        reasons.append(reason)
        if g:
            b_is = "other" if a_is == "primary" else "primary"
            gates_log.append({a_is: g["A"], b_is: g["B"]})
        if w == "TIE":
            votes.append("tie")
        elif w == "BOTH_FAIL":
            votes.append("both_fail")
        elif w == "ERROR":
            votes.append("error")
        else:
            a_won = w == "A"
            votes.append(a_is if a_won else ("other" if a_is == "primary" else "primary"))
    valid = [v for v in votes if v != "error"]
    if not valid:
        winner = "error"
    elif len(set(valid)) == 1:
        winner = valid[0]
    else:  # orders disagree -> position bias decided it: call it a tie
        winner = "tie"
    return PairVerdict(test_id, primary_name, other_name, winner, reasons, raws, gates_log)


def judge_all(
    judge: RemoteBackend,
    jobs: list[tuple[str, str, str, str, str, str]],
    both_orders: bool,
    concurrency: int,
    on_done: Callable[[PairVerdict], None] | None = None,
    temperature: float = 0.0,
) -> list[PairVerdict]:
    """jobs: (test_id, original, primary_name, primary_out, other_name, other_out)."""
    out: list[PairVerdict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(judge_pair, judge, *job, both_orders, temperature) for job in jobs]
        for f in as_completed(futures):
            v = f.result()
            out.append(v)
            if on_done:
                on_done(v)
    return out
