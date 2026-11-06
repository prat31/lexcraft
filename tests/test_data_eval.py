import json
from pathlib import Path

import pytest

from lexcraft.data import build
from lexcraft.evaluation.judge import PairVerdict, _gates, _parse, resolve
from lexcraft.evaluation.models import ModelsConfig, family_conflicts, guess_family
from lexcraft.evaluation.promptfoo import Row
from lexcraft.evaluation.report import summarize, wilson

ROOT = Path(__file__).resolve().parents[1]


def _pairs(n: int, cat: str) -> list[dict]:
    return [
        {
            "category": cat,
            "plain": f"Pay {i} dollars by Friday number {i}.",
            "legal": f"The sum of {i} dollars shall be paid by Friday, reference {i}.",
        }
        for i in range(n)
    ]


def test_build_split_is_stratified_and_deterministic(tmp_path: Path):
    seed = tmp_path / "seed"
    seed.mkdir()
    rows = _pairs(20, "a") + _pairs(10, "b")
    rows[1]["legal"] = "dropped facts"  # fails checks
    (seed / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    rep1 = build(seed, tmp_path / "out1")
    rep2 = build(seed, tmp_path / "out2")
    assert len(rep1.failed_checks) == 1
    assert rep1.splits == rep2.splits
    assert (tmp_path / "out1" / "test.jsonl").read_text() == (
        tmp_path / "out2" / "test.jsonl"
    ).read_text()
    assert set(rep1.by_category) == {"a", "b"}
    first = json.loads((tmp_path / "out1" / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]


def test_seed_dataset_is_clean(tmp_path: Path):
    rep = build(ROOT / "data" / "seed", tmp_path / "out")
    assert rep.loaded >= 1000
    assert not rep.invalid_schema and not rep.failed_checks


def test_models_yaml_and_family_guard():
    cfg = ModelsConfig.load(ROOT / "eval" / "models.yaml")
    systems, judge = cfg.select()
    assert cfg.primary in [s.name for s in systems]
    assert not family_conflicts(systems, judge)
    _, qwen_judge = cfg.select(judge="qwen3.8-max")
    assert family_conflicts(systems, qwen_judge)
    systems2, _ = cfg.select(systems=["lexcraft-lora"], extra=["opencode-go:grok-4.7"])
    assert [s.name for s in systems2] == ["lexcraft-lora", "grok-4.7"]
    assert systems2[1].family == "xai"
    with pytest.raises(ValueError):
        cfg.select(systems=["gpt-6-luna"])  # primary missing


def test_guess_family():
    assert guess_family("deepseek-v4-pro") == "deepseek"
    assert guess_family("Qwen/Qwen3-1.7B") == "qwen"
    assert guess_family("gemini-4-argon") == "google"


def test_judge_parse():
    assert _parse('{"winner": "A", "reason": "x"}') == ("A", "x")
    assert (
        _parse('<think>B is better?</think>```json\n{"winner":"tie","reason":"same"}\n```')[0]
        == "TIE"
    )
    assert _parse("garbage")[0] == "ERROR"


def _raw(a_legal, a_meaning, b_legal, b_meaning, winner):
    return json.dumps(
        {
            "a_legal": a_legal,
            "a_meaning": a_meaning,
            "b_legal": b_legal,
            "b_meaning": b_meaning,
            "winner": winner,
            "reason": "x",
        }
    )


def test_register_gate_overrides_judge_preference():
    # B keeps the meaning but is not legal register: it can never win.
    raw = _raw(True, True, False, True, "B")
    assert resolve(_parse(raw)[0], _gates(raw)) == "A"
    # A legal but changes meaning, B not legal: both fail, not a tie.
    raw = _raw(True, False, False, True, "B")
    assert resolve(_parse(raw)[0], _gates(raw)) == "BOTH_FAIL"
    # both pass the gates: the judge's preference stands
    raw = _raw(True, True, True, True, "B")
    assert resolve(_parse(raw)[0], _gates(raw)) == "B"
    # gates missing -> error, never a silent winner
    assert resolve("A", _gates('{"winner": "A"}')) == "ERROR"


def test_summary_and_wilson():
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi <= 1.0
    rows = [
        Row("p", "t1", "c", "s", "r", "o", 10.0, True, None, {"facts_preserved": 1.0}),
        Row("q", "t1", "c", "s", "r", "o", 20.0, False, None, {"facts_preserved": 0.0}),
    ]
    vs = [
        PairVerdict("t1", "p", "q", "primary", [], []),
        PairVerdict("t2", "p", "q", "tie", [], []),
        PairVerdict("t3", "p", "q", "both_fail", [], []),
    ]
    s = summarize(rows, vs, "p")
    assert list(summarize(rows[::-1], vs, "p", order=["p", "q"])["systems"]) == ["p", "q"]
    assert list(summarize(rows, vs, "p", order=["q", "p"])["systems"]) == ["q", "p"]
    pw = s["pairwise"]["q"]
    assert pw["win"] == 1 and pw["tie"] == 1 and pw["both_fail"] == 1 and pw["at_par_or_better"]
    assert s["systems"]["p"]["metrics"]["facts_preserved"] == 1.0


def test_early_stopping_and_best_checkpoint(tmp_path: Path):
    from lexcraft.train import EarlyStop, MetricsWriter, select_best_checkpoint

    mw = MetricsWriter(tmp_path / "m.jsonl", patience=2)
    mw.on_val_loss_report({"iteration": 0, "val_loss": 5.0})
    mw.on_val_loss_report({"iteration": 49, "val_loss": 0.7})
    mw.on_val_loss_report({"iteration": 99, "val_loss": 0.5})
    mw.on_val_loss_report({"iteration": 149, "val_loss": 0.55})
    with pytest.raises(EarlyStop):
        mw.on_val_loss_report({"iteration": 199, "val_loss": 0.6})
    assert mw.best == (99, 0.5)
    for step in (50, 100, 150, 200):
        (tmp_path / f"{step:07d}_adapters.safetensors").write_text(str(step))
    best = select_best_checkpoint(tmp_path, mw.best[0])
    assert best is not None and best.name == "0000100_adapters.safetensors"
    assert (tmp_path / "adapters.safetensors").read_text() == "100"


def test_ollama_modelfile_in_sync():
    from lexcraft.export import modelfile

    committed = (ROOT / "ollama" / "Modelfile").read_text()
    assert committed == modelfile("lexcraft-qwen3-1.7b-q8_0.gguf"), (
        "ollama/Modelfile is stale: regenerate with lexcraft export --gguf"
    )


def test_default_no_think(tmp_path):
    from lexcraft.export import _THINK_OPT_IN, _THINK_OPT_OUT, default_no_think

    (tmp_path / "chat_template.jinja").write_text(f"a{_THINK_OPT_OUT}b")
    default_no_think(tmp_path)
    assert (tmp_path / "chat_template.jinja").read_text() == f"a{_THINK_OPT_IN}b"
