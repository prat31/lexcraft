"""LoRA training.

- Apple Silicon: mlx-lm, in-process, with a metrics callback.
- NVIDIA CUDA:   PEFT + TRL SFTTrainer (install `lexcraft[cuda]`).

Both read the same chat-format JSONL produced by `lexcraft data build` and
write a metrics.jsonl next to the adapter for the eval report.
"""

from __future__ import annotations

import hashlib
import json
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexcraft.config import Settings, TrainConfig


@dataclass
class TrainSummary:
    backend: str
    base_model: str
    adapter_path: str
    iters: int
    seconds: float
    final_train_loss: float | None = None
    best_val_loss: float | None = None
    val_history: list[tuple[int, float]] = field(default_factory=list)
    peak_memory_gb: float | None = None
    stopped_early: bool = False
    best_checkpoint: str | None = None


def dataset_fingerprint(data_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("train.jsonl", "valid.jsonl"):
        p = data_dir / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _check_data(data_dir: Path) -> None:
    for name in ("train.jsonl", "valid.jsonl"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"{data_dir / name} not found. Run `lexcraft data build` first."
            )


def _write_meta(adapter: Path, settings: Settings, backend: str, cfg: TrainConfig) -> None:
    adapter.mkdir(parents=True, exist_ok=True)
    data_dir = settings.path(cfg.data_dir)
    assert data_dir
    meta = {
        "tool": "lexcraft",
        "backend": backend,
        "base_model": (settings.model.mlx_base or settings.model.base)
        if backend == "mlx"
        else settings.model.base,
        "dataset_fingerprint": dataset_fingerprint(data_dir),
        "train_config": json.loads(cfg.model_dump_json()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (adapter / "lexcraft_train.json").write_text(json.dumps(meta, indent=2))


class EarlyStop(Exception):
    """Raised from the metrics callback to stop training when val loss stops improving."""


class MetricsWriter:
    """Appends train/val metrics to metrics.jsonl, forwards them to a UI hook,
    and implements early stopping on validation loss."""

    def __init__(
        self,
        path: Path,
        on_event: Callable[[str, dict], None] | None = None,
        patience: int = 0,
    ):
        self.path = path
        self.path.write_text("")
        self.on_event = on_event
        self.patience = patience
        self.bad_evals = 0
        self.best: tuple[int, float] | None = None
        self.summary: dict[str, Any] = {"val": [], "train": []}

    def _write(self, kind: str, info: dict) -> None:
        row = {"kind": kind, "time": time.time(), **{k: float(v) for k, v in info.items()}}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.summary[kind].append(row)
        if self.on_event:
            self.on_event(kind, row)

    # mlx-lm TrainingCallback protocol
    def on_train_loss_report(self, info: dict) -> None:
        self._write("train", info)

    def on_val_loss_report(self, info: dict) -> None:
        self._write("val", info)
        it, loss = int(info["iteration"]), float(info["val_loss"])
        if it == 0:
            return  # baseline eval before any update
        if self.best is None or loss < self.best[1]:
            self.best, self.bad_evals = (it, loss), 0
        else:
            self.bad_evals += 1
            if self.patience and self.bad_evals >= self.patience:
                raise EarlyStop(f"val loss has not improved for {self.bad_evals} evals")


def select_best_checkpoint(adapter: Path, best_iter: int | None) -> Path | None:
    """Copy the checkpoint closest to (at or after) the best val step to adapters.safetensors."""
    import shutil

    if best_iter is None:
        return None
    ckpts = sorted(adapter.glob("*_adapters.safetensors"))
    steps = {int(p.name.split("_")[0]): p for p in ckpts if p.name.split("_")[0].isdigit()}
    if not steps:
        return None
    # mlx reports the val loss for step N as iteration N-1; its checkpoint is saved at step N.
    target = min(steps, key=lambda s: (abs(s - (best_iter + 1)), s))
    shutil.copyfile(steps[target], adapter / "adapters.safetensors")
    return steps[target]


def _summary(
    backend: str, settings: Settings, adapter: Path, cfg: TrainConfig, t0: float, mw: MetricsWriter
) -> TrainSummary:
    train, val = mw.summary["train"], mw.summary["val"]
    vh = [(int(v["iteration"]), v["val_loss"]) for v in val]
    return TrainSummary(
        backend=backend,
        base_model=settings.model.base,
        adapter_path=str(adapter),
        iters=cfg.iters,
        seconds=time.time() - t0,
        final_train_loss=train[-1]["train_loss"] if train else None,
        best_val_loss=min((v for _, v in vh), default=None),
        val_history=vh,
        peak_memory_gb=max((t.get("peak_memory", 0.0) for t in train), default=None),
    )


def mlx_args(settings: Settings, cfg: TrainConfig, adapter: Path) -> types.SimpleNamespace:
    from mlx_lm.lora import CONFIG_DEFAULTS

    data_dir = settings.path(cfg.data_dir)
    warmup = max(1, cfg.iters // 20)
    args = dict(CONFIG_DEFAULTS)
    args.update(
        model=settings.model.mlx_base or settings.model.base,
        train=True,
        data=str(data_dir),
        fine_tune_type="lora",
        optimizer="adamw",
        mask_prompt=True,  # loss only on the legal rewrite, not the prompt
        num_layers=cfg.num_layers,
        batch_size=cfg.batch_size,
        iters=cfg.iters,
        val_batches=-1,
        learning_rate=cfg.learning_rate,
        steps_per_report=cfg.steps_per_report,
        steps_per_eval=cfg.steps_per_eval,
        save_every=cfg.save_every,
        adapter_path=str(adapter),
        max_seq_length=cfg.max_seq_length,
        grad_checkpoint=cfg.grad_checkpoint,
        seed=cfg.seed,
        lr_schedule={
            "name": "cosine_decay",
            "warmup": warmup,
            "warmup_init": cfg.learning_rate / 100,
            "arguments": [cfg.learning_rate, cfg.iters - warmup, cfg.learning_rate / 20],
        },
        lora_parameters={
            "rank": cfg.lora_rank,
            "scale": cfg.lora_alpha / cfg.lora_rank,
            "dropout": cfg.lora_dropout,
        },
        test=False,
        report_to=None,
        project_name=None,
        trust_remote_code=False,
    )
    return types.SimpleNamespace(**args)


def train_mlx(
    settings: Settings,
    cfg: TrainConfig,
    adapter: Path,
    on_event: Callable[[str, dict], None] | None = None,
) -> TrainSummary:
    import os

    from mlx_lm import load
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    data_dir = settings.path(cfg.data_dir)
    assert data_dir
    _check_data(data_dir)
    _write_meta(adapter, settings, "mlx", cfg)
    args = mlx_args(settings, cfg, adapter)

    t0 = time.time()
    model, tokenizer = load(args.model)[:2]
    train_set, valid_set, _ = load_dataset(args, tokenizer)  # type: ignore[arg-type]
    mw = MetricsWriter(adapter / "metrics.jsonl", on_event, cfg.early_stopping_patience)
    stopped = False
    try:
        train_model(args, model, train_set, valid_set, training_callback=mw)  # type: ignore[arg-type]
    except EarlyStop:
        stopped = True
    summary = _summary("mlx", settings, adapter, cfg, t0, mw)
    best = select_best_checkpoint(adapter, mw.best[0] if mw.best else None)
    summary.stopped_early = stopped
    summary.best_checkpoint = best.name if best else None
    return summary


def train_cuda(
    settings: Settings,
    cfg: TrainConfig,
    adapter: Path,
    on_event: Callable[[str, dict], None] | None = None,
) -> TrainSummary:  # pragma: no cover - needs an NVIDIA GPU
    """PEFT/TRL LoRA training. Produces a PEFT adapter (adapter_model.safetensors)."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    data_dir = settings.path(cfg.data_dir)
    assert data_dir
    _check_data(data_dir)
    _write_meta(adapter, settings, "cuda", cfg)
    mw = MetricsWriter(adapter / "metrics.jsonl", on_event)

    class _CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" in logs:
                mw.on_train_loss_report(
                    {"iteration": state.global_step, "train_loss": logs["loss"]}
                )
            if "eval_loss" in logs:
                mw.on_val_loss_report(
                    {"iteration": state.global_step, "val_loss": logs["eval_loss"]}
                )

    ds = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "valid.jsonl"),
        },
    )
    tok = AutoTokenizer.from_pretrained(settings.model.base)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    sft = SFTConfig(
        output_dir=str(adapter),
        max_steps=cfg.iters,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=cfg.steps_per_report,
        eval_strategy="steps",
        eval_steps=cfg.steps_per_eval,
        save_steps=cfg.steps_per_eval,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=cfg.grad_checkpoint,
        max_length=cfg.max_seq_length,
        assistant_only_loss=True,
        seed=cfg.seed,
        report_to=[],
    )
    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        peft_config=peft_cfg,
        callbacks=[_CB()]
        + (
            [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
            if cfg.early_stopping_patience
            else []
        ),
    )
    trainer.train()
    trainer.save_model(str(adapter))
    return _summary("cuda", settings, adapter, cfg, t0, mw)
