"""Backend factory: pick the right runtime + model artefact for this machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lexcraft.backends.base import Backend, Messages, ModelDescription
from lexcraft.config import Settings
from lexcraft.hardware import HardwareInfo, detect, resolve_backend

__all__ = ["Backend", "Messages", "ModelChoice", "ModelDescription", "load_backend", "plan_model"]


@dataclass
class ModelChoice:
    backend: str
    model: str
    adapter: str | None
    note: str


def _adapter_ready(p: Path | None) -> bool:
    return bool(p and (p / "adapters.safetensors").exists())


def plan_model(
    settings: Settings,
    backend: str | None = None,
    model: str | None = None,
    adapter: str | None = None,
    no_adapter: bool = False,
    hw: HardwareInfo | None = None,
) -> ModelChoice:
    """Decide which backend and which model files to use, without loading anything."""
    name = resolve_backend(backend or settings.model.backend, hw or detect())
    m = settings.model

    if name == "remote":
        if not (m.remote_base_url and m.remote_model):
            raise RuntimeError("backend=remote needs model.remote_base_url and model.remote_model")
        return ModelChoice("remote", m.remote_model, None, m.remote_base_url)

    if name == "llamacpp":
        gguf = Path(model) if model else settings.path(m.gguf)
        if not gguf or not gguf.exists():
            raise RuntimeError(
                f"GGUF model not found at {gguf}. Run `lexcraft export --gguf` or pass --model."
            )
        return ModelChoice("llamacpp", str(gguf), None, "GGUF")

    # MLX
    if model:
        return ModelChoice("mlx", model, None if no_adapter else adapter, "explicit model")
    fused = settings.path(m.fused_mlx)
    adapter_path = Path(adapter) if adapter else settings.path(m.adapter)
    # Prefer base + adapter: re-quantizing a fused model can round away the LoRA delta.
    if not no_adapter and _adapter_ready(adapter_path):
        return ModelChoice("mlx", m.mlx_base or m.base, str(adapter_path), "base + LoRA adapter")
    if not no_adapter and not adapter and fused and (fused / "config.json").exists():
        return ModelChoice("mlx", str(fused), None, "fused LoRA model")
    note = (
        "base model only (no LoRA)" if no_adapter else "base model only — adapter not trained yet"
    )
    return ModelChoice("mlx", m.mlx_base or m.base, None, note)


def load_backend(choice: ModelChoice, settings: Settings) -> Backend:
    if choice.backend == "mlx":
        from lexcraft.backends.mlx_backend import MLXBackend

        return MLXBackend(choice.model, choice.adapter, choice.note)
    if choice.backend == "llamacpp":
        from lexcraft.backends.llamacpp_backend import LlamaCppBackend

        return LlamaCppBackend(choice.model, note=choice.note)
    from lexcraft.backends.remote import RemoteBackend

    m = settings.model
    assert m.remote_base_url and m.remote_model
    return RemoteBackend(m.remote_base_url, m.remote_model, m.remote_api_key_env)
