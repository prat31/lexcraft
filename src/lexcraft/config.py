"""Configuration: `lexcraft.toml` + environment (.env).

Lookup order: explicit path > $LEXCRAFT_CONFIG > ./lexcraft.toml > built-in defaults.
Relative paths in the file resolve against the file's directory.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BackendName = Literal["auto", "mlx", "llamacpp", "remote"]


class ModelConfig(BaseModel):
    base: str = "Qwen/Qwen3-1.7B"
    # MLX build of the same base (4-bit): QLoRA training + inference on Apple Silicon.
    mlx_base: str | None = "mlx-community/Qwen3-1.7B-4bit"
    adapter: Path | None = Path("artifacts/adapters/lexcraft-qwen3-1.7b")
    fused_mlx: Path | None = Path("artifacts/models/lexcraft-qwen3-1.7b-mlx-8bit")
    gguf: Path | None = Path("artifacts/models/lexcraft-qwen3-1.7b-q8_0.gguf")
    backend: BackendName = "auto"
    # Used only when backend = "remote" (any OpenAI-compatible endpoint).
    remote_base_url: str | None = None
    remote_model: str | None = None
    remote_api_key_env: str | None = None


class GenerationConfig(BaseModel):
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    # Retry a block once if the output loses facts/placeholders; then fall back to the original.
    verify: bool = True


class TrainConfig(BaseModel):
    data_dir: Path = Path("data/processed")
    iters: int = 600
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    num_layers: int = -1  # -1 = all transformer layers
    max_seq_length: int = 512
    grad_checkpoint: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 50
    save_every: int = 50  # keep equal to steps_per_eval so the best checkpoint exists
    early_stopping_patience: int = 2  # evals without improvement before stopping (0 = off)
    seed: int = 42


class Settings(BaseModel):
    root: Path = Field(default_factory=Path.cwd)
    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    train: TrainConfig = TrainConfig()

    def path(self, p: Path | str | None) -> Path | None:
        """Resolve a configured path against the config root."""
        if p is None:
            return None
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.root / p)


def find_config(explicit: Path | None = None) -> Path | None:
    if explicit:
        return explicit
    env = os.environ.get("LEXCRAFT_CONFIG")
    if env:
        return Path(env)
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "lexcraft.toml"
        if candidate.exists():
            return candidate
    return None


def load_env(root: Path | None = None) -> None:
    """Load secrets from .env (project root first, then cwd). Never overrides real env vars."""
    for d in filter(None, [root, Path.cwd()]):
        f = d / ".env"
        if f.exists():
            load_dotenv(f, override=False)


def load_settings(explicit: Path | None = None) -> Settings:
    cfg_path = find_config(explicit)
    if cfg_path is None:
        load_env()
        return Settings()
    cfg_path = cfg_path.resolve()
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)
    settings = Settings(root=cfg_path.parent, **raw)
    load_env(settings.root)
    return settings
