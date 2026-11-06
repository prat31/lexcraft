"""MLX backend (Apple Silicon)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lexcraft.backends.base import Backend, Messages, ModelDescription


class MLXBackend(Backend):
    """MLX streams are thread-local, so the model is loaded and run on one dedicated
    worker thread. Callers on any thread (CLI, HTTP server) are serialized through it."""

    name = "mlx"

    def __init__(self, model: str, adapter: str | None = None, note: str | None = None):
        self._model_id, self._adapter, self._note = model, adapter, note
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._pool.submit(self._load).result()

    def _load(self) -> None:
        from mlx_lm import load

        loaded: Any = load(self._model_id, adapter_path=self._adapter)
        self.model, self.tokenizer = loaded[0], loaded[1]

    def describe(self) -> ModelDescription:
        return ModelDescription("mlx", self._model_id, self._adapter, self._note)

    def _prompt(self, messages: Messages) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

    def _generate(self, messages: Messages, max_tokens: int, temperature: float) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        return generate(
            self.model,
            self.tokenizer,
            prompt=self._prompt(messages),
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )

    def generate(self, messages: Messages, max_tokens: int, temperature: float) -> str:
        return self._pool.submit(self._generate, messages, max_tokens, temperature).result()

    def close(self) -> None:
        self._pool.shutdown(wait=False)
