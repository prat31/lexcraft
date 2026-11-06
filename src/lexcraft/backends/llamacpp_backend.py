"""llama.cpp backend (GGUF; CPU, CUDA, Vulkan, Metal)."""

from __future__ import annotations

import threading

from lexcraft.backends.base import Backend, Messages, ModelDescription, chatml


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, gguf_path: str, n_ctx: int = 4096, note: str | None = None):
        from llama_cpp import Llama

        self._path, self._note = gguf_path, note
        # n_gpu_layers=-1 offloads everything the build supports (CUDA/Metal/Vulkan).
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        self._lock = threading.Lock()

    def describe(self) -> ModelDescription:
        return ModelDescription("llamacpp", self._path, None, self._note)

    def generate(self, messages: Messages, max_tokens: int, temperature: float) -> str:
        # Build the ChatML prompt ourselves so thinking stays disabled, exactly as in training.
        with self._lock:
            out = self.llm.create_completion(
                chatml(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
        return str(out["choices"][0]["text"])
