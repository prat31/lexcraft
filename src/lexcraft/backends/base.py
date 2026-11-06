"""Backend interface: anything that turns chat messages into text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

Messages = list[dict[str, str]]

# Qwen-family (ChatML) prompt with thinking disabled. Matches what the LoRA saw in training.
NO_THINK_SUFFIX = "<think>\n\n</think>\n\n"


def chatml(messages: Messages, no_think: bool = True) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    if no_think:
        parts.append(NO_THINK_SUFFIX)
    return "".join(parts)


@dataclass
class ModelDescription:
    backend: str
    model: str
    adapter: str | None = None
    note: str | None = None


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def describe(self) -> ModelDescription: ...

    @abstractmethod
    def generate(self, messages: Messages, max_tokens: int, temperature: float) -> str: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass
