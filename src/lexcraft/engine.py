"""Rewrite engine: segment -> rewrite each block -> verify -> reassemble."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from lexcraft.backends.base import Backend
from lexcraft.config import GenerationConfig
from lexcraft.prompts import build_messages
from lexcraft.structure import assemble, segment
from lexcraft.verify import CheckResult, check, clean_output


@dataclass
class BlockResult:
    index: int
    source: str
    output: str
    kind: str
    check: CheckResult
    attempts: int
    fell_back: bool
    seconds: float


@dataclass
class RewriteResult:
    text: str
    blocks: list[BlockResult] = field(default_factory=list)
    seconds: float = 0.0
    format_warnings: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        out = list(self.format_warnings)
        for b in self.blocks:
            if b.fell_back:
                out.append(f"block {b.index + 1}: kept original ({'; '.join(b.check.problems)})")
            elif b.check.problems:
                out.append(f"block {b.index + 1}: {'; '.join(b.check.problems)}")
        return out


ProgressFn = Callable[[int, int, BlockResult], None]


class Rewriter:
    def __init__(self, backend: Backend, gen: GenerationConfig | None = None):
        self.backend = backend
        self.gen = gen or GenerationConfig()

    def rewrite_block(self, text: str) -> tuple[str, CheckResult, int]:
        attempts, out, result = 0, text, check(text, text)
        max_attempts = 2 if self.gen.verify else 1
        while attempts < max_attempts:
            attempts += 1
            temperature = self.gen.temperature if attempts == 1 else max(self.gen.temperature, 0.4)
            raw = self.backend.generate(build_messages(text), self.gen.max_tokens, temperature)
            out = clean_output(raw)
            result = check(text, out)
            if result.ok and out:
                break
        return out, result, attempts

    def rewrite(self, document: str, progress: ProgressFn | None = None) -> RewriteResult:
        t0 = time.perf_counter()
        segments = segment(document)
        todo = [i for i, s in enumerate(segments) if s.translatable]
        outputs: dict[int, str] = {}
        blocks: list[BlockResult] = []
        for n, i in enumerate(todo, start=1):
            seg = segments[i]
            src = seg.text or ""
            tb = time.perf_counter()
            out, res, attempts = self.rewrite_block(src)
            fell_back = self.gen.verify and (not res.ok or not out)
            outputs[i] = src if fell_back else out
            block = BlockResult(
                index=n - 1,
                source=src,
                output=outputs[i],
                kind=seg.kind,
                check=res,
                attempts=attempts,
                fell_back=fell_back,
                seconds=time.perf_counter() - tb,
            )
            blocks.append(block)
            if progress:
                progress(n, len(todo), block)
        text = assemble(segments, outputs)
        lost = [
            f"bold/italic on '{phrase}' could not be re-applied"
            for seg in segments
            for phrase in seg.lost_emphasis
        ]
        return RewriteResult(text, blocks, time.perf_counter() - t0, lost)

    def count_blocks(self, document: str) -> int:
        return sum(1 for s in segment(document) if s.translatable)
