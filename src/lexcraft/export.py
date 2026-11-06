"""Export: fuse LoRA into the base model, quantize for MLX, convert to GGUF for llama.cpp."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportStep:
    name: str
    cmd: list[str]
    output: Path


def _py() -> str:
    return sys.executable


def _llama_cpp_tools() -> tuple[str | None, str | None]:
    """Locate llama.cpp's converter + quantizer ($LLAMA_CPP_DIR or PATH)."""
    root = os.environ.get("LLAMA_CPP_DIR")
    convert = quantize = None
    if root:
        c = Path(root) / "convert_hf_to_gguf.py"
        convert = str(c) if c.exists() else None
        for q in (Path(root) / "build" / "bin" / "llama-quantize", Path(root) / "llama-quantize"):
            if q.exists():
                quantize = str(q)
                break
    convert = convert or shutil.which("convert_hf_to_gguf.py")
    quantize = quantize or shutil.which("llama-quantize")
    return convert, quantize


def plan(
    base: str,
    adapter: Path,
    fused_dir: Path,
    mlx_out: Path,
    gguf_out: Path | None,
    q_bits: int = 8,
) -> tuple[list[ExportStep], list[str]]:
    """Build the export steps. Returns (steps, warnings)."""
    warnings: list[str] = []
    steps = [
        ExportStep(
            "Fuse LoRA into base (bf16)",
            [
                _py(),
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                base,
                "--adapter-path",
                str(adapter),
                "--save-path",
                str(fused_dir),
                "--dequantize",
            ],
            fused_dir,
        ),
        ExportStep(
            f"Quantize for MLX ({q_bits}-bit)",
            [
                _py(),
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                str(fused_dir),
                "--mlx-path",
                str(mlx_out),
                "-q",
                "--q-bits",
                str(q_bits),
            ],
            mlx_out,
        ),
    ]
    if gguf_out:
        convert, quantize = _llama_cpp_tools()
        if not convert:
            warnings.append(
                "GGUF skipped: llama.cpp converter not found. Clone llama.cpp and set "
                "LLAMA_CPP_DIR, or put convert_hf_to_gguf.py on PATH."
            )
        else:
            f16 = gguf_out.with_name(gguf_out.stem.replace("q8_0", "f16") + ".f16.gguf")
            cmd = (
                [convert]
                if os.access(convert, os.X_OK) and not convert.endswith(".py")
                else [_py(), convert]
            )
            steps.append(
                ExportStep(
                    "Convert to GGUF (f16)",
                    [*cmd, str(fused_dir), "--outfile", str(f16), "--outtype", "f16"],
                    f16,
                )
            )
            if quantize:
                steps.append(
                    ExportStep(
                        "Quantize GGUF (Q8_0)",
                        [quantize, str(f16), str(gguf_out), "Q8_0"],
                        gguf_out,
                    )
                )
            else:
                warnings.append("llama-quantize not found: leaving the f16 GGUF unquantized.")
    return steps, warnings


def run_steps(
    steps: list[ExportStep],
    on_step: Callable[[int, ExportStep], None] | None = None,
    log: Path | None = None,
) -> None:
    for i, step in enumerate(steps, start=1):
        if on_step:
            on_step(i, step)
        step.output.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(step.cmd, capture_output=True, text=True)
        if log:
            with log.open("a") as f:
                f.write(f"$ {' '.join(step.cmd)}\n{proc.stdout}\n{proc.stderr}\n")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
            raise RuntimeError(f"{step.name} failed:\n" + "\n".join(tail))


def modelfile(gguf_name: str) -> str:
    """Ollama Modelfile for the exported GGUF.

    Uses the exact system prompt and the no-think ChatML format the LoRA was trained on.
    """
    from lexcraft.prompts import SYSTEM_PROMPT

    return f'''# lexcraft — plain text to legal language (Qwen3 1.7B + LoRA)
# Build: ollama create lexcraft -f Modelfile
FROM ./{gguf_name}

TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

SYSTEM """{SYSTEM_PROMPT}"""

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
'''


def write_modelfile(gguf_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(modelfile(gguf_path.name))
    return out
