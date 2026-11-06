"""Hardware detection and backend selection."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class HardwareInfo:
    os: str
    arch: str
    cpu: str
    ram_gb: float
    apple_silicon: bool
    cuda: bool
    gpu_name: str | None = None
    vram_gb: float | None = None
    available_backends: list[str] = field(default_factory=list)
    recommended_backend: str = "llamacpp"
    notes: list[str] = field(default_factory=list)


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return 0.0


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _nvidia() -> tuple[bool, str | None, float | None]:
    if not shutil.which("nvidia-smi"):
        return False, None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return False, None, None
    if not out:
        return False, None, None
    name, mem = (x.strip() for x in out.splitlines()[0].split(","))
    return True, name, float(mem) / 1024


def detect() -> HardwareInfo:
    system, arch = platform.system(), platform.machine()
    apple = system == "Darwin" and arch == "arm64"
    cuda, gpu, vram = _nvidia()
    info = HardwareInfo(
        os=f"{system} {platform.release()}",
        arch=arch,
        cpu=_cpu_name(),
        ram_gb=round(_ram_gb(), 1),
        apple_silicon=apple,
        cuda=cuda,
        gpu_name=gpu if cuda else ("Apple GPU (Metal)" if apple else None),
        vram_gb=round(vram, 1) if vram else None,
    )

    if apple and _has("mlx_lm"):
        info.available_backends.append("mlx")
    if _has("llama_cpp"):
        info.available_backends.append("llamacpp")
    info.available_backends.append("remote")

    if "mlx" in info.available_backends:
        info.recommended_backend = "mlx"
    elif "llamacpp" in info.available_backends:
        info.recommended_backend = "llamacpp"
    else:
        info.recommended_backend = "remote"
        info.notes.append(
            "No local backend installed. Apple Silicon: `pip install lexcraft` (MLX). "
            "Others: `pip install 'lexcraft[llamacpp]'` (set CMAKE_ARGS=-DGGML_CUDA=on for CUDA)."
        )

    mem = info.vram_gb or info.ram_gb
    if mem and mem < 6:
        info.notes.append("Less than 6 GB memory: expect slow generation; use 4-bit models.")
    if cuda and "llamacpp" not in info.available_backends:
        info.notes.append("CUDA GPU found: install llama-cpp-python with CUDA for GPU inference.")
    return info


def resolve_backend(requested: str, info: HardwareInfo | None = None) -> str:
    info = info or detect()
    if requested != "auto":
        if requested not in info.available_backends:
            raise RuntimeError(
                f"Backend '{requested}' is not available here. "
                f"Available: {', '.join(info.available_backends)}"
            )
        return requested
    return info.recommended_backend
