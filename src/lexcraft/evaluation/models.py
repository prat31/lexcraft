"""Parse eval/models.yaml into typed, swappable model roles."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

FAMILY_PREFIXES = {
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
    "gemma": "google",
    "claude": "anthropic",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "kimi": "moonshot",
    "glm": "zhipu",
    "grok": "xai",
    "minimax": "minimax",
    "mimo": "xiaomi",
    "hy": "tencent",
    "step": "stepfun",
    "longcat": "meituan",
    "llama": "meta",
    "muse": "meta",
    "mistral": "mistral",
    "phi": "microsoft",
}


def guess_family(model: str) -> str:
    m = model.lower().split("/")[-1]
    for prefix, fam in FAMILY_PREFIXES.items():
        if m.startswith(prefix):
            return fam
    return m.split("-")[0]


class Endpoint(BaseModel):
    base_url: str
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class System(BaseModel):
    name: str
    family: str = ""
    # remote
    endpoint: str | None = None
    model: str | None = None
    # local
    no_adapter: bool = False
    local_model: str | None = None  # override model path / HF id
    adapter: str | None = None
    # some models accept only one temperature (e.g. kimi-k3.1-pro needs 1.0)
    temperature: float | None = None

    @property
    def is_local(self) -> bool:
        return self.endpoint is None

    def model_post_init(self, _ctx: object) -> None:
        if not self.family:
            self.family = guess_family(self.model or self.name)


class EvalSettings(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 4
    pairwise_both_orders: bool = True


class ModelsConfig(BaseModel):
    endpoints: dict[str, Endpoint] = Field(default_factory=dict)
    primary: str
    candidates: list[System] = Field(default_factory=list)
    baselines: list[System] = Field(default_factory=list)
    judge: System
    alt_judges: list[System] = Field(default_factory=list)
    settings: EvalSettings = EvalSettings()

    @classmethod
    def load(cls, path: Path) -> ModelsConfig:
        return cls(**yaml.safe_load(path.read_text()))

    @property
    def all_systems(self) -> list[System]:
        return [*self.candidates, *self.baselines]

    def endpoint(self, s: System) -> Endpoint:
        if s.endpoint not in self.endpoints:
            raise ValueError(f"{s.name}: unknown endpoint '{s.endpoint}'")
        return self.endpoints[s.endpoint]

    def select(
        self,
        systems: list[str] | None = None,
        judge: str | None = None,
        extra: list[str] | None = None,
    ) -> tuple[list[System], System]:
        """Resolve CLI choices. `extra` entries look like 'endpoint:model' for ad-hoc baselines."""
        pool = {s.name: s for s in self.all_systems}
        added: list[str] = []
        for spec in extra or []:
            ep, _, model = spec.partition(":")
            if not model or ep not in self.endpoints:
                raise ValueError(
                    f"--add expects endpoint:model (endpoints: {', '.join(self.endpoints)})"
                )
            pool[model] = System(name=model, endpoint=ep, model=model)
            added.append(model)
        names = list(systems) if systems else [s.name for s in self.all_systems]
        names += [a for a in added if a not in names]
        missing = [n for n in names if n not in pool]
        if missing:
            raise ValueError(f"unknown systems: {', '.join(missing)}. Known: {', '.join(pool)}")
        chosen = [pool[n] for n in names]
        if self.primary not in [s.name for s in chosen]:
            raise ValueError(f"primary system '{self.primary}' must be among the selected systems")

        j = self.judge
        if judge:
            judges = {x.name: x for x in [self.judge, *self.alt_judges]}
            if judge in judges:
                j = judges[judge]
            else:  # ad-hoc: same endpoint as the default judge
                j = System(name=judge, endpoint=self.judge.endpoint, model=judge)
        return chosen, j


def family_conflicts(systems: list[System], judge: System) -> list[str]:
    return [s.name for s in systems if s.family == judge.family]
