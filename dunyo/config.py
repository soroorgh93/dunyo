"""Configuration: defaults + environment variables (+ optional .env, editable from the app Settings drawer)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # data
    mode: str = field(default_factory=lambda: env("TS_MODE", "auto"))          # auto | live | replay
    cache_dir: Path = field(default_factory=lambda: ROOT / env("TS_CACHE_DIR", "runs/cache"))
    runs_dir: Path = field(default_factory=lambda: ROOT / "runs")
    http_timeout: float = float(os.environ.get("TS_HTTP_TIMEOUT", "25"))
    baseline_years: int = int(os.environ.get("TS_BASELINE_YEARS", "30"))
    # LLM (any OpenAI-compatible endpoint: OpenAI, Groq, Gemini-OpenAI, OpenRouter, Ollama, vLLM)
    llm_base_url: str = field(default_factory=lambda: env("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: env("LLM_MODEL", ""))
    # blockchain
    rpc_url: str = field(default_factory=lambda: env("RPC_URL", ""))
    private_key: str = field(default_factory=lambda: env("PRIVATE_KEY", ""))
    contract_address: str = field(default_factory=lambda: env("CONTRACT_ADDRESS", ""))
    chain_mode: str = field(default_factory=lambda: env("CHAIN_MODE", "auto"))  # auto | ledger | evm-local | evm-rpc
    # agent
    max_steps: int = int(os.environ.get("TS_MAX_STEPS", "16"))
    escalate_threshold: float = 0.70
    watch_threshold: float = 0.40
    # deep models
    deep_enabled: bool = field(default_factory=lambda: env("TS_DEEP", "1") != "0")
    deep_epochs: int = int(os.environ.get("TS_DEEP_EPOCHS", "6"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


def get_settings() -> Settings:
    s = Settings()
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    return s
