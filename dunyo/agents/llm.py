"""OpenAI-compatible chat client (works with OpenAI, Groq, Google Gemini's OpenAI endpoint, OpenRouter,
Together, Ollama, vLLM…) + a null client so everything runs without any key."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore


@dataclass
class LLMResponse:
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    ok: bool = True
    error: str = ""


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    cands = [m.group(1)] if m else []
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                cands.append(text[start:i + 1])
    for c in cands:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class LLMClient:
    is_real = False
    model = "none"

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 600, temperature: float = 0.2, json_mode: bool = False) -> LLMResponse:
        return LLMResponse("", ok=False, error="no LLM configured")


class OpenAICompatible(LLMClient):
    is_real = True

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0
        self.failures = 0

    def chat(self, messages, max_tokens=600, temperature=0.2, json_mode=False) -> LLMResponse:
        if requests is None:
            return LLMResponse("", ok=False, error="requests missing")
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        for attempt in range(2):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
                if r.status_code == 400 and json_mode:
                    body.pop("response_format", None)  # some providers do not support JSON mode
                    continue
                if r.status_code != 200:
                    self.failures += 1
                    if r.status_code in (429, 500, 502, 503) and attempt == 0:
                        time.sleep(2.0)
                        continue
                    return LLMResponse("", ok=False, error=f"HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                msg = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage", {})
                self.calls += 1
                self.tokens_in += int(usage.get("prompt_tokens", 0))
                self.tokens_out += int(usage.get("completion_tokens", 0))
                return LLMResponse(msg, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), (time.time() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
        return LLMResponse("", ok=False, error=locals().get("err", "failed"))


# ------------------------------------------------------------------------------------------- local Ollama (no key)
_OLLAMA: dict[str, Any] = {}
PREFERRED = ("llama3.2", "llama3.1", "qwen2.5", "qwen3", "gemma3", "gemma2", "mistral", "phi4", "phi3", "llama3")


def detect_ollama(host: str | None = None, force: bool = False) -> dict[str, Any] | None:
    """Find a local Ollama server (https://ollama.com) — free, private, no API key. Cached for 60 s."""
    import os
    host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    now = time.time()
    if not force and _OLLAMA.get("host") == host and now - _OLLAMA.get("t", 0) < 60:
        return _OLLAMA.get("info")
    info = None
    if requests is not None and os.environ.get("TS_OLLAMA", "1") != "0":
        try:
            r = requests.get(f"{host}/api/tags", timeout=0.8)
            models = [m.get("name") or m.get("model") for m in (r.json().get("models") or [])] if r.status_code == 200 else []
            models = [m for m in models if m and "embed" not in m]
            if models:
                want = os.environ.get("OLLAMA_MODEL", "")
                pick = want if want in models else next((m for p in PREFERRED for m in models if m.startswith(p)), models[0])
                info = {"host": host, "model": pick, "models": models}
        except Exception:  # noqa: BLE001
            info = None
    _OLLAMA.update(host=host, t=now, info=info)
    return info


def create_llm(settings, allow_local: bool = True) -> LLMClient:
    """Priority: an explicitly configured endpoint (.env / Settings) → a local Ollama found automatically → none."""
    if getattr(settings, "llm_enabled", False):
        c = OpenAICompatible(settings.llm_base_url, settings.llm_api_key or "none", settings.llm_model)
        c.provider = "configured"
        return c
    if allow_local:
        info = detect_ollama()
        if info:
            c = OpenAICompatible(info["host"] + "/v1", "ollama", info["model"], timeout=120.0)
            c.provider = "ollama"
            return c
    return LLMClient()
