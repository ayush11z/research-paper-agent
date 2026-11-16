"""Small local LLM (via Ollama) that turns the running conversation into search queries.

qwen3:4b-instruct answers in ~0.65 s on an M2 Pro with structured JSON output,
and writes much better queries than raw keyphrases.
"""

import json

import httpx

_PREFERRED = ["qwen3:4b-instruct", "qwen2.5:7b-instruct", "llama3.1:8b"]

_SYSTEM = """You listen to a live research discussion (speech-to-text, so expect transcription errors).
Find the research topic being discussed RIGHT NOW (weight the latest lines most) and write academic
search queries that would surface papers on it.
Rules:
- 1 or 2 queries, each 2-7 words of technical terms (no filler, no question form).
- Fix obvious transcription errors in technical terms.
- Do not repeat or trivially rephrase an already-searched query.
- relevant=false with no queries for small talk, logistics, or nothing new."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
    "required": ["relevant", "queries"],
}


class QueryLLM:
    def __init__(self, client: httpx.AsyncClient, host: str, model: str = ""):
        self.client = client
        self.host = host.rstrip("/")
        self.requested = model
        self.model = ""

    async def setup(self) -> str:
        """Pick an installed model and load it into memory. Returns '' if Ollama is unavailable."""
        try:
            r = await self.client.get(f"{self.host}/api/tags", timeout=3)
            installed = {m["name"] for m in r.json().get("models", [])}
        except Exception:
            return ""
        for name in ([self.requested] if self.requested else []) + _PREFERRED:
            if name in installed or f"{name}:latest" in installed:
                self.model = name
                break
        if self.model:
            try:
                await self.extract("warm up", [], [])
            except Exception:
                pass
        return self.model

    async def extract(self, latest: str, history: list[str], searched: list[str]) -> list[str]:
        if not self.model:
            return []
        user = (
            f"Already searched: {json.dumps(searched[-12:])}\n\n"
            "Earlier transcript:\n" + ("\n".join(history[-10:]) or "(none)") +
            f"\n\nLatest:\n{latest}"
        )
        r = await self.client.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model, "stream": False, "format": _SCHEMA, "keep_alive": "1h",
                "options": {"temperature": 0, "num_predict": 64},
                "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            },
            timeout=8,
        )
        r.raise_for_status()
        data = json.loads(r.json()["message"]["content"])
        if not data.get("relevant"):
            return []
        return [q.strip() for q in data.get("queries", []) if q and q.strip()][:2]
