"""Live meeting session: browser audio -> Deepgram -> topics -> papers, over one WebSocket.

Two query paths run side by side so suggestions appear fast and stay good:
  instant  keyphrases from the HF model, fired on final segments or on phrases
           that show up in two consecutive interim results (before the sentence ends)
  llm      a small local LLM reads the rolling transcript and writes cleaner queries
           when an utterance finishes
Each query searches all sources concurrently, and results are re-ranked against
the conversation and streamed as each source returns.
"""

import asyncio
import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx
import websockets
from fastapi import WebSocket, WebSocketDisconnect

from .llm import QueryLLM
from .models import Embedder, KeyphraseExtractor, normalize_phrase, pick_device
from .sources import PaperSearch


@dataclass
class Settings:
    deepgram_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    deepgram_model: str = field(default_factory=lambda: os.getenv("DEEPGRAM_MODEL", "nova-3"))
    openalex_key: str = field(default_factory=lambda: os.getenv("OPENALEX_API_KEY", ""))
    s2_key: str = field(default_factory=lambda: os.getenv("S2_API_KEY", ""))
    contact_email: str = field(default_factory=lambda: os.getenv("CONTACT_EMAIL", ""))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    live_llm: str = field(default_factory=lambda: os.getenv("LIVE_LLM", ""))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    keyphrase_model: str = field(default_factory=lambda: os.getenv(
        "KEYPHRASE_MODEL", "ml6team/keyphrase-extraction-distilbert-inspec"))
    min_relevance: float = field(default_factory=lambda: float(os.getenv("MIN_RELEVANCE", "0.6")))
    papers_per_topic: int = 4


class Services:
    """Process-wide, loaded once: models stay warm between sessions."""

    def __init__(self):
        self.settings = Settings()
        self.ready = False
        self.status: dict = {}

    async def start(self):
        s = self.settings
        device = pick_device()
        self.http = httpx.AsyncClient(http2=False, headers={"User-Agent": "research-paper-agent/live"})
        self.keyphrases, self.embedder = await asyncio.gather(
            asyncio.to_thread(KeyphraseExtractor, s.keyphrase_model, device),
            asyncio.to_thread(Embedder, s.embed_model, device),
        )
        await asyncio.to_thread(self.embedder.query, "warm up")
        self.search = PaperSearch(self.http, s.openalex_key, s.s2_key, s.contact_email)
        self.llm = QueryLLM(self.http, s.ollama_host, s.live_llm)
        llm_model = await self.llm.setup()
        self.status = {
            "device": device,
            "deepgram": bool(s.deepgram_key),
            "deepgram_model": s.deepgram_model,
            "llm": llm_model or None,
            "embed_model": s.embed_model,
            "keyphrase_model": s.keyphrase_model,
            "sources": self.search.sources,
        }
        self.ready = True

    async def close(self):
        await self.http.aclose()


class LiveSession:
    def __init__(self, ws: WebSocket, svc: Services):
        self.ws = ws
        self.svc = svc
        self.out: asyncio.Queue = asyncio.Queue()
        self.finals: deque = deque(maxlen=60)   # (timestamp, speaker, text)
        self.searched: dict[str, float] = {}   # normalized query -> time
        self.shown: dict[str, int] = {}        # paper key / title -> topic that showed it
        self.llm_cursor = 0                    # finals already read by the LLM
        self.total_finals = 0
        self.speculative: dict[int, str] = {}  # topic id -> phrase fired from an interim result
        self.interim_hits: Counter = Counter()
        self.topic_id = 0
        self.search_sem = asyncio.Semaphore(4)
        self.llm_busy = False
        self.llm_dirty = False
        self.words_since_llm = 0
        self.tasks: set[asyncio.Task] = set()
        self.topic_tasks: dict[int, asyncio.Task] = {}
        self.dg = None
        self.last_audio = 0.0

    # ── plumbing ──────────────────────────────────────────────────────────

    def send(self, type_: str, **data):
        self.out.put_nowait({"type": type_, **data})

    def spawn(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def _sender(self):
        while True:
            msg = await self.out.get()
            await self.ws.send_text(json.dumps(msg))

    async def run(self):
        sender = asyncio.create_task(self._sender())
        self.send("status", **self.svc.status)
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if self.dg:
                        self.last_audio = time.time()
                        await self.dg.send(msg["bytes"])
                elif msg.get("text"):
                    await self.on_control(json.loads(msg["text"]))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await self.stop_deepgram()
            for t in list(self.tasks):
                t.cancel()
            sender.cancel()

    async def on_control(self, m: dict):
        kind = m.get("type")
        if kind == "start":
            await self.start_deepgram(int(m.get("sample_rate", 16000)), m.get("keyterms", []))
        elif kind == "stop":
            await self.stop_deepgram()
        elif kind == "text":  # typed input, handled exactly like a finished spoken utterance
            text = (m.get("text") or "").strip()
            if text:
                t0 = time.perf_counter()
                self.send("transcript", text=text, is_final=True, speaker="you")
                await self.on_final(text, "you", t0, utterance_done=True)

    # ── Deepgram streaming ────────────────────────────────────────────────

    async def start_deepgram(self, sample_rate: int, keyterms: list[str]):
        s = self.svc.settings
        if not s.deepgram_key:
            self.send("error", message="DEEPGRAM_API_KEY is not set. Add it to .env; typing still works.")
            return
        await self.stop_deepgram()
        params = [
            ("model", s.deepgram_model), ("language", "en"),
            ("encoding", "linear16"), ("sample_rate", sample_rate), ("channels", 1),
            ("interim_results", "true"), ("smart_format", "true"), ("diarize", "true"),
            ("endpointing", 300), ("utterance_end_ms", 1000), ("vad_events", "true"),
        ] + [("keyterm", k.strip()) for k in keyterms if k.strip()][:50]
        try:
            self.dg = await websockets.connect(
                "wss://api.deepgram.com/v1/listen?" + urlencode(params),
                additional_headers={"Authorization": f"Token {s.deepgram_key}"},
                max_size=None,
            )
        except Exception as e:
            self.dg = None
            self.send("error", message=f"Could not connect to Deepgram: {e}")
            return
        self.last_audio = time.time()
        self.spawn(self._dg_reader(self.dg))
        self.spawn(self._dg_keepalive(self.dg))
        self.send("listening", on=True)

    async def stop_deepgram(self):
        dg, self.dg = self.dg, None
        if dg:
            try:
                await dg.send(json.dumps({"type": "CloseStream"}))
                await dg.close()
            except Exception:
                pass
            self.send("listening", on=False)

    async def _dg_keepalive(self, dg):
        while self.dg is dg:
            await asyncio.sleep(4)
            if time.time() - self.last_audio > 4:
                try:
                    await dg.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    return

    async def _dg_reader(self, dg):
        try:
            async for raw in dg:
                m = json.loads(raw)
                t0 = time.perf_counter()
                if m.get("type") == "Results":
                    alt = m["channel"]["alternatives"][0]
                    text = alt.get("transcript", "").strip()
                    if not text:
                        continue
                    spk = Counter(w.get("speaker") for w in alt.get("words", [])).most_common(1)
                    speaker = f"S{spk[0][0]}" if spk and spk[0][0] is not None else ""
                    final = bool(m.get("is_final"))
                    self.send("transcript", text=text, is_final=final, speaker=speaker)
                    if final:
                        await self.on_final(text, speaker, t0, utterance_done=bool(m.get("speech_final")))
                    else:
                        await self.on_interim(text, t0)
                elif m.get("type") == "UtteranceEnd":
                    self.maybe_run_llm(time.perf_counter(), force=True)
        except websockets.ConnectionClosed as e:
            if self.dg is dg and e.code != 1000:
                self.send("error", message=f"Deepgram closed the stream ({e.code}): {e.reason}")
        finally:
            if self.dg is dg:
                self.dg = None
                self.send("listening", on=False)

    # ── topic detection ───────────────────────────────────────────────────

    async def on_interim(self, text: str, t0: float):
        if len(text.split()) < 5:
            return
        phrases = await asyncio.to_thread(self.svc.keyphrases, text)
        for phrase, _ in phrases:
            self.interim_hits[phrase] += 1
            # stable across two interim results -> search before the sentence is even finished
            if self.interim_hits[phrase] == 2 and " " in phrase:
                tid = self.fire(phrase, "instant", t0)
                if tid:
                    self.speculative[tid] = phrase

    async def on_final(self, text: str, speaker: str, t0: float, utterance_done: bool):
        final_norm = normalize_phrase(text)
        for tid, phrase in list(self.speculative.items()):
            if phrase not in final_norm:  # interim was mis-heard (e.g. "weather" vs "whether")
                self.retract(tid, phrase)
        self.speculative.clear()
        self.finals.append((time.time(), speaker, text))
        self.total_finals += 1
        self.interim_hits.clear()
        phrases = await asyncio.to_thread(self.svc.keyphrases, text)
        if phrases:
            ranked = sorted(phrases, key=lambda x: -x[1])
            fresh = [p for p, _ in ranked if not self.recently_searched(p)]
            if fresh and len(fresh) == 1 and " " not in fresh[0] and self.svc.llm.model:
                fresh = []  # a lone word is too vague; let the LLM handle it
            if fresh:
                # pair the two strongest new phrases for a more specific query
                others = [p for p in fresh[1:] if p not in fresh[0] and fresh[0] not in p]
                self.fire(f"{fresh[0]} {others[0]}" if others else fresh[0], "instant", t0)
        self.words_since_llm += len(text.split())
        self.maybe_run_llm(t0, force=utterance_done)

    def maybe_run_llm(self, t0: float, force: bool = False):
        if not self.svc.llm.model or self.words_since_llm == 0:
            return
        if not force and self.words_since_llm < 25:
            return
        if self.llm_busy:
            self.llm_dirty = True
            return
        self.spawn(self._run_llm(t0))

    async def _run_llm(self, t0: float):
        self.llm_busy = True
        self.words_since_llm = 0
        try:
            lines = [f"{s + ': ' if s else ''}{t}" for _, s, t in self.finals]
            new = max(1, min(len(lines), self.llm_cursor_new()))
            self.llm_cursor = self.total_finals
            queries = await self.svc.llm.extract("\n".join(lines[-new:]), lines[:-new], list(self.searched))
            self.send("log", message=f"llm queries: {queries}")
            for q in queries:
                self.fire(q, "llm", t0)
        except Exception as e:
            self.send("log", message=f"LLM query step failed: {e}")
        finally:
            self.llm_busy = False
            if self.llm_dirty:
                self.llm_dirty = False
                self.maybe_run_llm(time.perf_counter(), force=True)

    def llm_cursor_new(self) -> int:
        return self.total_finals - self.llm_cursor

    @staticmethod
    def _tokens(q: str) -> set[str]:
        return {w.rstrip("s") for w in normalize_phrase(q).split() if len(w) > 2}

    def recently_searched(self, query: str) -> bool:
        """True if an earlier query covers this one: a superset of its words, or nearly the
        same words (Jaccard >= 0.6) without being more specific."""
        q = self._tokens(query)
        if not q:
            return True
        now = time.time()
        for prev, ts in self.searched.items():
            if now - ts > 600:
                continue
            p = self._tokens(prev)
            if q <= p or (len(q & p) / len(q | p) >= 0.6 and len(q) <= len(p)):
                return True
        return False

    def fire(self, query: str, via: str, t0: float) -> int | None:
        if self.recently_searched(query):
            return None
        self.searched[normalize_phrase(query)] = time.time()
        self.topic_id += 1
        self.topic_tasks[self.topic_id] = self.spawn(self.search_topic(self.topic_id, query, via, t0))
        return self.topic_id

    def retract(self, tid: int, phrase: str):
        task = self.topic_tasks.pop(tid, None)
        if task:
            task.cancel()
        self.searched.pop(normalize_phrase(phrase), None)
        self.shown = {k: v for k, v in self.shown.items() if v != tid}
        self.send("topic_retract", id=tid)

    # ── search + rank ─────────────────────────────────────────────────────

    def context_text(self, query: str) -> str:
        recent = " ".join(t for _, _, t in list(self.finals)[-2:])
        return f"{query}. {recent}"[:800]

    async def search_topic(self, tid: int, query: str, via: str, t0: float):
        svc = self.svc
        self.send("topic", id=tid, query=query, via=via)
        async with self.search_sem:
            qvec_task = asyncio.create_task(asyncio.to_thread(svc.embedder.query, self.context_text(query)))
            tasks = [asyncio.create_task(svc.search.search(src, query)) for src in svc.search.sources]
            pool: dict[str, dict] = {}
            titles: set[str] = set()
            sent_keys: list[str] = []
            first_ms = None
            try:
                for fut in asyncio.as_completed(tasks, timeout=4.5):
                    try:
                        papers = await fut
                    except Exception:
                        continue
                    for p in papers:  # same paper from two sources: keep the first
                        tkey = "t:" + normalize_phrase(p["title"])[:80]
                        if p["key"] not in pool and tkey not in titles:
                            pool[p["key"]] = p
                            titles.add(tkey)
                    candidates = [p for p in pool.values()
                                  if self.shown.get(p["key"], tid) == tid
                                  and self.shown.get("t:" + normalize_phrase(p["title"])[:80], tid) == tid]
                    if not candidates:
                        continue
                    ranked = await asyncio.to_thread(svc.embedder.rank, await qvec_task, candidates)
                    top = [p for p in ranked if p["relevance"] >= svc.settings.min_relevance]
                    top = top[: svc.settings.papers_per_topic]
                    keys = [p["key"] for p in top]
                    if keys and keys != sent_keys:
                        sent_keys = keys
                        for p in top:
                            self.shown[p["key"]] = tid
                            self.shown["t:" + normalize_phrase(p["title"])[:80]] = tid
                        ms = round((time.perf_counter() - t0) * 1000)
                        first_ms = first_ms or ms
                        self.send("papers", topic_id=tid, papers=top, latency_ms=ms, first_latency_ms=first_ms)
            except (TimeoutError, asyncio.TimeoutError):
                pass
            finally:
                for t in tasks:
                    t.cancel()
            self.send("topic_done", id=tid, found=len(sent_keys))
        self.topic_tasks.pop(tid, None)
