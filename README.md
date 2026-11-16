# Research Paper Finder

An AI research assistant with two modes:

- **Live mode:** listens to a meeting and suggests relevant papers *while people are talking*. It uses Deepgram for speech-to-text, local Hugging Face models for topic detection and ranking, and a small local LLM to write search queries.
- **Deep search:** takes a thesis statement and finds, ranks, summarizes and critiques the literature around it, including research gaps and contradicting papers.

Both run from one Python server.

---

## Quick start

```bash
# 1. Python environment
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# (or: python -m venv .venv && .venv/bin/pip install -r requirements.txt)

# 2. Local LLM
ollama pull qwen3:4b-instruct
ollama serve

# 3. Keys
cp .env.example .env     # add DEEPGRAM_API_KEY, OPENALEX_API_KEY, CONTACT_EMAIL

# 4. Run
.venv/bin/python app.py
# deep search: http://localhost:3000
# live mode:   http://localhost:3000/live.html
```

The first start downloads the two Hugging Face models (about 200 MB).

---

## Live mode

```
 Browser (public/live.html)
   mic and/or meeting tab audio ─► AudioWorklet ─► 16 kHz PCM, 100 ms frames
        │                                                 ▲ transcript, topics, papers (JSON)
        ▼  WebSocket /ws/live                             │
 app.py ─► live/engine.py (LiveSession)
   ├─ Deepgram nova-3 streaming (interim results, diarization, keyterm prompting)
   ├─ instant path: HF keyphrase model on every interim/final segment (~8 ms)
   │      fires when a phrase is stable in 2 interim results, before the sentence ends
   ├─ llm path: qwen3:4b-instruct reads the new lines + context ─► 1–2 clean queries (~0.65 s)
   ├─ each query: OpenAlex + arXiv (+ Semantic Scholar with key) in parallel, 3 s timeout, cached
   └─ re-rank vs. the conversation with bge-small embeddings (~35 ms / 25 papers)
        results stream to the page as each source returns
```

**Measured latency** (M2 Pro, from a finished sentence to papers on screen, simulated conversation):

| Path | Latency | What it's good at |
|---|---|---|
| instant (keyphrases) | **~0.2–0.65 s** (≈1 s on a cold cache) | quick results for explicit technical terms |
| llm (query writer) | **~1.2–2.7 s** | vague or conversational wording, fixing transcription errors, skipping small talk |

Deepgram adds roughly 0.3 s from speech to a final transcript. Interim triggering often starts the search before the sentence has finished.

**Behavior:**
- Small talk produces no suggestions.
- A query that is nearly the same as a recent one (by word overlap) is not searched again, and a paper is never suggested twice.
- The relevance bar uses cosine similarity to the conversation. Results below `MIN_RELEVANCE` (default 0.6) are hidden.
- "Meeting tab + mic" captures a Meet, Zoom or Teams browser tab via `getDisplayMedia` (tick "share tab audio"), so it hears everyone, not just you.
- The **jargon** box sends terms to Deepgram's `keyterm` prompting so field-specific words are transcribed correctly.
- Typed lines go through the same pipeline as speech. This is handy for testing without a Deepgram key.
- Papers can be pinned and copied as a Markdown list.

### Models used

| Role | Model | Why |
|---|---|---|
| Speech-to-text | Deepgram `nova-3` (`/v1/listen`) | streaming with interim results, diarization, keyterm prompting. (Deepgram's `flux` model targets voice-agent turn-taking, so nova-3 fits meeting transcription better.) |
| Keyphrases | [`ml6team/keyphrase-extraction-distilbert-inspec`](https://huggingface.co/ml6team/keyphrase-extraction-distilbert-inspec) | DistilBERT trained on scientific abstracts; about 8 ms per segment |
| Re-ranking | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 35 ms for 25 papers. [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) ranks better but took about 850 ms; set `EMBED_MODEL` to switch |
| Query writer | `qwen3:4b-instruct` via Ollama, JSON-schema output | about 0.65 s. `llama3.2:3b` was faster (0.4 s) but missed topics; `qwen2.5:7b` took 1.2 s |

---

## Deep search

```
 public/index.html ── POST /search (SSE) ──► app.py ──► citation_agent.py --server
   1. LLM writes 5 queries: direct, synonym, method, domain, opposing
   2. OpenAlex · Semantic Scholar · Crossref · arXiv → dedupe (DOI, arXiv ID, title overlap)
   3. rank: LLM match 35% · citations 20% · recency 15% · venue 15% · abstract directness 15%
   4. expand via citation graph + S2 recommendations → re-rank
   5. open-access PDF links, structured summary per paper, "why relevant"
   6. optional: Gemini reads figures/tables of top arXiv PDFs (vision_ranker.py)
   then: research gaps + contradicting paper pairs (analysis.py)
```

Each paper comes back with a score breakdown and a summary: `claim`, `method`, `dataset`, `result`, `supports_thesis`, `relevance_quote`. You can also run it in the terminal with `.venv/bin/python citation_agent.py`.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI server: static pages, `/search` (SSE), `/ws/live` (WebSocket), `/api/status` |
| `live/engine.py` | Live session: Deepgram stream, instant and LLM topic paths, novelty check, streaming search and ranking |
| `live/models.py` | Hugging Face keyphrase extractor and embedding ranker (kept warm, LRU-cached) |
| `live/sources.py` | Async OpenAlex / Semantic Scholar / arXiv clients tuned for latency |
| `live/llm.py` | Ollama query writer with JSON-schema output |
| `public/live.html` | Live UI: audio capture (AudioWorklet), transcript, suggestion feed, pins |
| `citation_agent.py` | Deep search pipeline |
| `citation_graph.py` | Expansion through Semantic Scholar references |
| `analysis.py` | Gap and contradiction finder |
| `vision_ranker.py` | Gemini figure/table analysis (`google-genai` SDK) |
| `public/index.html` | Deep search UI |

## Configuration

See `.env.example`. You need `DEEPGRAM_API_KEY` for live audio. `OPENALEX_API_KEY` is free, and OpenAlex says keys have been required since Feb 2026, although keyless requests still worked when tested. `CONTACT_EMAIL` turns on Unpaywall PDF links. Everything else is optional.

## Known issues and next steps

- Semantic Scholar without a key is almost always rate-limited (HTTP 429). Request a free key and set `S2_API_KEY`.
- The live path searches only by keyword. A local vector index (for example, arXiv metadata embedded with bge or SPECTER2) would make the first results nearly instant and work offline.
- The Gemini vision step hasn't been tested end to end on the current model ID.
- No automated tests yet.
