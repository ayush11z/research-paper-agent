# Research Paper Finder

An AI research assistant that takes a thesis statement and finds, ranks, summarizes and critiques the academic literature around it. It searches several scholarly databases, scores each paper with a local LLM (Llama 3 via Ollama) plus bibliometric signals, follows the citation graph to find more papers, and then points out **research gaps** and **contradictions** in what it found.

It runs as a command-line tool or as a small web app that streams results live.

---

## How it works

```
 Browser (public/index.html)
        │  POST /search  {thesis, relatedWork, field, keywords, topN}
        ▼
 Node/Express server (server.js)
        │  spawns: python3 citation_agent.py --server   (inputs passed as env vars)
        │  streams stdout back as Server-Sent Events
        ▼
 Python pipeline (citation_agent.py)
   1. Query generation  ── LLM writes 5 queries: direct, synonym, method, domain, opposing
   2. Multi-source search ─ OpenAlex · Semantic Scholar · Crossref · arXiv
                            → deduplicate by DOI, arXiv ID and title-word overlap (≥80%)
   3. Ranking ──────────── LLM scores + citation count + recency + venue
   4. Expansion ────────── citation_graph.py (Semantic Scholar references)
                            + S2 "recommended papers" for the top 3 → re-rank
   5. Enrichment ───────── open-access PDF links (Unpaywall / arXiv),
                            structured LLM summary per paper, one-line "why relevant"
   6. Vision (optional) ── vision_ranker.py: Gemini reads figures/tables in arXiv PDFs
   then: analysis.py ───── research gaps + contradicting paper pairs
```

### Ranking formula

Each candidate gets a `final_score` (0–10):

| Signal | Weight | Source |
|---|---|---|
| Semantic match to thesis | 35% | LLM (1–10) |
| Citation impact | 20% | >1000 → 10, >100 → 7, >10 → 4, else 2 |
| Recency | 15% | 2024+ → 10, 2022+ → 8, 2019+ → 6, else 3 |
| Venue quality | 15% | Top venues (NeurIPS, ICML, Nature, ACL…) → 10, others → 6 |
| Abstract directness | 15% | LLM (1–10) |

If a Gemini key is set, the top 5 arXiv papers also get a `vision_score`, and the list is re-sorted using it.

### Per-paper output

Each paper includes the title, authors, year, venue, source, URL, PDF link, the score breakdown, `why_relevant`, and a structured `summary`:
`claim`, `method`, `dataset`, `result`, `supports_thesis` (supports / contradicts / neutral), and `relevance_quote`.

---

## Files

| File | What it does |
|---|---|
| `citation_agent.py` | Main pipeline: query generation, the four search clients, dedup, scoring, summaries, Unpaywall, S2 recommendations. Runs as an interactive CLI or with `--server`. |
| `citation_graph.py` | Expands the paper set by pulling each paper's references from the Semantic Scholar graph API (adds up to 40 new papers). |
| `analysis.py` | LLM prompts that find research gaps (with suggestions for what would fill them) and pairs of papers that contradict each other. |
| `vision_ranker.py` | Optional: downloads arXiv PDFs and asks Gemini to judge whether each figure and table supports the thesis. |
| `server.js` | Express server. Serves the UI, runs the Python pipeline as a subprocess, and streams `log` / `result` / `gaps` / `contradictions` / `done` events over SSE. Has its own small `.env` loader. |
| `public/index.html` | Single-page UI: thesis, related work, keywords and field inputs; tabs for **Papers**, **Gaps** and **Contradictions**; paper cards with score bars, summaries and a supports/contradicts badge. |

---

## Setup

**Requirements:** Python 3.8+, Node.js 18+, and [Ollama](https://ollama.com) with Llama 3.

```bash
# LLM
ollama pull llama3
ollama serve

# Python dependencies
pip install arxiv requests langchain-ollama langchain-core
pip install google-generativeai     # optional, only for vision ranking

# Node dependencies
npm install
```

Optional `.env` in the project root:

```
GEMINI_API_KEY=your_key_here
PORT=3000
```

## Usage

**Web app**

```bash
node server.js
# open http://localhost:3000
```

**CLI**

```bash
python citation_agent.py
```

It asks for the thesis, related-work area, field, keywords and how many papers you want, then prints the ranked results.

---

## Git history

1. `e06d7f5` – initial commit: citation agent with LangChain + Ollama
2. `22d4909` – "post graphs change": reworked `citation_agent.py`, added Express dependencies
3. `d007206` – added citation graph expansion, gap detector, contradiction finder, improved ranking

---

## Known issues and next steps

- **`node_modules/` is committed to git.** Add it to `.gitignore` and run `git rm -r --cached node_modules`.
- **No `requirements.txt`.** The Python dependencies above should go in one.
- **`gemini-1.5-flash` is an old model ID** in `vision_ranker.py`, so vision ranking may fail silently. Switch it to a current model.
- **Placeholder contact email** (`research@finder.com`) is used for the Crossref and Unpaywall polite-pool requests. Replace it with a real one.
- **Speed:** the search runs one request at a time with `sleep` between calls to stay under rate limits, so a full run takes a few minutes. Running the searches in parallel would help.
- **LLM JSON parsing** falls back to neutral scores (5/10) when Llama 3 returns invalid JSON. Ollama's `format="json"` mode would make this more reliable.
- **Gaps and contradictions** use `all_expanded`, the pre-final-ranking pool, rather than the final ranked list.
- **No tests yet.**
