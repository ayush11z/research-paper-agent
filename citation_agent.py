"""
Research Paper Finder
======================
Run modes:
  python citation_agent.py            <- interactive CLI
  python citation_agent.py --server   <- called by Node server, reads env vars, prints JSON

Requirements:
    pip install langchain langchain-ollama arxiv requests

Ollama: run `ollama serve` in a separate terminal before starting the Node server.
"""

import arxiv
import requests
import json
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

SERVER_MODE = "--server" in sys.argv

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOllama(model="llama3", temperature=0)
parser = JsonOutputParser()

# ── Query generator ───────────────────────────────────────────────────────────
# Simplified prompt — ask for plain phrases, show an example, avoid boolean syntax
query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Respond with valid JSON only. No markdown, no explanation."),
    ("human", """Give me 4 short plain-English search phrases (3-6 words each) to find academic papers.
Do NOT use boolean operators like AND/OR/NOT, no site: prefixes, no special syntax.
Just simple keyword phrases like a human would type into Google Scholar.

Topic: {topic}

Example of correct output:
{{"queries": ["equivariant neural networks symmetry", "group theory deep learning", "symmetry convolutional networks", "invariant representations machine learning"]}}

Respond ONLY with the JSON for this topic:""")
])

def make_queries(thesis, related_work, field):
    """Generate search queries with robust fallback if LLM output is bad."""
    topic = f"{thesis} {related_work} {field}"
    try:
        raw = (query_prompt | llm).invoke({"topic": topic})
        text = raw.content if hasattr(raw, "content") else str(raw)
        text = text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        queries = [q.strip() for q in parsed.get("queries", []) if q.strip() and len(q.strip()) > 3]
        if queries:
            log(f"  Queries: {queries}")
            return queries
    except Exception as e:
        log(f"  LLM query generation failed ({e}), using keyword fallback.")

    # Fallback: extract keywords directly without LLM
    all_words = (thesis + " " + related_work).split()
    queries = [
        " ".join(all_words[:5]),
        " ".join(all_words[2:7]) if len(all_words) > 4 else related_work,
        related_work,
        field + " " + " ".join(all_words[:3])
    ]
    queries = [q.strip() for q in queries if q.strip() and len(q.strip()) > 3]
    log(f"  Fallback queries: {queries}")
    return queries


# ── Ranker chain ──────────────────────────────────────────────────────────────
rank_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Respond with valid JSON only. No markdown, no explanation."),
    ("human", """Rank these papers by relevance to the student's thesis. Keep only the top {top_n}.

Thesis: {thesis}
Related work: {related_work}

Papers:
{papers_json}

Respond ONLY with:
{{"ranked": [{{"title":"...","authors":"...","year":"...","source":"...","url":"...","why_relevant":"...","relevance_score":<1-10>}}]}}""")
])
rank_chain = rank_prompt | llm | parser


# ── Output helpers ─────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)

def emit(paper):
    if SERVER_MODE:
        print(json.dumps(paper), flush=True)


# ── Fetchers ───────────────────────────────────────────────────────────────────
def search_semantic_scholar(query, limit=5):
    if not query.strip():
        return []
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit, "fields": "title,authors,year,abstract,url"},
            timeout=10
        )
        resp.raise_for_status()
        papers = []
        for p in resp.json().get("data", []):
            papers.append({
                "title": p.get("title", ""),
                "authors": ", ".join(a["name"] for a in p.get("authors", [])[:3]) + (" et al." if len(p.get("authors", [])) > 3 else ""),
                "year": str(p.get("year", "N/A")),
                "abstract": (p.get("abstract") or "")[:300],
                "url": p.get("url", ""),
                "source": "Semantic Scholar"
            })
        return papers
    except Exception as e:
        log(f"  [Semantic Scholar error: {e}]")
        return []

def search_arxiv(query, limit=5):
    if not query.strip():
        return []
    try:
        papers = []
        for p in arxiv.Client().results(arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)):
            papers.append({
                "title": p.title,
                "authors": ", ".join(a.name for a in p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
                "year": str(p.published.year),
                "abstract": p.summary[:300],
                "url": p.entry_id,
                "source": "arXiv"
            })
        return papers
    except Exception as e:
        log(f"  [arXiv error: {e}]")
        return []

def deduplicate(papers):
    seen, unique = [], []
    for p in papers:
        key = p["title"].lower().strip()[:60]
        if not any(key in s or s in key for s in seen):
            seen.append(key)
            unique.append(p)
    return unique


# ── Main ───────────────────────────────────────────────────────────────────────
def find_papers(thesis, related_work, field, top_n=8):
    log("\n[1/4] Generating search queries...")
    queries = make_queries(thesis, related_work, field)

    log("\n[2/4] Searching Semantic Scholar + arXiv...")
    all_papers = []

    # arXiv in parallel (no rate limit)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for result in ex.map(lambda q: search_arxiv(q, 3), queries):
            all_papers.extend(result)

    # Semantic Scholar staggered (rate limited to ~1 req/sec)
    for q in []:
        all_papers.extend(search_semantic_scholar(q, 4))
        time.sleep(3.0)

    unique = deduplicate(all_papers)
    log(f"  Found {len(all_papers)} papers, {len(unique)} unique.")

    if not unique:
        log("No papers found. Try broader terms.")
        return

    log(f"\n[3/4] Ranking top {top_n} by relevance...")
    try:
        ranked = rank_chain.invoke({
            "thesis": thesis, "related_work": related_work,
            "top_n": top_n, "papers_json": json.dumps(unique[:25], indent=2)
        }).get("ranked", [])
    except Exception as e:
        log(f"  Ranking failed: {e}")
        ranked = unique[:top_n]

    log("\n[4/4] Analyzing PDF figures with Gemini vision...")
    try:
        from vision_ranker import rank_with_vision
        ranked = rank_with_vision(ranked, thesis)
    except Exception as e:
        log(f"  Vision analysis skipped: {e}")

    log(f"\n[done] {len(ranked)} papers found.")

    for paper in ranked:
        emit(paper)
        if not SERVER_MODE:
            score = paper.get("relevance_score", "?")
            print(f"\n  [{score}/10] {paper.get('title')}")
            print(f"  {paper.get('authors')} · {paper.get('year')} · {paper.get('source')}")
            print(f"  Why: {paper.get('why_relevant')}")
            print(f"  {paper.get('url')}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if SERVER_MODE:
        find_papers(
            os.environ.get("THESIS", ""),
            os.environ.get("RELATED_WORK", ""),
            os.environ.get("FIELD", "computer science"),
            int(os.environ.get("TOP_N", "8"))
        )
    else:
        print("\n=== Research Paper Finder ===")
        print("Make sure `ollama serve` is running.\n")
        thesis = input("Thesis:\n> ").strip()
        related_work = input("\nRelated work area:\n> ").strip()
        field = input("\nField: ").strip() or "computer science"
        top_n = int(t) if (t := input("\nHow many papers? (default 8): ").strip()).isdigit() else 8
        find_papers(thesis, related_work, field, top_n)
