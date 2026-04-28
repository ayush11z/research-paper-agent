"""
Research Paper Finder
======================
Workflow:
  1. You enter your thesis + related work description
  2. LLM generates smart search queries
  3. Searches Semantic Scholar + arXiv simultaneously
  4. LLM ranks results by relevance to your thesis
  5. Prints a clean ranked list with citation suggestions

Requirements:
    pip install langchain langchain-ollama arxiv requests

Ollama setup (one-time):
    brew install ollama
    ollama pull llama3
    ollama serve          <- run this in a separate terminal before running this script
"""

import arxiv
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatOllama(model="llama3", temperature=0)
parser = JsonOutputParser()


# ── Chain 1: Query generator ──────────────────────────────────────────────────
# LangChain concept: you can have multiple chains for different subtasks.
# This one takes your thesis and turns it into search queries.
query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Respond with valid JSON only. No markdown, no explanation."),
    ("human", """Given this thesis and related work area, generate 4 short search queries
to find relevant academic papers. Queries should be specific and varied.

Thesis: {thesis}
Related work area: {related_work}
Field: {field}

Respond ONLY with this JSON:
{{
  "queries": ["<query 1>", "<query 2>", "<query 3>", "<query 4>"]
}}""")
])

query_chain = query_prompt | llm | parser


# ── Chain 2: Paper ranker ─────────────────────────────────────────────────────
# This chain takes all fetched papers and ranks them by relevance.
rank_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Respond with valid JSON only. No markdown, no explanation."),
    ("human", """Rank these papers by relevance to the student's thesis.
Keep only the top {top_n} most relevant. For each, explain WHY it's relevant in one sentence.

Thesis: {thesis}
Related work area: {related_work}

Papers (JSON list):
{papers_json}

Respond ONLY with this JSON:
{{
  "ranked": [
    {{
      "title": "<exact title from input>",
      "authors": "<authors from input>",
      "year": "<year>",
      "source": "<source>",
      "url": "<url>",
      "why_relevant": "<one sentence explaining relevance to this specific thesis>",
      "relevance_score": <1-10>
    }}
  ]
}}""")
])

rank_chain = rank_prompt | llm | parser


# ── Semantic Scholar fetcher ──────────────────────────────────────────────────
def search_semantic_scholar(query: str, limit: int = 5) -> list:
    """Search Semantic Scholar API — free, no key needed."""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,externalIds,url"
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for p in data.get("data", []):
            papers.append({
                "title": p.get("title", ""),
                "authors": ", ".join(a["name"] for a in p.get("authors", [])[:3]) + (" et al." if len(p.get("authors", [])) > 3 else ""),
                "year": str(p.get("year", "N/A")),
                "abstract": p.get("abstract", "No abstract available.")[:300],
                "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
                "source": "Semantic Scholar"
            })
        return papers
    except Exception as e:
        print(f"  [Semantic Scholar error: {e}]")
        return []


# ── arXiv fetcher ─────────────────────────────────────────────────────────────
def search_arxiv(query: str, limit: int = 5) -> list:
    """Search arXiv — free, no key needed."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for p in arxiv.Client().results(search):
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
        print(f"  [arXiv error: {e}]")
        return []


# ── Deduplicator ──────────────────────────────────────────────────────────────
def deduplicate(papers: list) -> list:
    """Remove duplicate papers by fuzzy title match."""
    seen = []
    unique = []
    for p in papers:
        title_clean = p["title"].lower().strip()[:60]
        if not any(title_clean in s or s in title_clean for s in seen):
            seen.append(title_clean)
            unique.append(p)
    return unique


# ── Main agent ────────────────────────────────────────────────────────────────
def find_papers(thesis: str, related_work: str, field: str, top_n: int = 8):

    # Step 1: Generate search queries using LLM
    print("\n[1/3] Generating search queries...")
    try:
        query_result = query_chain.invoke({
            "thesis": thesis,
            "related_work": related_work,
            "field": field
        })
        queries = query_result.get("queries", [related_work])
    except Exception as e:
        print(f"  Query generation failed ({e}), using fallback.")
        queries = [related_work, thesis[:80]]

    print(f"  Queries: {queries}\n")

    # Step 2: Search both sources in parallel
    # LangChain concept: for I/O-bound tasks like API calls, run them concurrently.
    # In a full LangChain agent this would be a RunnableParallel.
    print("[2/3] Searching Semantic Scholar + arXiv in parallel...")
    all_papers = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        ss_futures = [executor.submit(search_semantic_scholar, q, 4) for q in queries]
        ax_futures = [executor.submit(search_arxiv, q, 3) for q in queries]

        for f in ss_futures + ax_futures:
            all_papers.extend(f.result())
            time.sleep(0.1)  # light rate limiting

    unique_papers = deduplicate(all_papers)
    print(f"  Found {len(all_papers)} papers, {len(unique_papers)} unique.\n")

    if not unique_papers:
        print("No papers found. Try broader search terms.")
        return

    # Step 3: LLM ranks papers by relevance
    print(f"[3/3] Ranking top {top_n} by relevance to your thesis...")
    try:
        rank_result = rank_chain.invoke({
            "thesis": thesis,
            "related_work": related_work,
            "top_n": top_n,
            "papers_json": json.dumps(unique_papers[:25], indent=2)  # cap at 25 to fit context
        })
        ranked = rank_result.get("ranked", [])
    except Exception as e:
        print(f"  Ranking failed ({e}), showing raw results.")
        ranked = unique_papers[:top_n]

    # Print results
    print(f"\n{'='*60}")
    print(f"  TOP {len(ranked)} PAPERS FOR YOUR THESIS")
    print(f"{'='*60}")

    for i, p in enumerate(ranked, 1):
        score = p.get("relevance_score", "")
        score_str = f"  [{score}/10]" if score else ""
        print(f"\n{i}. {p.get('title', 'Unknown title')}")
        print(f"   {p.get('authors', '')} · {p.get('year', '')} · {p.get('source', '')}{score_str}")
        print(f"   Why relevant: {p.get('why_relevant', '')}")
        print(f"   URL: {p.get('url', '')}")

    print(f"\n{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Research Paper Finder (LangChain + Ollama) ===")
    print("Make sure `ollama serve` is running in another terminal.\n")

    thesis = input("Your thesis / argument:\n> ").strip()
    related_work = input("\nDescribe your related work area (what kind of papers you need):\n> ").strip()
    field = input("\nField (e.g. machine learning, NLP, CV): ").strip() or "computer science"
    top_n_input = input("\nHow many papers to return? (default 8): ").strip()
    top_n = int(top_n_input) if top_n_input.isdigit() else 8

    find_papers(thesis, related_work, field, top_n)

    while True:
        again = input("Search again with a different related work area? (y/n): ").strip().lower()
        if again != "y":
            break
        related_work = input("\nNew related work area:\n> ").strip()
        find_papers(thesis, related_work, field, top_n)


if __name__ == "__main__":
    main()
