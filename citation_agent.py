"""
Research Paper Finder — Improved Pipeline
Run modes:
  python citation_agent.py            <- interactive CLI
  python citation_agent.py --server   <- called by Node server, reads env vars, prints JSON
"""

import arxiv
import requests
import json
import time
import sys
import os
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import citation_graph
import analysis

SERVER_MODE = "--server" in sys.argv

llm = ChatOllama(model="llama3", temperature=0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)

def emit(paper):
    if SERVER_MODE:
        print(json.dumps(paper), flush=True)

def _llm(system, human):
    raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    text = raw.content if hasattr(raw, "content") else str(raw)
    return text.strip().replace("```json", "").replace("```", "").strip()


# ── Query generation ──────────────────────────────────────────────────────────

def make_queries(thesis, related_work, field, keywords):
    topic = f"Thesis: {thesis}\nRelated work: {related_work}\nField: {field}\nKeywords: {keywords}"
    prompt = f"""Generate 5 search queries for academic papers.
Return ONLY valid JSON with exactly these 5 keys:
{{
  "direct": "thesis rephrased as a short plain query",
  "synonym": "rephrase the thesis using different vocabulary/synonyms",
  "method": "focus on the methodology or techniques involved",
  "domain": "focus on the problem domain and application area",
  "opposing": "limitations of or arguments against the thesis approach"
}}

Topic:
{topic}

Respond ONLY with the JSON:"""
    try:
        parsed = json.loads(_llm(
            "You are a research assistant. Respond with valid JSON only. No markdown.",
            prompt
        ))
        queries = {
            "direct": parsed.get("direct", thesis[:100]),
            "synonym": parsed.get("synonym", related_work[:100]),
            "method": parsed.get("method", ""),
            "domain": parsed.get("domain", field + " " + thesis[:50]),
            "opposing": parsed.get("opposing", "limitations of " + thesis[:80]),
        }
        log(f"  Queries: {list(queries.values())}")
        return queries
    except Exception as e:
        log(f"  LLM query generation failed ({e}), using fallback.")
        words = thesis.split()
        return {
            "direct": thesis[:100],
            "synonym": related_work[:100],
            "method": " ".join(words[:6]) if words else thesis[:60],
            "domain": field + " " + " ".join(words[:4]),
            "opposing": "limitations of " + " ".join(words[:5]),
        }


# ── Abstract reconstruction (OpenAlex inverted index) ─────────────────────────

def reconstruct_abstract(inv_index):
    if not inv_index:
        return ""
    try:
        pos_words = []
        for word, positions in inv_index.items():
            for pos in positions:
                pos_words.append((pos, word))
        pos_words.sort()
        return " ".join(w for _, w in pos_words)
    except Exception:
        return ""


# ── Fetchers ──────────────────────────────────────────────────────────────────

def search_openalex(query, limit=5):
    if not query.strip():
        return []
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": limit,
                "select": "title,authorships,publication_year,abstract_inverted_index,doi,cited_by_count,primary_location",
            },
            timeout=15,
        )
        resp.raise_for_status()
        papers = []
        for p in resp.json().get("results", []):
            auth_list = p.get("authorships", [])
            authors = ", ".join(
                a.get("author", {}).get("display_name", "") for a in auth_list[:3]
            ) + (" et al." if len(auth_list) > 3 else "")

            doi = (p.get("doi") or "").replace("https://doi.org/", "")
            primary_loc = p.get("primary_location") or {}
            source_obj = primary_loc.get("source") or {}
            venue = source_obj.get("display_name", "") or ""
            abstract = reconstruct_abstract(p.get("abstract_inverted_index"))
            year = p.get("publication_year")

            papers.append({
                "title": p.get("title", "") or "",
                "authors": authors,
                "year": str(year) if year else "N/A",
                "abstract": abstract[:400],
                "url": f"https://doi.org/{doi}" if doi else "",
                "doi": doi,
                "arxiv_id": "",
                "cited_by_count": p.get("cited_by_count", 0) or 0,
                "venue": venue,
                "source": "OpenAlex",
                "ss_id": "",
            })
        return papers
    except Exception as e:
        log(f"  [OpenAlex error: {e}]")
        return []


def search_semantic_scholar(query, limit=5):
    if not query.strip():
        return []
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,authors,year,abstract,url,externalIds,citationCount,venue,paperId",
            },
            timeout=15,
        )
        resp.raise_for_status()
        papers = []
        for p in resp.json().get("data", []):
            auth_list = p.get("authors", [])
            authors = ", ".join(a["name"] for a in auth_list[:3]) + (" et al." if len(auth_list) > 3 else "")
            ext = p.get("externalIds", {}) or {}
            doi = ext.get("DOI", "") or ""
            arxiv_id = ext.get("ArXiv", "") or ""
            papers.append({
                "title": p.get("title", "") or "",
                "authors": authors,
                "year": str(p.get("year", "N/A")),
                "abstract": (p.get("abstract") or "")[:400],
                "url": p.get("url", "") or "",
                "doi": doi,
                "arxiv_id": arxiv_id,
                "cited_by_count": p.get("citationCount", 0) or 0,
                "venue": p.get("venue", "") or "",
                "source": "Semantic Scholar",
                "ss_id": p.get("paperId", "") or "",
            })
        return papers
    except Exception as e:
        log(f"  [Semantic Scholar error: {e}]")
        return []


def search_crossref(query, limit=4):
    if not query.strip():
        return []
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={
                "query": query,
                "rows": limit,
                "select": "title,author,published,abstract,DOI,is-referenced-by-count,container-title",
            },
            headers={"User-Agent": "ResearchPaperFinder/1.0 (mailto:research@finder.com)"},
            timeout=15,
        )
        resp.raise_for_status()
        papers = []
        for p in resp.json().get("message", {}).get("items", []):
            title_list = p.get("title", [])
            title = title_list[0] if title_list else ""
            auth_list = p.get("author", [])
            authors = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip() for a in auth_list[:3]
            ) + (" et al." if len(auth_list) > 3 else "")
            pub = p.get("published", {}).get("date-parts", [[None]])
            year = str(pub[0][0]) if pub and pub[0] and pub[0][0] else "N/A"
            doi = p.get("DOI", "") or ""
            venue_list = p.get("container-title", [])
            venue = venue_list[0] if venue_list else ""
            abstract = re.sub(r"<[^>]+>", "", p.get("abstract", "") or "")[:400]
            papers.append({
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "url": f"https://doi.org/{doi}" if doi else "",
                "doi": doi,
                "arxiv_id": "",
                "cited_by_count": p.get("is-referenced-by-count", 0) or 0,
                "venue": venue,
                "source": "Crossref",
                "ss_id": "",
            })
        return papers
    except Exception as e:
        log(f"  [Crossref error: {e}]")
        return []


def search_arxiv(query, limit=3):
    if not query.strip():
        return []
    try:
        papers = []
        for p in arxiv.Client().results(
            arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
        ):
            arxiv_id = p.entry_id.split("/abs/")[-1] if "/abs/" in p.entry_id else ""
            papers.append({
                "title": p.title,
                "authors": ", ".join(a.name for a in p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
                "year": str(p.published.year),
                "abstract": p.summary[:400],
                "url": p.entry_id,
                "doi": p.doi or "",
                "arxiv_id": arxiv_id,
                "cited_by_count": 0,
                "venue": "arXiv",
                "source": "arXiv",
                "ss_id": "",
            })
        return papers
    except Exception as e:
        log(f"  [arXiv error: {e}]")
        return []


# ── Deduplication ─────────────────────────────────────────────────────────────

def _title_words(title):
    return set(re.sub(r"[^\w\s]", "", title.lower()).split())


def deduplicate(papers):
    seen_dois, seen_arxiv_ids, seen_title_sets = set(), set(), []
    unique = []
    for p in papers:
        doi = (p.get("doi") or "").strip().lower()
        arxiv_id = (p.get("arxiv_id") or "").strip()
        tw = _title_words(p.get("title", ""))

        if doi and doi in seen_dois:
            continue
        if arxiv_id and arxiv_id in seen_arxiv_ids:
            continue
        if len(tw) > 2 and any(
            len(tw & s) / max(len(tw), len(s)) >= 0.8 for s in seen_title_sets
        ):
            continue

        if doi:
            seen_dois.add(doi)
        if arxiv_id:
            seen_arxiv_ids.add(arxiv_id)
        if len(tw) > 2:
            seen_title_sets.append(tw)
        unique.append(p)
    return unique


# ── Scoring ───────────────────────────────────────────────────────────────────

_TOP_VENUES = {
    "nature", "science", "neurips", "nips", "icml", "iclr", "cvpr", "eccv",
    "iccv", "acl", "emnlp", "naacl", "aaai", "ijcai", "kdd", "www", "sigir",
    "cell", "lancet", "jama", "pnas", "plos", "ieee transactions",
}


def _citation_impact(n):
    n = n or 0
    if n > 1000: return 10
    if n > 100:  return 7
    if n > 10:   return 4
    return 2


def _recency(year_str):
    try:
        y = int(year_str)
    except (ValueError, TypeError):
        return 3
    if y >= 2024: return 10
    if y >= 2022: return 8
    if y >= 2019: return 6
    return 3


def _venue_quality(venue, source):
    if not venue:
        return 5 if source == "arXiv" else 4
    vl = venue.lower()
    if any(v in vl for v in _TOP_VENUES):
        return 10
    return 6


def _batch_llm_scores(papers, thesis):
    items = []
    for i, p in enumerate(papers):
        items.append(f"{i}. {p.get('title','')}\n   {(p.get('abstract') or '')[:180]}")
    prompt = f"""Score each paper on two criteria (integers 1-10):
- semantic_match: how well this paper matches the thesis topic
- abstract_directness: does the abstract directly address the thesis claim

Thesis: {thesis}

Papers:
{chr(10).join(items)}

Respond ONLY with a JSON array, one object per paper (same order):
[{{"semantic_match":7,"abstract_directness":6}}, ...]"""
    try:
        text = _llm("You are a research evaluator. Respond with valid JSON only.", prompt)
        scores = json.loads(text)
        if isinstance(scores, list) and len(scores) == len(papers):
            return scores
    except Exception as e:
        log(f"  Batch LLM scoring failed: {e}")
    return [{"semantic_match": 5, "abstract_directness": 5}] * len(papers)


def rank_papers(papers, thesis, top_n):
    if not papers:
        return []
    log(f"  Scoring {len(papers)} papers with LLM...")
    llm_scores = _batch_llm_scores(papers, thesis)
    scored = []
    for i, p in enumerate(papers):
        s = llm_scores[i] if i < len(llm_scores) else {}
        sem = float(s.get("semantic_match", 5))
        direct = float(s.get("abstract_directness", 5))
        imp = _citation_impact(p.get("cited_by_count", 0))
        rec = _recency(p.get("year", ""))
        vq = _venue_quality(p.get("venue", ""), p.get("source", ""))
        final = sem * 0.35 + imp * 0.20 + rec * 0.15 + vq * 0.15 + direct * 0.15
        scored.append({
            **p,
            "semantic_match": round(sem, 1),
            "citation_impact": imp,
            "recency": rec,
            "venue_quality": vq,
            "abstract_directness": round(direct, 1),
            "final_score": round(final, 2),
        })
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_n]


# ── Why relevant ──────────────────────────────────────────────────────────────

def generate_why_relevant(papers, thesis):
    items = [f"{i}. {p.get('title','')} — {(p.get('abstract') or '')[:150]}"
             for i, p in enumerate(papers)]
    prompt = f"""For each paper, write one sentence explaining why it is relevant to the thesis.

Thesis: {thesis}

Papers:
{chr(10).join(items)}

Respond ONLY with a JSON array of strings (one per paper, same order):
["reason 0", "reason 1", ...]"""
    try:
        reasons = json.loads(_llm("You are a research assistant. Respond with valid JSON only.", prompt))
        if isinstance(reasons, list):
            for i, p in enumerate(papers):
                if i < len(reasons):
                    p["why_relevant"] = reasons[i]
            return
    except Exception as e:
        log(f"  why_relevant generation failed: {e}")
    for p in papers:
        if not p.get("why_relevant"):
            p["why_relevant"] = ""


# ── Unpaywall ─────────────────────────────────────────────────────────────────

def get_pdf_url(doi):
    if not doi:
        return ""
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "research@finder.com"},
            timeout=8,
        )
        if resp.status_code == 200:
            for loc in resp.json().get("oa_locations", []):
                pdf = loc.get("url_for_pdf") or loc.get("pdf_url") or ""
                if pdf:
                    return pdf
    except Exception:
        pass
    return ""


# ── Paper summarizer ──────────────────────────────────────────────────────────

def summarize_paper(paper, thesis):
    prompt = f"""Summarize this paper's relevance to the thesis: {thesis}

Title: {paper.get('title', '')}
Abstract: {(paper.get('abstract') or '')[:400]}

Respond in JSON only:
{{
  "claim": "main claim of the paper in one sentence",
  "method": "methodology used",
  "dataset": "dataset or evaluation used, or N/A",
  "result": "key result or finding",
  "supports_thesis": "supports / contradicts / neutral",
  "relevance_quote": "most relevant sentence from the abstract"
}}"""
    try:
        return json.loads(_llm(
            "You are a research assistant. Respond with valid JSON only. No markdown.", prompt
        ))
    except Exception as e:
        log(f"  summarize_paper failed for '{paper.get('title','')[:40]}': {e}")
        return None


# ── S2 recommendations ────────────────────────────────────────────────────────

def get_s2_recommendations(ss_id):
    if not ss_id:
        return []
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{ss_id}",
            params={"fields": "title,authors,year,abstract,citationCount,url,externalIds,venue"},
            timeout=15,
        )
        if resp.status_code == 429:
            return []
        resp.raise_for_status()
        papers = []
        for p in resp.json().get("recommendedPapers", [])[:5]:
            auth_list = p.get("authors", [])
            authors = ", ".join(a["name"] for a in auth_list[:3]) + (" et al." if len(auth_list) > 3 else "")
            ext = p.get("externalIds", {}) or {}
            papers.append({
                "title": p.get("title", "") or "",
                "authors": authors,
                "year": str(p.get("year", "N/A")),
                "abstract": (p.get("abstract") or "")[:400],
                "url": p.get("url", "") or "",
                "doi": ext.get("DOI", "") or "",
                "arxiv_id": ext.get("ArXiv", "") or "",
                "cited_by_count": p.get("citationCount", 0) or 0,
                "venue": p.get("venue", "") or "",
                "source": "S2 Recommended",
                "ss_id": "",
            })
        return papers
    except Exception as e:
        log(f"  [S2 recommendations error: {e}]")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def find_papers(thesis, related_work, field, keywords, top_n=8):
    log("\n[1/6] Generating search queries...")
    queries = make_queries(thesis, related_work, field, keywords)
    query_list = list(queries.values())

    log("\n[2/6] Searching 4 sources...")
    all_papers = []

    log("  → OpenAlex")
    for q in query_list:
        results = search_openalex(q, limit=5)
        all_papers.extend(results)
        log(f"    '{q[:50]}': {len(results)}")
        time.sleep(1)

    log("  → Semantic Scholar (2 queries)")
    for q in query_list[:2]:
        results = search_semantic_scholar(q, limit=5)
        all_papers.extend(results)
        log(f"    '{q[:50]}': {len(results)}")
        time.sleep(2)

    log("  → Crossref")
    for q in query_list:
        results = search_crossref(q, limit=4)
        all_papers.extend(results)
        log(f"    '{q[:50]}': {len(results)}")
        time.sleep(1)

    log("  → arXiv")
    for q in query_list:
        results = search_arxiv(q, limit=3)
        all_papers.extend(results)
        log(f"    '{q[:50]}': {len(results)}")

    unique = deduplicate(all_papers)
    log(f"  {len(all_papers)} raw → {len(unique)} unique after deduplication.")

    if not unique:
        log("No papers found. Try broader terms.")
        return

    log(f"\n[3/6] Ranking {len(unique)} papers...")
    ranked = rank_papers(unique[:50], thesis, top_n * 3)

    log("\n[4/6] Expanding with citation graph + S2 recommendations...")
    expanded = citation_graph.expand(ranked[:top_n])
    log(f"  {len(expanded) - len(ranked[:top_n])} new via citation graph.")

    rec_papers = []
    for p in ranked[:3]:
        ss_id = p.get("ss_id", "")
        if not ss_id and "semanticscholar.org" in p.get("url", ""):
            ss_id = p["url"].rstrip("/").split("/")[-1]
        if ss_id:
            recs = get_s2_recommendations(ss_id)
            rec_papers.extend(recs)
            log(f"  S2 recs for '{p.get('title','')[:40]}': {len(recs)}")
            time.sleep(2)

    all_expanded = deduplicate(expanded + rec_papers)
    if len(all_expanded) > len(ranked[:top_n]):
        log(f"  Re-ranking {len(all_expanded)} papers...")
        ranked = rank_papers(all_expanded[:60], thesis, top_n)
    else:
        ranked = ranked[:top_n]

    log("\n[5/6] PDF links, summaries...")
    for i, p in enumerate(ranked):
        doi = p.get("doi", "")
        pdf_url = get_pdf_url(doi) if doi else ""
        if not pdf_url and p.get("source") == "arXiv" and p.get("url"):
            pdf_url = p["url"].replace("/abs/", "/pdf/") + ".pdf"
        p["pdf_url"] = pdf_url

        if i < 8:
            p["summary"] = summarize_paper(p, thesis)
            log(f"  Summarized [{i+1}/8]: {p.get('title','')[:50]}")
        else:
            p["summary"] = None

    generate_why_relevant(ranked, thesis)

    log("\n[6/6] Vision analysis...")
    try:
        from vision_ranker import rank_with_vision
        ranked = rank_with_vision(ranked, thesis)
    except Exception as e:
        log(f"  Vision analysis skipped: {e}")

    log(f"\n[done] Emitting {len(ranked)} papers.")
    for paper in ranked:
        emit(paper)
        if not SERVER_MODE:
            score = paper.get("final_score", "?")
            print(f"\n  [{score}] {paper.get('title')}")
            print(f"  {paper.get('authors')} · {paper.get('year')} · {paper.get('source')}")
            print(f"  Why: {paper.get('why_relevant')}")

    log("\nAnalyzing research gaps...")
    gaps = analysis.find_gaps(all_expanded[:20], thesis)
    log(f"  Found {len(gaps)} gaps.")

    log("\nFinding contradictions...")
    contradictions = analysis.find_contradictions(all_expanded[:20], thesis)
    log(f"  Found {len(contradictions)} contradictions.")

    if SERVER_MODE:
        print(json.dumps({"type": "gaps", "data": gaps}), flush=True)
        print(json.dumps({"type": "contradictions", "data": contradictions}), flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if SERVER_MODE:
        find_papers(
            os.environ.get("THESIS", ""),
            os.environ.get("RELATED_WORK", ""),
            os.environ.get("FIELD", "computer science"),
            os.environ.get("KEYWORDS", ""),
            int(os.environ.get("TOP_N", "8")),
        )
    else:
        print("\n=== Research Paper Finder ===")
        print("Make sure `ollama serve` is running.\n")
        thesis = input("Thesis:\n> ").strip()
        related_work = input("\nRelated work area:\n> ").strip()
        field = input("\nField: ").strip() or "computer science"
        keywords = input("\nKeywords (comma-separated): ").strip()
        top_n = int(t) if (t := input("\nHow many papers? (default 8): ").strip()).isdigit() else 8
        find_papers(thesis, related_work, field, keywords, top_n)
