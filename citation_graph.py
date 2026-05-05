import requests
import time


def _extract_ss_id(url):
    """Extract Semantic Scholar paper ID from a semanticscholar.org URL."""
    if not url or "semanticscholar.org" not in url:
        return None
    # URL format: https://www.semanticscholar.org/paper/Title/paperId
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-1]
    return None


def _fetch_references(paper_id):
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references",
            params={"fields": "title,authors,year,abstract,url"},
            timeout=15
        )
        if resp.status_code == 429:
            return []
        resp.raise_for_status()
        refs = []
        for item in resp.json().get("data", []):
            p = item.get("citedPaper", {})
            if not p.get("title"):
                continue
            authors = p.get("authors", [])
            refs.append({
                "title": p.get("title", ""),
                "authors": ", ".join(a["name"] for a in authors[:3]) + (" et al." if len(authors) > 3 else ""),
                "year": str(p.get("year", "N/A")),
                "abstract": (p.get("abstract") or "")[:300],
                "url": p.get("url", ""),
                "source": "citation_graph"
            })
        return refs
    except Exception:
        return []


def expand(papers):
    """Expand paper list via Semantic Scholar citation graph references."""
    existing_titles = {p["title"].lower().strip()[:60] for p in papers}
    new_papers = []

    for paper in papers:
        ss_id = _extract_ss_id(paper.get("url", ""))
        if not ss_id:
            continue  # skip arXiv-only or missing URLs silently

        refs = _fetch_references(ss_id)
        for ref in refs:
            key = ref["title"].lower().strip()[:60]
            if not any(key in t or t in key for t in existing_titles):
                existing_titles.add(key)
                new_papers.append(ref)

        time.sleep(2)

    return papers + new_papers[:40]
