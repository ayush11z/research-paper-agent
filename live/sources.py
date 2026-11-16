"""Async paper search clients tuned for latency (short timeouts, in-memory cache).

OpenAlex is the primary source. Semantic Scholar is used only when an API key is
set (keyless requests are almost always rate-limited). arXiv is throttled to its
documented one request per three seconds and is skipped rather than waited on.
"""

import asyncio
import re
import time
import xml.etree.ElementTree as ET

import httpx

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ARXIV_STOP = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with", "vs", "via", "by"}


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def paper_key(p: dict) -> str:
    return (p.get("doi") or "").lower() or p.get("arxiv_id") or _norm_title(p["title"])[:80]


def _authors(names: list[str]) -> str:
    names = [n for n in names if n]
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")


def _abstract_from_inverted(idx: dict | None) -> str:
    if not idx:
        return ""
    words = sorted((pos, w) for w, positions in idx.items() for pos in positions)
    return " ".join(w for _, w in words)


def _finish(p: dict) -> dict:
    p["key"] = paper_key(p)
    return p


class PaperSearch:
    def __init__(self, client: httpx.AsyncClient, openalex_key: str = "",
                 s2_key: str = "", mailto: str = "", timeout: float = 3.0):
        self.client = client
        self.openalex_key = openalex_key
        self.s2_key = s2_key
        self.mailto = mailto
        self.timeout = timeout
        self._cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}
        self._arxiv_last = 0.0
        self._arxiv_lock = asyncio.Lock()

    @property
    def sources(self) -> list[str]:
        return ["openalex", "arxiv"] + (["semantic_scholar"] if self.s2_key else [])

    async def search(self, source: str, query: str) -> list[dict]:
        ck = (source, query.lower())
        hit = self._cache.get(ck)
        if hit and time.time() - hit[0] < 900:
            return hit[1]
        fn = {"openalex": self.openalex, "semantic_scholar": self.semantic_scholar,
              "arxiv": self.arxiv}[source]
        papers = [_finish(p) for p in await fn(query) if p.get("title")]
        self._cache[ck] = (time.time(), papers)
        return papers

    async def openalex(self, q: str, n: int = 10) -> list[dict]:
        params = {
            "search": q, "per-page": n,
            "select": "id,doi,title,authorships,publication_year,abstract_inverted_index,"
                      "cited_by_count,primary_location,best_oa_location",
        }
        if self.openalex_key:
            params["api_key"] = self.openalex_key
        if self.mailto:
            params["mailto"] = self.mailto
        r = await self.client.get("https://api.openalex.org/works", params=params, timeout=self.timeout)
        r.raise_for_status()
        out = []
        for w in r.json().get("results", []):
            loc = w.get("primary_location") or {}
            oa = w.get("best_oa_location") or {}
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            out.append({
                "title": w.get("title") or "",
                "authors": _authors([(a.get("author") or {}).get("display_name", "")
                                     for a in w.get("authorships", [])]),
                "year": w.get("publication_year"),
                "venue": (loc.get("source") or {}).get("display_name") or "",
                "abstract": _abstract_from_inverted(w.get("abstract_inverted_index"))[:600],
                "doi": doi,
                "url": f"https://doi.org/{doi}" if doi else (loc.get("landing_page_url") or w.get("id", "")),
                "pdf_url": oa.get("pdf_url") or loc.get("pdf_url") or "",
                "cited_by_count": w.get("cited_by_count") or 0,
                "source": "OpenAlex",
            })
        return out

    async def semantic_scholar(self, q: str, n: int = 10) -> list[dict]:
        r = await self.client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": q, "limit": n,
                    "fields": "title,authors,year,abstract,url,externalIds,citationCount,venue,openAccessPdf"},
            headers={"x-api-key": self.s2_key}, timeout=self.timeout,
        )
        r.raise_for_status()
        out = []
        for p in r.json().get("data", []):
            ext = p.get("externalIds") or {}
            out.append({
                "title": p.get("title") or "",
                "authors": _authors([a.get("name", "") for a in p.get("authors", [])]),
                "year": p.get("year"),
                "venue": p.get("venue") or "",
                "abstract": (p.get("abstract") or "")[:600],
                "doi": ext.get("DOI") or "",
                "arxiv_id": ext.get("ArXiv") or "",
                "url": p.get("url") or "",
                "pdf_url": (p.get("openAccessPdf") or {}).get("url") or "",
                "cited_by_count": p.get("citationCount") or 0,
                "source": "Semantic Scholar",
            })
        return out

    async def arxiv(self, q: str, n: int = 6) -> list[dict]:
        async with self._arxiv_lock:
            if time.time() - self._arxiv_last < 3.0:
                return []  # never block the live path waiting on arXiv's rate limit
            self._arxiv_last = time.time()
        words = [w for w in re.findall(r"[A-Za-z0-9\-]+", q) if w.lower() not in _ARXIV_STOP][:6]
        if not words:
            return []
        r = await self.client.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": " AND ".join(f"all:{w}" for w in words),
                    "max_results": n, "sortBy": "relevance"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        out = []
        for e in ET.fromstring(r.text).findall("a:entry", _ATOM):
            abs_url = (e.findtext("a:id", "", _ATOM) or "").strip()
            arxiv_id = abs_url.rsplit("/abs/", 1)[-1]
            pdf = next((l.get("href") for l in e.findall("a:link", _ATOM) if l.get("title") == "pdf"), "")
            out.append({
                "title": " ".join((e.findtext("a:title", "", _ATOM) or "").split()),
                "authors": _authors([a.findtext("a:name", "", _ATOM) for a in e.findall("a:author", _ATOM)]),
                "year": int((e.findtext("a:published", "", _ATOM) or "0000")[:4]) or None,
                "venue": "arXiv",
                "abstract": " ".join((e.findtext("a:summary", "", _ATOM) or "").split())[:600],
                "doi": e.findtext("arxiv:doi", "", _ATOM) or "",
                "arxiv_id": re.sub(r"v\d+$", "", arxiv_id),
                "url": abs_url,
                "pdf_url": pdf,
                "cited_by_count": 0,
                "source": "arXiv",
            })
        return out
