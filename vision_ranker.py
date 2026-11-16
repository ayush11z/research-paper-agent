"""
PDF figure/table analysis with Gemini vision.
Requires: pip install google-genai requests
GEMINI_API_KEY must be set in the environment. GEMINI_MODEL overrides the model
(default gemini-3.8-flash; the old gemini-1.5-flash ID has been retired).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from google import genai
from google.genai import types

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

_PROMPT = (
    "This is a research paper. The student's thesis is: {thesis}. "
    "Look at every figure and table. For each one, decide if it supports the thesis."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "figures_found": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "figure_id": {"type": "string"},
                    "caption": {"type": "string"},
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["figure_id", "caption", "relevance", "reason"],
            },
        },
        "vision_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "summary": {"type": "string"},
    },
    "required": ["figures_found", "vision_score", "summary"],
}


def _to_pdf_url(arxiv_url: str) -> str:
    return arxiv_url.replace("arxiv.org/abs/", "arxiv.org/pdf/")


def _analyze(client, paper: dict, thesis: str) -> None:
    try:
        resp = requests.get(_to_pdf_url(paper["url"]), timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        if len(resp.content) > 20 * 1024 * 1024:
            return
        result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=resp.content, mime_type="application/pdf"),
                _PROMPT.format(thesis=thesis),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=_SCHEMA,
            ),
        )
        data = json.loads(result.text)
        paper["vision_score"] = int(data.get("vision_score", 0))
        paper["figures_found"] = data.get("figures_found", [])
        paper["vision_summary"] = data.get("summary", "")
    except Exception:
        pass


def rank_with_vision(papers: list, thesis: str) -> list:
    """
    Analyze the top 5 arXiv papers (in parallel) with Gemini vision.
    Adds vision_score, figures_found, vision_summary to each analyzed paper.
    Returns all papers sorted by (final_score + vision_score) descending.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return papers

    client = genai.Client(api_key=api_key)
    arxiv_papers = [p for p in papers if "arxiv.org" in (p.get("url") or "")][:5]
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda p: _analyze(client, p, thesis), arxiv_papers))

    def _sort_key(p):
        return (p.get("final_score") or 0) + (p.get("vision_score") or 0)

    return sorted(papers, key=_sort_key, reverse=True)
