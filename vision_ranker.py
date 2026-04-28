"""
PDF figure/table analysis using Gemini 1.5 Flash vision.
Requires: pip install google-generativeai requests
GEMINI_API_KEY must be set in the environment.
"""

import os
import json
import base64
import requests
import google.generativeai as genai

_PROMPT = (
    "This is a research paper. The student's thesis is: {thesis}. "
    "Look at every figure and table. For each one, decide if it supports the thesis. "
    "Respond in JSON only, no markdown: "
    '{{ "figures_found": [{{"figure_id": "...", "caption": "...", '
    '"relevance": "high/medium/low", "reason": "..."}}], '
    '"vision_score": <1-10>, "summary": "<one sentence>" }}'
)

def _to_pdf_url(arxiv_url: str) -> str:
    return arxiv_url.replace("arxiv.org/abs/", "arxiv.org/pdf/")

def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def rank_with_vision(papers: list, thesis: str) -> list:
    """
    Analyze the top 5 arXiv papers using Gemini 1.5 Flash vision.
    Adds vision_score, figures_found, vision_summary to each analyzed paper.
    Returns all papers sorted by (relevance_score + vision_score) descending.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return papers

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    arxiv_papers = [p for p in papers if "arxiv.org" in (p.get("url") or "")][:5]

    for paper in arxiv_papers:
        pdf_url = _to_pdf_url(paper["url"])
        try:
            resp = requests.get(
                pdf_url, timeout=30,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            pdf_bytes = resp.content
            if len(pdf_bytes) > 20 * 1024 * 1024:
                continue
        except Exception:
            continue

        prompt = _PROMPT.format(thesis=thesis)
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode()
        try:
            response = model.generate_content([
                {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
                prompt,
            ])
            vision_data = _parse_json(response.text)
            paper["vision_score"] = int(vision_data.get("vision_score", 0))
            paper["figures_found"] = vision_data.get("figures_found", [])
            paper["vision_summary"] = vision_data.get("summary", "")
        except Exception:
            pass

    def _sort_key(p):
        return (p.get("relevance_score") or 0) + (p.get("vision_score") or 0)

    return sorted(papers, key=_sort_key, reverse=True)
