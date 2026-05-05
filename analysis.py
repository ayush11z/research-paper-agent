import json
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)


def _paper_summaries(papers):
    lines = []
    for p in papers:
        abstract = (p.get("abstract") or "")[:150]
        lines.append(f"- {p.get('title', '')}: {abstract}")
    return "\n".join(lines)


def _invoke(system_msg, human_msg):
    from langchain_core.messages import SystemMessage, HumanMessage
    raw = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=human_msg)])
    text = raw.content if hasattr(raw, "content") else str(raw)
    return text.strip().replace("```json", "").replace("```", "").strip()


def find_gaps(papers, thesis):
    summaries = _paper_summaries(papers)
    human = f"""Given this thesis and these papers, what important arguments, methodologies, or perspectives are NOT covered in this literature? List 4-6 specific gaps. For each gap, suggest what kind of paper would fill it.

Thesis: {thesis}

Papers:
{summaries}

Respond ONLY with:
{{"gaps": [{{"gap": "...", "suggestion": "..."}}]}}"""
    try:
        text = _invoke(
            "You are a research assistant. Respond with valid JSON only. No markdown, no explanation.",
            human
        )
        return json.loads(text).get("gaps", [])
    except Exception:
        return []


def find_contradictions(papers, thesis):
    summaries = _paper_summaries(papers)
    human = f"""Identify pairs of papers from this list that make opposing or conflicting claims relevant to this thesis. For each contradiction, name both papers and explain what they disagree on.

Thesis: {thesis}

Papers:
{summaries}

Respond ONLY with:
{{"contradictions": [{{"paper_a": "...", "paper_b": "...", "conflict": "..."}}]}}"""
    try:
        text = _invoke(
            "You are a research assistant. Respond with valid JSON only. No markdown, no explanation.",
            human
        )
        return json.loads(text).get("contradictions", [])
    except Exception:
        return []
