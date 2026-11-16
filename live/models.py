"""Local Hugging Face models used on the live path.

Both models stay loaded in memory, so per-utterance cost is milliseconds
(measured on an M2 Pro / MPS):
  - keyphrase extraction  (ml6team/keyphrase-extraction-distilbert-inspec)  ~8 ms
  - embedding re-ranking   (BAAI/bge-small-en-v1.5, 25 papers)               ~35 ms
"""

import re
import threading
from collections import OrderedDict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

_BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

# Words that the keyphrase model sometimes tags in conversational speech but
# that make useless search queries on their own.
_STOP = set("""
a an the and or but so of to in on for with at by from as is are was were be been it this that these
those we you they i he she our your their my me us them just like yeah okay ok um uh right really
actually basically kind sort thing things stuff way lot lots people guy guys idea ideas paper papers
work works result results problem problems approach approaches method methods model models data
question questions point points time something anything everything someone meeting today week
""".split())


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_phrase(p: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\- ]", " ", p.lower())).strip()


class KeyphraseExtractor:
    """Token-classification keyphrase model trained on scientific abstracts (Inspec)."""

    def __init__(self, name: str, device: str):
        self.pipe = pipeline("token-classification", model=name,
                             aggregation_strategy="simple", device=device)
        self._lock = threading.Lock()

    def __call__(self, text: str) -> list[tuple[str, float]]:
        if len(text.split()) < 3:
            return []
        with self._lock:
            ents = self.pipe(text)
        out, seen = [], set()
        for e in ents:
            phrase = normalize_phrase(e["word"])
            words = phrase.split()
            if not words or phrase in seen:
                continue
            content = [w for w in words if w not in _STOP]
            if not content:
                continue
            # single words must be specific enough to be a useful query
            if len(content) == 1 and (len(content[0]) < 5 or e["score"] < 0.7):
                continue
            seen.add(phrase)
            out.append((phrase, float(e["score"])))
        return out


class Embedder:
    """Sentence-transformers embedder with an LRU cache of paper vectors."""

    def __init__(self, name: str, device: str, cache_size: int = 5000):
        self.model = SentenceTransformer(name, device=device)
        if "query" not in self.model.prompts and "bge" in name.lower():
            self.model.prompts["query"] = _BGE_QUERY_PROMPT
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def query(self, text: str) -> np.ndarray:
        with self._lock:
            return self.model.encode_query([text], normalize_embeddings=True)[0]

    def papers(self, papers: list[dict]) -> np.ndarray:
        missing = [p for p in papers if p["key"] not in self._cache]
        if missing:
            texts = [f"{p['title']}. {(p.get('abstract') or '')[:300]}" for p in missing]
            with self._lock:
                vecs = self.model.encode_document(texts, normalize_embeddings=True, batch_size=32)
            for p, v in zip(missing, vecs):
                self._cache[p["key"]] = v
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        for p in papers:
            self._cache.move_to_end(p["key"])
        return np.stack([self._cache[p["key"]] for p in papers])

    def rank(self, query_vec: np.ndarray, papers: list[dict]) -> list[dict]:
        if not papers:
            return []
        sims = self.papers(papers) @ query_vec
        ranked = []
        for p, s in zip(papers, sims):
            cites = p.get("cited_by_count") or 0
            prior = min(1.0, np.log10(1 + cites) / 4)  # 10k citations -> 1.0
            ranked.append({**p, "relevance": round(float(s), 3),
                           "score": round(float(s) * 0.85 + prior * 0.15, 3)})
        ranked.sort(key=lambda p: p["score"], reverse=True)
        return ranked
