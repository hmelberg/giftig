"""BM25 retrieval + synonym expansion over helsenorge.no/giftinformasjon articles.

Reads its corpus from Anvil Data Files (not Data Tables):

    corpus.pkl     — pickled dict with `articles`, `categories`, and a pre-
                     tokenised BM25 corpus. Produced by
                     seed/build_data_files.py.
    synonyms.json  — hand-authored domain synonyms.

At server-module import the files are loaded once; BM25Okapi is constructed
from the pre-tokenised corpus (fast — no stemming at startup).

Call `reload_data_files()` after uploading a new corpus.pkl to pick up
changes without restarting the worker.
"""

from __future__ import annotations

import json
import pickle
import re
import unicodedata
from dataclasses import dataclass

import anvil.server
from anvil.files import data_files
from rank_bm25 import BM25Okapi

try:
    import snowballstemmer
    _NO_STEMMER = snowballstemmer.stemmer("norwegian")
    _EN_STEMMER = snowballstemmer.stemmer("english")
except ImportError:  # pragma: no cover
    _NO_STEMMER = None
    _EN_STEMMER = None


_TOKEN_RE = re.compile(r"[a-zA-ZæøåÆØÅ0-9]{2,}", re.UNICODE)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text.lower())


def tokenize(text: str) -> list[str]:
    """Must match seed/build_data_files.py::tokenize exactly."""
    raw = _TOKEN_RE.findall(_fold(text))
    if not raw:
        return []
    if _NO_STEMMER is None:
        return raw
    return _NO_STEMMER.stemWords(raw) + _EN_STEMMER.stemWords(raw)


# ---------------------------------------------------------------------------
# Module-level state


@dataclass
class _Index:
    bm25: BM25Okapi | None
    docs: list[dict]


_corpus: dict | None = None
_articles_index: _Index = _Index(bm25=None, docs=[])
_synonyms_cache: dict[str, list[str]] = {}
_articles_by_id: dict[str, dict] = {}


def _load_corpus() -> None:
    global _corpus, _articles_index, _synonyms_cache, _articles_by_id

    with open(data_files["corpus.pkl"], "rb") as f:
        _corpus = pickle.load(f)

    bm25_tokens = (_corpus.get("bm25") or {}).get("articles") or []
    _articles_index = _Index(
        bm25=BM25Okapi(bm25_tokens) if bm25_tokens else None,
        docs=_corpus.get("articles") or [],
    )
    _articles_by_id = {a["id"]: a for a in _articles_index.docs}

    # Synonyms
    try:
        with open(data_files["synonyms.json"], "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        rows = []
    syn: dict[str, list[str]] = {}
    for row in rows:
        term = (row.get("term") or "").lower()
        syns = row.get("synonyms") or []
        if term and syns:
            syn.setdefault(term, []).extend(syns)
    _synonyms_cache = syn


def _ensure_loaded() -> None:
    if _corpus is None:
        _load_corpus()


@anvil.server.callable
def reload_data_files() -> dict:
    """Re-read corpus.pkl + synonyms.json after a fresh upload.

    Also busts the cached prompt prefix so any new content (category
    summaries) takes effect on the next request.
    """
    _load_corpus()
    try:
        import prompts
        prompts.refresh_cached_prefix()
    except Exception:
        pass
    return {
        "articles": len(_articles_index.docs),
        "categories": len((_corpus or {}).get("categories") or []),
        "synonyms": len(_synonyms_cache),
        "schema_version": (_corpus or {}).get("schema_version", 1),
    }


# ---------------------------------------------------------------------------
# Accessors used by prompts/endpoints


def get_corpus() -> dict:
    _ensure_loaded()
    assert _corpus is not None
    return _corpus


def articles_by_id() -> dict[str, dict]:
    _ensure_loaded()
    return _articles_by_id


def categories() -> list[dict]:
    _ensure_loaded()
    return (_corpus or {}).get("categories") or []


# ---------------------------------------------------------------------------
# Query helpers


def expand_synonyms(query: str) -> list[str]:
    _ensure_loaded()
    tokens = _TOKEN_RE.findall(_fold(query))
    expanded: list[str] = list(tokens)
    for tok in tokens:
        for syn in _synonyms_cache.get(tok, []):
            expanded.append(syn.lower())
    return expanded


def _bm25_top_k(idx: _Index, query_tokens: list[str], k: int) -> list[tuple[float, dict]]:
    if not idx.bm25 or not query_tokens:
        return []
    scores = idx.bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    out: list[tuple[float, dict]] = []
    for i, s in ranked[:k]:
        if s <= 0:
            break
        out.append((float(s), idx.docs[i]))
    return out


def search_articles(query: str, k: int = 8, category: str | None = None) -> list[dict]:
    """Return the top-k most relevant articles for `query`.

    Each row is the full article dict (title, url, sections, body_text,
    category, ...) so callers can both rank and cite. If `category` is
    provided, results are filtered to that category slug after scoring.
    """
    _ensure_loaded()
    tokens = tokenize(" ".join(expand_synonyms(query)))
    # Fetch a wider candidate set so the post-filter by category still has
    # enough to work with.
    hits = _bm25_top_k(_articles_index, tokens, k=k * 3 if category else k)
    if category:
        hits = [(s, d) for s, d in hits if d.get("category") == category]
    return [d for _, d in hits[:k]]


# ---------------------------------------------------------------------------
# Server-callable variants (used by /search endpoint if you expose one)


@anvil.server.callable
def server_search_articles(query: str, k: int = 8, category: str | None = None) -> list[dict]:
    hits = search_articles(query=query, k=k, category=category)
    return [
        {
            "id": h["id"],
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "category": h.get("category", ""),
            "category_label": h.get("category_label", ""),
            "last_updated": h.get("last_updated"),
            "severity_hints": h.get("severity_hints") or [],
        }
        for h in hits
    ]
