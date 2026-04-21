"""Claude call with prompt caching — gift-API Q&A.

Single entry point:
    answer_question(question) -> dict

No tool-use loop, no repair loop: the model gets the top-k articles up
front and answers in one shot. Keeps latency low (typical: 2-4 s).
"""

from __future__ import annotations

import json
import re

import anvil.secrets
from anthropic import Anthropic

import prompts
import retrieval

DEFAULT_MODEL = "claude-sonnet-4-6"


SAFETY_FOOTER = (
    "\n\nDette er ikke medisinsk rådgivning. Ved mistanke om forgiftning, "
    "ring Giftinformasjonen 22 59 13 00 (døgnåpen). Ved livstruende "
    "situasjoner ring 113."
)


def _client() -> Anthropic:
    api_key = anvil.secrets.get_secret("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key)


def _cached_prefix_block() -> dict:
    return {
        "type": "text",
        "text": prompts.cached_prefix(),
        "cache_control": {"type": "ephemeral"},
    }


_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except Exception:
        return None


def _recover_partial_json(raw: str) -> dict | None:
    if not raw:
        return None
    candidates = _JSON_OBJ_RE.findall(raw)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _ensure_safety_footer(answer: str) -> str:
    """Guarantee the safety footer appears, even if the model omitted it
    or the answer is empty. Idempotent — won't double-append if the
    footer is already present.
    """
    base = (answer or "").rstrip()
    if "22 59 13 00" in base:
        return base
    return (base + SAFETY_FOOTER).strip()


def _article_to_link(a: dict) -> dict:
    return {
        "id": a["id"],
        "title": a.get("title", ""),
        "url": a.get("url", ""),
        "category": a.get("category", ""),
        "category_label": a.get("category_label", ""),
        "last_updated": a.get("last_updated"),
    }


def _resolve_citations(citations_raw: list, articles_by_id: dict) -> list[dict]:
    """Turn the model's citation list (just ids + notes) into full link
    records. Filters out any id the model hallucinated.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for c in citations_raw or []:
        if not isinstance(c, dict):
            continue
        aid = c.get("article_id") or ""
        if not aid or aid in seen:
            continue
        seen.add(aid)
        art = articles_by_id.get(aid)
        if art is None:
            continue
        record = _article_to_link(art)
        note = (c.get("note") or "").strip()
        if note:
            record["note"] = note
        out.append(record)
    return out


def answer_question(question: str, k: int = 8) -> dict:
    """Answer a gift-related question and return {answer, citations,
    related_links, urgency, model, cache_stats, latency_ms_generation}.

    The `related_links` list is always the top-k BM25 results in rank
    order; `citations` is the subset the model explicitly drew from.
    Duplicates are removed.
    """
    articles = retrieval.search_articles(question, k=k)
    articles_by_id = {a["id"]: a for a in articles}

    related_links = [_article_to_link(a) for a in articles]

    if not articles:
        return {
            "answer": _ensure_safety_footer(
                "Jeg fant ingen artikler på helsenorge.no/giftinformasjon/ som "
                "dekker spørsmålet ditt. Ring Giftinformasjonen for å få hjelp."
            ),
            "citations": [],
            "related_links": related_links,
            "urgency": "info",
            "model": "",
            "cache_stats": {},
        }

    dynamic = prompts.render_retrieved_articles(articles)
    messages = [
        {
            "role": "user",
            "content": [
                _cached_prefix_block(),
                {
                    "type": "text",
                    "text": (
                        f"# Brukerens spørsmål\n\n{question}\n\n"
                        f"{dynamic}\n\n{prompts.ASK_OUTPUT_CONTRACT}"
                    ),
                },
            ],
        }
    ]

    client = _client()
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=700,
        system=prompts.SYSTEM_PROMPT,
        messages=messages,
    )

    text_out = ""
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            text_out = block.text
            break

    parsed = _parse_json_response(text_out) or _recover_partial_json(text_out)
    usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)

    if parsed is None:
        # Fall back to raw text as the answer so the user gets something.
        return {
            "answer": _ensure_safety_footer(text_out or ""),
            "citations": [],
            "related_links": related_links,
            "urgency": "info",
            "model": DEFAULT_MODEL,
            "cache_stats": usage,
        }

    citations = _resolve_citations(parsed.get("citations") or [], articles_by_id)
    urgency = parsed.get("urgency") if parsed.get("urgency") in ("info", "urgent") else "info"

    return {
        "answer": _ensure_safety_footer(parsed.get("answer") or ""),
        "citations": citations,
        "related_links": related_links,
        "urgency": urgency,
        "model": DEFAULT_MODEL,
        "cache_stats": usage,
    }
