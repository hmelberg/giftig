"""HTTP endpoints exposed by the gift-API Anvil app.

All endpoints are reachable at:

    https://<app>.anvil.app/_/api<path>

All endpoints require a valid X-API-Key header. Responses are JSON; 2xx on
success, 4xx on client errors (bad auth, rate limit, bad body), 5xx on
unexpected server errors.

Endpoints:
    POST /ask            — fritekst-spørsmål → {answer, citations, related_links, urgency}
    GET  /search         — BM25-oppslag uten LLM (debug / søkebokser i UI)
    GET  /health         — liveness probe (ingen auth)
"""

from __future__ import annotations

import json
import time

import anvil.server
from anvil.server import HttpResponse

import generation
import retrieval
import utils


def _json(body: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status=status,
        body=json.dumps(body, ensure_ascii=False),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


def _load_body() -> dict:
    req = anvil.server.request
    body = req.body_json
    if body is None and req.body:
        try:
            body = json.loads(req.body.get_bytes().decode("utf-8"))
        except Exception:
            body = None
    return body or {}


def _authenticate_or_fail():
    req = anvil.server.request
    alias = utils.authenticate(req)
    if not alias:
        return None, _json({"error": "invalid or missing X-API-Key"}, status=401)
    if not utils.check_rate_limit(alias):
        return None, _json({"error": "rate limit exceeded"}, status=429)
    return alias, None


# ---------------------------------------------------------------------------
# /ask


@anvil.server.http_endpoint("/ask", methods=["POST"], cross_site_session=False, enable_cors=True)
def http_ask():
    alias, err = _authenticate_or_fail()
    if err:
        return err

    body = _load_body()
    question = (body.get("question") or "").strip()
    if not question:
        return _json({"error": "missing 'question'"}, status=400)
    try:
        k = int(body.get("k", 8))
    except (TypeError, ValueError):
        k = 8
    k = max(3, min(k, 15))

    t0 = time.time()
    try:
        result = generation.answer_question(question=question, k=k)
    except Exception as exc:
        latency_ms = int((time.time() - t0) * 1000)
        utils.log_request(
            endpoint="/ask",
            question=question,
            latency_ms=latency_ms,
            api_key_alias=alias,
            error=f"{type(exc).__name__}: {exc}",
        )
        return _json({"error": "internal error", "detail": str(exc)}, status=500)
    latency_ms = int((time.time() - t0) * 1000)

    utils.log_request(
        endpoint="/ask",
        question=question,
        model=result.get("model", ""),
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        related_links=result.get("related_links", []),
        latency_ms=latency_ms,
        cache_stats=result.get("cache_stats") or {},
        api_key_alias=alias,
    )
    result["latency_ms"] = latency_ms
    return _json(result)


# ---------------------------------------------------------------------------
# /search  (BM25 only, no LLM — useful for a search-as-you-type UI)


@anvil.server.http_endpoint("/search", methods=["GET"], cross_site_session=False, enable_cors=True)
def http_search(**kwargs):
    alias, err = _authenticate_or_fail()
    if err:
        return err
    q = (kwargs.get("q") or "").strip()
    if not q:
        return _json({"error": "missing 'q'"}, status=400)
    category = (kwargs.get("category") or "").strip() or None
    try:
        k = int(kwargs.get("k", 10))
    except (TypeError, ValueError):
        k = 10
    k = max(1, min(k, 25))

    hits = retrieval.server_search_articles(query=q, k=k, category=category)
    return _json({"results": hits})


# ---------------------------------------------------------------------------
# /health  (no auth — simple liveness)


@anvil.server.http_endpoint("/health", methods=["GET"], cross_site_session=False, enable_cors=True)
def http_health():
    try:
        cats = retrieval.categories()
        n_articles = sum(c.get("count", 0) for c in cats)
        return _json({
            "status": "ok",
            "articles": n_articles,
            "categories": len(cats),
        })
    except Exception as exc:
        return _json({"status": "degraded", "error": str(exc)}, status=503)
