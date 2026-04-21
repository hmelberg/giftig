"""Static prompt content + assembly helpers for the gift-API.

The cached prefix (system prompt + safety rules + category overview) is
built once per worker from the in-memory corpus and re-used across
requests with Anthropic's cache_control. Per-request dynamic content
(retrieved articles, user question) is appended outside the cache
boundary.
"""

from __future__ import annotations

import retrieval


# ---------------------------------------------------------------------------
# Static content — stable across requests, lives inside the cache boundary

SYSTEM_PROMPT = """\
Du er en assistent som svarer på spørsmål om forgiftninger og giftige
stoffer, basert utelukkende på artikler fra helsenorge.no/giftinformasjon/
som du får siterte utdrag fra.

Regler:

1. Svar KUN på grunnlag av de siterte artiklene som følger med i brukerens
   tur. Finn aldri på detaljer. Hvis artiklene ikke dekker spørsmålet, si
   det tydelig og henvis til Giftinformasjonen.
2. Svar kort og konkret: 2-4 setninger. Brukeren får også en liste over
   relevante lenker ved siden av svaret ditt, så du trenger ikke gjenta
   URL-er i selve svar-teksten.
3. Svar alltid på norsk, selv om spørsmålet er på engelsk.
4. Ikke lag triage- eller alvorlighetsvurderinger på egen hånd (f.eks. "ring
   113" eller "dette er ikke alvorlig"). Nødnummer og ansvarsfraskrivelse
   legges til automatisk etter ditt svar.
5. Når du siterer, oppgi artikkelens `id`-felt (du ser det i hver candidate-
   artikkel). Ikke finn på id-er.

Du må svare med et JSON-objekt som matcher kontrakten i brukerens tur.
Ingen ekstra prosa utenfor JSON.
"""


SAFETY_RULES = """\
## Sikkerhetsregler (viktig)

- Ikke gi medisinsk behandlingsråd utover det som står i artiklene.
- Ikke anbefal at brukeren selv doserer motgift, fremkaller oppkast eller
  lar være å oppsøke hjelp.
- Hvis spørsmålet virker akutt (mistanke om alvorlig forgiftning, små
  barn som har svelget noe giftig, bevisstløshet, pustebesvær), si
  eksplisitt at brukeren bør ringe Giftinformasjonen (22 59 13 00) eller
  113 ved livstruende symptomer.
- Ved usikkerhet: henvis til Giftinformasjonen. Det er aldri galt.
"""


RESPONSE_STYLE = """\
## Stil på svaret

- 2-4 korte setninger, konkret og nyttig.
- Ingen overskrifter, punktlister eller markdown i selve `answer`-feltet.
- Bruk aktivt språk ("Skyll munnen med vann og drikk et glass melk") — ikke
  passive "det anbefales at".
- Gjenta aldri brukerens spørsmål. Gå rett på svaret.
"""


def build_categories_overview() -> str:
    """Render the category list so the model knows what the corpus covers."""
    cats = retrieval.categories()
    if not cats:
        return ""
    lines = [
        "## Kategorier i artikkelbasen",
        "",
        "Artiklene er organisert i disse kategoriene (antall artikler i parentes):",
        "",
    ]
    for c in cats:
        lines.append(f"- **{c.get('label', '')}** (`{c.get('slug', '')}`, {c.get('count', 0)} artikler)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cached prefix assembly


_cached_prefix: str | None = None


def cached_prefix() -> str:
    global _cached_prefix
    if _cached_prefix is None:
        _cached_prefix = "\n\n".join(
            filter(
                None,
                [
                    SAFETY_RULES,
                    RESPONSE_STYLE,
                    build_categories_overview(),
                ],
            )
        )
    return _cached_prefix


def refresh_cached_prefix() -> None:
    """Call after retrieval.reload_data_files() so a new corpus takes effect."""
    global _cached_prefix
    _cached_prefix = None


# ---------------------------------------------------------------------------
# Per-request dynamic content


def render_retrieved_articles(rows: list[dict], max_chars_per_section: int = 1200) -> str:
    """Compose the article payload for the model's context window.

    We prefer the structured `sections` dict when present (cleaner than raw
    body_text) and fall back to body_text if sections are empty. Each article
    is prefixed with `<article id="...">` tags so the model can cite by id
    without URL regex.
    """
    if not rows:
        return "## Kildeartikler\n\n(Ingen artikler matchet spørsmålet. Henvis brukeren til Giftinformasjonen.)"

    parts = ["## Kildeartikler", ""]
    for r in rows:
        parts.append(f'<article id="{r["id"]}">')
        parts.append(f"**Tittel:** {r.get('title', '')}")
        parts.append(f"**Kategori:** {r.get('category_label', '')}")
        if r.get("last_updated"):
            parts.append(f"**Sist oppdatert:** {r['last_updated']}")
        sections = r.get("sections") or {}
        body_parts: list[str] = []
        if sections:
            for key in ("symptomer", "forstehjelp", "giftighet", "beskrivelse",
                        "risiko", "virkestoff", "forvekslinger", "behandling"):
                val = sections.get(key)
                if val:
                    trimmed = val if len(val) <= max_chars_per_section else val[:max_chars_per_section] + " …"
                    body_parts.append(f"**{key}:** {trimmed}")
        if not body_parts:
            fallback = r.get("body_text") or ""
            if len(fallback) > max_chars_per_section:
                fallback = fallback[:max_chars_per_section] + " …"
            body_parts.append(fallback)
        parts.append("\n\n".join(body_parts))
        parts.append("</article>")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output contract

ASK_OUTPUT_CONTRACT = """\
Svar med et JSON-objekt på én linje (ingen markdown-fencing, ingen ekstra
prosa utenfor JSON):

{
  "answer": "2-4 korte setninger med svaret, på norsk",
  "citations": [
    {"article_id": "<id fra en <article id='...'> tag>", "note": "kort hint om hvorfor, f.eks. 'symptomer' eller 'førstehjelp'"}
  ],
  "urgency": "info" | "urgent"
}

- `citations`: kun artikkel-id-er du FAKTISK brukte til å formulere svaret.
  Maks 3 citations. Ikke finn på id-er — bruk eksakt streng fra <article id="...">.
- `urgency`: "urgent" hvis spørsmålet tyder på akutt forgiftning (små barn,
  bevisstløshet, pustebesvær, alvorlige symptomer). Ellers "info".
"""
