"""AI service for intelligent email and lead decisions.

Supports two backends behind one interface:

* **anthropic** — Claude via the Anthropic Messages API (the historical default).
* **openrouter** — the OpenAI-compatible *provider chain* owned by
  :mod:`app.services.llm_providers`: OpenRouter, then Google Gemini, Groq and
  Cerebras, each tried in turn until one answers.

The name "openrouter" is kept for that second backend because it is what
``AI_PROVIDER``, the per-org agent configs and the existing logs all say; it now
means "the OpenAI-compatible chain, headed by OpenRouter" rather than "OpenRouter
alone". A single provider's quota wall used to take every AI feature down to
static templates at once — see :mod:`app.services.llm_providers` for why that
kept going unnoticed.

The backend is chosen in :meth:`__init__` from settings (see ``ai_provider`` /
``openrouter_api_key`` / ``llm_provider_chain`` in :mod:`app.config`). All the
public methods build the same prompts regardless of backend and route the actual
call through :meth:`_chat` / :meth:`_chat_completion`, which normalise the
request and response shapes so the rest of the code never sees the difference.
"""

import logging
import json
import re
from datetime import date, datetime, timezone
from typing import Optional

import anthropic
import httpx  # noqa: F401  (kept importable for tests that patch httpx.AsyncClient)
from app.config import get_settings
from app.services.llm_providers import (
    LLMChainExhausted,
    build_provider_chain,
    chat_completion as chain_chat_completion,
    has_any_provider_key,
    to_openai_message,
)

logger = logging.getLogger(__name__)

# The five routing categories (+ the internal 'unknown' fallback) used by
# ``classify_reply``. Kept as a module constant so the task and tests share it.
REPLY_CATEGORIES = (
    "interested",
    "not_interested",
    "out_of_office",
    "wrong_person",
    "unsubscribe",
    "bounce",
    "unknown",
)

# --- Cheap keyword pre-filter (no LLM) -----------------------------------------
# High-precision phrases only. These are the compliance-critical / autoresponder
# signals we must never miss on an AI outage, so the patterns are deliberately
# narrow (explicit opt-out language, unambiguous OOO markers) to keep false
# positives near zero. Anything not matched here falls through to the LLM.

# The Anthropic model, named once. Used both as the native backend's model and
# as the cross-provider fallback when OpenRouter's free-tier quota is spent —
# ``self.model`` holds an OpenRouter model id in that case, which Anthropic
# would reject.
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Where a client stops the sender's own words and starts quoting ours. Used by
# ``strip_quoted_reply``; see it for why scanning quoted text is destructive.
# One pattern per client family actually seen in this mailbox:
#   Gmail / Apple Mail  "On Tue, Jul 22, 2026 at 4:02 PM Suman <s@x> wrote:"
#   Outlook (EN)        "-----Original Message-----" / "From: … Sent: …"
#   Outlook (web)       a long underscore rule above the quoted header
#   Our own footer      the sentinel that made this necessary at all
_QUOTE_HEADER_RE = re.compile(
    r"(^On\b.{0,200}\bwrote:\s*$)"
    r"|(^-{2,}\s*Original Message\s*-{2,})"
    r"|(^_{10,}$)"
    r"|(^From:\s.+)"
    r"|(^\s*Don'?t want these emails\?)",
    re.IGNORECASE,
)

# Explicit, legally-meaningful opt-out language. Matching any of these ALWAYS
# wins — unsubscribe is never gated by AI confidence.
_UNSUBSCRIBE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bunsubscribe\b",
        r"\bopt[\s-]?out\b",
        r"\bopt (?:me|us) out\b",
        r"\bremove me\b",
        r"\btake me off\b",
        r"\bremove (?:me )?from (?:your )?(?:the )?(?:mailing |email |distribution )?list\b",
        r"\bstop (?:emailing|contacting|messaging) me\b",
        r"\bdo not (?:email|contact) me\b",
        r"\bdon'?t (?:email|contact) me\b",
    )
]

# Delivery-status / bounce markers. A bounce is machine-generated and, unlike a
# human reply, ECHOES OUR ORIGINAL EMAIL — including its unsubscribe footer — so
# it MUST be detected before the unsubscribe pass or a dead address gets logged
# as an opt-out. Detection is deliberately high-precision: the sender is a mail
# daemon, or the subject is an unambiguous DSN subject.
_BOUNCE_SENDER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"mailer-daemon@",
        r"postmaster@",
        r"^mail delivery (?:subsystem|system)\b",
        r"no-?reply@.*(?:mailer|postmaster)",
    )
]
_BOUNCE_SUBJECT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"delivery status notification",
        r"undelivered mail returned to sender",
        r"mail delivery fail(?:ed|ure)",
        r"delivery (?:has )?failed",
        r"returned mail",
        r"\bfailure notice\b",
        r"\bundeliverable\b",
        r"delivery incomplete",
        r"could not be delivered",
    )
]


def is_bounce(
    subject: Optional[str], body_text: Optional[str], from_email: Optional[str]
) -> bool:
    """True when a message is a delivery-status / bounce notification.

    High precision: the sender is a mail daemon/postmaster, OR the subject is an
    unambiguous DSN subject. Body text is deliberately NOT scanned for phrases
    like "550" because a forwarded/quoted thread can contain them innocently.
    """
    sender = (from_email or "").strip()
    if any(p.search(sender) for p in _BOUNCE_SENDER_PATTERNS):
        return True
    if subject and any(p.search(subject) for p in _BOUNCE_SUBJECT_PATTERNS):
        return True
    return False


# Automated out-of-office / vacation autoresponder markers.
_OOO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bout of (?:the )?office\b",
        r"\bon vacation\b",
        r"\bon (?:annual |parental |medical |sick )?leave\b",
        r"\bon holiday\b",
        r"\bauto[\s-]?reply\b",
        r"\bautomatic reply\b",
        r"\bautoresponder\b",
        r"\baway from (?:my|the) (?:desk|office)\b",
        r"\bcurrently (?:away|travelling|traveling|unavailable)\b",
        r"\bi am away\b",
        r"\breturning to the office\b",
    )
]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def strip_quoted_reply(body_text: Optional[str]) -> str:
    """Return only what the sender actually typed, dropping the quoted thread.

    A mail client quotes the message being replied to, so the text of *our own
    email* — signature, CAN-SPAM footer and all — arrives inside every genuine
    reply. Scanning that for intent reads our words as theirs.

    This is not hypothetical. The outbound footer reads "Don't want these
    emails? Reply UNSUBSCRIBE or click: …", so a bare ``\\bunsubscribe\\b`` scan
    matched on every quoted reply. In production a real prospect answer to
    "Suman, quick question about Apprend Tech…" was classified ``unsubscribe``
    with confidence 1.0, the contact was suppressed, and its enrollments were
    stopped — the reply the whole system exists to catch, converted into an
    opt-out and an audit notification. Observed 2026-07-24 while auditing why
    Dialtone had no replies.

    Cuts at the first quote marker, covering the shapes every major client
    emits. Deliberately conservative: an unrecognised marker means we scan a bit
    of quoted text (the old behaviour), never that we drop something the sender
    wrote.
    """
    if not body_text:
        return ""
    lines = body_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            return "\n".join(lines[:i])
        if _QUOTE_HEADER_RE.search(stripped):
            return "\n".join(lines[:i])
    return body_text


def keyword_prefilter(
    subject: Optional[str],
    body_text: Optional[str],
    from_email: Optional[str] = None,
) -> Optional[str]:
    """Return a high-precision category from keywords alone, or ``None``.

    Runs before any LLM call. Returns ``"bounce"`` for delivery-status
    notifications, ``"unsubscribe"`` for explicit opt-out language, or
    ``"out_of_office"`` for autoresponder markers; ``None`` when nothing matched
    (the caller then asks the LLM).

    Order matters: **bounce is checked first**, because a DSN echoes our original
    email — footer and all — so scanning it for "unsubscribe" would suppress a
    dead address as if it had opted out. Unsubscribe is checked before OOO so an
    autoreply that also says "unsubscribe" is treated as the compliance-critical
    opt-out.

    The opt-out pass reads only the *unquoted* portion, for the same reason the
    bounce check runs first: our own footer invites the reader to "Reply
    UNSUBSCRIBE", and a quoted copy of it is our sentence, not theirs. A human
    who actually types "unsubscribe" types it above the quote, so no true
    positive is lost. OOO keeps scanning the whole body — an autoresponder's
    marker is machine-generated and its quoting is unpredictable, and the action
    it triggers (re-queue the next touch) is recoverable, unlike suppression.
    """
    if is_bounce(subject, body_text, from_email):
        return "bounce"
    authored = f"{subject or ''}\n{strip_quoted_reply(body_text)}"
    if any(p.search(authored) for p in _UNSUBSCRIBE_PATTERNS):
        return "unsubscribe"
    text = f"{subject or ''}\n{body_text or ''}"
    if any(p.search(text) for p in _OOO_PATTERNS):
        return "out_of_office"
    return None


def extract_return_date(text: Optional[str], *, today: Optional[date] = None) -> Optional[str]:
    """Best-effort parse of an OOO return date → ISO ``YYYY-MM-DD`` or ``None``.

    Handles the common autoresponder shapes ("back on August 3", "returning
    2026-08-15", "until 08/20") without a heavyweight date dependency. Ambiguous
    or missing dates return ``None`` so the caller can apply the safe +7-day
    default. ``today`` is injectable for deterministic tests.
    """
    if not text:
        return None
    today = today or datetime.now(timezone.utc).date()

    # ISO: 2026-08-15
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            return None

    # Month name: "August 3", "Aug 3rd", "3 August"
    month_alt = "|".join(_MONTHS.keys())
    m = re.search(rf"\b({month_alt})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text, re.IGNORECASE)
    if not m:
        m2 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_alt})\b", text, re.IGNORECASE)
        if m2:
            day_num, mon = int(m2.group(1)), _MONTHS[m2.group(2).lower()]
            return _resolve_forward(today, mon, day_num)
    else:
        mon, day_num = _MONTHS[m.group(1).lower()], int(m.group(2))
        return _resolve_forward(today, mon, day_num)

    # Numeric MM/DD or MM/DD/YYYY
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        if m.group(3):
            y = int(m.group(3))
            y = y + 2000 if y < 100 else y
            try:
                return date(y, mo, d).isoformat()
            except ValueError:
                return None
        return _resolve_forward(today, mo, d)
    return None


def _resolve_forward(today: date, month: int, day: int) -> Optional[str]:
    """Resolve a month/day (no year) to the next such date on/after ``today``."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if candidate >= today:
            return candidate.isoformat()
    return None


class AIService:
    """Makes intelligent decisions about outreach and leads via an LLM backend.

    Anthropic calls use AsyncAnthropic; OpenRouter calls use httpx directly
    against the OpenAI-compatible endpoint. Both are async so they do not block
    the FastAPI event loop. Celery tasks call these via _run_async() wrappers
    which create a new event loop, so the await is resolved there too.
    """

    def __init__(
        self,
        *,
        openrouter_model: str | None = None,
        openrouter_api_key: str | None = None,
        force_openrouter: bool = False,
    ):
        """``openrouter_model`` overrides the model **only on the OpenRouter
        backend** — the name says so because an OpenRouter model id is
        meaningless to Anthropic. Callers that need a stronger model for one job
        (see ``ai_email_generator``) pass it here and still get plain
        claude-sonnet-4-6 if the backend resolves to Anthropic.

        ``openrouter_api_key`` lets a caller run this instance on a *specific*
        OpenRouter credential rather than the process-wide one — used by the
        Social Listening, Deal Progression, and Client Success agents, which
        carry their own per-org key. Passing a non-empty key pins the backend to
        OpenRouter with that key + model, regardless of ``AI_PROVIDER``.

        ``force_openrouter`` pins OpenRouter even when no key is supplied here
        (the global env key is then used as the fallback) — the Deal Progression
        Agent uses this so its AI touches always route through OpenRouter's
        OpenAI-compatible API. Either way, this instance is a throwaway, not the
        shared singleton.
        """
        settings = get_settings()

        # An explicit per-caller OpenRouter key always pins the OpenRouter
        # backend — the caller has chosen the credential and the model.
        forced_key = (openrouter_api_key or "").strip()

        # Resolve the backend. "auto" prefers the OpenAI-compatible chain when
        # ANY of its providers is keyed (e.g. Anthropic credits depleted) and
        # otherwise stays on Anthropic. Asking only about the OpenRouter key here
        # — which is what this used to do — means a deployment carrying only a
        # Gemini or Groq key silently answers every call from a static template.
        if forced_key or force_openrouter:
            provider = "openrouter"
        else:
            provider = (getattr(settings, "ai_provider", "auto") or "auto").lower()
            if provider == "auto":
                provider = "openrouter" if has_any_provider_key(settings) else "anthropic"
        self.provider = provider
        self._anthropic_key = settings.anthropic_api_key
        # Which provider/model actually answered the most recent call. Populated
        # by _chat_completion; surfaced by GET /api/v1/ai/providers. Before this,
        # "which provider wrote this email?" was unanswerable.
        self.last_provider: str | None = None
        self.last_model: str | None = None

        if self.provider == "openrouter":
            self._openrouter_key = forced_key or settings.openrouter_api_key
            self.model = openrouter_model or settings.openrouter_model
            # AsyncAnthropic is still constructed lazily-free (no network) so any
            # stray direct references don't explode; it is simply never called
            # unless the whole provider chain is exhausted.
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or "unused")
            logger.info(
                "AIService using the OpenAI-compatible provider chain, "
                "OpenRouter model=%s", self.model,
            )
        else:
            self._openrouter_key = ""
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            self.model = ANTHROPIC_MODEL
            logger.info("AIService using Anthropic backend, model=%s", self.model)

    @property
    def is_configured(self) -> bool:
        """True when the *active* backend has a credential it can use.

        Callers gate on this instead of on ``anthropic_api_key`` directly: with
        AI_PROVIDER=openrouter the Anthropic key is legitimately absent, and
        checking the wrong one silently degrades a working backend to templates.

        On the chain backend this is true when ANY chain provider is keyed —
        including the Anthropic cross-fallback — not just OpenRouter, for the
        same reason.
        """
        if self.provider == "openrouter":
            if self._openrouter_key or self._anthropic_key:
                return True
            return has_any_provider_key(get_settings())
        return bool(self._anthropic_key)

    # ------------------------------------------------------------------
    # Backend transport
    # ------------------------------------------------------------------
    async def _chat_completion(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: float | None = None,
        system: str | None = None,
    ) -> str:
        """Send chat ``messages`` to the active backend and return the reply text.

        ``messages`` use the Anthropic/OpenAI-common shape
        ``[{"role": "user", "content": <str | list-of-blocks>}]``. For the
        OpenRouter backend, Anthropic-style image blocks are translated to the
        OpenAI ``image_url`` form. ``system`` is a system prompt, which the two
        APIs carry differently: Anthropic takes a top-level ``system`` kwarg,
        while OpenAI-compatible endpoints want a leading ``role: "system"``
        message. Raises on transport/HTTP errors so callers keep their existing
        try/except degradation.
        """
        if self.provider == "openrouter":
            settings = get_settings()
            # A reasoning model bills its chain-of-thought against the same
            # max_tokens budget as the answer, and emits it first — so an
            # unbounded trace returns an empty answer with
            # finish_reason="length" rather than an error. Give the answer a
            # floor. See ``openrouter_min_max_tokens``.
            budget = max(max_tokens, settings.openrouter_min_max_tokens)
            providers = build_provider_chain(
                settings,
                openrouter_key=self._openrouter_key,
                openrouter_models=self._model_chain(),
            )
            last_exc: Exception | None = None
            if providers:
                try:
                    content, served_by, served_model = await chain_chat_completion(
                        messages,
                        providers=providers,
                        max_tokens=budget,
                        temperature=temperature,
                        system=system,
                    )
                    self.last_provider = served_by
                    self.last_model = served_model
                    return content
                except LLMChainExhausted as exc:
                    last_exc = exc
            else:
                last_exc = LLMChainExhausted("no OpenAI-compatible provider is configured")

            # Every provider in the chain is exhausted. Before giving up, cross
            # to Anthropic if a key is configured.
            #
            # A free tier's daily quota is a *routine* condition, not an
            # incident — it is reached every day that outreach runs at any
            # volume, and the provider answers 429 to everything after that.
            # Without a cross-provider hop, each exhaustion silently downgrades
            # every email to the static mail-merge and every reply
            # classification to keyword-only: the "healthy-looking degradation"
            # this whole chain exists to end.
            if self._anthropic_key:
                logger.warning(
                    "LLM provider chain exhausted (%s) — falling back to Anthropic "
                    "for this call",
                    last_exc,
                )
                content = await self._anthropic_completion(
                    messages, max_tokens=max_tokens, temperature=temperature, system=system
                )
                self.last_provider = "anthropic"
                self.last_model = ANTHROPIC_MODEL
                logger.info("LLM request served by anthropic (model=%s)", ANTHROPIC_MODEL)
                return content
            # Nothing left. Raise so the caller's existing degradation path
            # (static copy, neutral score) runs, exactly as before.
            raise last_exc if last_exc else LLMChainExhausted("no LLM provider configured")

        content = await self._anthropic_completion(
            messages, max_tokens=max_tokens, temperature=temperature, system=system
        )
        self.last_provider = "anthropic"
        self.last_model = self.model
        logger.info("LLM request served by anthropic (model=%s)", self.model)
        return content

    async def _anthropic_completion(
        self,
        messages: list,
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        system: str | None = None,
    ) -> str:
        """Call the Anthropic backend. Shared by the native path and the fallback."""
        # ``self.model`` names an OpenRouter model when that is the active
        # backend, and Anthropic would reject it — so the fallback names its own.
        model = self.model if self.provider == "anthropic" else ANTHROPIC_MODEL
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        message = await self.client.messages.create(**kwargs)
        return message.content[0].text

    def _model_chain(self) -> list[str]:
        """The OpenRouter models to try, in order: primary then configured fallbacks.

        De-duplicated (preserving order) so listing the primary among the
        fallbacks doesn't retry a model that already answered 429.
        """
        chain = [self.model]
        raw = getattr(get_settings(), "openrouter_fallback_models", "") or ""
        chain.extend(m.strip() for m in raw.split(",") if m.strip())
        seen: set[str] = set()
        return [m for m in chain if m and not (m in seen or seen.add(m))]

    @staticmethod
    def _to_openai_message(message: dict) -> dict:
        """Translate one Anthropic-shaped message to OpenAI chat format.

        The implementation moved to :mod:`app.services.llm_providers` when the
        transport did; this stays as the name the rest of the codebase (and the
        tests) already reach for.
        """
        return to_openai_message(message)

    async def _chat(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
        system: str | None = None,
    ) -> str:
        """Convenience wrapper: send a single user-text prompt, return reply text."""
        return await self._chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from Claude response, handling markdown code blocks and
        any stray prose the model puts before/after the JSON object."""
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fall back to the first {...} / [...] span, so a stray preamble like
            # "Here is the JSON:" doesn't blow up an otherwise-valid response.
            start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
            end = max(text.rfind("}"), text.rfind("]"))
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    async def complete_json(
        self, prompt: str, max_tokens: int = 1024, system: str | None = None
    ) -> dict:
        """Public single-shot JSON completion.

        Thin wrapper over :meth:`_complete_json` for other services (deal
        forecasting, smart suggestions, cold-email copy) that need a structured
        reply and want the same fence-tolerant parsing + never-raises contract.
        """
        return await self._complete_json(prompt, max_tokens=max_tokens, system=system)

    async def _complete_json(
        self, prompt: str, max_tokens: int = 1024, system: str | None = None
    ) -> dict:
        """Send a single-shot prompt to the backend and parse the JSON reply.

        Centralizes the chat + markdown-fence-tolerant JSON parse used by the
        playbook/tactics helpers. Returns ``{"error": ...}`` on any failure so
        callers never raise into the request handler; on a parse failure the
        offending reply is returned under ``"raw"`` so the caller can log it.
        """
        response_text = ""
        try:
            response_text = await self._chat(prompt, max_tokens=max_tokens, system=system)
            return self._parse_json_response(response_text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI JSON response: %s", str(e))
            return {"error": "Failed to parse AI response", "raw": response_text}
        except Exception as e:
            logger.error("AI completion failed: %s", str(e))
            return {"error": str(e)}

    async def complete_text(self, prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        """Public single-shot free-text completion.

        Used by features that need prose rather than JSON (RAG natural-language
        answers, voice-call summaries). Never raises — returns an empty string
        on failure so callers can degrade gracefully.
        """
        try:
            return await self._chat(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            logger.error("AI text completion failed: %s", str(e))
            return ""

    async def generate_email(
        self,
        company_name: str,
        contact_name: str,
        contact_title: str,
        industry: str,
        purpose: str = "initial outreach",
        tone: str = "professional but warm",
        additional_context: str = "",
        rag_context: str = "",
    ) -> dict:
        """Generate a personalized email using Claude.

        ``rag_context`` (feature 3.4) lets a caller inject retrieved
        institutional knowledge — e.g. an email that worked for a similar shop —
        which the model may draw on. It is advisory context, appended after the
        per-lead ``additional_context``.
        """
        response_text = ""
        try:
            ctx_line = ""
            if additional_context:
                ctx_line = f"Additional Context: {additional_context}"
            if rag_context:
                ctx_line += (
                    f"\n\nRelevant past context (draw on what worked before, do NOT copy verbatim):\n{rag_context}"
                )

            prompt = f"""You're Suman from Apprend Technologies. You build manufacturing software — shop floor monitoring, quoting tools, ERP integrations, quality tracking.

Write a cold email to:

Name: {contact_name}
Title: {contact_title}
Company: {company_name}
Industry: {industry}
{ctx_line}

HOW TO WRITE THIS:
- You're a person, not a department. Write like you'd talk to someone at a trade show
- 4-5 sentences. Under 100 words. That's it
- Open with something about THEIR world — their industry, their role, a real challenge they probably face. Not about you
- Use contractions. Short sentences. The occasional dash
- One ask at the end: a 15-min call. Frame it as a question, not a demand
- Subject line: lowercase, short, like you'd text a colleague

BANNED:
- "I hope this finds you well", "I came across", "I noticed that", "I wanted to reach out"
- "leverage", "synergy", "cutting-edge", "revolutionary", "transform", "streamline", "innovative"
- Exclamation marks
- Bullet points or lists in the body

GOOD example:
Subject: quick question about your quoting process
Body: Hey {contact_name.split()[0] if contact_name else 'there'}, saw that {company_name} does [specific thing]. Curious — are you still running quotes through spreadsheets? We built a tool for a shop like yours that got their turnaround from 2 days to 20 minutes. Would a 15-min call make sense to see if something similar could work for you guys?

Return ONLY valid JSON: {{"subject": "...", "body": "...", "follow_up_suggestion": "..."}}"""

            response_text = await self._chat(prompt, max_tokens=1024)
            result = self._parse_json_response(response_text)
            return {
                "subject": result.get("subject", ""),
                "body": result.get("body", ""),
                "follow_up_suggestion": result.get("follow_up_suggestion", ""),
                "model": self.model,
            }
        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI response as JSON: %s", str(e))
            return {"error": "Failed to parse AI response", "raw": response_text}
        except Exception as e:
            logger.error("AI email generation failed: %s", str(e))
            return {"error": str(e)}

    async def improve_email(self, subject: str, body: str, instruction: str = "") -> dict:
        """Improve an existing email draft using Claude."""
        try:
            instr_line = "General improvement: better subject line, tighter copy, stronger CTA."
            if instruction:
                instr_line = f"Specific instructions: {instruction}"

            prompt = f"""Tighten up this outreach email. Make it sound like a real person wrote it quickly, not like a committee drafted it.

Current Subject: {subject}
Current Body:
{body}

{instr_line}

What to do:
- Cut it to under 100 words. Kill any sentence that doesn't earn its spot
- Subject line: short, lowercase, something you'd actually type — not a headline
- Read it out loud — if you wouldn't say it to someone at a coffee shop, rewrite it
- Strip out anything corporate-sounding. Contractions are good. Dashes are fine
- One clear ask at the end
- Keep the core message, just make it sound human

Return ONLY valid JSON: {{"subject": "...", "body": "...", "changes_made": "..."}}"""

            response_text = await self._chat(prompt, max_tokens=1024)
            return self._parse_json_response(response_text)
        except Exception as e:
            logger.error("AI email improvement failed: %s", str(e))
            return {"error": str(e)}

    @staticmethod
    def lead_score_prompt(
        company_name: str,
        industry: str,
        employee_count: Optional[int],
        location: str,
        engagement_history: str = "",
        notes: str = "",
    ) -> str:
        """The lead-scoring prompt, shared with the per-org fallback in
        ``app.tasks.enrichment`` so both paths ask the model the same thing."""
        emp_str = str(employee_count) if employee_count else "Unknown"
        eng_str = engagement_history if engagement_history else "None yet"
        notes_str = notes if notes else "None"

        prompt = f"""Score this B2B lead on a scale of 1-100 and explain why.

Company: {company_name}
Industry: {industry}
Employee Count: {emp_str}
Location: {location}
Engagement History: {eng_str}
Notes: {notes_str}

Context: We are Apprend Technologies, a custom manufacturing software company. Our sweet spot:
- Mid-size manufacturers (50-500 employees)
- Industries: plastics, CNC machining, injection molding, metal stamping, aerospace parts, marine parts
- US and Canada
- Companies still using spreadsheets, paper, or outdated ERP systems
- Decision makers: VPs of Operations, Plant Managers, CTOs, Owners

Scoring criteria:
- Industry relevance (is this a manufacturer we would want to work with?)
- Company size (sweet spot is 50-500 employees)
- Location (US/Canada preferred)
- Engagement signals (opens, clicks, replies are positive)
- Title/role fit of contacts

Return ONLY valid JSON: {{"score": 75, "grade": "A/B/C/D/F", "reasoning": "...", "recommended_action": "...", "priority": "high/medium/low"}}"""
        return prompt

    async def score_lead(
        self,
        company_name: str,
        industry: str,
        employee_count: Optional[int],
        location: str,
        engagement_history: str = "",
        notes: str = "",
    ) -> dict:
        """Use Claude to intelligently score a lead."""
        try:
            prompt = self.lead_score_prompt(
                company_name, industry, employee_count, location, engagement_history, notes
            )

            response_text = await self._chat(prompt, max_tokens=512)
            result = self._parse_json_response(response_text)
            # A parse failure or an error dict means we couldn't get a usable
            # score — degrade gracefully rather than block the batch.
            if not isinstance(result, dict) or "score" not in result:
                return self._default_lead_score("AI returned no usable score")
            return result
        except Exception as e:
            # Graceful degradation: if the AI backend is unavailable (e.g. the
            # provider is out of credits or rate-limited), assign a neutral
            # mid-range score so the lead is still recorded and reviewable,
            # instead of failing the whole auto-scoring batch.
            logger.error("AI lead scoring failed, using default mid-range score: %s", str(e))
            return self._default_lead_score(str(e))

    @staticmethod
    def _default_lead_score(reason: str) -> dict:
        """Neutral fallback score used when AI scoring is unavailable.

        Returns a valid score dict (no ``error`` key) so downstream persistence
        treats it as a real, if provisional, score. ``degraded`` flags that a
        human should re-score once AI is back.
        """
        return {
            "score": 50,
            "grade": "C",
            "reasoning": f"AI scoring unavailable ({reason}); assigned a default mid-range score pending review.",
            "recommended_action": "Re-score once AI scoring is restored, or review manually.",
            "priority": "medium",
            "degraded": True,
        }

    async def analyze_company(self, company_data: dict) -> dict:
        """Analyze a company and suggest outreach strategy."""
        try:
            company_json = json.dumps(company_data, indent=2, default=str)
            prompt = f"""Analyze this company for B2B outreach and suggest a strategy.

Company Data:
{company_json}

Context: We are Apprend Technologies, a custom manufacturing software company. We build quoting tools, ERP integrations, shop floor dashboards, and quality tracking systems for mid-size manufacturers. Analyze:
1. How good a fit is this company for our services?
2. What is the best approach to reach out?
3. What pain points might they have that we could address?
4. Who should we target (which roles)?
5. What messaging angle would resonate?

Return ONLY valid JSON:
{{
  "fit_score": 1,
  "fit_reasoning": "...",
  "recommended_approach": "...",
  "likely_pain_points": ["..."],
  "target_roles": ["..."],
  "messaging_angle": "...",
  "suggested_subject_line": "...",
  "red_flags": ["..."]
}}"""

            response_text = await self._chat(prompt, max_tokens=1024)
            return self._parse_json_response(response_text)
        except Exception as e:
            logger.error("AI company analysis failed: %s", str(e))
            return {"error": str(e)}

    async def generate_followup(
        self,
        original_subject: str,
        original_body: str,
        step_number: int,
        contact_name: str,
        days_since_last: int,
        booking_url: str = "",
    ) -> dict:
        """Generate an intelligent follow-up email that does not feel spammy.

        ``booking_url`` (Feature 4) is passed in *already interest-gated* by the
        caller — when set, the model is instructed to weave the link in once,
        conversationally, never as a hard CTA. A post-render check collapses the
        body to a single link if the model ignores the "once" instruction.
        """
        try:
            is_final = step_number >= 3
            final_note = "This is the final follow-up. Make it a graceful breakup email — keep it classy, leave the door open." if is_final else ""

            booking_note = ""
            if booking_url:
                booking_note = (
                    f'8. They\'ve shown interest — include this booking link EXACTLY ONCE, conversationally '
                    f'(e.g. "happy to grab 15 min — here\'s my calendar: {booking_url}"). '
                    f'Never as a hard CTA block, never more than once: {booking_url}'
                )

            prompt = f"""Write follow-up #{step_number} for a cold email I sent {days_since_last} days ago.

Original subject: {original_subject}
Contact: {contact_name}

THE GOAL: Get a reply. Not annoy them.

RULES:
1. Under 50 words. Seriously — the shorter the follow-up, the higher the reply rate
2. Follow-up #1: Float the thread back up. Add one new thing — a quick stat, a thought about their industry, or the demo link. Keep it casual. "Hey {contact_name.split()[0] if contact_name else 'there'}, wanted to bump this" energy
3. Follow-up #2: Completely different angle. Don't reference the old email. Try a specific pain point, a result from a similar shop, or a question about their process
4. Follow-up #3: Breakup. "Totally fine if the timing's off — just didn't want to keep pinging you without checking." Two sentences. Leave the door open. Done
5. After #3: Set should_stop_sequence to true
6. NEVER say: "just checking in", "circling back", "following up on my previous email", "touching base", "wanted to circle back"
7. Sound like you dashed this off in 30 seconds, because you did
{booking_note}
{final_note}

Return ONLY valid JSON: {{"subject": "...", "body": "...", "is_final": {str(is_final).lower()}, "should_stop_sequence": {str(is_final).lower()}}}"""

            response_text = await self._chat(prompt, max_tokens=512)
            result = self._parse_json_response(response_text)

            # Post-render safety: never let more than one booking link through.
            if booking_url and isinstance(result, dict) and result.get("body"):
                from app.services.booking import collapse_booking_links

                result["body"] = collapse_booking_links(result["body"], booking_url)
            return result
        except Exception as e:
            logger.error("AI follow-up generation failed: %s", str(e))
            return {"error": str(e)}

    async def analyze_reply(self, reply_body: str, original_subject: str, original_body: str, contact_name: str, company_name: str) -> dict:
        """Analyze a lead's reply to understand intent and draft a response focused on scheduling appointments."""
        try:
            settings = get_settings()
            scheduling_link = getattr(settings, "scheduling_link", "")
            scheduling_instruction = ""
            if scheduling_link:
                scheduling_instruction = f"""
IMPORTANT: We have a scheduling link available: {scheduling_link}
Include this link naturally in your response when suggesting a call or meeting."""

            prompt = f"""You're Suman. Someone replied to your cold email. Read what they said and write back.

THEIR REPLY:
{reply_body[:1000]}

YOUR ORIGINAL EMAIL:
Subject: {original_subject}

Who they are: {contact_name} at {company_name}

{scheduling_instruction}

FIRST: Figure out what they mean:
- "interested" = they want to learn more or talk
- "question" = they asked something specific
- "not_now" = bad timing but not a no
- "not_interested" = clear no
- "referral" = pointing you to someone else
- "auto_reply" = out of office / automated

THEN write your reply:

- MAX 60 words. You're replying from your phone
- Match their vibe. Casual reply to a casual email. Slightly more polished if they're formal
- Get to the point immediately. No "thanks for getting back to me" throat-clearing
- Interested/question: suggest hopping on a quick call. Ask what times work for them, or propose 2-3 slots
- Not now: "totally get it" + when should you check back? One sentence
- Not interested: "appreciate you letting me know" and done. Don't grovel
- Auto-reply: don't respond at all. Set should_auto_send to false
- Sign off: "- Suman" or just "Suman"

Return ONLY valid JSON:
{{
  "intent": "interested|not_now|not_interested|question|referral|auto_reply",
  "confidence": 0.0,
  "summary": "one line summary",
  "draft_reply": {{
    "subject": "Re: ...",
    "body": "the reply"
  }},
  "should_send": true,
  "should_auto_send": true,
  "reasoning": "why",
  "suggested_action": "schedule_meeting|follow_up_later|close_lead|ask_for_referral_intro|no_action",
  "follow_up_days": null
}}

Auto-send: true ONLY for "interested" and "question". false for everything else. NEVER for "auto_reply".
"""

            response_text = await self._chat(prompt, max_tokens=1024)
            return self._parse_json_response(response_text)
        except Exception as e:
            logger.error("AI reply analysis failed: %s", str(e))
            return {"error": str(e), "should_send": False, "should_auto_send": False}

    async def classify_reply(
        self,
        subject: Optional[str],
        body_text: Optional[str],
        original_subject: str = "",
        contact_name: str = "",
        company_name: str = "",
        from_email: Optional[str] = None,
    ) -> dict:
        """Classify an incoming reply into one routing category. Never raises.

        A dedicated, cheap *routing* pass (distinct from the richer, drafting-
        oriented :meth:`analyze_reply`). It runs a high-precision keyword
        pre-filter first — compliance-critical opt-outs and OOO autoresponders
        never depend on the LLM — and only falls through to ``complete_json`` for
        ambiguous replies. On an AI outage the keyword result stands, else the
        reply is ``unknown`` (manual review). Output shape::

            {
              "category": "interested|not_interested|out_of_office|"
                          "wrong_person|unsubscribe|bounce|unknown",
              "confidence": 0.0-1.0,
              "reason": "one-line justification",
              "return_date": "YYYY-MM-DD" | None,   # OOO only
              "referral": {"name": None, "email": None},  # wrong_person only
              "unsubscribe_explicit": bool,
              "source": "keyword|jev|ai|fallback",
              "model": "<model or None>",
              "jev": {...},   # Jev's label + probabilities, when it was asked
            }
        """
        combined = f"{subject or ''}\n{body_text or ''}"

        # 1. Keyword pre-filter — high precision, no LLM, guarantees the
        #    compliance-critical categories survive an AI outage.
        kw = keyword_prefilter(subject, body_text, from_email)
        if kw == "bounce":
            return {
                "category": "bounce",
                "confidence": 1.0,
                "reason": "Delivery-status notification detected (mail daemon / DSN subject).",
                "return_date": None,
                "referral": {"name": None, "email": None},
                "unsubscribe_explicit": False,
                "source": "keyword",
                "model": None,
            }
        if kw == "unsubscribe":
            return {
                "category": "unsubscribe",
                "confidence": 1.0,
                "reason": "Explicit opt-out language detected (keyword match).",
                "return_date": None,
                "referral": {"name": None, "email": None},
                "unsubscribe_explicit": True,
                "source": "keyword",
                "model": None,
            }
        if kw == "out_of_office":
            return {
                "category": "out_of_office",
                "confidence": 0.9,
                "reason": "Out-of-office / autoresponder markers detected (keyword match).",
                "return_date": extract_return_date(combined),
                "referral": {"name": None, "email": None},
                "unsubscribe_explicit": False,
                "source": "keyword",
                "model": None,
            }

        # 2. Jev (TypeSafe System One) — a structured classifier with calibrated
        #    probabilities. A confident label is the answer; a low-confidence
        #    one, or ``wrong_person`` (whose referral only the LLM extracts),
        #    falls through to the LLM and is kept as the answer if the LLM is
        #    unavailable. Never raises; ``None`` when Jev is off or down.
        jev = await self._jev_classify_reply(
            subject, body_text, from_email, contact_name, company_name, combined
        )
        if jev is not None and jev["category"] != "wrong_person" and (
            jev["confidence"] >= get_settings().jev_reply_min_confidence
        ):
            return jev

        # 3. Ambiguous → ask the LLM. temperature=0 for stability.
        prompt = f"""Classify this reply to a cold sales email into EXACTLY ONE routing category.
Your job is to CLASSIFY ONLY — do NOT draft a reply.

CATEGORIES:
- interested: positive intent — wants to talk, learn more, asks a qualifying question, or requests a call/demo.
- not_interested: a clear rejection ("no thanks", "not a fit", "we're all set"). NO legal opt-out language.
- out_of_office: an automated out-of-office / vacation autoresponder (even if the body sounds positive).
- wrong_person: the recipient is not the right contact ("I don't handle this", "try our CTO", "no longer with the company").
- unsubscribe: an explicit opt-out request ("unsubscribe", "remove me", "take me off your list").
- bounce: an automated delivery-failure / bounce notification from a mail server (mailer-daemon, "delivery status notification", "address not found", "user unknown"). NOT a human reply.
- unknown: genuinely unclear.

TIE-BREAKERS (apply in order):
1. A mail-server delivery-failure notice is ALWAYS bounce, even if it quotes our original email (footer included).
2. Explicit opt-out / unsubscribe language ALWAYS wins over not_interested.
3. An autoresponder is ALWAYS out_of_office even if the wording sounds interested.
4. If it points you to a different person/role, it is wrong_person (capture the referral).

THE REPLY:
Subject: {subject or ''}
Body: {(body_text or '')[:1200]}

CONTEXT (our original outreach):
Original subject: {original_subject}
Who they are: {contact_name or 'unknown'} at {company_name or 'unknown'}

Return ONLY valid JSON:
{{
  "category": "interested|not_interested|out_of_office|wrong_person|unsubscribe|bounce|unknown",
  "confidence": 0.0,
  "reason": "one-line justification",
  "return_date": null,
  "referral": {{"name": null, "email": null}},
  "unsubscribe_explicit": false
}}
- confidence is your 0..1 certainty in the category.
- return_date: ISO date (YYYY-MM-DD) ONLY for out_of_office when a return date is stated, else null.
- referral: name/email ONLY for wrong_person when they point elsewhere, else nulls.
- unsubscribe_explicit: true ONLY for legally-explicit opt-out language."""

        result = await self.complete_json(prompt, max_tokens=400)
        normalized = self._normalize_classification(result)
        if normalized["source"] == "fallback" and jev is not None:
            return jev
        if jev is not None:
            normalized["jev"] = jev["jev"]
        return normalized

    async def _jev_classify_reply(
        self,
        subject: Optional[str],
        body_text: Optional[str],
        from_email: Optional[str],
        contact_name: str,
        company_name: str,
        combined: str,
    ) -> Optional[dict]:
        """Jev's reply intent in :meth:`classify_reply`'s output shape, or ``None``.

        ``source`` is ``"jev"``, so the confidence gate in
        ``app.tasks.replies`` applies exactly as it does to an LLM label.
        """
        from app.services import jev_service

        choice = await jev_service.classify_reply_intent(
            subject, body_text,
            from_email=from_email, contact_name=contact_name, company_name=company_name,
        )
        if choice is None or choice.label not in REPLY_CATEGORIES:
            return None
        category = choice.label
        return {
            "category": category,
            "confidence": choice.confidence,
            "reason": f"Jev classified as {category} (confidence {choice.confidence:.2f}).",
            "return_date": extract_return_date(combined) if category == "out_of_office" else None,
            "referral": {"name": None, "email": None},
            "unsubscribe_explicit": category == "unsubscribe",
            "source": "jev",
            "model": choice.model,
            "jev": choice.as_dict(),
        }

    def _normalize_classification(self, result: dict) -> dict:
        """Validate/coerce a raw LLM classification into the canonical shape.

        A parse error, missing category, or out-of-vocabulary category degrades
        to ``unknown`` (confidence 0) so a bad AI response is surfaced for manual
        review rather than mis-routing a reply.
        """
        if not isinstance(result, dict) or result.get("error") or "category" not in result:
            return {
                "category": "unknown",
                "confidence": 0.0,
                "reason": "Classifier unavailable or returned no category; needs manual review.",
                "return_date": None,
                "referral": {"name": None, "email": None},
                "unsubscribe_explicit": False,
                "source": "fallback",
                "model": None,
            }

        category = str(result.get("category", "unknown")).strip().lower()
        if category not in REPLY_CATEGORIES:
            category = "unknown"

        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        referral = result.get("referral") or {}
        if not isinstance(referral, dict):
            referral = {}

        return_date = result.get("return_date")
        if category != "out_of_office":
            return_date = None

        unsub_explicit = bool(result.get("unsubscribe_explicit", False))
        # A model that labels something unsubscribe must have seen explicit
        # opt-out language; keep the flag consistent for the compliance path.
        if category == "unsubscribe":
            unsub_explicit = True

        return {
            "category": category,
            "confidence": confidence,
            "reason": str(result.get("reason", ""))[:500],
            "return_date": return_date if isinstance(return_date, str) else None,
            "referral": {"name": referral.get("name"), "email": referral.get("email")},
            "unsubscribe_explicit": unsub_explicit,
            "source": "ai",
            "model": self.model,
        }

    # ------------------------------------------------------------------
    # Sales tactics / playbook generators
    #
    # These power the frontend "Tactics" tab — multi-channel outreach assets
    # for a single lead. All return plain dicts (never raise) and share the
    # same Apprend voice rules as the cold-email generator above.
    # ------------------------------------------------------------------

    _VOICE_RULES = (
        "VOICE: Write like a real person, not a marketing team. Short, concrete, "
        "conversational (contractions OK). NO buzzwords (leverage, synergy, "
        "cutting-edge, revolutionary, transform, streamline). NO generic openers "
        '("I hope this finds you well", "I came across"). One clear ask.'
    )

    async def linkedin_messages(
        self,
        company_name: str,
        contact_name: str,
        contact_title: str = "",
        company_industry: str = "",
        company_domain: str = "",
        topic: str = "",
    ) -> dict:
        """Draft LinkedIn connection note, follow-up, and a post-comment."""
        topic_line = f"Angle/topic to lead with: {topic}" if topic else ""
        prompt = f"""You are Suman from Apprend Technologies (custom manufacturing software —
quoting tools, ERP integrations, shop-floor dashboards, quality tracking).

Write LinkedIn outreach to:
Name: {contact_name}
Title: {contact_title}
Company: {company_name}
Industry: {company_industry}
{topic_line}

{self._VOICE_RULES}

Produce three assets:
1. connection_note: a connection request note, HARD MAX 300 characters, no link.
2. follow_up_message: a short message to send after they accept (2-3 sentences, one soft ask).
3. comment_draft: a thoughtful comment to leave on one of their recent posts (1-2 sentences, adds value, not salesy).

Return ONLY valid JSON: {{"connection_note": "...", "follow_up_message": "...", "comment_draft": "..."}}"""
        return await self._complete_json(prompt, max_tokens=768)

    async def phone_script(
        self,
        company_name: str,
        contact_name: str,
        contact_title: str = "",
        company_industry: str = "",
        product: str = "",
    ) -> dict:
        """Build a cold-call script: opener, value prop, discovery, objections, close, voicemail."""
        product_line = f"Product/service to pitch: {product}" if product else "Pitch our custom manufacturing software broadly."
        prompt = f"""You are coaching Suman from Apprend Technologies for a cold call.

Prospect: {contact_name}, {contact_title} at {company_name} ({company_industry}).
{product_line}

{self._VOICE_RULES}
Keep every spoken line natural and brief — this is talked, not read aloud.

Return ONLY valid JSON with these fields:
{{
  "opener": "first 2 sentences when they pick up — pattern-interrupt, not 'how are you'",
  "value_proposition": "one tight sentence on the outcome we deliver for shops like theirs",
  "discovery_questions": ["3-5 open questions about their current process/pain"],
  "objection_responses": {{"we already have a system": "...", "no budget": "...", "send me an email": "..."}},
  "close": "the specific ask — a 15-min call/demo",
  "voicemail_script": "a 15-second voicemail if they don't answer"
}}"""
        return await self._complete_json(prompt, max_tokens=1024)

    async def referral_request(
        self,
        company_name: str,
        referrer_name: str,
        target_name: str,
        target_title: str = "",
        company_industry: str = "",
    ) -> dict:
        """Draft a warm-intro request to a mutual connection + a forwardable blurb."""
        prompt = f"""You are Suman from Apprend Technologies (custom manufacturing software).

You want an introduction to {target_name} ({target_title}) at {company_name} ({company_industry}).
You'll ask {referrer_name}, a mutual connection, to make the intro.

{self._VOICE_RULES}

Produce:
1. referral_email: a short note to {referrer_name} asking for the intro — make it EASY to say yes, give context, no pressure.
2. forwardable_blurb: a 2-3 sentence blurb {referrer_name} can paste/forward straight to {target_name} describing who you are and why it's worth 15 minutes.

Return ONLY valid JSON: {{"referral_email": "...", "forwardable_blurb": "..."}}"""
        return await self._complete_json(prompt, max_tokens=768)

    async def event_outreach(
        self,
        company_name: str,
        contact_name: str,
        event_name: str,
        event_type: str = "conference",
        company_industry: str = "",
    ) -> dict:
        """Draft a 3-touch event sequence: pre-event, at-event, post-event."""
        prompt = f"""You are Suman from Apprend Technologies (custom manufacturing software).

You and {contact_name} from {company_name} ({company_industry}) will both be at
{event_name} (a {event_type}). Write a 3-touch sequence around the event.

{self._VOICE_RULES}

Produce:
1. pre_event_email: before the event — suggest meeting up / grabbing coffee there.
2. at_event_message: a short text/LinkedIn message to send during the event to connect in person.
3. post_event_followup: after the event — reference the event, propose next step.

Return ONLY valid JSON: {{"pre_event_email": "...", "at_event_message": "...", "post_event_followup": "..."}}"""
        return await self._complete_json(prompt, max_tokens=1024)

    async def multi_touch_plan(
        self,
        company_name: str,
        company_industry: str = "",
        employee_count: Optional[int] = None,
        location: str = "",
        contacts_summary: str = "",
        fit_score: Optional[int] = None,
    ) -> dict:
        """Generate a sequenced multi-channel campaign plan for one company."""
        emp = str(employee_count) if employee_count else "Unknown"
        score_line = f"Current fit score: {fit_score}/100." if fit_score is not None else ""
        prompt = f"""You are a B2B sales strategist for Apprend Technologies (custom manufacturing
software — quoting, ERP integrations, shop-floor dashboards, quality tracking).

Design a realistic multi-channel outreach campaign for this account:
Company: {company_name}
Industry: {company_industry}
Employees: {emp}
Location: {location}
Known contacts: {contacts_summary or 'none yet — discovery needed'}
{score_line}

Mix channels (email, linkedin, phone, referral) across ~2-4 weeks. Be specific and
sequenced — each step builds on the last. {self._VOICE_RULES}

Return ONLY valid JSON:
{{
  "summary": "one-paragraph strategy overview tailored to this account",
  "steps": [
    {{"channel": "email|linkedin|phone|referral", "timing": "e.g. Day 1", "action": "exactly what to do/say"}}
  ]
}}
Provide 5-8 steps."""
        return await self._complete_json(prompt, max_tokens=1500)

    async def extract_business_card(self, image_base64: str, media_type: str = "image/jpeg") -> dict:
        """Extract contact information from a business card image using Claude vision.

        Returns the extracted fields plus a ``_confidence`` map (field -> 0..1)
        and an ``_overall_confidence`` score so the UI can flag fields that need
        a human check. Handles international/multilingual cards and multiple
        phone numbers in mixed formats. Never raises — returns ``{"error": ...}``
        on failure so the request handler can surface it cleanly.
        """
        try:
            messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": """You are a precise OCR engine for business cards. Extract every piece of contact information from this card image.

Return ONLY valid JSON with these fields (use an empty string "" if a field is not present on the card):
{
  "first_name": "",
  "last_name": "",
  "title": "",
  "department": "",
  "company_name": "",
  "email": "",
  "phone": "",
  "mobile": "",
  "fax": "",
  "phones": [{"label": "", "number": ""}],
  "website": "",
  "address": "",
  "city": "",
  "state": "",
  "postal_code": "",
  "country": "",
  "linkedin_url": "",
  "notes": "",
  "_confidence": {},
  "_overall_confidence": 0.0
}

Extraction rules:
- PHONES: Capture EVERY phone number. Put the main/office number in "phone", a
  cell/mobile number in "mobile", and a fax number in "fax". List ALL numbers
  (including those three) in the "phones" array, each with a short label such as
  "Office", "Mobile", "Fax", "Direct", "Toll-free", "Tel", "WhatsApp" based on
  the card's own labels/icons. If a number has no label, use "Phone". Preserve
  international formats verbatim, INCLUDING the country code and "+" (e.g.
  "+49 30 1234567", "+81-3-1234-5678"). Do not reformat or strip digits.
- NAMES: Split the person's name into first/last. For names written
  family-name-first (common on East-Asian cards), still map the personal/given
  name to "first_name" and the family name to "last_name". If the card shows the
  name in a non-Latin script with a Latin transliteration, use the Latin form.
- MULTILINGUAL: Cards may be in any language. Translate the FIELD MEANINGS, not
  the values — keep proper nouns (names, company, street) as printed; you may
  normalize the country to its common English name.
- ADDRESS: "address" is the street line(s) only; put city, state/province,
  ZIP/postal code and country in their own fields. Map regions sensibly
  (e.g. a German "PLZ" is the postal_code).
- EMAIL/WEB: Fix obvious OCR slips in emails/URLs only when unambiguous (e.g.
  a clearly-misread "@"); otherwise transcribe exactly. Do not invent domains.
- Be thorough — read ALL text including small print, taglines and back-of-card
  details. Put anything useful that has no field (slogan, hours, second person)
  into "notes".

Confidence rules:
- "_confidence" maps EACH field you populated (non-empty) to a number from 0 to
  1 reflecting how sure you are it is correct and legible (1 = crystal clear,
  0.5 = partly obscured/ambiguous, <0.3 = a guess). Omit fields you left empty.
- "_overall_confidence" is your overall read quality for the whole card (0..1),
  factoring in image clarity, glare, blur and how much you had to guess.""",
                        },
                    ],
                }]

            response_text = await self._chat_completion(messages, max_tokens=1536, temperature=0)
            result = self._parse_json_response(response_text)
            if not isinstance(result, dict):
                return {"error": "Failed to parse extracted data"}
            return result
        except json.JSONDecodeError as e:
            logger.error("Failed to parse card extraction: %s", str(e))
            return {"error": "Failed to parse extracted data"}
        except Exception as e:
            logger.error("Business card extraction failed: %s", str(e))
            return {"error": str(e)}


_ai_service = None


def get_ai_service() -> AIService:
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service
