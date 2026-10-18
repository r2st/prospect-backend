"""Application configuration via pydantic-settings."""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # .env also carries compose-only vars (POSTGRES_USER, ...); don't crash on them.
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://landline:landline_secret@localhost:5432/landline_db"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Short-TTL response cache for hot dashboard/analytics aggregates. Disabled in
    # the test suite (see conftest) so cached rows can't leak across tests.
    response_cache_enabled: bool = True
    response_cache_ttl: int = 30  # seconds

    # AWS SES (fallback email transport when no Gmail account is connected).
    # SES sends from its own DNS-verified domain, so this stays on the
    # SES-verified apprendtechnologies.com identity even though the *primary*
    # sender is the connected apprend.tech Gmail account (see primary_sender_email).
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    ses_sender_email: str = "landline@apprendtechnologies.com"

    # Shared secret an external ESP signs its event webhook with. Empty means no
    # ESP is wired up, and the public event endpoint accepts nothing anonymous —
    # not "accepts everything", which is what it did before this existed. See
    # ``app.routers.outreach.require_signed_webhook_or_user`` for why an
    # unauthenticated write there is a send-accounting problem, not a data one.
    outreach_webhook_secret: str = ""

    # The canonical primary sender for outreach. Real sending goes through the
    # connected Gmail OAuth account whose address matches this (Google Workspace
    # signs DKIM for the domain automatically), so this is both the operator's
    # intended default sender and the domain reported by the deliverability /
    # test-send endpoints before any Gmail account is connected.
    primary_sender_email: str = "suman.das@apprend.tech"

    # Google / Gmail OAuth (per-user email sending)
    # Create these in Google Cloud Console — see deploy/GOOGLE-OAUTH-SETUP.md
    google_client_id: str = ""
    google_client_secret: str = ""
    # Must EXACTLY match an Authorized redirect URI configured in Google Cloud.
    # e.g. https://prospect.doaide.com/api/v1/gmail/oauth/callback
    google_oauth_redirect_uri: str = "http://localhost:8000/api/v1/gmail/oauth/callback"
    # Postmaster Tools OAuth callback (P2 audit). Empty = derive from the Gmail
    # redirect by swapping the path (see postmaster_oauth._redirect_uri).
    postmaster_oauth_redirect_uri: str = ""
    # Where to send the user's browser back to after a connect attempt (the frontend).
    # The OAuth callback closes the popup and notifies this origin via postMessage.
    frontend_url: str = "http://localhost:5173"
    # Fernet key used to encrypt stored OAuth refresh/access tokens at rest.
    # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    token_encryption_key: str = ""

    # Claude API
    anthropic_api_key: str = ""

    # --- AI provider selection ---
    # Which LLM backend AIService uses. "auto" (default) picks OpenRouter when an
    # OPENROUTER_API_KEY is set, else Anthropic — so setting the key is enough to
    # fail over when Anthropic credits are depleted. Force a backend with
    # AI_PROVIDER=anthropic or AI_PROVIDER=openrouter.
    ai_provider: str = "auto"
    # OpenRouter (OpenAI-compatible) — a drop-in fallback that can serve free-tier
    # models. Base URL is the OpenAI-compatible endpoint; the default model is a
    # free-tier model so scoring keeps working at zero cost.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # NOTE: a ":free" model id is only valid while OpenRouter actually serves that
    # variant, and they retire them without notice — the previous default here
    # (meta-llama/llama-3.3-70b-instruct:free, and before it llama-3.1-8b) had
    # stopped existing, which 404s every call and looks exactly like an outage:
    # silent template fallback + a hard 404 on conversation.analyze_thread.
    # Check a candidate against https://openrouter.ai/api/v1/models before setting.
    openrouter_model: str = "openai/gpt-oss-20b:free"
    # Cold-email copy is the quality-sensitive call, so it gets its own knob. It
    # currently resolves to the same model as everything else: on the free tier the
    # daily quota is per-account and shared across models, so a smaller model buys
    # no headroom, and the email system prompt is a long list of *negative*
    # constraints (no buzzwords, no fabricated proof, hold a tone) that small
    # models comply with poorly — an 8B will happily invent the customer story the
    # guardrail bans. Kept separate so email can move to a paid model without
    # dragging scoring/classification along (e.g. openai/gpt-4o-mini).
    openrouter_email_model: str = "openai/gpt-oss-20b:free"
    # Comma-separated model ids tried, in order, when the primary model answers
    # 429 (free-tier quota exhausted) or 5xx. Without this a single 429 silently
    # drops every AI call back to the static mail-merge — which is how the whole
    # sequence engine ended up sending un-personalised template copy while
    # looking healthy. Leave empty to disable fallback. Paid ids belong here: the
    # free tier shares one quota across models, so a second ":free" id buys
    # nothing when the 429 is the account-level daily cap — which is exactly why
    # ``llm_provider_chain`` (below) exists. This knob is now for reaching a
    # *paid* OpenRouter model before crossing to another provider; if you only
    # want free headroom, add provider keys instead of model ids.
    openrouter_fallback_models: str = ""
    # Reasoning budget for reasoning models (gpt-oss, nemotron-*-reasoning, o-series).
    #
    # This is not a tuning knob, it is a correctness one. OpenRouter bills a
    # reasoning model's chain-of-thought against the SAME ``max_tokens`` budget
    # as its visible answer, and reasoning is emitted FIRST. The default model
    # here (openai/gpt-oss-20b) reasons hard about a long instruction list, so
    # against this repo's cold-email system prompt at max_tokens=800 it spent
    # 797 tokens thinking and returned an EMPTY content string with
    # finish_reason="length" — measured in production 2026-07-24. Every AI email
    # was silently falling back to the static mail-merge, which is precisely the
    # un-personalised copy the whole compose path exists to replace.
    #
    # "low" caps deliberation (~105 reasoning tokens on the same prompt) and
    # leaves the budget for the answer. Set "" to omit the parameter entirely —
    # correct for a non-reasoning model, which has no chain-of-thought to bill.
    # Note ``reasoning.exclude`` is NOT a substitute: it only hides the trace
    # from the response, it still burns the budget.
    openrouter_reasoning_effort: str = "low"
    # Floor applied to max_tokens on the OpenRouter backend. A reasoning model
    # needs headroom for the trace *plus* the answer; callers asking for 256
    # (classification, scoring) would otherwise spend the entire allowance
    # thinking and return "". Only ever raises a caller's budget, never lowers it.
    openrouter_min_max_tokens: int = 1024

    # --- Multi-provider LLM fallback chain ---
    # OpenAI-compatible providers tried IN ORDER by app.services.llm_providers.
    # A provider with no API key is skipped when the chain is built, so leaving
    # a key unset simply removes that hop.
    #
    # This exists because a single backend's quota wall is indistinguishable, in
    # the logs, from a healthy system: every AI call raises, every caller catches,
    # and outreach quietly reverts to static mail-merge copy and keyword-only
    # reply routing. It has happened twice — Anthropic credits, then OpenRouter's
    # free-tier daily cap. ``openrouter_fallback_models`` cannot fix it, because
    # on the free tier the quota is ACCOUNT-level: a second ":free" model id
    # shares the same exhausted allowance. Independent providers do fix it.
    #
    # Gemini, Groq and Cerebras all offer free tiers that need no credit card.
    llm_provider_chain: str = "openrouter,gemini,groq,cerebras"
    # Google Gemini via its OpenAI-compatibility endpoint (note the trailing
    # "/openai/" — the path is the compatibility shim, not the native API).
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.0-flash"
    # Discovery document features (upload → extract → draft) call Gemini with
    # the *org's own* key from org_llm_keys — never the platform chain above.
    # Comma-separated, tried in order: a 404 moves to the next one. Google
    # closes older models to *new* accounts while still listing them — keys
    # created in AI Studio from late 2026 (the "AQ." format) get a 404 on
    # gemini-2.5-flash that older keys don't — so a single pinned id breaks
    # every newly onboarded org. The "-latest" alias is Google-maintained.
    discovery_gemini_model: str = "gemini-3.5-flash,gemini-3.6-flash"
    # 2.5-flash is a thinking model; its reasoning tokens share max_tokens with
    # the answer. "low" keeps structured-extraction answers from being starved.
    discovery_gemini_reasoning_effort: str = "low"
    # Discovery *search* needs Google Search grounding, which only the native
    # generateContent API exposes — not the OpenAI shim above.
    gemini_native_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    # Thinking tokens for the grounded search call; 0 disables thinking.
    discovery_search_thinking_budget: int = 1024
    # Fallbacks after Gemini for the Discovery features, on keys the org saves
    # under Settings → API Keys (provider "groq" / "openrouter"). Comma lists,
    # tried in order. Neither has search grounding: Discovery search on them is
    # training-data only, and says so in its response.
    discovery_groq_models: str = "qwen/qwen3.8-27b,openai/gpt-oss-120b"
    discovery_openrouter_models: str = (
        "meta-llama/llama-3.3-70b-instruct:free,google/gemini-2.0-flash-exp:free"
    )
    # Groq — free tier, very fast, generous per-day request cap.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Claude as an optional Discovery provider, on the org's own Anthropic key,
    # via Anthropic's OpenAI-compatible endpoint (so it goes through the same
    # chat_completion chain as every other provider — no SDK client).
    anthropic_openai_base_url: str = "https://api.anthropic.com/v1"
    discovery_anthropic_models: str = "claude-haiku-5-5,claude-sonnet-5-5"
    # Web search behind Discovery when Gemini's Google Search grounding is
    # unavailable: the org's Tavily / Serper / Brave key finds pages, and an LLM
    # reads only those pages. Queries per discovery run, and results per query.
    discovery_web_queries: int = 4
    discovery_web_results_per_query: int = 10
    groq_model: str = "llama-3.3-70b-versatile"
    # Cerebras — free tier, same Llama 3.3 70B under a different model id.
    cerebras_api_key: str = ""
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model: str = "llama-3.3-70b"
    # Circuit breaker. A provider that fails this many times IN A ROW is skipped
    # for the cooldown. Without it, a provider that is out of quota for the rest
    # of the day is re-tried on every call and pays its full timeout each time —
    # on a batch send that is a minute of added latency per email rather than a
    # few hundred milliseconds. Per-process, by design: it is a latency
    # optimisation, not a distributed lock.
    llm_breaker_threshold: int = 3
    llm_breaker_cooldown_seconds: int = 300
    # Client-side pacing, per provider, so we stay under the free-tier caps
    # rather than discovering them. The breaker is the wrong instrument for this:
    # it needs three consecutive failures, so a batch send hits the cap and then
    # burns three more round trips before anything stops. See
    # app.services.llm_rate_limit.
    #
    # "name:rpm" pairs. The caps differ by an order of magnitude between these
    # providers, and one global number set low enough for the tightest of them
    # throttles the whole chain to its worst hop.
    llm_provider_rpm: str = "openrouter:20,gemini:15,groq:30,cerebras:30"
    llm_default_rpm: int = 20
    # How long a single call will wait for a paced provider before moving to the
    # next one. Small on purpose: the premise of the chain is that another
    # provider is idle with an independent quota, so blocking a worker on the
    # busy one trades throughput for latency and gains nothing. 0 = never wait,
    # always move on.
    llm_rate_limit_max_wait_seconds: float = 3.0
    # ...unless EVERY provider is paced out at once, in which case moving on
    # means degrading to static mail-merge copy because we were two seconds
    # early. Then the chain waits out the soonest one, up to this long.
    llm_rate_limit_max_stall_seconds: float = 30.0

    # Apollo.io
    apollo_api_key: str = ""
    apollo_base_url: str = "https://api.apollo.io/v1"
    # Client-side token bucket, so we back off before Apollo has to 429 us.
    apollo_max_requests_per_minute: int = 50
    # Credit guardrails. The gate is checked *before* each call, so an
    # exhausted budget costs nothing. 0 = unlimited (escape hatch).
    apollo_daily_credit_budget: int = 200
    apollo_monthly_credit_budget: int = 5000
    # Notify once when spend crosses this fraction of a budget.
    prospecting_credit_alert_threshold: float = 0.8

    # --- Hunter.io email verification (P3) ---
    # Per-mailbox SMTP verification, layered behind the free syntax + MX check.
    # MX proves a *domain* accepts mail; Hunter proves the *mailbox* exists —
    # which is what keeps a bad Apollo batch from tripping the Feature 2 bounce
    # auto-pause. Empty key = capability disabled; the pipeline degrades to
    # today's MX-only behaviour and never blocks.
    hunter_api_key: str = ""
    hunter_base_url: str = "https://api.hunter.io/v2"
    # A 'deliverable' below this confidence still sends, but is flagged
    # low-confidence and is excluded from the auto-enroll high-confidence band.
    hunter_min_score: int = 80
    # Re-verify a contact whose verification is older than this. People leave;
    # addresses rot. Only applied to contacts we could actually still email.
    hunter_reverify_days: int = 90
    # Guardrail (not billing truth) on Hunter calls per UTC day. 0 = unlimited.
    hunter_daily_credit_budget: int = 100
    # Verify newly-imported contacts before they can enter a sequence.
    hunter_verify_on_import: bool = True
    # Last-resort gate: verify at send time if a contact somehow reached a send
    # unverified. Set False if verification latency ever becomes a bottleneck —
    # the async task still covers the normal path.
    hunter_verify_before_send: bool = True
    # Max contacts accepted by one bulk-verify request, and how many Hunter calls
    # run concurrently within a batch (Hunter rate-limits per-second).
    hunter_max_batch: int = 100
    hunter_concurrency: int = 5
    # The 15-minute verify_pending_contacts and weekly reverify sweeps. Off by
    # default: on a free Hunter plan (100 verifications/month) the pending sweep
    # spends the month on the existing contact backlog in its first day, and the
    # key arrives through the credential store, where no env flag can follow it.
    # Off leaves the allowance to the paths that verify what is about to be
    # mailed (Autopilot campaigns, verify-before-send, import). Turn it on with
    # a paid plan.
    hunter_background_sweeps: bool = False

    # --- DIY email pattern discovery ---
    # The free fallback under Hunter's email-finder: derive likely addresses from
    # a name + the employer's domain (see app/services/email_pattern_generator.py).
    # Generation is always free and offline; what needs configuring is how — or
    # whether — a guess gets *proven*, because an unproven guess is never
    # promoted past email_status='unknown'.
    pattern_discovery_enabled: bool = True
    # A Reacher instance (https://reacher.email) — self-hostable, so unlike
    # Hunter it costs no per-address credit. Empty = no probe layer, which
    # degrades to generate-only: candidates are surfaced for a human, never
    # auto-marked valid.
    reacher_url: str = ""
    reacher_api_key: str = ""
    # Raw SMTP RCPT-TO probing as a last resort. **Off by default, deliberately.**
    # Two reasons, both load-bearing: (1) AWS blocks outbound port 25 from EC2
    # unless you file a request, so on this deployment it would simply time out
    # into 'unknown'; (2) probing from a sending IP is the behaviour DNSBLs
    # classify as address harvesting — the exact reputation damage
    # blacklist_check and domain_health exist to catch. Enable only from an IP
    # you do not send from.
    pattern_smtp_fallback_enabled: bool = False
    # MAIL FROM for the probe. A real, monitored address on a domain you own:
    # a bogus sender is itself a spam signal to the receiving MTA.
    pattern_smtp_from_address: str = ""
    # Probes per person. Seven patterns exist, but each probe is an SMTP
    # conversation with a stranger's mail server; stopping early keeps a bulk
    # import from looking like a dictionary attack.
    pattern_max_probes: int = 4

    # --- Google Places SMB discovery (P4) ---
    # Places API (New) — the legacy Places API is deprecated for new projects.
    # Apollo's coverage of small international manufacturers is thin; Places has
    # near-exhaustive local-business coverage. What it does *not* have is emails
    # or named people, so a Places result is a lead on a prospect, not a
    # prospect — see app/services/places_discovery.py. Empty key = capability
    # disabled; searches return 'skipped' and never raise.
    google_places_api_key: str = ""
    google_places_base_url: str = "https://places.googleapis.com/v1"
    # Guardrail on Places calls per UTC day. Broad coverage needs many narrow
    # queries (a grid over geography x category) and each page is billed, so
    # this is the one that stops a scheduled search from quietly compounding.
    # 0 = unlimited.
    google_places_daily_budget: int = 100
    google_places_monthly_budget: int = 2000
    # Client-side token bucket, so we back off before Google has to 429 us.
    google_places_max_requests_per_minute: int = 60
    # Places caps a text search at 20 results/page and 60 results total (3
    # pages). Encoded here so the pagination loop and its test agree on the
    # ceiling rather than both hardcoding it.
    google_places_page_size: int = 20
    google_places_max_results: int = 60

    # --- Multi-channel step execution (P2 audit) ---
    # LinkedIn and phone/SMS channels execute through provider stubs until a
    # real integration lands. An empty credential = "not configured": the
    # handler simulates the touch (status="simulated") instead of hitting an
    # API, so a cadence authored across channels still runs end-to-end.
    linkedin_api_token: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    # Census Bureau
    census_api_key: str = ""

    # App
    app_name: str = "DoAide Prospect"
    app_env: str = "development"

    # API key authentication (empty = disabled for backward compat)
    api_key: str = ""

    # Global sending kill switch — default state when the switch has never been
    # explicitly set in system_settings. Ships True (sending allowed) so existing
    # deployments and tests are unaffected; set SENDING_ENABLED_DEFAULT=false in
    # production so a fresh install stays PAUSED until an operator turns sending on
    # (belt-and-suspenders with the seeded "off" system_settings row, migration 059).
    sending_enabled_default: bool = True

    # JWT auth
    jwt_secret_key: str = ""  # Falls back to secret_key or token_encryption_key
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_minutes: int = 10080  # 7 days

    # Base URL for email tracking pixel / click-redirect endpoints
    tracking_base_url: str = "http://localhost:8000"
    # Instrument outreach *sequence* sends with an open pixel + wrapped links.
    # Without this the sequence path emits no engagement events at all, so the
    # send_condition branches, the optimizer, and per-variant A/B stats all read
    # counts that are structurally zero. Kept behind a flag because a pixel is a
    # (small) deliverability cost: turning it off restores the prior behaviour.
    sequence_tracking_enabled: bool = True
    # Instrument the FIRST touch of a sequence with an open pixel + wrapped
    # links. Off by default: the cold first email is the one most likely to be
    # flagged by a spam filter, and an open pixel + link-rewrapping is a strong
    # bulk-mail signal on an address with no prior relationship. Dropping tracking
    # on step 1 keeps it as plain as a personal note (which lands), and the
    # engagement signal it would have carried is recovered on steps 2+, which
    # stay tracked. Set True to restore uniform tracking across every step.
    track_first_step: bool = False

    # --- Manual (non-email) sequence steps ---
    # A "call" / "linkedin" / "video" / "direct_mail" step is a human touch. The
    # cadence holds at that step until the person marks the task done, so the
    # next email is spaced from when the call *happened* rather than from when
    # the reminder was filed.
    #
    # Holding needs a deadline or a forgotten task strands the enrollment
    # forever, which is a worse failure than a skipped touch: the contact goes
    # silent mid-sequence and nothing reports it. After this long the step
    # expires — the cadence resumes and the touch is recorded as *not made*,
    # which is the honest thing for the funnel to count.
    #
    # 72h rather than 24: a rep who queues Friday's calls should not lose them to
    # a weekend, and ``SequenceStep.task_timeout_hours`` overrides it per step
    # for the cadences where a same-day call is the whole point.
    manual_step_timeout_hours: int = 72
    # Set False to restore the old behaviour: file the task and advance
    # immediately, regardless of whether anyone ever does it.
    manual_step_blocks_cadence: bool = True

    # --- Repeated send failures ---
    # A send that throws all the way through its Celery retries is one failure
    # incident. After this many *consecutive* incidents the enrollment is paused
    # (``pause_reason='send_failures'``) instead of being re-dispatched every
    # sweep forever, which is what a permanent fault used to cost.
    #
    # 5 rather than 1: transient faults are the common case — a provider blip, a
    # timeout, a token refresh landing badly — and pausing a live sequence on the
    # first one would turn a self-healing hiccup into manual work. 5 consecutive
    # incidents, spread over the backoff below, is roughly a day of a thing
    # genuinely not working.
    #
    # 0 disables quarantine; the backoff still applies. See
    # ``app.services.send_failures``.
    send_failure_quarantine_threshold: int = 5
    # Base for the exponential backoff between incidents (15m, 30m, 1h, 2h, 4h…),
    # clamped to [10m, 12h]. The floor is deliberately above the 5-minute queue
    # sweep: a backoff shorter than the sweep interval does nothing.
    send_failure_backoff_base_minutes: int = 15

    # --- CAN-SPAM sender identity ---
    # Legal sender name + a FULL physical postal address are required on every
    # commercial email (CAN-SPAM / CASL). ``company_postal_address`` must be a
    # complete address (street/suite or registered PO box + city + region +
    # postal code + country), not just a city. Override via env for the real
    # registered address.
    company_name: str = "Apprend Technologies"
    company_postal_address: str = "Bangalore, Karnataka 560001, India"

    # Secret key for signing tokens (unsubscribe links, etc.)
    secret_key: str = ""
    # Mailbox that receives RFC 2369 mailto: unsubscribe requests. When set, it is
    # added to the List-Unsubscribe header alongside the one-click https URL —
    # Gmail/Yahoo bulk-sender guidelines recommend offering both methods. Leave
    # empty to use the sending account's own (monitored, reply-synced) address;
    # only set this to a mailbox that can actually receive mail (has MX records).
    unsubscribe_mailto: str = ""
    # Only send during these hours (sender-local) on weekdays — outreach that lands
    # at 3 AM reads as a bot and hurts reply rates. 24h clock; set send_window_enabled
    # False to send around the clock.
    send_window_enabled: bool = True
    send_window_start_hour: int = 8       # inclusive, e.g. 8 = 8:00am
    send_window_end_hour: int = 18        # exclusive, e.g. 18 = 6:00pm
    send_window_timezone: str = "America/Toronto"
    send_window_weekdays_only: bool = True
    notification_email: str = ""
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"


    # Anti-spam / sending limits
    max_emails_per_contact_per_sequence: int = 5       # Max steps in any sequence
    max_emails_per_day: int = 100                       # Steady-state daily cap (== fully-warmed week-3 volume)
    min_days_between_followups: int = 3                 # Minimum gap between follow-ups to same contact
    max_emails_per_contact_per_month: int = 8           # Hard cap per contact per 30 days
    # Account-level (per-company) sending discipline. The per-contact caps above
    # are keyed on an *address* and cannot see that info@, sales@ and two named
    # people are the same buyer. See ``SendLimiter.can_send_to_company``.
    min_hours_between_company_sends: int = 24           # Gap between touching two people at one company
    max_contacts_per_company_per_signal: int = 2        # Fan-out cap when a signal surfaces a company
    # At-most-once sending (``services.send_claim``). Every gate above reasons
    # about state in the database, and the window between "Gmail has the
    # message" and "the transaction that says so committed" has not written any
    # of it — so a retry in that window re-sends. A claim taken before the wire
    # and committed separately is the only fact that survives it.
    #
    # ``grace`` is how long a claim is presumed to belong to a worker still
    # running: longer than a research + LLM composition + SMTP round trip, or a
    # healthy concurrent send gets reconciled out from under it. Floored in the
    # module at 10 minutes.
    send_claim_grace_minutes: int = 15
    # How long a queued send task is presumed to still be coming. The sweep
    # marked nothing when it dispatched, so a worker running behind was handed
    # another copy of its whole backlog every five minutes. Must exceed the
    # sweep interval (300s) to have any effect at all.
    dispatch_lease_minutes: int = 15
    warmup_enabled: bool = True                         # Gradual ramp-up
    # Conservative warm-up ramp for a brand-new sending domain/mailbox:
    # 20/day week 1 → 50/day week 2 → 100/day week 3+ (then steady at max_emails_per_day).
    warmup_daily_limit_week1: int = 20                  # Week 1 daily limit
    warmup_daily_limit_week2: int = 50                  # Week 2 daily limit
    warmup_daily_limit_week3: int = 100                 # Week 3+ daily limit
    # ISO date the warm-up ramp starts from. Defaults to the launch date; the
    # persisted system_settings value (auto-stamped on first send, or reset from
    # the UI) takes precedence over this static default.
    warmup_start_date: str = "2026-07-15"

    # Per-mailbox send pacing (app/services/send_pacing.py). The warm-up ramp
    # caps how MUCH a mailbox sends per day; this caps how FAST. Without it the
    # 5-minute queue sweep dispatches every due enrollment at once and a single
    # mailbox can put ten messages on the wire inside a second — the burst shape
    # bulk-send detection keys on, at a daily volume that looks perfectly safe.
    # The gap is drawn from [min_gap, min_gap + jitter] because a *fixed* gap is
    # its own fingerprint. Defaults: 60-180s, i.e. ~20-60 sends/hour/mailbox,
    # comfortably above the warm-up ceiling of 100/day.
    send_pacing_enabled: bool = True
    send_min_gap_seconds: int = 60
    send_gap_jitter_seconds: int = 120

    # Pre-send spam gate (app/services/spam_guard.py). The deterministic content
    # scan runs on the final rendered copy of every outbound message.
    #  - flag  (>= flag threshold): send proceeds, score recorded on the send.
    #  - block (>= block threshold AND block_enabled): soft-skip, enrollment held.
    # Blocking is OFF by default on purpose. A content heuristic that stops
    # outreach on its own judgement is a worse failure than the spam it prevents,
    # so an operator turns it on after watching the flag lane for a while.
    spam_guard_enabled: bool = True
    spam_guard_flag_threshold: int = 25                  # "medium" risk band
    spam_guard_block_threshold: int = 50                 # "high" risk band
    spam_guard_block_enabled: bool = False

    # Per-sender auto-pause on danger signals. When only individual accounts
    # breach (no global-scope breach), pause just those mailboxes instead of
    # halting all sending — a single bad list on one inbox shouldn't stop the
    # other senders. A global-scope breach still trips the global kill switch,
    # because shared domain/IP reputation endangers everything.
    danger_account_auto_pause_enabled: bool = True

    # Smart auto-reply
    auto_reply_enabled: bool = True                      # Auto-send AI-drafted replies for clear positive intents
    # Safety floor for auto-send (P1 fix). An AI-drafted reply is only auto-sent
    # when the reply analysis is at least this confident about the intent; below
    # it we draft-only and leave the send to a human. Guards against replying to
    # a misread "please stop" or a borderline classification.
    auto_reply_min_confidence: float = 0.75

    # Complaint / feedback-loop (FBL) ingestion (P1 fix). Rolling lookback window
    # the mailbox sweep scans for abuse/complaint reports, mirroring the bounce
    # sweep. A complained-about recipient is suppressed immediately, like a hard
    # bounce.
    complaint_lookback_days: int = 3

    # Autonomous follow-up + re-scoring (Celery beat)
    stale_lead_followup_days: int = 7                      # Active-pipeline lead with no touch in N days → auto follow-up reminder
    stale_lead_followup_stages: List[str] = [             # Stages worth chasing automatically
        "contacted", "engaged", "qualified", "opportunity",
    ]
    auto_followup_batch_limit: int = 50                    # Max reminders created per beat run
    rescore_batch_limit: int = 50                          # Max companies re-scored per beat run

    # Scheduling link (Calendly or similar) — included in auto-replies when set.
    # Also the global fallback for the per-sender {{booking_link}} (Feature 4):
    # sender booking_url → scheduling_link → empty.
    scheduling_link: str = ""

    # --- Calendar Booking Injection (Feature 4) ---
    # Allowlist of scheduler hosts a booking URL may point at. A saved booking_url
    # must be https and its host must equal (or be a subdomain of) one of these —
    # defense against injecting an arbitrary link into outbound mail.
    booking_scheduler_hosts: List[str] = [
        "calendly.com",
        "cal.com",
        "acuityscheduling.com",
        "savvycal.com",
        "hubspot.com",          # meetings.hubspot.com
        "chilipiper.com",
        "youcanbook.me",
        "oncehub.com",          # ...scheduleonce
        "tidycal.com",
        "zcal.co",
        "koalendar.com",
        "outlook.office365.com", # Microsoft Bookings
        "calendar.google.com",   # Google appointment schedules
        "calendar.app.google",
    ]
    # Soft interest signal: how many opens of a contact's mail gate booking-link
    # injection (a repeated open reads as warming interest).
    booking_min_opens: int = 2

    # --- Daily autonomous-activity summary ---
    # Timezone whose calendar day the daily summary covers (the beat runs at
    # 6 PM in this zone). Also drives the day-boundary windowing.
    daily_summary_timezone: str = "Asia/Kolkata"
    # Rewrite the narrative into warmer prose via Claude (falls back to the
    # deterministic rule-based text if the model is unavailable).
    daily_summary_use_ai: bool = True
    # Email the daily digest to notification_email when True.
    daily_summary_email_enabled: bool = False

    # --- Phase 1 AI features ---
    # 1.2 AI-Optimized Sequence Engine: when enabled, the optimizer may adjust
    # timing, rewrite content, switch channel, or skip a step before each send.
    # Falls back to sending the static step verbatim on any failure.
    sequence_optimizer_enabled: bool = True
    # AI-composed outreach: ground the copy in the company's research brief
    # instead of sending static mail-merge. Always falls back to the static
    # template on any failure, so this can be turned off at any time without
    # stalling outreach.
    ai_compose_enabled: bool = True
    # Compose every step, not just the first touch. On: the sequence's ``tone``
    # only shapes copy the copywriter writes, so with this off a sequence spoke
    # in its own voice for one email and in the static templates' voice for the
    # other three. Each step writes under its own prompt hint (see
    # ``ai_compose.STEP_TEMPLATE_TYPES``) so the arc survives — a follow-up is
    # still written as a follow-up, not as a second first touch. The optimizer
    # still rewrites off engagement history afterwards.
    #
    # Default ON since 2026-07-24. Off, steps 2-4 shipped byte-identical copy to
    # every recipient — production sends of the same step differed only in the
    # greeting, which is the "reads identically to 500 other shops" failure the
    # copy prompt explicitly forbids. The review gate below still applies per
    # step, so turning this on queues drafts for a human before it sends any.
    ai_compose_all_steps: bool = True
    # How many AI-composed emails per sequence *step* go to a human before that
    # step is trusted to send them unattended. Cold outreach is irreversible and
    # charged against the sending domain, so the first batch gets read. Counted
    # per step, not per sequence: a quota the first touch can exhaust on its own
    # would let every follow-up the model has ever written send unread.
    ai_compose_review_batch_size: int = 3
    # Restrict that review quota to the first touch.
    #
    # This is what makes ``ai_compose_all_steps`` safe to leave on. The two used
    # to be welded together: composing every step also *gated* every step, so a
    # follow-up with no reviewer sat in ``pending_review`` and froze the whole
    # enrollment mid-sequence — which is why all-steps was turned back off, and
    # why steps 2-4 then shipped byte-identical copy to every recipient.
    #
    # Step 1 is the touch that deserves the gate: it is the first impression, it
    # goes to someone who has never heard from us, and it is unrecoverable. A
    # follow-up lands in a thread the prospect has already seen, and it still
    # passes the spam scan, the truthfulness guardrail, and the static fallback
    # before it ships. Set False to gate every step (the old coupled behaviour).
    ai_compose_review_first_step_only: bool = True
    # A ``pending_review`` draft parks its enrollment (the outreach queue only
    # dispatches ``active``). If no human clears it, the whole enrollment stalls
    # indefinitely. After this many hours the auto-expire sweep releases the
    # draft — it does NOT send the unreviewed AI copy; it falls back to the
    # static template a human already wrote and un-parks the enrollment, so a
    # missed review costs a plainer email, never a frozen prospect. 0 disables.
    ai_compose_review_expiry_hours: int = 24
    # 1.4 AI Company Research Agent: auto-trigger a research brief when a company
    # is created / before first outreach.
    auto_research_enabled: bool = True
    # 1.5 Conversation Intelligence: batch analysis of active threads.
    conversation_intelligence_enabled: bool = True
    # Threads with activity in the last N days are considered "active" for the
    # analyze_active_threads beat task.
    conversation_active_days: int = 14

    # Signal-based outreach timing: when fresh company news (funding, a new hire,
    # expansion) lands, prospects at that company are at their most receptive.
    # The signal-enrollment sweep reacts to those signals by surfacing the
    # company's un-enrolled, emailable contacts for priority outreach. Fresh =
    # discovered within this many hours (the relevance window closes fast — a
    # funding round is old news in a week). ``signal_enrollment_auto_enroll``
    # controls whether a matched prospect is enrolled automatically (into the
    # company's resolved vertical sequence) or only flagged as a priority
    # follow-up for a human to send — flagging is the safe default so a news
    # blip can never fire an unreviewed cold send.
    signal_enrollment_enabled: bool = True
    signal_enrollment_fresh_hours: int = 48
    signal_enrollment_auto_enroll: bool = False
    # Only news scored at least this relevant (1-10) counts as an outreach
    # trigger, so routine PR noise doesn't surface prospects.
    signal_enrollment_min_relevance: int = 5
    # The same sweep also reacts to scored buyer-intent signals (a Quality
    # Engineer role posted, a raise, a new plant). Separate threshold because the
    # scales are different: intent blends relevance and urgency 0-100 each, so
    # reusing ``min_relevance`` above would admit every signal ever recorded.
    signal_enrollment_include_intent: bool = True
    signal_enrollment_min_intent_score: float = 45.0

    # --- Phase 3 AI features ---
    # 3.1 Autonomous Prospecting Agent.
    prospecting_enabled: bool = True
    prospecting_target_volume: int = 25            # candidates evaluated per run
    prospecting_daily_max: int = 25                # hard cap on companies enrolled per day
    prospecting_auto_enroll_threshold: int = 65    # fit >= → auto-enroll into outreach
    prospecting_review_threshold: int = 50         # fit >= → human review queue
    # Auto-pause the agent if the rolling enroll→engaged conversion drops below
    # this fraction (guardrail against enrolling junk). 0 disables the check.
    prospecting_min_conversion_rate: float = 0.0
    # Places businesses arrive with no email, so the prospector enriches them
    # inline (Hunter, then a scrape) to have anyone to enrol. Both hops are slow
    # and `POST /prospecting/run` awaits the run, so this caps how many a single
    # run will pay for. The nightly sweep in tasks/places_discovery.py picks up
    # whatever is left over, unbounded by the request clock. 0 disables inline
    # enrichment and defers entirely to that sweep.
    prospecting_places_enrich_max: int = 10

    # --- Market policy ---
    # Countries we actually import and prospect into. Narrower than the schema
    # allowlist (``app.constants.ALLOWED_COUNTRIES``) and tunable without a
    # migration. A record whose country is outside this list is **rejected**,
    # never relabelled — a mislabelled company is worse than a rejected one.
    # Must stay a subset of ALLOWED_COUNTRIES or inserts fail the CHECK.
    supported_countries: list[str] = ["US", "CA", "IN"]

    # 3.3 Deliverability Optimizer.
    deliverability_optimizer_enabled: bool = True
    sender_bounce_rate_throttle: float = 0.05      # 7d bounce rate above this → throttle
    sender_complaint_rate_throttle: float = 0.003  # 7d complaint rate above this → throttle
    sender_min_reputation: int = 40                # below this the account is 'critical'

    # Feature 2 — Auto-Pause on Danger Signals (deliverability safety rail).
    # A 5-minute monitor task computes 24h rolling bounce/complaint metrics per
    # sending account and globally, and auto-engages the kill switch on breach.
    danger_signals_enabled: bool = True
    # 24h bounce rate: >5% auto-pauses, >3% warns (needs the min sample below).
    danger_bounce_rate_threshold: float = 0.05
    danger_bounce_rate_warn: float = 0.03
    # 24h spam-complaint rate: >0.1% auto-pauses, >0.05% warns (min sample too).
    danger_complaint_rate_threshold: float = 0.001
    danger_complaint_rate_warn: float = 0.0005
    # Absolute hard-bounce floor in 24h — trips regardless of sample size, so a
    # bad list burns the domain even during low-volume warm-up (rates lie at
    # small N). >=5 auto-pauses, >=3 warns.
    danger_hard_bounce_count_threshold: int = 5
    danger_hard_bounce_count_warn: int = 3
    # Minimum sends in the window before a *rate* trip is trusted (rates are
    # meaningless at low volume — 1 bounce of 20 is 5%). The hard-bounce floor is
    # the low-volume backstop and is NOT gated by this.
    danger_min_sample_size: int = 20

    # 3.4 RAG Knowledge System.
    rag_enabled: bool = True
    rag_top_k: int = 5                             # retrieved chunks per query
    rag_min_similarity: float = 0.05               # drop weak matches below this cosine sim

    # --- Autonomous lead-gen enhancements ---
    # Per-account warm-up: a newly connected mailbox ramps its own daily cap
    # based on how long ago it was connected, independent of the global warm-up
    # date. Applied by send_limiter.account_capacity() alongside the reputation
    # throttle. Disable to fall back to the reputation-only cap.
    account_warmup_enabled: bool = True
    account_warmup_daily_limit_week1: int = 20   # days 0-6 since connected_at
    account_warmup_daily_limit_week2: int = 50   # days 7-13
    account_warmup_daily_limit_week3: int = 100  # days 14-20 (then uncapped by warm-up)
    # Bounce handling: after this many soft bounces a contact is suppressed
    # (treated like a hard bounce). A single hard bounce suppresses immediately.
    soft_bounce_suppress_threshold: int = 3
    # Write off a whole recipient domain once this many *distinct* addresses
    # there have hard-bounced and nothing at the domain has ever engaged. The
    # per-address gates cannot see that the same wrong address convention is
    # failing for everybody at one company. Below 1 disables the rule.
    # See ``app.services.domain_reputation``.
    dead_domain_bounce_threshold: int = 3
    # DKIM has no directory record; we probe these common selectors at
    # <selector>._domainkey.<domain> when no explicit selector is supplied.
    dkim_default_selectors: List[str] = ["google", "default", "selector1", "selector2", "k1"]

    # --- Feature 5: Domain Health Dashboard ---
    # DNSBL zones queried per sending domain/IP by the daily domain-health check.
    # IP-based zones (reversed-IP.<zone> A-record lookup). For Gmail domains the
    # sending IP is Google's shared pool, so these are informational only.
    dnsbl_zones: List[str] = ["zen.spamhaus.org", "b.barracudacentral.org", "bl.spamcop.net"]
    # Domain-based zones (<domain>.<zone>) — the meaningful signal for shared-pool
    # senders; these DO count toward the health score.
    dnsbl_domain_zones: List[str] = ["dbl.spamhaus.org"]
    # Spamhaus Data Query Service key (26 chars). Optional. Spamhaus refuses
    # public-mirror queries that arrive via an open/public resolver — which is
    # exactly what our EC2 box uses — answering 127.255.255.254 ("no data for
    # you"), so every spamhaus.org zone currently reports `unknown`. A DQS key
    # authenticates the query and makes the answer real. When set, the
    # spamhaus.org zones above are transparently queried as
    # <key>.<zone>.dq.spamhaus.net (see blacklist_check.dqs_zone); non-Spamhaus
    # zones (Barracuda, SpamCop) are unaffected. Note the free DQS tier is
    # non-commercial-only, so a commercial deployment wants the trial or a paid
    # subscription — see .env.example. Treat the key as a credential.
    spamhaus_dqs_key: str = ""
    # Composite health-score weights (unknown/unverifiable components are dropped
    # from the denominator so the score reflects only what we could verify).
    domain_health_weight_spf: int = 20
    domain_health_weight_dkim: int = 20
    domain_health_weight_dmarc: int = 25
    domain_health_weight_blacklists: int = 30
    # Band boundaries: >= healthy_min → healthy; >= watch_min → watch; else at_risk.
    domain_health_band_healthy_min: int = 90
    domain_health_band_watch_min: int = 70
    # On-demand recheck cooldown (seconds) to be polite to the DNSBL zones.
    domain_health_recheck_cooldown_seconds: int = 60

    # --- Feature 6: Smart Send-Time Optimization ---
    # Master switch. When off, the send queue keeps today's naive next-window
    # timing for every enrollment (no scheduling layer, no A/B split).
    smart_send_timing_enabled: bool = True
    # A contact needs at least this many prior opens before its PERSONAL pattern
    # is trusted; below it we blend to the cohort/best-practice default. Cold
    # lists have 0-2 opens, so the default is load-bearing.
    smart_timing_min_opens: int = 3
    # Hard ceiling on how far past a step's due time smart timing may defer it, so
    # optimization can never stall a sequence. Warm-up caps + danger-pause always
    # take precedence over this.
    max_smart_delay_hours: int = 48
    # Best-practice default window (recipient-local) when personal data is thin:
    # Tue-Thu, 9-11 AM. Days are 0=Sunday..6=Saturday (matches send_timing).
    smart_timing_default_dows: List[int] = [2, 3, 4]   # Tue, Wed, Thu
    smart_timing_default_hour_start: int = 9            # inclusive
    smart_timing_default_hour_end: int = 11            # exclusive
    # Width (hours) of the window built around a recipient's personal best hour.
    smart_timing_personal_window_hours: int = 2
    # How many opens one reply is worth when picking a recipient's best (dow, hour)
    # bucket. A reply is a far stronger "they were engaged at this local time"
    # signal than an open, so it dominates the peak selection. Raw event counts
    # (not this weight) still drive the smart_timing_min_opens trust threshold.
    smart_timing_reply_weight: int = 3
    # Fraction of newly-scheduled enrollments assigned to the 'optimized' arm; the
    # rest are 'control' (naive timing) so the lift can be measured. 0.5 = even.
    smart_timing_optimized_fraction: float = 0.5

    # 3.5 Voice Intelligence.
    voice_intelligence_enabled: bool = True
    # OpenAI key for Whisper transcription (optional — a transcript can also be
    # supplied directly on upload, and analysis works from any transcript).
    openai_api_key: str = ""
    whisper_model: str = "whisper-1"

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            value = v.strip()
            if not value:
                return []
            # Accept a JSON array or a plain comma-separated list.
            if value.startswith("["):
                import json
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return v

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
