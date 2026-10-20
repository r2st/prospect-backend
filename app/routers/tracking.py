"""Email open and click tracking endpoints."""

import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from html import escape
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from itsdangerous import URLSafeTimedSerializer

from app.config import get_settings
from app.database import get_db
from app.models.email_template import EmailSend
from app.models.contact import Contact
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.org_context import org_scope
from app.services import opt_out

logger = logging.getLogger(__name__)

# Random per-process fallback key — used only when secret_key is not configured.
# Tokens signed with this key will NOT survive a process restart, but that is
# preferable to using a hardcoded, guessable secret.
_FALLBACK_KEY = secrets.token_urlsafe(32)
router = APIRouter(prefix="/track", tags=["tracking"])

# Status hierarchy — higher number = more advanced state; never downgrade
_STATUS_RANK = {
    "sent": 0,
    "delivered": 1,
    "opened": 2,
    "clicked": 3,
    "replied": 4,
}

# 1x1 transparent GIF
PIXEL_GIF = bytes([
    0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00,
    0x01, 0x00, 0x80, 0x00, 0x00, 0xff, 0xff, 0xff,
    0x00, 0x00, 0x00, 0x21, 0xf9, 0x04, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44,
    0x01, 0x00, 0x3b
])


def _enqueue_signal_rescore(*, contact_id) -> None:
    """Best-effort trigger for real-time lead re-routing on an engagement signal.

    Lazily imported and fully guarded so the tracking pixel/redirect never fails
    (and stays fast) when no Celery broker is reachable, e.g. in tests.
    """
    try:
        from app.tasks.lead_routing import enqueue_signal_rescore
        enqueue_signal_rescore(contact_id=contact_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Signal rescore enqueue skipped: %s", exc)


def _get_serializer():
    settings = get_settings()
    key = settings.secret_key
    if not key:
        logger.warning("secret_key not set — tracking tokens will use a random per-process key")
        key = _FALLBACK_KEY
    return URLSafeTimedSerializer(key)


def _get_unsubscribe_serializer():
    """Delegates to the one place unsubscribe tokens are minted and verified.

    Kept as a thin alias because tests and older callers import it by name. The
    body used to be a second, independently-drifting copy of the sender's key
    resolution — see :func:`app.services.opt_out.unsubscribe_serializer`.
    """
    return opt_out.unsubscribe_serializer()


def generate_tracking_pixel_url(send_id: str) -> str:
    """Generate a tracking pixel URL for an email send."""
    serializer = _get_serializer()
    token = serializer.dumps(send_id, salt="open-track")
    settings = get_settings()
    base = settings.tracking_base_url.rstrip("/")
    return f"{base}/api/v1/track/open/{token}"


def generate_click_url(send_id: str, original_url: str) -> str:
    """Wrap a URL for click tracking."""
    serializer = _get_serializer()
    token = serializer.dumps({"send_id": send_id, "url": original_url}, salt="click-track")
    settings = get_settings()
    base = settings.tracking_base_url.rstrip("/")
    return f"{base}/api/v1/track/click/{token}"


# ---------------------------------------------------------------------------
# Sequence (outreach) tracking
# ---------------------------------------------------------------------------
# The sequence path records no ``EmailSend`` row — its send log is
# ``OutreachEvent`` keyed by (enrollment_id, step_number). So sequence tokens
# carry that pair instead of a send id, and resolve to an ``OutreachEvent``
# rather than an ``EmailSend`` update. Both token shapes share the endpoints and
# salts below; they're told apart by payload type (str → EmailSend, dict with
# ``eid`` → enrollment), which keeps existing links working unchanged.


def generate_sequence_pixel_url(enrollment_id: str, step_number: int) -> str:
    """Tracking pixel URL for one sequence step send."""
    serializer = _get_serializer()
    token = serializer.dumps(
        {"eid": str(enrollment_id), "step": int(step_number)}, salt="open-track"
    )
    base = get_settings().tracking_base_url.rstrip("/")
    return f"{base}/api/v1/track/open/{token}"


def generate_sequence_click_url(enrollment_id: str, step_number: int, original_url: str) -> str:
    """Wrap a URL for click tracking within a sequence step send."""
    serializer = _get_serializer()
    token = serializer.dumps(
        {"eid": str(enrollment_id), "step": int(step_number), "url": original_url},
        salt="click-track",
    )
    base = get_settings().tracking_base_url.rstrip("/")
    return f"{base}/api/v1/track/click/{token}"


async def _engagement_exists(
    db: AsyncSession, enrollment_id, step_number: int, event_type: str
) -> bool:
    """Whether this (enrollment, step, event_type) has already been recorded.

    The cheap check that answers the overwhelmingly common case — a mail client
    re-fetching a pixel minutes or days later. It is *not* what makes the dedup
    hold; the partial unique index from migration 117 is. Its own function so a
    test can make it miss and exercise the losing side of the race, which is the
    one path a single-connection test database cannot produce on its own.
    """
    return (
        await db.execute(
            select(OutreachEvent.id).where(
                OutreachEvent.enrollment_id == enrollment_id,
                OutreachEvent.step_number == int(step_number),
                OutreachEvent.event_type == event_type,
            ).limit(1)
        )
    ).scalars().first() is not None


async def _record_outreach_engagement(
    db: AsyncSession, enrollment_id: str, step_number: int, event_type: str
) -> bool:
    """Record an ``opened``/``clicked`` OutreachEvent for a sequence step.

    Deduplicated per (enrollment, step, event_type): a pixel is fetched once per
    open but mail clients re-fetch it on every re-render, and the optimizer reads
    these as *counts*, so writing one row per fetch would inflate engagement.
    First occurrence wins; repeats are dropped.

    **The read below is the fast path, not the guarantee.** A SELECT followed by
    an INSERT is not a constraint, and this endpoint is public, unauthenticated
    and served concurrently — a phone and a laptop rendering the same message, a
    provider prefetch landing beside the human's open, or the click handler
    recording the implied ``opened`` while the pixel records the real one. Two
    requests both read no row and both wrote one. The duplicate is not only a
    count: ``adaptive_delay.register_engagement`` runs per recorded event, so a
    doubled open shortened the cadence twice and pulled the next step in further
    than the signal justified. Migration 117 adds the partial unique index; the
    savepoint below is how losing that race is turned back into "already
    recorded" rather than a 500 on a tracking pixel.

    Returns True when a new row was written. Never raises — tracking must not
    break the pixel response.
    """
    try:
        # The pixel is public (no auth), so no org context is bound. Load the
        # enrollment unfiltered to learn which tenant it belongs to, then bind
        # that org so the OutreachEvent is stamped to the RIGHT organization —
        # otherwise engagement would be misattributed to the first-party org and
        # the real tenant's analytics / send-time optimizer would never see it.
        enrollment = (
            await db.execute(
                select(OutreachEnrollment)
                .where(OutreachEnrollment.id == uuid.UUID(str(enrollment_id)))
                .execution_options(skip_org_filter=True)
            )
        ).scalar_one_or_none()
        if enrollment is None:
            return False

        async with org_scope(enrollment.org_id):
            if await _engagement_exists(
                db, enrollment.id, int(step_number), event_type
            ):
                return False

            try:
                async with db.begin_nested():
                    db.add(
                        OutreachEvent(
                            enrollment_id=enrollment.id,
                            step_number=int(step_number),
                            event_type=event_type,
                            occurred_at=datetime.now(timezone.utc),
                            event_metadata={"source": "tracking_pixel"},
                        )
                    )
                    await db.flush()
            except IntegrityError:
                # A concurrent fetch of the same pixel got there first. The
                # savepoint takes the row back and leaves the session usable, so
                # this returns the same "already recorded" the SELECT above
                # would have — and, critically, returns *before* the engagement
                # registration below, which must run once per real signal.
                logger.debug(
                    "Concurrent %s already recorded for enrollment %s step %s",
                    event_type, enrollment_id, step_number,
                )
                return False

            # Adaptive Step Delays (Feature 3): an open/click is a warm signal —
            # shorten the enrollment's cadence and pull in the pending next step.
            from app.services import adaptive_delay

            await adaptive_delay.register_engagement(db, enrollment)
            await db.commit()

        # An open/click is an engagement signal — re-route the lead now.
        if enrollment.contact_id:
            _enqueue_signal_rescore(contact_id=enrollment.contact_id)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to record %s for enrollment %s: %s", event_type, enrollment_id, exc)
        return False


@router.get("/open/{token}")
async def track_open(token: str, db: AsyncSession = Depends(get_db)):
    """Record an email open event and return a 1x1 transparent GIF."""
    try:
        serializer = _get_serializer()
        payload = serializer.loads(token, salt="open-track", max_age=86400 * 90)  # 90 days

        # Sequence-step token → record an OutreachEvent and stop here.
        if isinstance(payload, dict) and payload.get("eid"):
            await _record_outreach_engagement(
                db, payload["eid"], payload.get("step", 1), "opened"
            )
            return Response(content=PIXEL_GIF, media_type="image/gif", headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            })

        send_id = payload

        # Fetch current record to check status before updating
        result = await db.execute(
            select(EmailSend).where(EmailSend.id == uuid.UUID(send_id))
        )
        send_record = result.scalar_one_or_none()

        if send_record:
            current_rank = _STATUS_RANK.get(send_record.status, 0)
            new_rank = _STATUS_RANK.get("opened", 2)

            updates = {}
            # Only set opened_at on first open
            if send_record.opened_at is None:
                updates["opened_at"] = datetime.now(timezone.utc)
            # Only upgrade status, never downgrade
            if new_rank > current_rank:
                updates["status"] = "opened"

            if updates:
                await db.execute(
                    update(EmailSend)
                    .where(EmailSend.id == uuid.UUID(send_id))
                    .values(**updates)
                )
                await db.commit()
                # Real-time lead prioritization: an open is an engagement signal,
                # so re-route this lead immediately (best-effort, never blocks).
                if send_record.contact_id:
                    _enqueue_signal_rescore(contact_id=send_record.contact_id)

        logger.info("Tracked email open for send %s", send_id)
    except Exception as e:
        logger.debug("Open tracking failed: %s", str(e))

    return Response(content=PIXEL_GIF, media_type="image/gif", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    })


@router.get("/click/{token}")
async def track_click(token: str, db: AsyncSession = Depends(get_db)):
    """Record a click event and redirect to the original URL."""
    try:
        serializer = _get_serializer()
        data = serializer.loads(token, salt="click-track", max_age=86400 * 90)
        original_url = data["url"]

        # Sequence-step token → record OutreachEvents and redirect. A click
        # implies an open, and mail clients that block images mean the open
        # pixel may never have fired — so record both.
        if data.get("eid"):
            step = data.get("step", 1)
            await _record_outreach_engagement(db, data["eid"], step, "opened")
            await _record_outreach_engagement(db, data["eid"], step, "clicked")
            return RedirectResponse(url=original_url, status_code=302)

        send_id = data["send_id"]

        # Fetch current record to check status before updating
        result = await db.execute(
            select(EmailSend).where(EmailSend.id == uuid.UUID(send_id))
        )
        send_record = result.scalar_one_or_none()

        if send_record:
            current_rank = _STATUS_RANK.get(send_record.status, 0)
            new_rank = _STATUS_RANK.get("clicked", 3)

            updates = {}
            # Always record clicked_at on first click
            if send_record.clicked_at is None:
                updates["clicked_at"] = datetime.now(timezone.utc)
            # Also set opened_at if not already set (click implies open)
            if send_record.opened_at is None:
                updates["opened_at"] = datetime.now(timezone.utc)
            # Only upgrade status, never downgrade
            if new_rank > current_rank:
                updates["status"] = "clicked"

            if updates:
                await db.execute(
                    update(EmailSend)
                    .where(EmailSend.id == uuid.UUID(send_id))
                    .values(**updates)
                )
                await db.commit()
                # A click is a strong buying signal — re-route the lead now.
                if send_record.contact_id:
                    _enqueue_signal_rescore(contact_id=send_record.contact_id)

        logger.info("Tracked click for send %s -> %s", send_id, original_url)
        return RedirectResponse(url=original_url, status_code=302)
    except Exception as e:
        logger.warning("Click tracking failed: %s", str(e))
        return RedirectResponse(url="https://apprendtechnologies.com", status_code=302)


def _decode_unsubscribe_token(token: str) -> tuple[Optional[str], Optional[uuid.UUID]]:
    """Verify an unsubscribe token into ``(email, org_id)``. Raises if invalid.

    Two shapes, because tokens outlive deploys. Current tokens are a dict
    carrying the sending org (``{"e": ..., "o": ...}``) so the opt-out lands on
    that sender's suppression list. Tokens minted before that are a bare address
    string and are valid for a year — they are sitting in real inboxes right now,
    and breaking them would break the one-click unsubscribe Gmail requires of us.
    They decode to ``org_id=None``, which the opt-out reads as "every org holding
    this address".
    """
    payload = _get_unsubscribe_serializer().loads(
        token, salt="unsubscribe", max_age=86400 * 365  # 1 year
    )
    if isinstance(payload, dict):
        raw_org = payload.get("o")
        org_id = None
        if raw_org:
            try:
                org_id = uuid.UUID(str(raw_org))
            except (ValueError, AttributeError, TypeError):
                # A signed token we minted, so this shouldn't happen — but an
                # unreadable org must not cost the recipient their opt-out.
                logger.warning("Unsubscribe token carried an unreadable org id")
        return payload.get("e"), org_id
    return payload, None


async def _process_unsubscribe(token: str, db: AsyncSession) -> bool:
    """Verify an unsubscribe token and stop all mail to that address.

    Raises when the token fails to verify; the callers turn that into the error
    page / a 400. Shared by the GET (a human clicks the footer link) and POST
    (RFC 8058 one-click) handlers so both do exactly the same thing.

    Delegates to :mod:`app.services.opt_out`, which is the point of this
    function's existence. It used to set ``contact.unsubscribed`` and stop — a
    column no send gate reads. The sequence engine checks ``contact.suppressed``
    and the org suppression list, so the sequence carried on to step 2, 3, 4 and
    5 after the recipient had opted out. See that module for the rest.
    """
    email, org_id = _decode_unsubscribe_token(token)
    result = await opt_out.record(
        db, email, org_id=org_id, reason="unsubscribe", source="auto",
        basis="unsubscribe link",
    )
    if result is None:
        # The signature verified but the payload isn't an address. Nothing to
        # suppress, and nothing the recipient can do about it — don't show them
        # a success page for an opt-out that recorded nothing.
        logger.warning("Unsubscribe token verified but carried no usable address")
        return False
    return True


# The page a human lands on from the footer link. Server-rendered with no
# script: it has to work in whatever browser a mail client opens, and the two
# actions are plain form POSTs so nothing happens until a person presses one.
_PAGE_CSS = """
:root { color-scheme: light dark; --bg:#f6f7f9; --card:#ffffff; --fg:#1d2330;
  --muted:#5b6475; --border:#e3e6eb; --accent:#2f5bd3; --accent-fg:#ffffff;
  --ok:#1f7a4d; }
@media (prefers-color-scheme: dark) { :root { --bg:#12151b; --card:#1b2029;
  --fg:#e8ebf0; --muted:#9aa3b2; --border:#2b3240; --accent:#7a9cff;
  --accent-fg:#0d1220; --ok:#5fcf98; } }
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; display:flex; align-items:center;
  justify-content:center; padding:24px 16px; background:var(--bg); color:var(--fg);
  font:16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { width:100%; max-width:440px; background:var(--card);
  border:1px solid var(--border); border-radius:14px; padding:32px 24px;
  text-align:center; }
h1 { font-size:1.35rem; font-weight:600; margin:0 0 8px; }
p { margin:0 0 20px; color:var(--muted); }
.email { display:block; margin:0 0 24px; font-weight:600; color:var(--fg);
  word-break:break-all; }
.ok { color:var(--ok); }
button { width:100%; min-height:48px; padding:12px 20px; font:inherit;
  font-weight:600; border-radius:10px; cursor:pointer; border:1px solid var(--accent);
  background:var(--accent); color:var(--accent-fg); }
button.secondary { background:transparent; color:var(--accent); }
button:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
"""


def _unsub_page(title: str, body_html: str, status_code: int = 200) -> HTMLResponse:
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta name='robots' content='noindex, nofollow, noarchive'>"
        "<meta name='referrer' content='no-referrer'>"
        f"<title>{escape(title)}</title><style>{_PAGE_CSS}</style></head>"
        f"<body><main>{body_html}</main></body></html>"
    )
    return HTMLResponse(
        html,
        status_code=status_code,
        headers={
            # The token in the URL is a capability over this address's
            # subscription. Keep it out of search indexes, caches and the
            # Referer header of anything the page might link to.
            "X-Robots-Tag": "noindex, nofollow, noarchive",
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
        },
    )


def _status_page(token: str, state: "opt_out.SubscriptionStatus", *, just_changed: bool = False) -> HTMLResponse:
    email_html = f"<span class='email'>{escape(state.email)}</span>"
    q = escape(quote(token, safe=""))
    if not state.unsubscribed:
        if just_changed:
            heading = "<h1 class='ok'>You're subscribed again</h1>"
            lead = "<p>Welcome back. You'll keep receiving our emails at</p>"
        else:
            heading = "<h1>Unsubscribe from our emails?</h1>"
            lead = "<p>Press the button below to stop all emails to</p>"
        return _unsub_page(
            "Email preferences",
            heading + lead + email_html
            + f"<form method='post' action='unsubscribe?token={q}'>"
            "<input type='hidden' name='source' value='page'>"
            "<button type='submit'>Unsubscribe</button></form>",
        )
    heading = "<h1 class='ok'>You're unsubscribed</h1>"
    lead = "<p>We won't send any more emails to</p>"
    if not state.can_resubscribe:
        return _unsub_page("Unsubscribed", heading + lead + email_html)
    return _unsub_page(
        "Unsubscribed",
        heading + lead + email_html
        + "<p>Unsubscribed by mistake?</p>"
        + f"<form method='post' action='resubscribe?token={q}'>"
        "<button type='submit' class='secondary'>Re-subscribe</button></form>",
    )


def _error_page(status_code: int = 400) -> HTMLResponse:
    return _unsub_page(
        "Link not valid",
        "<h1>That link didn't work</h1>"
        "<p>It may have expired. Just reply to any of our emails with the word "
        "UNSUBSCRIBE and we'll take care of it.</p>",
        status_code=status_code,
    )


async def _token_status(token: str, db: AsyncSession):
    """Verify ``token`` and read its address's state. ``None`` if not genuine."""
    try:
        email, org_id = _decode_unsubscribe_token(token)
    except Exception:
        return None, None, None
    state = await opt_out.status(db, email, org_id=org_id)
    return email, org_id, state


@router.get("/unsubscribe")
async def unsubscribe(token: str, db: AsyncSession = Depends(get_db)):
    """Show the address and an Unsubscribe button. Changes nothing.

    This used to unsubscribe on load. A GET is not consent: mail security
    scanners (Safe Links, Mimecast, Proofpoint) fetch every link in a message
    before the human sees it, so an opt-out on GET unsubscribes recipients who
    never clicked. The opt-out is the POST the button sends — and the RFC 8058
    one-click POST below, which a mailbox provider only sends on a user's click.

    Status is shown only for a token that verifies; anything else gets the
    error page and learns nothing about any address.
    """
    _, _, state = await _token_status(token, db)
    if state is None:
        return _error_page()
    return _status_page(token, state)


@router.post("/unsubscribe")
async def unsubscribe_one_click(
    request: Request, token: str, db: AsyncSession = Depends(get_db)
):
    """Handle RFC 8058 one-click unsubscribe, and the page's Unsubscribe button.

    Gmail/Yahoo bulk-sender guidelines (June 2024) require a ``List-Unsubscribe:
    <https…>`` URL paired with ``List-Unsubscribe-Post: List-Unsubscribe=One-Click``.
    The mailbox provider then POSTs to that URL (no human ever loads the page), so
    this must accept POST or the provider gets a 405 and the one-click affordance
    is treated as broken. That caller gets a small 200 body (ignored).

    The page's button posts here too, with ``source=page`` in the form, and gets
    the confirmation page back. Same URL on purpose: one opt-out code path.
    """
    from_page = False
    try:
        form = await request.form()
        from_page = form.get("source") == "page"
    except Exception:
        pass

    try:
        ok = await _process_unsubscribe(token, db)
    except Exception as e:
        logger.warning("Unsubscribe failed: %s", str(e))
        ok = False
    if not from_page:
        # A 400 tells the provider the token was bad; the address stays subscribed.
        return Response(status_code=200 if ok else 400)
    if not ok:
        return _error_page()
    # Re-read rather than assume: the page promises what the database says.
    _, _, state = await _token_status(token, db)
    if state is None or not state.unsubscribed:
        return _error_page()
    return _status_page(token, state, just_changed=True)


@router.post("/resubscribe")
async def resubscribe(token: str, db: AsyncSession = Depends(get_db)):
    """Undo this link's opt-out. 403 unless the token is genuine.

    Re-subscribing re-opens mail to a person, so unlike the opt-out — where
    being lenient errs towards sending less — a token that does not verify is
    refused outright. What it may lift is narrow; see ``opt_out.resubscribe``.
    """
    try:
        email, org_id = _decode_unsubscribe_token(token)
    except Exception:
        return _error_page(status_code=403)
    state = await opt_out.resubscribe(db, email, org_id=org_id)
    if state is None:
        return _error_page(status_code=403)
    if state.unsubscribed:
        # On the list for a reason this link did not put there (bounce,
        # complaint, erasure, manual). Nothing changed; say so plainly.
        return _status_page(token, state)
    return _status_page(token, state, just_changed=True)
