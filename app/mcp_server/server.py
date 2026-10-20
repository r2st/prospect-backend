"""The FastMCP app: registers the tools and runs each call in its own session.

Every call:

1. opens a fresh session (a long-lived one would hold a transaction, and stale
   identity-map rows, across a whole Claude conversation);
2. wraps itself in ``org_scope(None)`` so the org that ``authenticate`` binds
   is undone when the call returns, whatever happens in between;
3. re-authenticates the key, so a revoke takes effect on the next call;
4. commits on success and rolls back on any failure.

Refusals (:class:`ToolFailure`) reach the model verbatim. Anything else is
logged with its traceback to stderr and reported as a short internal error, so
a stack trace with SQL in it never ends up in someone's chat transcript.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any, Awaitable, Callable, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from app.mcp_server import tools
from app.mcp_server.auth import ToolFailure, authenticate
from app.org_context import org_scope

logger = logging.getLogger(__name__)

API_KEY_ENV = "PROSPECT_API_KEY"

INSTRUCTIONS = """\
DoAide Prospect: B2B lead generation and outreach. "Leads" are contacts, and
each one belongs to a company with a pipeline stage (new_lead → researching →
outreach → engaged → negotiating → client → churned). Campaigns are Autopilot
Outreach campaigns. Each one is built from uploaded discovery documents: it
finds prospects, drafts emails and sends them from a connected mailbox.

Ids come back as UUID strings. Feed them back exactly as they came. To build a
campaign: call list_discovery_documents (status 'ready'), then create_campaign,
then add_leads_to_campaign with ids from search_leads. These tools never send
email directly. With review on (the default), drafts wait for a human in the
Prospect app.
"""

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                        openWorldHint=True)

ToolFn = Callable[..., Awaitable[Any]]


def build_server(
    *,
    session_factory=None,
    api_key: Optional[str] = None,
) -> FastMCP:
    """Build the server. Tests pass a session factory and key; the CLI uses
    ``app.database`` and ``$PROSPECT_API_KEY``."""
    mcp = FastMCP("doaide-prospect", instructions=INSTRUCTIONS)

    async def call(fn: ToolFn, **kwargs) -> Any:
        if session_factory is not None:
            factory = session_factory
        else:
            from app.database import async_session_factory as factory
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        async with org_scope(None):
            async with factory() as db:
                try:
                    ctx = await authenticate(db, key)
                    # Keep last_used_at even if the tool itself fails.
                    await db.commit()
                    result = await fn(db, ctx, **kwargs)
                    await db.commit()
                    return result
                except ToolFailure as exc:
                    await db.rollback()
                    raise ToolError(str(exc)) from None
                except Exception as exc:
                    await db.rollback()
                    logger.exception("MCP tool %s failed", getattr(fn, "__name__", fn))
                    raise ToolError(
                        f"Internal error in Prospect ({type(exc).__name__}). "
                        "The server log has details."
                    ) from None

    # -- leads ---------------------------------------------------------------

    @mcp.tool(annotations=READ)
    async def search_leads(
        query: Annotated[Optional[str], Field(
            description="Free text matched against name, email, title, company name and domain. "
                        "Every word must match something.")] = None,
        company_id: Annotated[Optional[str], Field(description="Only leads at this company.")] = None,
        pipeline_stage: Annotated[Optional[str], Field(
            description="Company pipeline stage: new_lead, researching, outreach, engaged, "
                        "negotiating, client, churned.")] = None,
        has_email: Annotated[Optional[bool], Field(
            description="true = only leads with an email address; false = only without.")] = None,
        limit: Annotated[int, Field(ge=1, le=tools.MAX_PAGE)] = 25,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> dict:
        """Search and filter leads (contacts) in your organization, newest first."""
        return await call(tools.search_leads, query=query, company_id=company_id,
                          pipeline_stage=pipeline_stage, has_email=has_email,
                          limit=limit, offset=offset)

    @mcp.tool(annotations=READ)
    async def get_lead_details(
        lead_id: Annotated[str, Field(description="The lead's id, from search_leads.")],
    ) -> dict:
        """Everything about one lead: contact fields, company, the campaigns and
        sequences they're in, when they were last emailed, and whether they may be
        emailed at all (suppression and unsubscribe status)."""
        return await call(tools.get_lead_details, lead_id=lead_id)

    # -- campaigns -----------------------------------------------------------

    @mcp.tool(annotations=READ)
    async def list_discovery_documents(
        status: Annotated[Optional[str], Field(
            description="parsing, ready or failed. Only 'ready' documents can build a campaign.")] = None,
    ) -> dict:
        """List uploaded discovery documents, which are the input to create_campaign."""
        return await call(tools.list_discovery_documents, status=status)

    @mcp.tool(annotations=READ)
    async def list_campaigns(
        status: Annotated[Optional[str], Field(description="active, paused or stopped.")] = None,
    ) -> dict:
        """List email campaigns with their status, sender and send settings."""
        return await call(tools.list_campaigns, status=status)

    @mcp.tool(annotations=READ)
    async def get_campaign_stats(
        campaign_id: Annotated[str, Field(description="The campaign id, from list_campaigns.")],
    ) -> dict:
        """Campaign performance: prospects by status, sent, opens, clicks, replies,
        bounces and their rates, today's send cap, and the last 14 days."""
        return await call(tools.get_campaign_stats, campaign_id=campaign_id)

    @mcp.tool(annotations=WRITE)
    async def create_campaign(
        name: Annotated[str, Field(min_length=1, max_length=255)],
        document_ids: Annotated[list[str], Field(
            min_length=1, max_length=20,
            description="Ready discovery document ids (list_discovery_documents). "
                        "The AI reads them into the ideal-customer profile.")],
        sender_account_id: Annotated[Optional[str], Field(
            description="Mailbox id to send from (search_mailboxes). Default: the key "
                        "owner's own connected mailbox.")] = None,
        daily_send_limit: Annotated[Optional[int], Field(ge=1)] = None,
        review_before_send: Annotated[Optional[bool], Field(
            description="Hold drafts for human review before sending. Defaults to true.")] = None,
        send_window_start: Annotated[Optional[str], Field(description="HH:MM, e.g. 09:00")] = None,
        send_window_end: Annotated[Optional[str], Field(description="HH:MM, e.g. 18:00")] = None,
        timezone: Annotated[Optional[str], Field(description="IANA zone, e.g. America/Toronto")] = None,
        send_on_weekends: Optional[bool] = None,
        template_id: Annotated[Optional[str], Field(
            description="Email template to draft from. Default: the org's default template.")] = None,
    ) -> dict:
        """Create a new Autopilot email campaign from discovery documents. It goes
        live right away and starts finding prospects. Unless review is turned off,
        nothing is sent until a human approves the drafts in the app."""
        return await call(
            tools.create_campaign, name=name, document_ids=document_ids,
            sender_account_id=sender_account_id, daily_send_limit=daily_send_limit,
            review_before_send=review_before_send, send_window_start=send_window_start,
            send_window_end=send_window_end, timezone=timezone,
            send_on_weekends=send_on_weekends, template_id=template_id,
        )

    @mcp.tool(annotations=WRITE)
    async def add_leads_to_campaign(
        campaign_id: Annotated[str, Field(description="The campaign id.")],
        lead_ids: Annotated[list[str], Field(
            min_length=1, max_length=500, description="Lead ids from search_leads.")],
    ) -> dict:
        """Add leads to an existing campaign. Addresses are verified before
        anything is drafted. Leads who are unsubscribed, suppressed, already
        mailed, or have no usable email are skipped, and the result says why."""
        return await call(tools.add_leads_to_campaign, campaign_id=campaign_id, lead_ids=lead_ids)

    # -- mailboxes & compliance ---------------------------------------------

    @mcp.tool(annotations=READ)
    async def search_mailboxes(
        query: Annotated[Optional[str], Field(description="Match on address or display name.")] = None,
        status: Annotated[Optional[str], Field(description="e.g. connected, revoked, error")] = None,
        sending_enabled: Optional[bool] = None,
    ) -> dict:
        """List connected Gmail and SMTP mailboxes: status, whether sending is on,
        today's send count and owner."""
        return await call(tools.search_mailboxes, query=query, status=status,
                          sending_enabled=sending_enabled)

    @mcp.tool(annotations=READ)
    async def get_unsubscribes(
        reason: Annotated[str, Field(
            description="'unsubscribe' (default), 'bounce', 'complaint', 'manual', "
                        "'erasure', or 'all' for the whole do-not-contact list.")] = "unsubscribe",
        query: Annotated[Optional[str], Field(description="Substring of the email address.")] = None,
        since: Annotated[Optional[str], Field(description="ISO date: only entries on or after it.")] = None,
        limit: Annotated[int, Field(ge=1, le=tools.MAX_PAGE)] = 100,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> dict:
        """List email addresses that unsubscribed, or are otherwise on the
        do-not-contact list, newest first."""
        return await call(tools.get_unsubscribes, reason=reason, query=query, since=since,
                          limit=limit, offset=offset)

    return mcp


TOOL_NAMES = (
    "search_leads",
    "get_lead_details",
    "list_discovery_documents",
    "list_campaigns",
    "get_campaign_stats",
    "create_campaign",
    "add_leads_to_campaign",
    "search_mailboxes",
    "get_unsubscribes",
)
