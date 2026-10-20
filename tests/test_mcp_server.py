"""The Prospect MCP server (``app.mcp_server``).

Every tool is driven through a real MCP client session (in-memory transport),
so these tests exercise registration, argument schemas, the per-call
session/auth wrapper and error reporting, not just the tool bodies.

Each tool is checked for:
- the happy path;
- isolation: a key from another org sees none of this one's rows;
- the scope it needs.
Write tools also refuse when there is no acting user.
"""

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

# `mcp` needs pydantic>=2.11 and the API image pins 2.10.3, so it lives in
# requirements-mcp.txt rather than requirements.txt. Run this file from a venv
# built from that file; a prod-pinned venv reports it as skipped, not passed.
pytest.importorskip("mcp", reason="install requirements-mcp.txt to test the MCP server")

from cryptography.fernet import Fernet  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.mcp_server.server import TOOL_NAMES, build_server
from app.models.api_key import ALL_SCOPES, ApiKey
from app.models.campaign import Campaign, CampaignProspect
from app.models.company import Company
from app.models.contact import Contact
from app.models.discovery_document import STATUS_READY, DiscoveryDocument
from app.models.email_template import EmailSend
from app.models.gmail_account import GmailAccount
from app.models.org_llm_key import OrgLlmKey
from app.models.smtp_account import SmtpAccount
from app.models.suppression import SuppressionEntry
from app.models.tenant import Organization
from app.models.user import UserRole
from app.org_context import APPREND_ORG_ID, get_current_org_id
from app.services import crypto
from app.services.api_key_auth import generate_api_key

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _crypto_key():
    settings = get_settings()
    prev = settings.token_encryption_key
    settings.token_encryption_key = Fernet.generate_key().decode()
    crypto._fernet.cache_clear()
    yield
    settings.token_encryption_key = prev
    crypto._fernet.cache_clear()


@pytest.fixture(autouse=True)
def _mx_valid(monkeypatch):
    from app.services import email_validation

    monkeypatch.setattr(email_validation, "check_mx", lambda domain: "valid")


@pytest.fixture
def factory(db_engine):
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def owner(make_user):
    return await make_user("keyowner@landline.app", role=UserRole.ORG_ADMIN.value)


@pytest_asyncio.fixture
async def other_org(db_session):
    org = Organization(id=uuid.uuid4(), name="Rival Co", slug=f"rival-{uuid.uuid4().hex[:6]}",
                       is_active=True, billing_plan="pro")
    db_session.add(org)
    await db_session.commit()
    return org


async def _mint(db, *, org_id=APPREND_ORG_ID, scopes=ALL_SCOPES, created_by=None, revoked=False):
    raw, key_hash, prefix = generate_api_key()
    db.add(ApiKey(
        org_id=org_id, name="mcp", key_hash=key_hash, key_prefix=prefix,
        scopes=list(scopes), rate_limit=100000, created_by=created_by,
        revoked_at=datetime.now(timezone.utc) if revoked else None,
    ))
    await db.commit()
    return raw


async def _call(factory, key, tool, args=None):
    """Call ``tool`` over a real MCP session. Returns ``(is_error, payload)``."""
    server = build_server(session_factory=factory, api_key=key)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool(tool, args or {})
    text = result.content[0].text if result.content else ""
    if result.isError:
        return True, text
    return False, json.loads(text)


async def _company(db, name, *, org_id=APPREND_ORG_ID, stage="new_lead", domain=None):
    co = Company(name=name, domain=domain or f"{uuid.uuid4().hex[:8]}.com", source="test",
                 pipeline_stage=stage, org_id=org_id)
    db.add(co)
    await db.flush()
    return co


async def _contact(db, company, first, email, *, org_id=APPREND_ORG_ID, **kw):
    ct = Contact(company_id=company.id, first_name=first, last_name=kw.pop("last", "Tester"),
                 email=email, source="test", org_id=org_id, **kw)
    db.add(ct)
    await db.flush()
    return ct


def _gmail(**kw):
    defaults = dict(
        email=f"sender-{uuid.uuid4().hex[:6]}@example.com", google_sub=uuid.uuid4().hex,
        refresh_token_encrypted="SECRET-REFRESH", access_token_encrypted="SECRET-ACCESS",
        status="connected", sending_enabled=True, is_active=True,
    )
    defaults.update(kw)
    return GmailAccount(**defaults)


async def _campaign(db, *, org_id=APPREND_ORG_ID, user_id=None, name="Q4 push"):
    acct = _gmail(org_id=org_id, user_id=user_id)
    db.add(acct)
    await db.flush()
    c = Campaign(
        name=name, status="active", document_ids=[], icp_json={"summary": "widgets"},
        gmail_account_id=acct.id, daily_send_limit=20, send_window_start="09:00",
        send_window_end="18:00", timezone="America/Toronto", send_on_weekends=False,
        review_before_send=True, attach_documents=False, warmup_day=1,
        org_id=org_id, user_id=user_id,
    )
    db.add(c)
    await db.commit()
    return c


# --- registration & plumbing -------------------------------------------------


async def test_server_registers_exactly_the_documented_tools(factory):
    server = build_server(session_factory=factory, api_key="unused")
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        listed = (await client.list_tools()).tools
    assert sorted(t.name for t in listed) == sorted(TOOL_NAMES)
    by_name = {t.name: t for t in listed}
    for read_only in ("search_leads", "get_lead_details", "list_campaigns", "get_campaign_stats",
                      "search_mailboxes", "get_unsubscribes", "list_discovery_documents"):
        assert by_name[read_only].annotations.readOnlyHint is True, read_only
    for write in ("create_campaign", "add_leads_to_campaign"):
        assert by_name[write].annotations.readOnlyHint is False, write
    # Required arguments are advertised as such, so the model knows to ask.
    assert by_name["get_lead_details"].inputSchema["required"] == ["lead_id"]


async def test_a_missing_or_revoked_key_is_refused(factory, db_session):
    err, msg = await _call(factory, "", "search_leads")
    assert err and "PROSPECT_API_KEY" in msg

    err, msg = await _call(factory, "llk_not-a-real-key", "search_leads")
    assert err and "Invalid API key" in msg

    revoked = await _mint(db_session, revoked=True)
    err, msg = await _call(factory, revoked, "search_leads")
    assert err and "revoked" in msg


async def test_a_suspended_org_key_is_refused(factory, db_session, other_org):
    key = await _mint(db_session, org_id=other_org.id)
    other_org.is_active = False
    await db_session.commit()
    err, msg = await _call(factory, key, "list_campaigns")
    assert err and "deactivated" in msg


async def test_the_org_binding_does_not_outlive_the_call(factory, db_session):
    """``get_api_key_context`` binds the key's org and leaves it bound; in a
    long-lived process that would scope (or mis-scope) whatever runs next."""
    key = await _mint(db_session)
    server = build_server(session_factory=factory, api_key=key)
    assert get_current_org_id() is None
    await server.call_tool("search_leads", {})
    assert get_current_org_id() is None


# --- search_leads --------------------------------------------------------------


async def test_search_leads_filters_and_stays_in_its_org(factory, db_session, other_org):
    acme = await _company(db_session, "Acme Stamping", stage="engaged")
    globex = await _company(db_session, "Globex")
    await _contact(db_session, acme, "Ada", "ada@acme.com", last="Lovelace", title="VP Ops")
    await _contact(db_session, globex, "Grace", "grace@globex.com", last="Hopper")
    await _contact(db_session, globex, "Nomail", None)
    rival = await _company(db_session, "Acme Rival", org_id=other_org.id)
    await _contact(db_session, rival, "Ada", "ada@rival.com", org_id=other_org.id)
    await db_session.commit()
    key = await _mint(db_session)

    err, out = await _call(factory, key, "search_leads", {"query": "ada acme"})
    assert not err, out
    assert [lead["email"] for lead in out["leads"]] == ["ada@acme.com"]
    assert out["leads"][0]["company"]["name"] == "Acme Stamping"

    err, out = await _call(factory, key, "search_leads", {"pipeline_stage": "engaged"})
    assert [lead["first_name"] for lead in out["leads"]] == ["Ada"]

    err, out = await _call(factory, key, "search_leads", {"has_email": False})
    assert [lead["first_name"] for lead in out["leads"]] == ["Nomail"]

    err, out = await _call(factory, key, "search_leads", {"limit": 2})
    assert out["total"] == 3 and len(out["leads"]) == 2

    rival_key = await _mint(db_session, org_id=other_org.id)
    err, out = await _call(factory, rival_key, "search_leads", {"query": "ada"})
    assert [lead["email"] for lead in out["leads"]] == ["ada@rival.com"]


async def test_search_leads_rejects_an_unknown_pipeline_stage(factory, db_session):
    key = await _mint(db_session)
    err, msg = await _call(factory, key, "search_leads", {"pipeline_stage": "won"})
    assert err and "pipeline_stage" in msg


async def test_lead_tools_need_contacts_read(factory, db_session):
    key = await _mint(db_session, scopes=["campaigns:read"])
    err, msg = await _call(factory, key, "search_leads")
    assert err and "contacts:read" in msg


# --- get_lead_details ------------------------------------------------------------


async def test_get_lead_details_reports_campaigns_last_send_and_do_not_contact(factory, db_session):
    co = await _company(db_session, "Acme")
    ada = await _contact(db_session, co, "Ada", "Ada@Acme.com")
    sent_at = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)
    db_session.add(EmailSend(contact_id=ada.id, to_email=ada.email, subject="Hi",
                             status="sent", sent_at=sent_at))
    # On the org list but with clean flags: a lead that looks mailable unless
    # the list is read too.
    db_session.add(SuppressionEntry(email="ada@acme.com", reason="unsubscribe", source="link"))
    c = await _campaign(db_session)
    db_session.add(CampaignProspect(campaign_id=c.id, contact_id=ada.id, to_email="ada@acme.com",
                                    status="sent", sent_at=sent_at))
    await db_session.commit()
    key = await _mint(db_session)

    err, out = await _call(factory, key, "get_lead_details", {"lead_id": str(ada.id)})
    assert not err, out
    assert out["company"]["name"] == "Acme"
    assert out["last_emailed_at"].startswith("2026-10-01")
    assert out["mailable"] is False and out["do_not_contact_reason"]
    assert [(m["campaign_name"], m["status"]) for m in out["campaigns"]] == [("Q4 push", "sent")]


async def test_get_lead_details_cannot_read_another_orgs_lead(factory, db_session, other_org):
    co = await _company(db_session, "Rival", org_id=other_org.id)
    theirs = await _contact(db_session, co, "Eve", "eve@rival.com", org_id=other_org.id)
    await db_session.commit()
    key = await _mint(db_session)

    err, msg = await _call(factory, key, "get_lead_details", {"lead_id": str(theirs.id)})
    assert err and "not found" in msg
    err, msg = await _call(factory, key, "get_lead_details", {"lead_id": "nope"})
    assert err and "valid lead id" in msg


# --- list_campaigns / get_campaign_stats ---------------------------------------------


async def test_list_campaigns_and_status_filter(factory, db_session, other_org):
    await _campaign(db_session, name="Live")
    paused = await _campaign(db_session, name="Held")
    paused.status = "paused"
    await _campaign(db_session, org_id=other_org.id, name="Theirs")
    await db_session.commit()
    key = await _mint(db_session)

    err, out = await _call(factory, key, "list_campaigns")
    assert not err, out
    assert sorted(c["name"] for c in out["campaigns"]) == ["Held", "Live"]
    assert "icp_json" not in out["campaigns"][0]

    err, out = await _call(factory, key, "list_campaigns", {"status": "paused"})
    assert [c["name"] for c in out["campaigns"]] == ["Held"]

    no_scope = await _mint(db_session, scopes=["contacts:read"])
    err, msg = await _call(factory, no_scope, "list_campaigns")
    assert err and "campaigns:read" in msg


async def test_campaign_stats_count_opens_clicks_replies_and_bounces(factory, db_session, other_org):
    c = await _campaign(db_session)
    co = await _company(db_session, "Acme")
    now = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc)
    rows = [
        # (status, opened, clicked)
        ("sent", True, False),
        ("replied", True, True),
        ("bounced", False, False),
        ("queued", None, None),
    ]
    for i, (status, opened, clicked) in enumerate(rows):
        ct = await _contact(db_session, co, f"P{i}", f"p{i}@acme.com")
        send_id = None
        if opened is not None:
            send = EmailSend(contact_id=ct.id, to_email=ct.email, status="sent", sent_at=now,
                             opened_at=now if opened else None, clicked_at=now if clicked else None)
            db_session.add(send)
            await db_session.flush()
            send_id = send.id
        db_session.add(CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email=ct.email,
                                        status=status, email_send_id=send_id))
    await db_session.commit()
    key = await _mint(db_session)

    err, out = await _call(factory, key, "get_campaign_stats", {"campaign_id": str(c.id)})
    assert not err, out
    stats = out["stats"]
    assert stats["emails_sent"] == 3
    assert (stats["opens"], stats["clicks"], stats["replies"], stats["bounces"]) == (2, 1, 1, 1)
    assert stats["open_rate"] == round(2 / 3, 4)
    assert stats["by_status"]["queued"] == 1

    rival_key = await _mint(db_session, org_id=other_org.id)
    err, msg = await _call(factory, rival_key, "get_campaign_stats", {"campaign_id": str(c.id)})
    assert err and "not found" in msg


# --- create_campaign ------------------------------------------------------------------


@pytest_asyncio.fixture
async def campaign_inputs(db_session, owner, monkeypatch):
    """A ready document, an AI key, the owner's mailbox; ICP and Celery stubbed."""
    from app.services import campaigns as svc
    from app.tasks import campaigns as campaign_tasks

    acct = _gmail(user_id=owner.id)
    doc = DiscoveryDocument(filename="pitch.txt", file_type="txt", size_bytes=10,
                            status=STATUS_READY, content=b"We sell widgets",
                            extracted={"candidates": []})
    db_session.add_all([acct, doc, OrgLlmKey(
        org_id=APPREND_ORG_ID, provider="gemini", key_encrypted=crypto.encrypt("k"), key_hint="••••",
    )])
    await db_session.commit()

    async def fake_icp(db, org_id, docs):
        return svc.Icp(summary="Widgets for factories", target_industries=["Manufacturing"]), "Widgets"

    dispatched = []
    monkeypatch.setattr(svc, "build_icp", fake_icp)
    monkeypatch.setattr(campaign_tasks.run_campaign_now, "delay", lambda *a: dispatched.append(a))
    return {"doc": doc, "account": acct, "dispatched": dispatched}


async def test_create_campaign_goes_through_the_campaign_handler(factory, db_session, owner, campaign_inputs):
    key = await _mint(db_session, created_by=owner.id)

    err, docs = await _call(factory, key, "list_discovery_documents", {"status": "ready"})
    assert not err and [d["filename"] for d in docs["documents"]] == ["pitch.txt"]

    err, out = await _call(factory, key, "create_campaign", {
        "name": "Q4 push", "document_ids": [str(campaign_inputs["doc"].id)], "daily_send_limit": 40,
    })
    assert not err, out
    assert out["user_id"] == str(owner.id)
    assert out["gmail_account_id"] == str(campaign_inputs["account"].id)
    assert out["icp_json"]["target_industries"] == ["Manufacturing"]
    # The handler's safe default: drafts wait for a human.
    assert out["review_before_send"] is True and "review" in out["next_steps"]
    assert campaign_inputs["dispatched"] == [(str(APPREND_ORG_ID), out["id"])]

    err, listed = await _call(factory, key, "list_campaigns")
    assert [c["id"] for c in listed["campaigns"]] == [out["id"]]


async def test_create_campaign_reports_the_handlers_refusals(factory, db_session, owner, campaign_inputs):
    key = await _mint(db_session, created_by=owner.id)
    err, msg = await _call(factory, key, "create_campaign", {
        "name": "x", "document_ids": [str(uuid.uuid4())],
    })
    assert err and "Documents not found" in msg

    err, msg = await _call(factory, key, "create_campaign", {
        "name": "x", "document_ids": [str(campaign_inputs["doc"].id)], "timezone": "Mars/Base",
    })
    assert err and "timezone" in msg.lower()


async def test_write_tools_need_an_acting_user_and_the_write_scope(factory, db_session, owner, campaign_inputs):
    args = {"name": "x", "document_ids": [str(campaign_inputs["doc"].id)]}

    orphan = await _mint(db_session, created_by=None)
    err, msg = await _call(factory, orphan, "create_campaign", args)
    assert err and "no owning user" in msg

    read_only = await _mint(db_session, created_by=owner.id, scopes=["campaigns:read"])
    err, msg = await _call(factory, read_only, "create_campaign", args)
    assert err and "campaigns:write" in msg

    owner.is_active = False
    await db_session.commit()
    inactive = await _mint(db_session, created_by=owner.id)
    err, msg = await _call(factory, inactive, "create_campaign", args)
    assert err and "no longer an active member" in msg
    assert campaign_inputs["dispatched"] == []


# --- add_leads_to_campaign --------------------------------------------------------------


async def test_add_leads_queues_mailable_leads_and_says_why_others_were_skipped(
    factory, db_session, owner, monkeypatch, other_org,
):
    from app.tasks import campaigns as campaign_tasks

    processing = []
    monkeypatch.setattr(campaign_tasks.process_new_prospects, "delay", lambda *a: processing.append(a))
    c = await _campaign(db_session, user_id=owner.id)
    co = await _company(db_session, "Lead Co", domain="lead.com")
    fresh = await _contact(db_session, co, "Fresh", "fresh@lead.com")
    gone = await _contact(db_session, co, "Gone", "gone@lead.com")
    gone.suppressed = True
    await db_session.commit()
    key = await _mint(db_session, created_by=owner.id)

    err, out = await _call(factory, key, "add_leads_to_campaign", {
        "campaign_id": str(c.id), "lead_ids": [str(fresh.id), str(gone.id)],
    })
    assert not err, out
    assert out["queued"] == 1 and out["processing"] == "dispatched"
    assert {s["email"]: s["reason"] for s in out["skipped"]} == {"gone@lead.com": "blocked"}
    assert processing == [(str(c.org_id), str(c.id))]

    # Another org's campaign id is not found; its lead ids aren't ours to add.
    rival_c = await _campaign(db_session, org_id=other_org.id)
    err, msg = await _call(factory, key, "add_leads_to_campaign", {
        "campaign_id": str(rival_c.id), "lead_ids": [str(fresh.id)],
    })
    assert err and "not found" in msg.lower()


# --- search_mailboxes -------------------------------------------------------------------


async def test_search_mailboxes_lists_both_kinds_without_secrets(factory, db_session, owner, other_org):
    db_session.add(_gmail(email="sales@ourco.com", display_name="Sales", user_id=owner.id))
    db_session.add(SmtpAccount(email="ops@ourco.com", host="smtp.ourco.com", username="ops",
                               password_encrypted="SECRET-PASSWORD", status="error",
                               sending_enabled=False, last_error="auth failed"))
    db_session.add(_gmail(email="them@rival.com", org_id=other_org.id))
    await db_session.commit()
    key = await _mint(db_session)

    err, out = await _call(factory, key, "search_mailboxes")
    assert not err, out
    by_email = {m["email"]: m for m in out["mailboxes"]}
    assert set(by_email) == {"sales@ourco.com", "ops@ourco.com"}
    assert by_email["sales@ourco.com"]["owner_email"] == "keyowner@landline.app"
    assert by_email["ops@ourco.com"]["kind"] == "smtp"
    assert by_email["ops@ourco.com"]["last_error"] == "auth failed"
    assert "SECRET" not in json.dumps(out)

    err, out = await _call(factory, key, "search_mailboxes", {"sending_enabled": True})
    assert [m["email"] for m in out["mailboxes"]] == ["sales@ourco.com"]

    no_scope = await _mint(db_session, scopes=["contacts:read"])
    err, msg = await _call(factory, no_scope, "search_mailboxes")
    assert err and "mailboxes:read" in msg


# --- get_unsubscribes -------------------------------------------------------------------


async def test_get_unsubscribes_reads_the_list_and_legacy_flags(factory, db_session, other_org):
    db_session.add_all([
        SuppressionEntry(email="listed@a.com", reason="unsubscribe", source="link"),
        SuppressionEntry(email="bounced@a.com", reason="bounce", source="dsn"),
        SuppressionEntry(email="theirs@a.com", reason="unsubscribe", source="link",
                         org_id=other_org.id),
    ])
    co = await _company(db_session, "Acme")
    # Opted out through the old link handler: the flag, no list entry — and
    # duplicated, because contacts.email is not unique.
    await _contact(db_session, co, "Old", "Legacy@a.com", unsubscribed=True)
    await _contact(db_session, co, "Old2", "legacy@a.com", unsubscribed=True)
    # Flagged *and* listed: listed once, not twice.
    await _contact(db_session, co, "Both", "listed@a.com", unsubscribed=True)
    rival_co = await _company(db_session, "Rival", org_id=other_org.id)
    await _contact(db_session, rival_co, "R", "rival-legacy@a.com", org_id=other_org.id,
                   unsubscribed=True)
    await db_session.commit()
    key = await _mint(db_session)

    err, out = await _call(factory, key, "get_unsubscribes")
    assert not err, out
    assert sorted(e["email"] for e in out["entries"]) == ["legacy@a.com", "listed@a.com"]
    assert out["total"] == 2
    sources = {e["email"]: e["source"] for e in out["entries"]}
    assert sources == {"legacy@a.com": "contact_flag", "listed@a.com": "link"}

    err, out = await _call(factory, key, "get_unsubscribes", {"reason": "all"})
    assert sorted(e["email"] for e in out["entries"]) == ["bounced@a.com", "legacy@a.com", "listed@a.com"]

    err, out = await _call(factory, key, "get_unsubscribes", {"reason": "bounce"})
    assert [e["email"] for e in out["entries"]] == ["bounced@a.com"]

    err, out = await _call(factory, key, "get_unsubscribes", {"query": "legacy"})
    assert [e["email"] for e in out["entries"]] == ["legacy@a.com"]

    err, msg = await _call(factory, key, "get_unsubscribes", {"reason": "spam"})
    assert err and "reason must be" in msg

    no_scope = await _mint(db_session, scopes=["contacts:read"])
    err, msg = await _call(factory, no_scope, "get_unsubscribes")
    assert err and "suppression:read" in msg


# --- the stdio entry point ------------------------------------------------------------


def test_stray_stdout_cannot_reach_the_protocol_stream():
    """On stdio, fd 1 is the protocol. A print — or SQLAlchemy echo — after
    isolation must land on stderr; only the protocol stream writes to fd 1."""
    script = (
        "import os\n"
        "from app.mcp_server.__main__ import isolate_stdout\n"
        "proto = isolate_stdout()\n"
        "print('stray python print')\n"
        "os.write(1, b'stray fd write\\n')\n"
        "proto.write('PROTOCOL\\n'); proto.flush()\n"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "PROTOCOL\n"
    assert "stray python print" in done.stderr and "stray fd write" in done.stderr


def test_sql_echo_is_turned_off():
    from app.database import engine
    from app.mcp_server.__main__ import silence_sql_echo

    prev = engine.sync_engine.echo
    engine.sync_engine.echo = True
    try:
        silence_sql_echo()
        assert engine.sync_engine.echo is False
    finally:
        engine.sync_engine.echo = prev


async def test_the_real_entry_point_speaks_mcp_over_stdio():
    """Launch ``python -m app.mcp_server`` the way Claude Desktop does and
    complete a handshake. Any byte on stdout that isn't protocol breaks this."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {k: v for k, v in os.environ.items() if k != "PROSPECT_API_KEY"}
    # Echo on: the engine would log SQL to stdout if the entry point let it.
    env["APP_ENV"] = "development"
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"],
                                   cwd=REPO_ROOT, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert names == set(TOOL_NAMES)
            result = await session.call_tool("search_leads", {})
    assert result.isError and "PROSPECT_API_KEY" in result.content[0].text
