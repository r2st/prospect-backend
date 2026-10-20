"""The unsubscribe page: look first, act on a button, and undo it.

Three properties:

* **A GET changes nothing.** Mail security scanners (Safe Links, Mimecast,
  Proofpoint) fetch every link in a message before a human sees it, so an
  opt-out on GET unsubscribes people who never clicked.
* **Status is shown only for a genuine token**, and re-subscribing — which
  re-opens mail to a person — is a 403 on anything else.
* **Re-subscribe lifts only this link's opt-out.** A bounce, a complaint, an
  operator's manual entry or an erasure on the list is not the recipient's to
  lift from a footer link.

The RFC 8058 one-click POST (no ``source=page`` form field) is untouched and
covered in ``test_unsubscribe_oneclick.py`` / ``test_opt_out.py``.
"""

import uuid

import pytest
from sqlalchemy import select

from app.models.company import Company
from app.models.consent import ConsentRecord
from app.models.contact import Contact
from app.models.outreach import OutreachEnrollment
from app.models.suppression import SuppressionEntry
from app.org_context import APPREND_ORG_ID
from app.services import opt_out, suppression

UNSUB = "/api/v1/track/unsubscribe"
RESUB = "/api/v1/track/resubscribe"


def _token(email: str, org_id=APPREND_ORG_ID) -> str:
    return opt_out.make_unsubscribe_token(email, org_id)


def _legacy_token(email: str) -> str:
    return opt_out.unsubscribe_serializer().dumps(email, salt="unsubscribe")


async def _make_contact(db_session, email, *, org_id=None):
    kw = {"org_id": org_id} if org_id is not None else {}
    company = Company(id=uuid.uuid4(), name="Acme", country="US", source="test", **kw)
    db_session.add(company)
    await db_session.flush()
    contact = Contact(
        id=uuid.uuid4(), company_id=company.id, email=email,
        first_name="Test", last_name="Lead", **kw,
    )
    db_session.add(contact)
    await db_session.flush()
    enrollment = OutreachEnrollment(
        id=uuid.uuid4(), contact_id=contact.id, sequence_id=uuid.uuid4(),
        status="active", current_step=2, **kw,
    )
    db_session.add(enrollment)
    await db_session.commit()
    return contact, enrollment


async def _entries(db_session, email):
    return (
        await db_session.execute(
            select(SuppressionEntry)
            .where(SuppressionEntry.email == email)
            .execution_options(skip_org_filter=True)
        )
    ).scalars().all()


def _unsubscribe_from_page(client, token):
    return client.post(f"{UNSUB}?token={token}", data={"source": "page"})


# --------------------------------------------------------------------------
# GET is a look, not an action.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_shows_the_address_and_a_button_and_changes_nothing(client, db_session):
    contact, enrollment = await _make_contact(db_session, "look@example.com")

    resp = await client.get(f"{UNSUB}?token={_token('look@example.com')}")
    assert resp.status_code == 200
    assert "look@example.com" in resp.text
    assert ">Unsubscribe</button>" in resp.text
    assert "Re-subscribe" not in resp.text

    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is False
    assert contact.unsubscribed is False
    assert enrollment.status == "active"
    assert await _entries(db_session, "look@example.com") == []


@pytest.mark.asyncio
async def test_repeated_gets_as_a_scanner_would_still_change_nothing(client, db_session):
    contact, _ = await _make_contact(db_session, "scanned@example.com")
    url = f"{UNSUB}?token={_token('scanned@example.com')}"
    for _ in range(3):
        assert (await client.get(url)).status_code == 200
    await db_session.refresh(contact)
    assert contact.suppressed is False


@pytest.mark.asyncio
async def test_get_with_a_tampered_token_reveals_nothing(client, db_session):
    await _make_contact(db_session, "secret@example.com")
    good = _token("secret@example.com")
    tampered = good[:-3] + ("aaa" if not good.endswith("aaa") else "bbb")

    for token in (tampered, "not-a-real-token"):
        resp = await client.get(f"{UNSUB}?token={token}")
        assert resp.status_code == 400
        assert "didn't work" in resp.text
        assert "secret@example.com" not in resp.text
        assert "<button" not in resp.text


@pytest.mark.asyncio
async def test_get_shows_unsubscribed_state_for_an_already_opted_out_address(client, db_session):
    await _make_contact(db_session, "already@example.com")
    await opt_out.record(db_session, "already@example.com", org_id=APPREND_ORG_ID)

    resp = await client.get(f"{UNSUB}?token={_token('already@example.com')}")
    assert resp.status_code == 200
    assert "You're unsubscribed" in resp.text
    assert "Re-subscribe" in resp.text


# --------------------------------------------------------------------------
# The full round trip: unsubscribe → re-subscribe → unsubscribe button again.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_trip(client, db_session):
    contact, enrollment = await _make_contact(db_session, "trip@example.com")
    token = _token("trip@example.com")

    resp = await _unsubscribe_from_page(client, token)
    assert resp.status_code == 200
    assert "You're unsubscribed" in resp.text
    assert f"resubscribe?token={token}" in resp.text
    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is True
    assert enrollment.status == "unsubscribed"
    assert await suppression.address_blocking_reason(db_session, "trip@example.com")

    resp = await client.post(f"{RESUB}?token={token}")
    assert resp.status_code == 200
    assert "subscribed again" in resp.text
    assert ">Unsubscribe</button>" in resp.text
    assert f"unsubscribe?token={token}" in resp.text
    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is False
    assert contact.unsubscribed is False
    assert await _entries(db_session, "trip@example.com") == []
    assert await suppression.address_blocking_reason(db_session, "trip@example.com") is None
    # Re-subscribing permits mail; it does not restart a stopped cadence.
    assert enrollment.status == "unsubscribed"

    granted = (
        await db_session.execute(
            select(ConsentRecord).where(
                ConsentRecord.contact_id == contact.id,
                ConsentRecord.consent_type == "express",
            )
        )
    ).scalars().all()
    assert len(granted) == 1

    # And the page offers Unsubscribe again, which works again.
    resp = await client.get(f"{UNSUB}?token={token}")
    assert ">Unsubscribe</button>" in resp.text
    resp = await _unsubscribe_from_page(client, token)
    assert "You're unsubscribed" in resp.text
    await db_session.refresh(contact)
    assert contact.suppressed is True


@pytest.mark.asyncio
async def test_legacy_bare_address_token_round_trips(client, db_session):
    contact, _ = await _make_contact(db_session, "legacy@example.com")
    token = _legacy_token("legacy@example.com")

    assert "You're unsubscribed" in (await _unsubscribe_from_page(client, token)).text
    resp = await client.post(f"{RESUB}?token={token}")
    assert resp.status_code == 200
    await db_session.refresh(contact)
    assert contact.suppressed is False


# --------------------------------------------------------------------------
# Re-subscribe needs a genuine link, and lifts only its own opt-out.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resubscribe_with_a_tampered_token_is_403_and_changes_nothing(client, db_session):
    contact, _ = await _make_contact(db_session, "keepout@example.com")
    good = _token("keepout@example.com")
    await _unsubscribe_from_page(client, good)
    tampered = good[:-3] + ("aaa" if not good.endswith("aaa") else "bbb")

    for token in (tampered, "not-a-real-token"):
        resp = await client.post(f"{RESUB}?token={token}")
        assert resp.status_code == 403
        assert "keepout@example.com" not in resp.text

    await db_session.refresh(contact)
    assert contact.suppressed is True
    assert len(await _entries(db_session, "keepout@example.com")) == 1


@pytest.mark.asyncio
async def test_resubscribe_with_a_token_signed_for_another_purpose_is_403(client, db_session):
    await _make_contact(db_session, "salt@example.com")
    wrong_salt = opt_out.unsubscribe_serializer().dumps("salt@example.com", salt="other")
    resp = await client.post(f"{RESUB}?token={wrong_salt}")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_resubscribe_with_a_non_address_payload_is_403(client):
    token = opt_out.unsubscribe_serializer().dumps("not-an-email", salt="unsubscribe")
    assert (await client.post(f"{RESUB}?token={token}")).status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["bounce", "complaint", "manual", "erasure"])
async def test_resubscribe_does_not_lift_other_reasons(client, db_session, reason):
    contact, _ = await _make_contact(db_session, f"{reason}@example.com")
    email = f"{reason}@example.com"
    await opt_out.record(db_session, email, org_id=APPREND_ORG_ID)
    # (org, email) is unique, so put the other reason in place of the
    # unsubscribe entry, as a bounce handler writing first would have left it.
    entry = (await _entries(db_session, email))[0]
    entry.reason = reason
    await db_session.commit()

    token = _token(email)
    page = await client.get(f"{UNSUB}?token={token}")
    assert "You're unsubscribed" in page.text
    assert "Re-subscribe" not in page.text

    resp = await client.post(f"{RESUB}?token={token}")
    assert resp.status_code == 200
    assert "Re-subscribe" not in resp.text
    await db_session.refresh(contact)
    assert contact.suppressed is True
    assert [e.reason for e in await _entries(db_session, email)] == [reason]


@pytest.mark.asyncio
async def test_org_scoped_resubscribe_leaves_other_senders_alone(client, db_session):
    other_org = uuid.uuid4()
    mine, _ = await _make_contact(db_session, "two@example.com")
    theirs, _ = await _make_contact(db_session, "two@example.com", org_id=other_org)
    await opt_out.record(db_session, "two@example.com", org_id=APPREND_ORG_ID)
    await opt_out.record(db_session, "two@example.com", org_id=other_org)

    resp = await client.post(f"{RESUB}?token={_token('two@example.com', APPREND_ORG_ID)}")
    assert resp.status_code == 200

    await db_session.refresh(mine)
    await db_session.refresh(theirs)
    assert mine.suppressed is False
    assert theirs.suppressed is True
    assert [e.org_id for e in await _entries(db_session, "two@example.com")] == [other_org]


# --------------------------------------------------------------------------
# The page itself.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_page_is_noindex_themed_and_mobile_ready(client, db_session, valid):
    await _make_contact(db_session, "page@example.com")
    token = _token("page@example.com") if valid else "bogus"
    resp = await client.get(f"{UNSUB}?token={token}")

    assert "noindex" in resp.headers["x-robots-tag"]
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["referrer-policy"] == "no-referrer"
    html = resp.text
    assert "<meta name='robots' content='noindex" in html
    assert "name='viewport'" in html
    assert "prefers-color-scheme: dark" in html
    assert "<script" not in html


@pytest.mark.asyncio
async def test_resubscribe_page_is_noindex_too(client, db_session):
    resp = await client.post(f"{RESUB}?token=bogus")
    assert "noindex" in resp.headers["x-robots-tag"]


@pytest.mark.asyncio
async def test_address_is_html_escaped(client, db_session):
    email = "o'<b>x@example.com"
    token = _token(email)
    resp = await client.get(f"{UNSUB}?token={token}")
    # normalize_email may reject this; either way no raw markup is echoed.
    assert "<b>x@" not in resp.text


@pytest.mark.asyncio
async def test_one_click_post_without_the_page_field_still_returns_an_empty_200(client, db_session):
    contact, _ = await _make_contact(db_session, "provider@example.com")
    resp = await client.post(
        f"{UNSUB}?token={_token('provider@example.com')}",
        data={"List-Unsubscribe": "One-Click"},
    )
    assert resp.status_code == 200
    assert resp.text == ""
    await db_session.refresh(contact)
    assert contact.suppressed is True
