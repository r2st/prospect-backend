"""An opt-out has to stop the mail — everywhere, and for good.

The bug these cover: the unsubscribe link in the footer of every email, and the
RFC 8058 one-click header Gmail and Yahoo require, set ``contact.unsubscribed``
and nothing else. The send path gates on ``contact.suppressed`` and the org
suppression list, and has never read ``unsubscribed`` — so the only opt-out a
recipient can actually reach was the only one that let the sequence carry on.

Three properties, tested at the level each one lives at:

* the endpoint records an opt-out that the *send path* recognises;
* the record survives the contact row (the scraper makes new rows);
* the token the sender mints is one the endpoint accepts.

Mutation-tested — each of these fails at least one test above: not setting
``suppressed``; not stopping enrollments; not writing the suppression entry;
matching the address case-sensitively; taking only the first of several contact
rows; dropping the first-party org fallback; accepting a non-address; ignoring
``org_id``; not setting ``unsubscribed``; skipping the consent record; treating
``completed`` as actionable; dropping the org from a minted token; removing the
``unsubscribed`` check from either send gate; reporting success unconditionally;
ignoring a refusal in the GET handler; and dropping legacy bare-string tokens.
"""

import uuid

import pytest
from sqlalchemy import select

from app.models.company import Company
from app.models.consent import ConsentRecord
from app.models.contact import Contact
from app.models.outreach import OutreachEnrollment
from app.models.suppression import SuppressionEntry
from app.org_context import APPREND_ORG_ID, org_scope
from app.routers.tracking import _get_unsubscribe_serializer
from app.services import opt_out, suppression


def _legacy_token(email: str) -> str:
    """A token in the shape minted before opt-outs carried an org."""
    return _get_unsubscribe_serializer().dumps(email, salt="unsubscribe")


async def _make_contact(db_session, email, *, org_id=None, enrolled="active"):
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
    enrollment = None
    if enrolled:
        enrollment = OutreachEnrollment(
            id=uuid.uuid4(), contact_id=contact.id, sequence_id=uuid.uuid4(),
            status=enrolled, current_step=2, **kw,
        )
        db_session.add(enrollment)
    await db_session.commit()
    return contact, enrollment


# --------------------------------------------------------------------------
# The headline: clicking unsubscribe stops the sequence.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_click_sets_the_flag_the_send_path_reads(client, db_session):
    """``suppressed``, not just ``unsubscribed`` — the send gate reads the former."""
    contact, _ = await _make_contact(db_session, "victim@example.com")

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('victim@example.com')}"
    )
    assert resp.status_code == 200

    await db_session.refresh(contact)
    assert contact.suppressed is True
    assert contact.unsubscribed is True


@pytest.mark.asyncio
async def test_one_click_stops_the_live_sequence(client, db_session):
    """Steps 2-5 of a five-step sequence must not follow the opt-out."""
    _, enrollment = await _make_contact(db_session, "midflight@example.com")

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('midflight@example.com')}"
    )
    assert resp.status_code == 200

    await db_session.refresh(enrollment)
    # ``unsubscribed``, never ``bounced`` — a compliance stop is not a delivery
    # failure and must not move sender reputation metrics.
    assert enrollment.status == "unsubscribed"
    assert enrollment.completed_at is not None


@pytest.mark.asyncio
async def test_one_click_puts_the_address_on_the_suppression_list(client, db_session):
    """``is_suppressed`` is what the dispatcher asks; it must now say yes."""
    await _make_contact(db_session, "listed@example.com")

    await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('listed@example.com')}"
    )

    assert await suppression.is_suppressed(db_session, "listed@example.com") is True
    entry = (
        await db_session.execute(
            select(SuppressionEntry).where(SuppressionEntry.email == "listed@example.com")
        )
    ).scalar_one()
    assert entry.reason == "unsubscribe"


@pytest.mark.asyncio
async def test_one_click_writes_a_consent_record(client, db_session):
    """The audit trail the reply-based path already wrote."""
    contact, _ = await _make_contact(db_session, "audited@example.com")

    await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('audited@example.com')}"
    )

    record = (
        await db_session.execute(
            select(ConsentRecord).where(ConsentRecord.contact_id == contact.id)
        )
    ).scalar_one()
    assert record.consent_type == "none"
    assert record.revoked_at is not None
    assert record.granted_at is None


@pytest.mark.asyncio
async def test_get_path_does_the_same_thing_as_post(client, db_session):
    """A human pressing the page's button and a provider POSTing must not differ.

    (The GET itself only renders the page — see ``test_unsubscribe_page.py``.)
    """
    contact, enrollment = await _make_contact(db_session, "human@example.com")

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('human@example.com')}",
        data={"source": "page"},
    )
    assert resp.status_code == 200
    assert "unsubscribed" in resp.text.lower()

    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is True
    assert enrollment.status == "unsubscribed"


# --------------------------------------------------------------------------
# The cases that used to raise, lie, or silently miss.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_contacts_one_address_both_opt_out(client, db_session):
    """``contacts.email`` has no unique constraint; a re-scrape makes duplicates.

    This used to raise ``MultipleResultsFound`` out of ``scalar_one_or_none``,
    which the handler turned into a 400 — the recipient was told the link didn't
    work and *neither* row was unsubscribed.
    """
    first, first_enr = await _make_contact(db_session, "dupe@example.com")
    second, second_enr = await _make_contact(db_session, "dupe@example.com")

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('dupe@example.com')}"
    )
    assert resp.status_code == 200

    for contact in (first, second):
        await db_session.refresh(contact)
        assert contact.suppressed is True
    for enr in (first_enr, second_enr):
        await db_session.refresh(enr)
        assert enr.status == "unsubscribed"


@pytest.mark.asyncio
async def test_address_is_matched_case_insensitively(client, db_session):
    """Addresses are case-insensitive everywhere else; an opt-out is no exception."""
    contact, enrollment = await _make_contact(db_session, "Mixed.Case@Example.com")

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('mixed.case@example.com')}"
    )
    assert resp.status_code == 200

    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is True
    assert enrollment.status == "unsubscribed"


@pytest.mark.asyncio
async def test_opt_out_with_no_contact_row_is_still_recorded(client, db_session):
    """The confirmation page is a promise; it must not be made for nothing.

    No contact row used to mean: nothing written anywhere, and a cheerful
    "You're unsubscribed". The address now lands on the list regardless, so a
    contact created for it later is already suppressed.
    """
    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('ghost@example.com')}"
    )
    assert resp.status_code == 200
    assert await suppression.is_suppressed(db_session, "ghost@example.com") is True


@pytest.mark.asyncio
async def test_opt_out_survives_a_new_contact_row(db_session):
    """The scraper rediscovering someone must not re-open the door.

    The flags live on a row; the list is keyed by address. This is the whole
    reason both are written.
    """
    await opt_out.record(db_session, "rediscovered@example.com")

    # ...a month later, a different source finds them again.
    fresh, _ = await _make_contact(db_session, "rediscovered@example.com", enrolled=None)
    assert fresh.suppressed is False  # the row itself knows nothing
    assert await suppression.is_suppressed(db_session, "rediscovered@example.com") is True


@pytest.mark.asyncio
async def test_a_completed_enrollment_is_left_alone(client, db_session):
    """Only actionable states are interrupted; finished ones keep their history."""
    _, enrollment = await _make_contact(
        db_session, "done@example.com", enrolled="completed"
    )

    await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('done@example.com')}"
    )

    await db_session.refresh(enrollment)
    assert enrollment.status == "completed"


@pytest.mark.asyncio
async def test_a_paused_enrollment_is_stopped(client, db_session):
    """Paused is actionable — it resumes on its own, so leaving it is a leak."""
    _, enrollment = await _make_contact(
        db_session, "paused@example.com", enrolled="paused"
    )

    await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('paused@example.com')}"
    )

    await db_session.refresh(enrollment)
    assert enrollment.status == "unsubscribed"


@pytest.mark.asyncio
async def test_repeat_opt_out_is_idempotent(client, db_session):
    """Providers retry one-click POSTs; a second one must not 500 on the unique key."""
    await _make_contact(db_session, "twice@example.com")
    token = _legacy_token("twice@example.com")

    first = await client.post(f"/api/v1/track/unsubscribe?token={token}")
    second = await client.post(f"/api/v1/track/unsubscribe?token={token}")
    assert first.status_code == 200
    assert second.status_code == 200

    entries = (
        await db_session.execute(
            select(SuppressionEntry).where(SuppressionEntry.email == "twice@example.com")
        )
    ).scalars().all()
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_a_forged_token_changes_nothing(client, db_session):
    """The signature is the only thing standing between a stranger and the list."""
    contact, enrollment = await _make_contact(db_session, "safe@example.com")

    resp = await client.post("/api/v1/track/unsubscribe?token=not-a-real-token")
    assert resp.status_code == 400

    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is False
    assert enrollment.status == "active"
    assert await suppression.is_suppressed(db_session, "safe@example.com") is False


@pytest.mark.asyncio
async def test_a_token_carrying_a_non_address_is_refused(client, db_session):
    """A verified signature over junk is not an opt-out — don't claim it is."""
    token = _get_unsubscribe_serializer().dumps("not-an-email", salt="unsubscribe")

    resp = await client.post(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 400

    resp = await client.get(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 400
    assert "didn't work" in resp.text


# --------------------------------------------------------------------------
# Tokens: what the sender mints, the endpoint must accept.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_minted_token_round_trips_through_the_endpoint(client, db_session):
    """The sender and the verifier used to disagree about the signing key.

    ``_make_unsubscribe_url`` fell back to a hardcoded secret; the endpoint fell
    back to a random per-process key. With no ``secret_key`` configured that made
    every unsubscribe link we sent unverifiable. One factory now serves both.
    """
    from app.services.email_dispatch import _make_unsubscribe_url

    contact, enrollment = await _make_contact(db_session, "roundtrip@example.com")
    url = _make_unsubscribe_url("roundtrip@example.com")
    token = url.split("token=", 1)[1]

    resp = await client.post(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 200

    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is True
    assert enrollment.status == "unsubscribed"


@pytest.mark.asyncio
async def test_minted_token_carries_the_sending_org(db_session):
    """So the opt-out lands on the sender's list, not on everyone's."""
    from app.services.email_dispatch import _make_unsubscribe_url

    other_org = uuid.uuid4()
    async with org_scope(other_org):
        token = _make_unsubscribe_url("scoped@example.com").split("token=", 1)[1]

    payload = _get_unsubscribe_serializer().loads(token, salt="unsubscribe")
    assert payload["e"] == "scoped@example.com"
    assert payload["o"] == str(other_org)


@pytest.mark.asyncio
async def test_ses_footer_link_actually_unsubscribes(client, db_session):
    """The SES footer was a second copy of both halves, and both were wrong.

    It signed with the hardcoded ``"landline-unsubscribe-secret"`` fallback the
    endpoint would never accept, and pointed at
    ``landline.apprendtechnologies.com/unsubscribe`` — a frontend route that has
    never existed. A recipient who followed the only opt-out that email offered
    got a 404, behind which sat a token that would not have verified anyway.
    """
    from app.services.email_sender import EmailSender

    contact, enrollment = await _make_contact(db_session, "sesfooter@example.com")
    footer = EmailSender()._build_unsubscribe_footer("sesfooter@example.com")

    assert "landline.apprendtechnologies.com/unsubscribe" not in footer
    assert opt_out._unsubscribe_endpoint() in footer

    token = footer.split("token=", 1)[1].split()[0]
    resp = await client.post(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 200

    await db_session.refresh(contact)
    await db_session.refresh(enrollment)
    assert contact.suppressed is True
    assert enrollment.status == "unsubscribed"


@pytest.mark.asyncio
async def test_both_senders_mint_against_the_same_endpoint(db_session):
    """One URL builder, for the same reason there is one serializer."""
    from app.services.email_dispatch import _make_unsubscribe_url
    from app.services.email_sender import EmailSender

    dispatch_url = _make_unsubscribe_url("same@example.com")
    ses_footer = EmailSender()._build_unsubscribe_footer("same@example.com")

    assert dispatch_url.split("?")[0] == opt_out._unsubscribe_endpoint()
    assert ses_footer.split("token=")[0].endswith(opt_out._unsubscribe_endpoint() + "?")


@pytest.mark.asyncio
async def test_legacy_bare_string_tokens_still_verify(client, db_session):
    """Year-valid tokens are sitting in real inboxes; breaking them breaks one-click."""
    contact, _ = await _make_contact(db_session, "oldtoken@example.com")

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('oldtoken@example.com')}"
    )
    assert resp.status_code == 200

    await db_session.refresh(contact)
    assert contact.suppressed is True


@pytest.mark.asyncio
async def test_org_scoped_token_suppresses_only_that_sender(db_session):
    """An opt-out is addressed to a sender, not to the platform."""
    other_org = uuid.uuid4()
    mine, mine_enr = await _make_contact(db_session, "shared@example.com")
    theirs, theirs_enr = await _make_contact(
        db_session, "shared@example.com", org_id=other_org
    )

    await opt_out.record(db_session, "shared@example.com", org_id=APPREND_ORG_ID)

    await db_session.refresh(mine)
    await db_session.refresh(theirs)
    await db_session.refresh(mine_enr)
    await db_session.refresh(theirs_enr)
    assert mine.suppressed is True
    assert mine_enr.status == "unsubscribed"
    # The other org never mailed them; its enrollment is not ours to stop.
    assert theirs.suppressed is False
    assert theirs_enr.status == "active"


@pytest.mark.asyncio
async def test_legacy_token_without_an_org_stops_every_sender(db_session):
    """No org in the token and no way to ask — take the broad reading."""
    other_org = uuid.uuid4()
    mine, _ = await _make_contact(db_session, "broad@example.com")
    theirs, theirs_enr = await _make_contact(
        db_session, "broad@example.com", org_id=other_org
    )

    await opt_out.record(db_session, "broad@example.com", org_id=None)

    await db_session.refresh(mine)
    await db_session.refresh(theirs)
    await db_session.refresh(theirs_enr)
    assert mine.suppressed is True
    assert theirs.suppressed is True
    assert theirs_enr.status == "unsubscribed"


# --------------------------------------------------------------------------
# The service, directly.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_reports_what_it_changed(db_session):
    await _make_contact(db_session, "counted@example.com")
    await _make_contact(db_session, "counted@example.com")

    result = await opt_out.record(db_session, "counted@example.com")

    assert result.email == "counted@example.com"
    assert result.contacts == 2
    assert result.enrollments_stopped == 2
    assert result.recorded is True


@pytest.mark.asyncio
async def test_record_normalises_the_address(db_session):
    result = await opt_out.record(db_session, "  LOUD@Example.COM  ")
    assert result.email == "loud@example.com"


@pytest.mark.asyncio
async def test_record_refuses_a_non_address(db_session):
    assert await opt_out.record(db_session, "not-an-email") is None
    assert await opt_out.record(db_session, "") is None
    assert await opt_out.record(db_session, None) is None


@pytest.mark.asyncio
async def test_record_falls_back_to_the_first_party_org(db_session):
    """No contact, no org in the token — the list still has to hold it somewhere."""
    result = await opt_out.record(db_session, "nowhere@example.com")
    assert result.orgs == (APPREND_ORG_ID,)


# --------------------------------------------------------------------------
# The reply-based opt-out: correct on the flags, previously not durable.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_path_refuses_a_contact_flagged_unsubscribed_only(
    db_session, sample_company, sample_contact, monkeypatch
):
    """Defence in depth for the column that used to be written on its own.

    ``opt_out.record`` sets both flags, so this state should no longer arise from
    a link click — but three writers have set ``unsubscribed`` alone at various
    points, and rows in that state are in the database right now from every click
    that happened before this fix. The send gate reads it directly rather than
    trusting that nothing will ever write it alone again.
    """
    import app.tasks.outreach as outreach_tasks
    from tests.test_outreach_soft_skip import _enrollment_due_now, _stub_pipeline

    enrollment = await _enrollment_due_now(db_session, sample_company, sample_contact)
    enrollment_id = enrollment.id

    sample_contact.unsubscribed = True
    sample_contact.suppressed = False  # the pre-fix shape exactly
    await db_session.commit()

    calls = _stub_pipeline(monkeypatch, db_session, {"status": "sent"})
    result = await outreach_tasks.send_sequence_step.run(str(enrollment_id))

    assert result["reason"] == "contact_suppressed"
    assert calls == [], "a step went on the wire to someone who had opted out"

    stopped = (
        await db_session.execute(
            select(OutreachEnrollment).where(OutreachEnrollment.id == enrollment_id)
        )
    ).scalar_one()
    assert stopped.status == "unsubscribed"


@pytest.mark.asyncio
async def test_sweep_does_not_dispatch_a_contact_flagged_unsubscribed_only(
    db_session, sample_company, sample_contact, monkeypatch
):
    """The sweep's own copy of the gate. Cheaper to stop it before the task exists."""
    import app.tasks.outreach as outreach_tasks
    from tests.test_dispatch_lease import _due_enrollment, _stub_sweep

    await _due_enrollment(db_session, sample_contact)
    sample_contact.unsubscribed = True
    sample_contact.suppressed = False
    await db_session.commit()

    dispatched = _stub_sweep(monkeypatch, db_session)
    await outreach_tasks.process_outreach_queue.run()

    assert dispatched == [], "the sweep queued a send to someone who had opted out"


@pytest.mark.asyncio
async def test_org_scoped_token_through_the_endpoint_spares_the_other_sender(
    client, db_session
):
    """End-to-end on the token's org: mint scoped, POST it, only that org stops."""
    from app.services.email_dispatch import _make_unsubscribe_url

    other_org = uuid.uuid4()
    mine, mine_enr = await _make_contact(db_session, "tenant@example.com")
    theirs, theirs_enr = await _make_contact(
        db_session, "tenant@example.com", org_id=other_org
    )

    async with org_scope(APPREND_ORG_ID):
        token = _make_unsubscribe_url("tenant@example.com").split("token=", 1)[1]

    resp = await client.post(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 200

    for obj in (mine, theirs, mine_enr, theirs_enr):
        await db_session.refresh(obj)
    assert mine.suppressed is True
    assert mine_enr.status == "unsubscribed"
    assert theirs.suppressed is False
    assert theirs_enr.status == "active"


@pytest.mark.asyncio
async def test_reply_unsubscribe_also_lands_on_the_list(db_session):
    """Otherwise a re-scrape re-opens someone who replied "UNSUBSCRIBE"."""
    from app.tasks.replies import suppress_contact

    contact, enrollment = await _make_contact(db_session, "replied@example.com")

    stopped = await suppress_contact(db_session, contact, "replied unsubscribe")
    await db_session.commit()

    assert stopped == 1
    await db_session.refresh(enrollment)
    assert enrollment.status == "unsubscribed"
    assert await suppression.is_suppressed(db_session, "replied@example.com") is True


# --------------------------------------------------------------------------
# The opt-out has to be *visible*, not just effective.
#
# Every campaign surface — the campaign table, the operator dashboard, the
# bounce/funnel summary — sources `unsubscribed` from `OutreachEvent`, and
# nothing in the product wrote one. The stop worked perfectly and reported
# itself as never having happened: `unsubscribe_rate` was a structural 0.0% on
# every campaign, forever, which is the reading that keeps a bad campaign
# running.
#
# Mutation-tested — each of these fails at least one test below: not writing the
# event; writing it on every stopped enrollment instead of one per org; writing
# it for a contact that was never mailed; stamping the ambient org instead of
# the contact's; writing a second one on a repeat click; and leaving the reply
# path out.
# --------------------------------------------------------------------------


async def _sent_event(db_session, enrollment, *, step=1, org_id=None):
    from datetime import datetime, timezone

    from app.models.outreach import OutreachEvent

    kw = {"org_id": org_id} if org_id is not None else {}
    db_session.add(OutreachEvent(
        id=uuid.uuid4(), enrollment_id=enrollment.id, step_number=step,
        event_type="sent", occurred_at=datetime.now(timezone.utc),
        event_metadata={}, **kw,
    ))
    await db_session.commit()


async def _unsub_events(db_session, enrollment_id=None):
    from app.models.outreach import OutreachEvent

    q = select(OutreachEvent).where(OutreachEvent.event_type == "unsubscribed")
    if enrollment_id is not None:
        q = q.where(OutreachEvent.enrollment_id == enrollment_id)
    return list((await db_session.execute(
        q.execution_options(skip_org_filter=True)
    )).scalars().all())


@pytest.mark.asyncio
async def test_unsubscribe_is_booked_against_the_campaign_that_caused_it(
    client, db_session
):
    """The campaign table reported nobody opting out, forever."""
    contact, enrollment = await _make_contact(db_session, "seen@example.com")
    await _sent_event(db_session, enrollment, step=3)

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('seen@example.com')}"
    )
    assert resp.status_code == 200

    events = await _unsub_events(db_session, enrollment.id)
    assert len(events) == 1, "the opt-out left no trace on any campaign surface"
    # The step the recipient was actually on — the row is read next to the
    # funnel, where "they left at step 3" is the whole point of having it.
    assert events[0].step_number == 3
    assert events[0].org_id == contact.org_id
    assert events[0].event_metadata.get("source") == "opt_out"


@pytest.mark.asyncio
async def test_one_optout_is_one_event_however_many_campaigns_were_mailing(
    client, db_session
):
    """Three live campaigns and one human is one unsubscribe, not three.

    The portfolio totals sum the campaign rows, so per-enrollment attribution
    would report more unsubscribes than there were people.
    """
    contact, first = await _make_contact(db_session, "busy@example.com")
    await _sent_event(db_session, first, step=1)

    others = []
    for step in (2, 3):
        enr = OutreachEnrollment(
            id=uuid.uuid4(), contact_id=contact.id, sequence_id=uuid.uuid4(),
            status="active", current_step=1, org_id=contact.org_id,
        )
        db_session.add(enr)
        await db_session.flush()
        await _sent_event(db_session, enr, step=step)
        others.append(enr)

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('busy@example.com')}"
    )
    assert resp.status_code == 200

    events = await _unsub_events(db_session)
    assert len(events) == 1
    # The campaign that most recently put mail in front of them. Which one
    # actually prompted the click isn't knowable — the RFC 8058 token carries an
    # address and an org, never an enrollment — so the last send is the guess,
    # and it is the one an operator reading the row would make anyway.
    assert events[0].enrollment_id == others[-1].id


@pytest.mark.asyncio
async def test_repeat_one_click_does_not_double_count(client, db_session):
    """Providers POST one-click and the human clicks the link too."""
    _, enrollment = await _make_contact(db_session, "twice@example.com")
    await _sent_event(db_session, enrollment)

    token = _legacy_token("twice@example.com")
    for _ in range(3):
        assert (await client.post(f"/api/v1/track/unsubscribe?token={token}")).status_code == 200

    assert len(await _unsub_events(db_session)) == 1


@pytest.mark.asyncio
async def test_never_mailed_contact_books_nothing(client, db_session):
    """A campaign with no sends is off the table and its rate divides by zero."""
    _, _ = await _make_contact(db_session, "silent@example.com")

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('silent@example.com')}"
    )
    assert resp.status_code == 200
    assert await _unsub_events(db_session) == []


@pytest.mark.asyncio
async def test_reply_unsubscribe_counts_the_same_as_the_link(db_session):
    """Same event to the recipient, so it has to be the same event to us."""
    from app.tasks.replies import suppress_contact

    contact, enrollment = await _make_contact(db_session, "toldyou@example.com")
    await _sent_event(db_session, enrollment, step=2)

    await suppress_contact(db_session, contact, "replied unsubscribe")
    await db_session.commit()

    events = await _unsub_events(db_session, enrollment.id)
    assert len(events) == 1
    assert events[0].step_number == 2


@pytest.mark.asyncio
async def test_optout_events_do_not_cross_tenants(client, db_session):
    """A legacy token opts out everywhere; each org books its own row.

    This runs with no org bound, where the insert stamp falls back to the
    first-party org — so an unstamped write would file a stranger's opt-out
    under ours and leave their own campaign reading zero.
    """
    from app.models.tenant import Organization

    other_org = uuid.uuid4()
    db_session.add(Organization(id=other_org, name="Other Co", slug="other-co"))
    await db_session.flush()

    mine, mine_enr = await _make_contact(db_session, "shared@example.com")
    theirs, theirs_enr = await _make_contact(
        db_session, "shared@example.com", org_id=other_org
    )
    await _sent_event(db_session, mine_enr, org_id=mine.org_id)
    await _sent_event(db_session, theirs_enr, org_id=other_org)

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={_legacy_token('shared@example.com')}"
    )
    assert resp.status_code == 200

    events = await _unsub_events(db_session)
    assert {e.org_id for e in events} == {mine.org_id, other_org}
    assert {e.enrollment_id for e in events} == {mine_enr.id, theirs_enr.id}


@pytest.mark.asyncio
async def test_only_a_real_opt_out_lands_on_the_unsubscribe_rate(db_session):
    """A bounce is a dead mailbox, not a person objecting.

    `record` takes a `reason` and every do-not-contact writer could route
    through it. If it booked an `unsubscribed` event whatever the reason, the
    column an operator reads as "how many people we annoyed" would fill up with
    bad addresses — and it sits next to the bounce column, so the same recipient
    would be counted twice in two different meanings.
    """
    contact, enrollment = await _make_contact(db_session, "bounced@example.com")
    await _sent_event(db_session, enrollment)

    await opt_out.record(
        db_session, "bounced@example.com", org_id=contact.org_id, reason="bounce"
    )

    assert await _unsub_events(db_session) == []
    # The stop itself still happened — this is about what gets *counted*.
    await db_session.refresh(enrollment)
    assert enrollment.status == "unsubscribed"
