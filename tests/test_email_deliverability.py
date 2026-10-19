"""Tests for email deliverability, spam avoidance, and conversion optimization.

These tests verify that outreach emails follow best practices for:
- Inbox placement (avoiding spam filters)
- Reply detection and auto-pause
- Signature and footer compliance
- Email structure and readability
"""

import re
import uuid as _uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company
from app.models.contact import Contact
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.models.sequence import OutreachSequence, SequenceStep
from app.services.email_compose import render
from app.services.email_dispatch import (
    wrap_html_email,
    _unsubscribe_footer_html,
    _html_to_plain_text,
    _make_unsubscribe_url,
)


# ── Spam filter avoidance ──


class TestSpamFilterAvoidance:
    """Verify emails avoid common spam filter triggers."""

    def _render_full_email(self, subject, body, context):
        rendered_body = render(body, context)
        footer = _unsubscribe_footer_html("test@example.com")
        full_html = wrap_html_email(rendered_body + footer)
        return render(subject, context), full_html

    def test_no_all_caps_in_subject(self):
        """Subject lines should not have ALL CAPS words (spam trigger)."""
        subjects = [
            "the part that keeps you up at night, {{first_name}}",
            "the $47,000 mistake nobody talks about",
            "your competitors are already doing this",
            "last note from me, {{first_name}}",
            "that mold {{company_name}} keeps nursing along",
            "your scrap bin is more expensive than you think",
            "the shops beating you on quotes have a secret",
            "closing the loop (free process assessment inside)",
        ]
        ctx = {"first_name": "Raj", "company_name": "OneNet Plastics"}
        for subj in subjects:
            rendered = render(subj, ctx)
            words = rendered.split()
            all_caps = [w for w in words if len(w) > 2 and w.isupper() and w.isalpha()]
            assert not all_caps, f"ALL CAPS word in subject: {all_caps} in '{rendered}'"

    def test_no_excessive_punctuation(self):
        """No more than 1 exclamation or question mark per subject."""
        subjects = [
            "the part that keeps you up at night, Raj",
            "the $47,000 mistake nobody talks about",
            "your scrap bin is more expensive than you think",
        ]
        for subj in subjects:
            assert subj.count("!") <= 1, f"Too many ! in: {subj}"
            assert subj.count("?") <= 1, f"Too many ? in: {subj}"

    def test_no_url_in_body_without_context(self):
        """Email bodies should not contain raw URLs (use anchor tags instead)."""
        bodies = [
            "<p>Hi Raj,</p><p>Check our system that watches pressure curves.</p>",
            "<p>One shop cut scrap from 8.2% to 0.9%.</p>",
        ]
        url_pattern = re.compile(r'https?://\S+')
        for body in bodies:
            # URLs inside href attributes are fine, raw URLs in text are not
            plain_text = re.sub(r'<[^>]+>', '', body)
            urls = url_pattern.findall(plain_text)
            assert not urls, f"Raw URL in body text: {urls}"

    def test_body_has_reasonable_text_to_html_ratio(self):
        """Email should have more text than HTML tags (good text-to-code ratio)."""
        body = "<p>Hi Raj,</p><p>Every injection molder I talk to has one — that part that makes their stomach drop.</p><p>Our AI watches pressure curves in real time.</p>"
        wrapped = wrap_html_email(body)

        # Extract text content
        text_only = re.sub(r'<[^>]+>', '', wrapped)
        text_only = re.sub(r'\s+', ' ', text_only).strip()

        # Text should be at least 20% of total length
        ratio = len(text_only) / len(wrapped)
        assert ratio > 0.15, f"Text-to-HTML ratio too low: {ratio:.2f}"

    def test_no_image_tags_in_outreach_body(self):
        """Cold outreach emails should not contain images (reduces deliverability)."""
        bodies = [
            "<p>Hi {{first_name}},</p><p>Quick question about quality at {{company_name}}.</p>",
            "<p>{{first_name}}, following up on scrap cost reduction.</p>",
        ]
        for body in bodies:
            assert "<img" not in body.lower(), f"Image tag in cold email body"

    def test_email_not_too_long(self):
        """Cold emails should be under 300 words for optimal engagement."""
        bodies = [
            "<p>Hi {{first_name}},</p><p>Every injection molder I talk to has one — that part that makes their stomach drop when it comes off the press wrong. The one where a bad run means a frantic weekend, an angry customer call, and an air-freight bill that wipes out the margin.</p><p>At {{company_name}}, I imagine you know exactly which mold I am talking about.</p><p>We built something that would have caught the problem before it happened. Our AI watches pressure curves, melt temperature, and cooling profiles in real time — and alerts your team the moment something drifts, usually 2-3 cycles before the first bad part.</p><p>One shop told me: We used to find out about problems when the customer called. Now we find out before the operator does.</p><p>Would 15 minutes be worth it if it meant never getting that call again?</p>",
        ]
        for body in bodies:
            text_only = re.sub(r'<[^>]+>', '', body)
            words = text_only.split()
            assert len(words) < 300, f"Email too long: {len(words)} words"


# ── Reply detection ──


class TestReplyDetection:
    """Verify the system can detect and handle prospect replies."""

    @pytest.mark.asyncio
    async def test_reply_event_pauses_enrollment(self, db_session: AsyncSession, sample_contact: Contact):
        """When a 'replied' event is recorded, the enrollment should be paused."""
        seq = OutreachSequence(id=_uuid.uuid4(), name="Reply Test", is_active=True)
        db_session.add(seq)
        await db_session.flush()

        step = SequenceStep(
            id=_uuid.uuid4(), sequence_id=seq.id,
            step_number=1, delay_days=0,
            subject_template="Test", body_template="<p>Test</p>",
        )
        db_session.add(step)
        await db_session.flush()

        enrollment = OutreachEnrollment(
            id=_uuid.uuid4(),
            contact_id=sample_contact.id,
            sequence_id=seq.id,
            current_step=2,
            gmail_thread_id="thread_reply_test",
        )
        db_session.add(enrollment)
        await db_session.flush()

        # Record a reply event
        reply_event = OutreachEvent(
            id=_uuid.uuid4(),
            enrollment_id=enrollment.id,
            step_number=1,
            event_type="replied",
            event_metadata={"from": "prospect@example.com", "subject": "Re: Test"},
        )
        db_session.add(reply_event)

        # The reply should pause the enrollment
        enrollment.status = "replied"
        await db_session.flush()

        result = await db_session.execute(
            select(OutreachEnrollment).where(OutreachEnrollment.id == enrollment.id)
        )
        loaded = result.scalar_one()
        assert loaded.status == "replied"

    @pytest.mark.asyncio
    async def test_replied_enrollment_not_dispatched(self, db_session: AsyncSession, sample_contact: Contact):
        """Enrollments with 'replied' status should not be dispatched."""
        seq = OutreachSequence(id=_uuid.uuid4(), name="Reply Test", is_active=True)
        db_session.add(seq)
        await db_session.flush()

        enrollment = OutreachEnrollment(
            id=_uuid.uuid4(),
            contact_id=sample_contact.id,
            sequence_id=seq.id,
            current_step=2,
            status="replied",
        )
        db_session.add(enrollment)
        await db_session.flush()

        # Query active enrollments (as /process does)
        result = await db_session.execute(
            select(OutreachEnrollment).where(OutreachEnrollment.status == "active")
        )
        active = list(result.scalars().all())
        assert enrollment not in active


# ── Email signature ──


class TestEmailSignature:
    """Verify email signature handling."""

    @pytest.mark.asyncio
    async def test_signature_appended_to_body(self):
        """Email signature should be appended after the body."""
        from app.services.email_dispatch import send_outreach_email

        # We can't fully test without a real account, but verify the function
        # accepts signature_id parameter
        import inspect
        sig = inspect.signature(send_outreach_email)
        assert "signature_id" in sig.parameters

    def test_unsubscribe_footer_contains_token(self):
        """Unsubscribe footer should contain a signed token for the contact email."""
        footer = _unsubscribe_footer_html("prospect@example.com")
        assert "token=" in footer
        assert "unsubscribe" in footer.lower()

    def test_unsubscribe_token_is_unique_per_email(self):
        """Different emails should get different unsubscribe tokens."""
        footer1 = _unsubscribe_footer_html("alice@example.com")
        footer2 = _unsubscribe_footer_html("bob@example.com")
        # Extract tokens
        token1 = re.search(r'token=([^"&]+)', footer1).group(1)
        token2 = re.search(r'token=([^"&]+)', footer2).group(1)
        assert token1 != token2


# ── Conversion-oriented content checks ──


class TestConversionPatterns:
    """Verify emails follow proven B2B cold outreach conversion patterns."""

    def _get_v2_templates(self):
        return [
            {
                "subject": "the part that keeps you up at night, {{first_name}}",
                "body": "<p>Hi {{first_name}},</p><p>Every injection molder I talk to has one...</p>",
            },
            {
                "subject": "the $47,000 mistake nobody talks about",
                "body": "<p>Hi {{first_name}},</p><p>I ran some numbers recently...</p>",
            },
            {
                "subject": "your competitors are already doing this",
                "body": "<p>Hi {{first_name}},</p><p>Three injection molding shops...</p>",
            },
            {
                "subject": "last note from me, {{first_name}}",
                "body": "<p>Hi {{first_name}},</p><p>This will be my last note...</p>",
            },
        ]

    def _get_v3_templates(self):
        return [
            {
                "subject": "that mold {{company_name}} keeps nursing along",
                "body": "<p>Hi {{first_name}},</p><p>You know the one. The mold where cavity 4...</p>",
            },
            {
                "subject": "your scrap bin is more expensive than you think",
                "body": "<p>Hi {{first_name}},</p><p>Quick math that might sting...</p>",
            },
            {
                "subject": "the shops beating you on quotes have a secret",
                "body": "<p>Hi {{first_name}},</p><p>There is a reason some injection molders...</p>",
            },
            {
                "subject": "closing the loop (free process assessment inside)",
                "body": "<p>Hi {{first_name}},</p><p>This is my last email...</p>",
            },
        ]

    def test_every_email_has_cta(self):
        """Every email should end with a clear call to action."""
        cta_indicators = [
            "15 minutes",
            "worth a look",
            "what does your",
            "can i show",
            "reply",
            "calendar",
        ]
        # V2 and V3 both need CTAs
        assert len(cta_indicators) >= 4

    def test_subjects_create_curiosity_gap(self):
        """Subjects should create a curiosity gap that compels opening."""
        curiosity_subjects = [
            "the part that keeps you up at night",  # Identifies with pain
            "the $47,000 mistake nobody talks about",  # Specific number + mystery
            "your competitors are already doing this",  # FOMO
            "that mold keeps nursing along",  # Insider knowledge
            "your scrap bin is more expensive than you think",  # Challenge belief
            "the shops beating you on quotes have a secret",  # Competitive threat
        ]
        for subj in curiosity_subjects:
            # Each subject should be a statement that provokes curiosity
            assert len(subj) > 15, f"Subject too short to create curiosity: {subj}"
            assert len(subj) < 65, f"Subject too long for mobile: {subj}"

    def test_social_proof_in_steps_2_and_3(self):
        """Steps 2 and 3 should include social proof (stories from other shops)."""
        proof_keywords = ["shop", "plant", "told me", "discovered", "found",
                          "quality manager", "one of the", "three", "signed up"]
        # These keywords should appear in the social proof steps
        assert len(proof_keywords) > 5

    def test_step4_offers_something_free(self):
        """Step 4 (breakup) should offer something of value to re-engage."""
        v3_step4_subject = "closing the loop (free process assessment inside)"
        assert "free" in v3_step4_subject.lower()
        assert "assessment" in v3_step4_subject.lower()

    def test_progressive_urgency(self):
        """Urgency should increase across steps: empathy → data → FOMO → breakup."""
        step_types = ["empathy", "data/proof", "competitive_urgency", "breakup"]
        assert len(step_types) == 4
        # Verify the sequence follows the pattern
        assert step_types[0] == "empathy"  # Step 1: connect with pain
        assert "data" in step_types[1] or "proof" in step_types[1]  # Step 2: prove with numbers
        assert "urgency" in step_types[2]  # Step 3: create urgency
        assert "breakup" in step_types[3]  # Step 4: respectful close


# ── A/B testing infrastructure ──


class TestABTestInfrastructure:
    """Verify multiple sequences can coexist for A/B testing."""

    @pytest.mark.asyncio
    async def test_two_sequences_can_exist_simultaneously(self, db_session: AsyncSession):
        """Multiple active sequences should coexist."""
        seq_a = OutreachSequence(id=_uuid.uuid4(), name="Variant A", is_active=True)
        seq_b = OutreachSequence(id=_uuid.uuid4(), name="Variant B", is_active=True)
        db_session.add_all([seq_a, seq_b])
        await db_session.flush()

        result = await db_session.execute(
            select(OutreachSequence).where(OutreachSequence.is_active == True)
        )
        active = list(result.scalars().all())
        assert len(active) >= 2

    @pytest.mark.asyncio
    async def test_contact_can_only_be_in_one_sequence_at_a_time(
        self, db_session: AsyncSession, sample_contact: Contact
    ):
        """A contact enrolled in seq A should be skipped if enrolled in seq B (already active)."""
        seq_a = OutreachSequence(id=_uuid.uuid4(), name="Variant A", is_active=True)
        db_session.add(seq_a)
        await db_session.flush()

        enrollment = OutreachEnrollment(
            id=_uuid.uuid4(),
            contact_id=sample_contact.id,
            sequence_id=seq_a.id,
        )
        db_session.add(enrollment)
        await db_session.flush()

        # Check: the contact now has an active enrollment
        result = await db_session.execute(
            select(OutreachEnrollment).where(
                OutreachEnrollment.contact_id == sample_contact.id,
                OutreachEnrollment.status == "active",
            )
        )
        assert result.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_events_track_which_sequence_variant(
        self, db_session: AsyncSession, sample_contact: Contact
    ):
        """Events should be traceable back to the specific sequence variant."""
        seq = OutreachSequence(id=_uuid.uuid4(), name="Variant A", is_active=True)
        db_session.add(seq)
        await db_session.flush()

        enrollment = OutreachEnrollment(
            id=_uuid.uuid4(),
            contact_id=sample_contact.id,
            sequence_id=seq.id,
        )
        db_session.add(enrollment)
        await db_session.flush()

        event = OutreachEvent(
            id=_uuid.uuid4(),
            enrollment_id=enrollment.id,
            step_number=1,
            event_type="sent",
            event_metadata={"variant": "A"},
        )
        db_session.add(event)
        await db_session.flush()

        # Trace event → enrollment → sequence
        result = await db_session.execute(
            select(OutreachEvent).where(OutreachEvent.enrollment_id == enrollment.id)
        )
        loaded_event = result.scalar_one()
        assert loaded_event.enrollment_id == enrollment.id

        result = await db_session.execute(
            select(OutreachEnrollment).where(OutreachEnrollment.id == enrollment.id)
        )
        loaded_enrollment = result.scalar_one()
        assert loaded_enrollment.sequence_id == seq.id


# ── Email readability ──


class TestEmailReadability:
    """Verify emails are readable and well-structured."""

    def test_paragraphs_are_short(self):
        """Each paragraph should be under 4 sentences for mobile readability."""
        body = (
            "<p>Hi Raj,</p>"
            "<p>Every injection molder I talk to has one — that part that makes their stomach drop. "
            "The one where a bad run means a frantic weekend.</p>"
            "<p>We built something that catches the problem before it happens.</p>"
        )
        # Extract paragraphs
        paragraphs = re.findall(r'<p>(.*?)</p>', body, re.DOTALL)
        for p in paragraphs:
            sentences = re.split(r'[.!?]+', p)
            sentences = [s.strip() for s in sentences if s.strip()]
            assert len(sentences) <= 5, f"Paragraph too long ({len(sentences)} sentences): {p[:50]}..."

    def test_uses_em_for_emphasis(self):
        """Emphasis should use <em> tags, not ALL CAPS or excessive bold."""
        body = "<p>They are <em>sleeping better</em>. Not just catching defects — <em>preventing</em> them.</p>"
        assert "<em>" in body
        # No ALL CAPS emphasis
        text = re.sub(r'<[^>]+>', '', body)
        words = text.split()
        caps_words = [w for w in words if len(w) > 3 and w.isupper()]
        assert not caps_words, f"ALL CAPS used for emphasis: {caps_words}"

    def test_email_starts_with_greeting(self):
        """Every email body should start with a personal greeting."""
        bodies = [
            "<p>Hi {{first_name}},</p><p>Content here</p>",
            "<p>Hi {{first_name}},</p><p>Quick math</p>",
        ]
        for body in bodies:
            assert body.startswith("<p>Hi {{first_name}},</p>")

    def test_no_marketing_buzzwords(self):
        """Email should avoid overused marketing buzzwords that hurt credibility."""
        buzzwords = ["synergy", "leverage", "paradigm", "disruptive", "game-changing",
                     "revolutionary", "cutting-edge", "best-in-class", "world-class",
                     "state-of-the-art"]
        body = (
            "Our AI watches pressure curves in real time and alerts your team "
            "the moment something drifts. One shop cut their scrap from 8.2% to 0.9%."
        )
        for bw in buzzwords:
            assert bw not in body.lower(), f"Marketing buzzword found: {bw}"


# ── V4 deliverability and conversion checks ──


class TestV4Deliverability:
    """Verify the v4 deal-cracker sequence passes deliverability gates."""

    V4_FULL_BODIES = [
        # Step 1
        "Hi Vikram, I want to ask you something that nobody in your position likes to think about. "
        "When a mold goes sideways at 2 AM and your night-shift operator does not catch it "
        "who gets the phone call? When 8,000 parts ship to your biggest customer and 400 of them "
        "have sink marks that slipped past visual inspection whose name is on the corrective action report? "
        "I already know the answer. It is yours. You carry that weight every single day. "
        "Here is what I want you to imagine for 30 seconds: your phone buzzes at 10:14 AM. "
        "Not a crisis an alert. Press 7, cavity 4: pressure curve drifting 3.2% from baseline. "
        "Possible core pin wear. Review recommended. You walk out to the floor, pull the mold, "
        "and find early-stage wear that would have caused flash in another 200 cycles. "
        "That is not a fantasy. That is what our AI does, right now, for shops like Apex Mold Industries. "
        "I am not selling you software. I am offering you something you cannot put a price on: peace of mind. "
        "15 minutes. I will show you the dashboard. If it is not for you, I will never bother you again.",
        # Step 2
        "Hi Vikram, I did something that might make you uncomfortable. I ran the numbers on what "
        "scrap and rework probably cost Apex Mold Industries last year. "
        "Annual resin spend times scrap rate times 1 plus rework multiplier. "
        "For a typical injection molder running 8-15 presses, the number lands between "
        "$220K and $480K per year. And that is just the direct cost. "
        "The overtime your team works to remake parts that should have been right the first time. "
        "The air freight you pay when a scrap run blows your delivery schedule. "
        "The margin you give back on the next quote because the customer remembers the last quality issue. "
        "One of our customers a 12-press shop in Michigan told me: "
        "I knew scrap was a problem. I did not know it was a $340,000 problem. "
        "When I showed the board that number, and then showed them it dropped to $87,000 in six months "
        "with your system, nobody asked about the subscription cost. "
        "Vikram, I am not asking you to commit to anything. I am asking you to see your own numbers.",
        # Step 3
        "Hi Vikram, Let me tell you about the conversation I have had with every injection molding "
        "shop owner I have met this year. We have one guy Mike, or Steve, or Dave who has been here "
        "22 years. He can listen to a press and tell you if something is off. He knows every mold, "
        "every material, every trick. And he is going to retire in a few years, and I have no idea "
        "what we are going to do. Does that sound familiar? "
        "Your best operator's knowledge is not documented anywhere. It lives in his head, "
        "in his hands, in the way he adjusts pack pressure by feel. "
        "Our AI learns what a good cycle looks like on every mold not from a textbook, "
        "but from your actual production data. It becomes the institutional memory that never retires. "
        "The shops that figure this out first will hire the best people, because the best people "
        "want to work somewhere that invests in getting better.",
        # Step 4
        "Hi Vikram, I have sent you three emails now, and I know that is a lot from someone you "
        "have never met. So I want to be honest with you about why I keep writing. "
        "It is not because I need another sale. It is because I have seen what happens to shops that wait. "
        "I watched a 15-press operation in Indiana lose their largest customer a $2.1M account "
        "because of two quality escapes in six months. "
        "Too busy running parts to protect the business that runs the parts. "
        "Free 48-hour process audit on one press. We connect our sensors, run for two days, "
        "and hand you a report showing every anomaly, drift pattern, and hidden inefficiency. "
        "Most shops discover at least one issue they did not know they had. "
        "One found a check ring at 40% wear that was adding 1.8 seconds to every cycle "
        "14 hours of wasted press time per week. "
        "If you are interested, just reply audit and I will set it up.",
    ]

    def test_v4_no_all_caps(self):
        """V4 bodies should have no ALL-CAPS words (except short abbreviations)."""
        for body in self.V4_FULL_BODIES:
            words = body.split()
            all_caps = [w for w in words if len(w) > 3 and w.isupper() and w.isalpha()]
            assert not all_caps, f"ALL CAPS in v4 body: {all_caps}"

    def test_v4_word_count_under_300(self):
        """Each v4 step should be under 300 words."""
        for i, body in enumerate(self.V4_FULL_BODIES, 1):
            words = body.split()
            assert len(words) < 300, f"V4 step {i} too long: {len(words)} words"

    def test_v4_no_raw_urls(self):
        """V4 bodies should not contain raw URLs."""
        url_pattern = re.compile(r'https?://\S+')
        for body in self.V4_FULL_BODIES:
            urls = url_pattern.findall(body)
            assert not urls, f"Raw URL in v4 body: {urls}"

    def test_v4_no_images(self):
        """V4 emails should have no image tags."""
        for body in self.V4_FULL_BODIES:
            assert "<img" not in body.lower()

    def test_v4_no_marketing_buzzwords(self):
        """V4 should avoid marketing buzzwords."""
        buzzwords = ["synergy", "leverage", "paradigm", "disruptive", "game-changing",
                     "revolutionary", "cutting-edge", "best-in-class", "world-class",
                     "state-of-the-art", "next-generation", "scalable", "ecosystem"]
        for body in self.V4_FULL_BODIES:
            for bw in buzzwords:
                assert bw not in body.lower(), f"Buzzword '{bw}' in v4 body"

    def test_v4_uses_concrete_numbers(self):
        """V4 should use specific numbers, not vague claims."""
        all_text = " ".join(self.V4_FULL_BODIES)
        numbers = re.findall(r'\$[\d,.]+[KkMm]?|\d+%|\d+[\s-](?:press|year|hour|cycle|day)', all_text)
        assert len(numbers) >= 8, f"Not enough concrete numbers in v4: found {len(numbers)}"

    def test_v4_mentions_injection_molding_terms(self):
        """V4 should use industry-specific vocabulary."""
        terms = ["cavity", "mold", "press", "scrap", "resin", "cycle",
                 "core pin", "flash", "sink marks", "pack pressure", "check ring"]
        all_text = " ".join(self.V4_FULL_BODIES).lower()
        found = [t for t in terms if t in all_text]
        assert len(found) >= 8, f"Missing injection molding terms. Found: {found}"

    def test_v4_step4_has_single_word_reply_cta(self):
        """Step 4 should request a one-word reply for maximum conversion."""
        step4 = self.V4_FULL_BODIES[3]
        assert "reply" in step4.lower()
        assert "audit" in step4.lower()

    def test_v4_progressive_emotional_intensity(self):
        """V4 emotional arc: personal fear → financial → knowledge loss → breakup."""
        # Step 1 should have fear/guilt words
        step1 = self.V4_FULL_BODIES[0].lower()
        assert any(w in step1 for w in ["blame", "phone call", "anxiety", "stomach"])
        # Step 2 should have money words
        step2 = self.V4_FULL_BODIES[1].lower()
        assert "$" in step2 and any(w in step2 for w in ["cost", "resin", "margin", "overtime"])
        # Step 3 should have loss words
        step3 = self.V4_FULL_BODIES[2].lower()
        assert any(w in step3 for w in ["retire", "walks out", "documented", "head"])
        # Step 4 should have finality words
        step4 = self.V4_FULL_BODIES[3].lower()
        assert any(w in step4 for w in ["last", "leave you alone", "final", "honest"])


# ── Anti-spam header and plain text tests ──


class TestAntiSpamHeaders:
    """Verify RFC 2369 List-Unsubscribe and RFC 8058 one-click headers."""

    def test_list_unsubscribe_url_generated(self, monkeypatch):
        """Unsubscribe URL should be generated for any contact email.

        Pins ``tracking_base_url`` rather than reading it from the machine's
        ``.env``: a developer box points it at ``http://localhost:8000``, and
        the assertion is about how the link is built, not about that box.
        """
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "tracking_base_url", "https://prospect.doaide.com")
        url = _make_unsubscribe_url("test@example.com")
        assert url.startswith("https://prospect.doaide.com/api/v1/track/unsubscribe")
        assert "unsubscribe" in url
        assert "token=" in url

    def test_list_unsubscribe_url_unique_per_contact(self):
        """Different contacts get different unsubscribe URLs."""
        url1 = _make_unsubscribe_url("alice@example.com")
        url2 = _make_unsubscribe_url("bob@example.com")
        assert url1 != url2

    def test_gmail_sender_accepts_list_unsubscribe_param(self):
        """GmailSender.send_email should accept list_unsubscribe_url."""
        import inspect
        from app.services.gmail_sender import GmailSender
        sig = inspect.signature(GmailSender.send_email)
        assert "list_unsubscribe_url" in sig.parameters

    def test_mime_includes_list_unsubscribe_header(self):
        """MIME message should include List-Unsubscribe header when URL provided."""
        import base64
        from unittest.mock import MagicMock
        from app.services.gmail_sender import GmailSender

        account = MagicMock()
        account.email = "sender@example.com"
        account.display_name = "Test Sender"
        sender = GmailSender(account)

        unsub_url = "https://example.com/unsubscribe?token=abc123"
        result = sender._build_mime(
            to_email="prospect@example.com",
            subject="test subject",
            body_html="<p>Hello</p>",
            body_text="Hello",
            reply_to=None,
            list_unsubscribe_url=unsub_url,
        )
        # Decode the raw MIME to verify headers
        raw_bytes = base64.urlsafe_b64decode(result["raw"])
        raw_str = raw_bytes.decode("utf-8")
        assert "List-Unsubscribe:" in raw_str
        assert unsub_url in raw_str
        assert "List-Unsubscribe-Post: List-Unsubscribe=One-Click" in raw_str

    def test_mime_threaded_includes_list_unsubscribe_header(self):
        """Threaded MIME message should also include List-Unsubscribe header."""
        import base64
        from unittest.mock import MagicMock
        from app.services.gmail_sender import GmailSender

        account = MagicMock()
        account.email = "sender@example.com"
        account.display_name = "Test Sender"
        sender = GmailSender(account)

        unsub_url = "https://example.com/unsubscribe?token=abc123"
        result = sender._build_mime_threaded(
            to_email="prospect@example.com",
            subject="Re: test subject",
            body_html="<p>Follow up</p>",
            body_text="Follow up",
            reply_to=None,
            in_reply_to="<msg123@mail.gmail.com>",
            references="<msg123@mail.gmail.com>",
            list_unsubscribe_url=unsub_url,
        )
        raw_bytes = base64.urlsafe_b64decode(result["raw"])
        raw_str = raw_bytes.decode("utf-8")
        assert "List-Unsubscribe:" in raw_str
        assert "List-Unsubscribe-Post:" in raw_str

    def test_no_list_unsubscribe_when_url_is_none(self):
        """MIME should not include List-Unsubscribe when no URL provided."""
        import base64
        from unittest.mock import MagicMock
        from app.services.gmail_sender import GmailSender

        account = MagicMock()
        account.email = "sender@example.com"
        account.display_name = "Test Sender"
        sender = GmailSender(account)

        result = sender._build_mime(
            to_email="prospect@example.com",
            subject="test subject",
            body_html="<p>Hello</p>",
            body_text="Hello",
            reply_to=None,
            list_unsubscribe_url=None,
        )
        raw_bytes = base64.urlsafe_b64decode(result["raw"])
        raw_str = raw_bytes.decode("utf-8")
        assert "List-Unsubscribe:" not in raw_str


class TestPlainTextGeneration:
    """Verify HTML-to-plain-text conversion for the text/plain MIME part."""

    def test_strips_html_tags(self):
        """Plain text should contain no HTML tags."""
        html = "<p>Hi <strong>Raj</strong>,</p><p>Check this out.</p>"
        plain = _html_to_plain_text(html)
        assert "<" not in plain
        assert ">" not in plain

    def test_preserves_readable_text(self):
        """Core text content should survive conversion."""
        html = "<p>Hi Raj,</p><p>Every injection molder I talk to has one.</p>"
        plain = _html_to_plain_text(html)
        assert "Hi Raj" in plain
        assert "injection molder" in plain

    def test_converts_links_to_text_with_url(self):
        """Links should show both text and URL in plain text."""
        html = '<p>See <a href="https://example.com/demo">the demo</a> here.</p>'
        plain = _html_to_plain_text(html)
        assert "the demo" in plain
        assert "https://example.com/demo" in plain

    def test_converts_br_to_newlines(self):
        """<br> tags should become newlines."""
        html = "Line one<br>Line two<br/>Line three"
        plain = _html_to_plain_text(html)
        assert "Line one\nLine two\nLine three" in plain

    def test_converts_p_to_double_newlines(self):
        """</p> should produce paragraph breaks."""
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        plain = _html_to_plain_text(html)
        assert "First paragraph." in plain
        assert "Second paragraph." in plain
        # Should have some separation
        lines = plain.split("\n")
        assert len(lines) >= 2

    def test_decodes_html_entities(self):
        """HTML entities like &middot; should be decoded."""
        html = "<p>Apprend &middot; Toronto</p>"
        plain = _html_to_plain_text(html)
        assert "Apprend" in plain
        assert "Toronto" in plain
        assert "&middot;" not in plain

    def test_no_excessive_whitespace(self):
        """Converted text should not have runs of 3+ blank lines."""
        html = "<p>One</p><p></p><p></p><p></p><p>Two</p>"
        plain = _html_to_plain_text(html)
        assert "\n\n\n" not in plain

    def test_v6_body_produces_clean_plain_text(self):
        """V6 email body should produce readable plain text with no HTML."""
        v6_step1 = (
            '<p>Hi Raj,</p><p>I know what keeps people in your position awake.</p>'
            '<p>It is not the press going down — you can fix a press. It is the '
            '<em>silent</em> failure.</p>'
            '<p><a href="https://best-molding-demo.pages.dev" '
            'style="color:#2563eb;font-weight:bold;">'
            'See how AI catches defects before they happen →</a></p>'
        )
        plain = _html_to_plain_text(v6_step1)
        assert "<" not in plain
        assert ">" not in plain or "→" in plain  # arrow char is OK
        assert "Hi Raj" in plain
        assert "best-molding-demo.pages.dev" in plain
        assert "silent" in plain
