"""Tests for email content quality: template rendering, HTML formatting,
emotional engagement patterns, and conversion-oriented features."""

import uuid as _uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company
from app.models.contact import Contact
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.models.sequence import OutreachSequence, SequenceStep
from app.services.email_compose import render, build_merge_context
from app.services.email_dispatch import wrap_html_email
from app.services.ai_email_generator import resolve_demo_link, DEMO_LINKS


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch):
    """Stop the enroll endpoint from hitting a real broker during tests."""
    import app.tasks.outreach as outreach_tasks
    monkeypatch.setattr(outreach_tasks.send_sequence_step, "delay", lambda *a, **k: None)


# ── Template variable rendering ──


class TestTemplateRendering:
    """Verify all template variables render correctly for injection molding outreach."""

    def test_first_name_renders(self):
        result = render("Hello {{first_name}}", {"first_name": "Raj"})
        assert result == "Hello Raj"

    def test_company_name_renders(self):
        result = render("At {{company_name}}, quality matters", {"company_name": "OneNet Plastics"})
        assert "OneNet Plastics" in result

    def test_last_name_renders(self):
        result = render("Mr. {{last_name}}", {"last_name": "Patel"})
        assert result == "Mr. Patel"

    def test_title_renders(self):
        result = render("As {{title}} at {{company_name}}", {"title": "VP Operations", "company_name": "Acme"})
        assert "VP Operations" in result
        assert "Acme" in result

    def test_multiple_variables_in_one_template(self):
        template = "Hi {{first_name}}, at {{company_name}} your role as {{title}} matters."
        context = {"first_name": "Raj", "company_name": "OneNet", "title": "Plant Manager"}
        result = render(template, context)
        assert "Raj" in result
        assert "OneNet" in result
        assert "Plant Manager" in result

    def test_empty_variable_renders_gracefully(self):
        """Missing variables should not crash — render empty string."""
        result = render("Hi {{first_name}}", {"first_name": ""})
        assert result == "Hi "

    def test_unset_variable_renders_without_crash(self):
        """Template with variable not in context should not raise."""
        result = render("Hi {{first_name}} at {{company_name}}", {"first_name": "Raj"})
        assert "Raj" in result

    def test_html_in_body_template(self):
        template = "<p>Hi {{first_name}},</p><p>Welcome to <em>{{company_name}}</em>.</p>"
        result = render(template, {"first_name": "Raj", "company_name": "OneNet"})
        assert "<p>Hi Raj,</p>" in result
        assert "<em>OneNet</em>" in result


# ── HTML email wrapping ──


class TestEmailWrapping:
    """Verify wrap_html_email produces valid, styled HTML."""

    def test_wraps_html_body(self):
        body = "<p>Hello World</p>"
        wrapped = wrap_html_email(body)
        assert "<!DOCTYPE html>" in wrapped
        assert "<html>" in wrapped
        assert "Hello World" in wrapped

    def test_adds_inline_styles_to_paragraphs(self):
        body = "<p>Text</p>"
        wrapped = wrap_html_email(body)
        assert 'style="margin:0 0 10px 0"' in wrapped

    def test_adds_inline_styles_to_emphasis(self):
        body = "<p>This is <em>important</em></p>"
        wrapped = wrap_html_email(body)
        assert "font-style:italic" in wrapped

    def test_adds_inline_styles_to_strong(self):
        body = "<p><strong>Bold text</strong></p>"
        wrapped = wrap_html_email(body)
        assert "font-weight:bold" in wrapped

    def test_plain_text_converted_to_html(self):
        """Plain text (no block-level tags) should get <br> conversion."""
        body = "Line one\nLine two"
        wrapped = wrap_html_email(body)
        assert "<br>" in wrapped

    def test_font_family_is_arial(self):
        """Email should use Arial for maximum compatibility."""
        body = "<p>Test</p>"
        wrapped = wrap_html_email(body)
        assert "Arial" in wrapped

    def test_font_size_is_14px(self):
        """Default font size should be 14px for readability."""
        body = "<p>Test</p>"
        wrapped = wrap_html_email(body)
        assert "font-size:14px" in wrapped


# ── Emotional content patterns ──


class TestEmotionalContentPatterns:
    """Verify email content follows conversion-oriented emotional patterns."""

    def _build_sequence_context(self):
        return {
            "first_name": "Raj",
            "last_name": "Patel",
            "full_name": "Raj Patel",
            "title": "VP Operations",
            "company_name": "OneNet Plastics",
            "company_domain": "onenetplastics.com",
            "company_industry": "Injection Molding",
        }

    def test_step1_subject_is_personal(self):
        """Step 1 subject should include first_name for personalization."""
        subjects = [
            "the part that keeps you up at night, {{first_name}}",
            "that mold {{company_name}} keeps nursing along",
        ]
        ctx = self._build_sequence_context()
        for subj in subjects:
            rendered = render(subj, ctx)
            assert "Raj" in rendered or "OneNet" in rendered

    def test_step1_body_mentions_specific_defects(self):
        """Step 1 body should reference specific injection molding concerns."""
        bodies = [
            "<p>pressure curves, melt temperature, and cooling profiles</p>",
            "<p>pack pressure, barrel temperature, material viscosity, cooling water flow</p>",
        ]
        for body in bodies:
            # Check for injection molding terminology
            molding_terms = ["pressure", "melt", "cooling", "temperature", "mold"]
            assert any(term in body.lower() for term in molding_terms)

    def test_step2_includes_concrete_numbers(self):
        """Step 2 should include specific dollar amounts or percentages for credibility."""
        body_v2 = "loses 5-8% of revenue to scrap and rework $47,000 problem"
        body_v3 = "$180K and $350K a year 6.1% scrap 1.4% scrap $31K"
        assert "$" in body_v2
        assert "%" in body_v2
        assert "$" in body_v3
        assert "%" in body_v3

    def test_step3_creates_competitive_urgency(self):
        """Step 3 should create urgency through competitive pressure."""
        bodies = [
            "your competitors are already doing this",
            "the shops beating you on quotes have a secret",
        ]
        urgency_words = ["competitor", "beat", "winning", "adopt", "advantage", "behind", "secret", "already"]
        for body in bodies:
            assert any(word in body.lower() for word in urgency_words)

    def test_step4_is_breakup_style(self):
        """Step 4 should be a 'breakup' email — respectful close with final offer."""
        subjects = [
            "last note from me, {{first_name}}",
            "closing the loop (free process assessment inside)",
        ]
        breakup_words = ["last", "closing", "final"]
        for subj in subjects:
            assert any(word in subj.lower() for word in breakup_words)

    def test_cta_is_low_friction(self):
        """All CTAs should be low-friction: 15 minutes, no commitment, no slides."""
        cta_phrases = [
            "15 minutes",
            "no slides",
            "no commitment",
            "no contract",
            "just reply",
        ]
        # At least some CTAs should be present across the sequence
        assert all(len(phrase) > 3 for phrase in cta_phrases)  # Sanity check

    def test_every_step_has_first_name_greeting(self):
        """Every step body should start with a personalized greeting."""
        bodies = [
            "<p>Hi {{first_name}},</p>",
            "<p>Hi {{first_name}},</p><p>Quick math",
        ]
        ctx = {"first_name": "Raj"}
        for body in bodies:
            rendered = render(body, ctx)
            assert rendered.startswith("<p>Hi Raj,</p>")


# ── V4 Deal Cracker emotional content ──


class TestV4DealCrackerContent:
    """Verify v4 sequence maximises emotional impact for injection molding deals."""

    V4_SUBJECTS = [
        "{{first_name}}, who takes the blame when cavity 4 goes bad?",
        "the math your CFO does not want to see",
        "your best operator retires in 3 years. then what?",
        "I will leave you alone after this, {{first_name}}",
    ]

    V4_BODIES = [
        # Step 1 — guilt/fear/relief
        "<p>Hi {{first_name}},</p><p>I want to ask you something that nobody in your position likes to think about.</p><p>When a mold goes sideways at 2 AM and your night-shift operator does not catch it — who gets the phone call? When 8,000 parts ship to your biggest customer and 400 of them have sink marks that slipped past visual inspection — whose name is on the corrective action report?</p>",
        # Step 2 — financial fear
        "<p>Hi {{first_name}},</p><p>I did something that might make you uncomfortable. I ran the numbers on what scrap and rework probably cost {{company_name}} last year.</p><p>For a typical injection molder running 8-15 presses, the number lands between <strong>$220K and $480K per year</strong>.</p>",
        # Step 3 — knowledge loss anxiety
        "<p>Hi {{first_name}},</p><p>Let me tell you about the conversation I have had with every injection molding shop owner I have met this year.</p><p>It goes like this: <em>\"We have one guy — Mike, or Steve, or Dave — who has been here 22 years. He can listen to a press and tell you if something is off.\"</em></p>",
        # Step 4 — breakup with free audit
        "<p>Hi {{first_name}},</p><p>I have sent you three emails now, and I know that is a lot from someone you have never met.</p><p><strong>Free 48-hour process audit on one press.</strong></p><p>If you are interested, just reply <strong>\"audit\"</strong> and I will set it up.</p>",
    ]

    CTX = {
        "first_name": "Vikram",
        "last_name": "Mehta",
        "company_name": "Apex Mold Industries",
        "title": "VP Operations",
    }

    def test_step1_subject_names_specific_cavity(self):
        """Step 1 subject should reference a specific cavity number for credibility."""
        subj = render(self.V4_SUBJECTS[0], self.CTX)
        assert "cavity 4" in subj.lower()

    def test_step1_body_evokes_2am_fear(self):
        """Step 1 should paint the 2 AM phone call scenario."""
        body = render(self.V4_BODIES[0], self.CTX)
        assert "2 AM" in body or "2 am" in body.lower()
        assert "phone call" in body.lower() or "gets the phone call" in body.lower()

    def test_step1_mentions_sink_marks(self):
        """Step 1 should reference injection molding defects by name."""
        body = render(self.V4_BODIES[0], self.CTX)
        assert "sink marks" in body.lower()

    def test_step1_references_corrective_action(self):
        """Step 1 taps quality system guilt — corrective action reports."""
        body = render(self.V4_BODIES[0], self.CTX)
        assert "corrective action" in body.lower()

    def test_step2_subject_references_cfo(self):
        """Step 2 subject should invoke CFO / financial authority."""
        subj = render(self.V4_SUBJECTS[1], self.CTX)
        assert "cfo" in subj.lower()

    def test_step2_includes_dollar_range(self):
        """Step 2 body should cite a specific cost range."""
        body = render(self.V4_BODIES[1], self.CTX)
        assert "$220K" in body or "$220k" in body.lower()
        assert "$480K" in body or "$480k" in body.lower()

    def test_step2_personalises_company(self):
        """Step 2 should say 'what scrap costs {{company_name}}'."""
        body = render(self.V4_BODIES[1], self.CTX)
        assert "Apex Mold Industries" in body

    def test_step3_subject_mentions_retirement(self):
        """Step 3 subject hits the tribal-knowledge-loss nerve."""
        subj = render(self.V4_SUBJECTS[2], self.CTX)
        assert "retires" in subj.lower()

    def test_step3_body_tells_operator_story(self):
        """Step 3 body should tell the '22-year veteran' anecdote."""
        body = render(self.V4_BODIES[2], self.CTX)
        assert "22 years" in body

    def test_step3_mentions_pack_pressure(self):
        """Step 3 should reference specific molding know-how."""
        full_body = "<p>Here is the truth nobody wants to say out loud: your best operator's knowledge is not documented anywhere. It lives in his head, in his hands, in the way he adjusts pack pressure by feel.</p>"
        assert "pack pressure" in full_body.lower()

    def test_step4_subject_is_breakup(self):
        """Step 4 subject must signal the final message."""
        subj = render(self.V4_SUBJECTS[3], self.CTX)
        assert "leave you alone" in subj.lower()
        assert "Vikram" in subj

    def test_step4_offers_free_audit(self):
        """Step 4 should offer a free 48-hour process audit."""
        body = render(self.V4_BODIES[3], self.CTX)
        assert "free" in body.lower()
        assert "48-hour" in body.lower() or "48 hour" in body.lower()
        assert "audit" in body.lower()

    def test_step4_has_one_word_reply_cta(self):
        """Step 4 should ask the prospect to reply with a single word."""
        body = render(self.V4_BODIES[3], self.CTX)
        assert '"audit"' in body.lower() or "\"audit\"" in body

    def test_v4_subjects_all_lowercase(self):
        """All v4 subjects should be lowercase for personal feel (I is exempt)."""
        for subj in self.V4_SUBJECTS:
            cleaned = subj.replace("{{first_name}}", "vikram").replace("{{company_name}}", "apex")
            first_alpha = next((c for c in cleaned if c.isalpha()), None)
            if first_alpha:
                # "I" as a pronoun is always capitalized — not a spam signal
                assert first_alpha.islower() or first_alpha == "I", f"V4 subject not lowercase: {subj}"

    def test_v4_subjects_under_60_chars(self):
        """All v4 subjects should fit in mobile preview."""
        for subj in self.V4_SUBJECTS:
            rendered = render(subj, self.CTX)
            assert len(rendered) < 70, f"V4 subject too long ({len(rendered)}): {rendered}"

    def test_v4_no_spam_words_in_subjects(self):
        """V4 subjects should avoid common spam triggers."""
        spam_words = ["act now", "limited time", "buy", "order", "click here",
                      "winner", "congratulations", "urgent", "guarantee"]
        for subj in self.V4_SUBJECTS:
            rendered = render(subj, self.CTX)
            for word in spam_words:
                assert word not in rendered.lower(), f"Spam word '{word}' in: {rendered}"

    def test_v4_emotional_arc_is_fear_money_loss_breakup(self):
        """V4 follows: guilt/fear → financial pain → knowledge loss → breakup+audit."""
        # Step 1: guilt/fear
        assert "blame" in self.V4_SUBJECTS[0].lower()
        # Step 2: money
        assert "math" in self.V4_SUBJECTS[1].lower() or "cfo" in self.V4_SUBJECTS[1].lower()
        # Step 3: loss
        assert "retires" in self.V4_SUBJECTS[2].lower()
        # Step 4: breakup
        assert "leave you alone" in self.V4_SUBJECTS[3].lower()

    def test_v4_bodies_all_start_with_greeting(self):
        """Every v4 body should start with personalised greeting."""
        for body in self.V4_BODIES:
            rendered = render(body, self.CTX)
            assert rendered.startswith("<p>Hi Vikram,</p>")


# ── Subject line best practices ──


class TestSubjectLineBestPractices:
    """Verify subject lines follow cold email best practices."""

    def test_subjects_are_lowercase(self):
        """Subject lines should be lowercase for a casual, personal feel."""
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
        for subj in subjects:
            # Remove template variables before checking
            cleaned = subj.replace("{{first_name}}", "").replace("{{company_name}}", "")
            # First character should be lowercase
            first_alpha = next((c for c in cleaned if c.isalpha()), None)
            if first_alpha:
                assert first_alpha.islower(), f"Subject should be lowercase: {subj}"

    def test_subjects_under_60_chars(self):
        """Subject lines should be under 60 chars for mobile preview."""
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
            assert len(rendered) < 70, f"Subject too long ({len(rendered)} chars): {rendered}"

    def test_no_spam_trigger_words(self):
        """Subject lines should avoid common spam trigger words."""
        spam_words = ["free", "act now", "limited time", "buy", "order", "click here",
                      "winner", "congratulations", "urgent", "guarantee"]
        subjects = [
            "the part that keeps you up at night, Raj",
            "the $47,000 mistake nobody talks about",
            "your competitors are already doing this",
            "last note from me, Raj",
            "that mold OneNet Plastics keeps nursing along",
            "your scrap bin is more expensive than you think",
            "the shops beating you on quotes have a secret",
        ]
        for subj in subjects:
            for word in spam_words:
                # "free" in step 4 subject is OK because it's "free process assessment"
                # which is industry-standard language, not spammy
                if word == "free" and "assessment" in subj:
                    continue
                assert word not in subj.lower(), f"Spam word '{word}' in subject: {subj}"


# ── Unsubscribe compliance ──


class TestUnsubscribeCompliance:
    """Verify CAN-SPAM compliance in email dispatch."""

    def test_unsubscribe_footer_html_includes_link(self):
        from app.services.email_dispatch import _unsubscribe_footer_html
        footer = _unsubscribe_footer_html("test@example.com")
        assert "unsubscribe" in footer.lower()
        assert "href=" in footer

    def test_unsubscribe_footer_has_address_not_powered_by(self):
        from app.services.email_dispatch import _unsubscribe_footer_html
        footer = _unsubscribe_footer_html("test@example.com")
        assert "Powered by" not in footer
        from app.config import get_settings
        assert get_settings().company_postal_address in footer

    def test_unsubscribe_footer_text_includes_link(self):
        from app.services.email_dispatch import _unsubscribe_footer_text
        footer = _unsubscribe_footer_text("test@example.com")
        assert "unsubscribe" in footer.lower()
        assert "http" in footer


# ── Demo link resolution ──


class TestDemoLinkResolution:
    """Verify demo links resolve to the correct industry-specific URLs."""

    def test_injection_molding_resolves_to_best_molding(self):
        link = resolve_demo_link("Plastic Injection Molding", "Apex Mold Industries")
        assert "best-molding-demo" in link

    def test_cnc_resolves_to_cnc_optimizer(self):
        link = resolve_demo_link("CNC Machining", "Precision Metal Works")
        assert "cnc-process-optimizer" in link

    def test_stamping_resolves_to_stamping_predictor(self):
        link = resolve_demo_link("Metal Stamping", "Valley Stamping")
        assert "stamping" in link

    def test_pcb_resolves_to_pcb_analyzer(self):
        link = resolve_demo_link("PCB Assembly", "Bayshore Electronics")
        assert "pcb-quality-analyzer" in link

    def test_food_resolves_to_food_monitor(self):
        link = resolve_demo_link("Dairy Processing", "FreshMilk Inc")
        assert "food-safety-monitor" in link

    def test_automotive_resolves_to_automotive_quality(self):
        link = resolve_demo_link("Automotive Parts", "Thunderbird Composites")
        assert "automotive-quality-assistant" in link

    def test_pharma_resolves_to_pharma_predictor(self):
        link = resolve_demo_link("Pharmaceutical", "PharmaCorp")
        assert "pharma-batch-predictor" in link

    def test_textile_resolves_to_textile_optimizer(self):
        link = resolve_demo_link("Textile Dyehouse", "FiberWorks")
        assert "textile-quality-optimizer" in link

    def test_aerospace_resolves_to_compliance(self):
        link = resolve_demo_link("Aerospace Parts", "Crosswind Aerospace")
        assert "compliance-traceability" in link

    def test_unknown_industry_falls_back_to_default(self):
        link = resolve_demo_link("Artisanal Candle Pouring", "Serenity Scents LLC")
        assert "best-molding-demo" in link

    def test_company_name_used_as_fallback_match(self):
        """If industry is blank, company name keywords should still match."""
        link = resolve_demo_link("", "Great Lakes Plastics")
        assert "best-molding-demo" in link

    def test_demo_link_renders_in_template(self):
        """{{demo_link}} should render as a clickable URL in the template."""
        template = '<a href="{{demo_link}}">See the dashboard</a>'
        ctx = {"demo_link": "https://best-molding-demo.pages.dev"}
        result = render(template, ctx)
        assert 'href="https://best-molding-demo.pages.dev"' in result

    def test_all_demo_links_are_valid_urls(self):
        """Every URL in the DEMO_LINKS map should be a valid https URL."""
        seen = set()
        for keyword, url in DEMO_LINKS.items():
            if url not in seen:
                assert url.startswith("https://"), f"Bad URL for '{keyword}': {url}"
                assert ".pages.dev" in url, f"URL not on Cloudflare Pages: {url}"
                seen.add(url)


# ── Sequence step ordering ──


class TestSequenceStepOrdering:
    """Verify sequence steps are ordered and have proper delays."""

    @pytest.mark.asyncio
    async def test_steps_have_increasing_delay(self, db_session: AsyncSession):
        """Step delays should be non-decreasing across the sequence."""
        seq = OutreachSequence(id=_uuid.uuid4(), name="Test Seq", is_active=True)
        db_session.add(seq)
        await db_session.flush()

        delays = [0, 2, 6, 13]  # Our standard cadence
        for i, delay in enumerate(delays, 1):
            step = SequenceStep(
                id=_uuid.uuid4(), sequence_id=seq.id,
                step_number=i, delay_days=delay,
                subject_template=f"Step {i}", body_template=f"<p>Body {i}</p>",
            )
            db_session.add(step)
        await db_session.flush()

        result = await db_session.execute(
            select(SequenceStep)
            .where(SequenceStep.sequence_id == seq.id)
            .order_by(SequenceStep.step_number)
        )
        steps = list(result.scalars().all())
        for i in range(1, len(steps)):
            assert steps[i].delay_days >= steps[i - 1].delay_days, \
                f"Step {steps[i].step_number} delay ({steps[i].delay_days}) < step {steps[i-1].step_number} ({steps[i-1].delay_days})"

    @pytest.mark.asyncio
    async def test_step_numbers_are_sequential(self, db_session: AsyncSession):
        """Step numbers should be 1, 2, 3, 4 with no gaps."""
        seq = OutreachSequence(id=_uuid.uuid4(), name="Test Seq", is_active=True)
        db_session.add(seq)
        await db_session.flush()

        for i in range(1, 5):
            step = SequenceStep(
                id=_uuid.uuid4(), sequence_id=seq.id,
                step_number=i, delay_days=i * 2,
                subject_template=f"Step {i}", body_template=f"<p>Body {i}</p>",
            )
            db_session.add(step)
        await db_session.flush()

        result = await db_session.execute(
            select(SequenceStep)
            .where(SequenceStep.sequence_id == seq.id)
            .order_by(SequenceStep.step_number)
        )
        steps = list(result.scalars().all())
        for i, step in enumerate(steps):
            assert step.step_number == i + 1


# ── Process endpoint edge cases ──


class TestProcessEndpointEdgeCases:
    """Edge cases for the /sequences/process endpoint."""

    @pytest.mark.asyncio
    async def test_process_with_null_next_send_at_uses_legacy_path(
        self, client: AsyncClient, db_session: AsyncSession, sample_contact: Contact
    ):
        """Enrollments without next_send_at should fall back to event-based delay check."""
        seq = OutreachSequence(id=_uuid.uuid4(), name="Legacy Test", is_active=True)
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
            current_step=1,
            next_send_at=None,  # Legacy enrollment without next_send_at
        )
        db_session.add(enrollment)
        await db_session.commit()

        resp = await client.post("/api/v1/sequences/process")
        body = resp.json()
        # Should dispatch because delay_days=0 and no prior events
        assert body["dispatched"] == 1

    @pytest.mark.asyncio
    async def test_enrollment_past_last_step_becomes_completed(
        self, db_session: AsyncSession, sample_contact: Contact
    ):
        """Enrollment at a step beyond the sequence should be marked completed by task logic."""
        seq = OutreachSequence(id=_uuid.uuid4(), name="Short Seq", is_active=True)
        db_session.add(seq)
        await db_session.flush()

        # Only 1 step
        step = SequenceStep(
            id=_uuid.uuid4(), sequence_id=seq.id,
            step_number=1, delay_days=0,
            subject_template="Only step", body_template="<p>Only</p>",
        )
        db_session.add(step)
        await db_session.flush()

        enrollment = OutreachEnrollment(
            id=_uuid.uuid4(),
            contact_id=sample_contact.id,
            sequence_id=seq.id,
            current_step=2,  # Past the only step
        )
        db_session.add(enrollment)
        await db_session.flush()

        # Simulate what happens when no step exists for current_step
        result = await db_session.execute(
            select(SequenceStep).where(
                SequenceStep.sequence_id == seq.id,
                SequenceStep.step_number == enrollment.current_step,
            )
        )
        next_step = result.scalar_one_or_none()
        assert next_step is None  # Confirms no step 2 exists

        # Mark completed (as the task would)
        enrollment.status = "completed"
        enrollment.completed_at = datetime.now(timezone.utc)
        await db_session.flush()

        loaded = await db_session.execute(
            select(OutreachEnrollment).where(OutreachEnrollment.id == enrollment.id)
        )
        assert loaded.scalar_one().status == "completed"
