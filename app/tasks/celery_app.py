"""Celery application configuration."""

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "landline",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=300,
    task_time_limit=600,
    task_routes={
        "app.tasks.scraping.*": {"queue": "scraping"},
        "app.tasks.enrichment.*": {"queue": "enrichment"},
        "app.tasks.outreach.*": {"queue": "outreach"},
        "app.tasks.scheduled_sends.*": {"queue": "outreach"},
        "app.tasks.campaigns.*": {"queue": "outreach"},
        "app.tasks.news.*": {"queue": "enrichment"},
        "app.tasks.followups.*": {"queue": "enrichment"},
        "app.tasks.calls.*": {"queue": "enrichment"},
        "app.tasks.research.*": {"queue": "enrichment"},
        "app.tasks.conversation.*": {"queue": "enrichment"},
        "app.tasks.intent.*": {"queue": "enrichment"},
        "app.tasks.coaching.*": {"queue": "enrichment"},
        "app.tasks.competitive.*": {"queue": "enrichment"},
        "app.tasks.orchestration.*": {"queue": "outreach"},
        "app.tasks.pipeline_predictions.*": {"queue": "enrichment"},
        "app.tasks.lead_routing.*": {"queue": "enrichment"},
        "app.tasks.smart_suggestions_task.*": {"queue": "enrichment"},
        "app.tasks.prospecting.*": {"queue": "enrichment"},
        "app.tasks.client_health.*": {"queue": "enrichment"},
        "app.tasks.client_success.*": {"queue": "enrichment"},
        "app.tasks.deliverability.*": {"queue": "enrichment"},
        "app.tasks.bounces.*": {"queue": "enrichment"},
        "app.tasks.complaints.*": {"queue": "enrichment"},
        "app.tasks.danger_signals.*": {"queue": "enrichment"},
        "app.tasks.domain_health.*": {"queue": "enrichment"},
        "app.tasks.postmaster.*": {"queue": "enrichment"},
        "app.tasks.rag.*": {"queue": "enrichment"},
        "app.tasks.voice.*": {"queue": "enrichment"},
        "app.tasks.replies.*": {"queue": "enrichment"},
        "app.tasks.ab_testing.*": {"queue": "enrichment"},
        "app.tasks.send_timing.*": {"queue": "enrichment"},
        "app.tasks.verification.*": {"queue": "enrichment"},
        "app.tasks.places_discovery.*": {"queue": "enrichment"},
        "app.tasks.phone_enrichment.*": {"queue": "enrichment"},
        "app.tasks.deal_agent.*": {"queue": "enrichment"},
        "app.tasks.social_listening.*": {"queue": "enrichment"},
        "app.tasks.meeting_booking.*": {"queue": "enrichment"},
        "app.tasks.inbox.*": {"queue": "enrichment"},
        "app.tasks.lead_monitor.*": {"queue": "enrichment"},
    },
    beat_schedule={
        # Credential overrides set from the dashboard are applied per process,
        # so without this every *other* worker would keep using the retired key
        # until the next deploy — the UI reporting success while the beat worker
        # went on failing. Five minutes: these change about once a year.
        "refresh-credentials": {
            "task": "app.tasks.credentials.refresh_credentials",
            "schedule": 300.0,
        },
        "process-outreach-queue": {
            "task": "app.tasks.outreach.process_outreach_queue",
            "schedule": 300.0,  # every 5 minutes
        },
        "check-email-replies": {
            "task": "app.tasks.outreach.check_email_replies",
            "schedule": 600.0,  # every 10 minutes
        },
        # Put "send later" batches on the wire. Nothing read
        # ``EmailSend.status == 'scheduled'`` before this entry existed, so the
        # compose endpoint's scheduling option queued mail that was never sent.
        # Every 5 minutes: the UI schedules to the minute, and a send window /
        # daily cap can defer a row past any one pass anyway.
        "send-due-scheduled-emails": {
            "task": "app.tasks.scheduled_sends.send_due_scheduled_emails",
            "schedule": 300.0,  # every 5 minutes
        },
        # Release AI-compose drafts that have sat in pending_review too long so a
        # missed review can't freeze an enrollment. Hourly is fine — the expiry
        # window is measured in hours (default 24h).
        "auto-expire-review-drafts": {
            "task": "app.tasks.outreach.auto_expire_review_drafts",
            "schedule": 3600.0,  # hourly
        },
        # Bring back sequences paused by an out-of-office autoresponder. Without
        # this the pause is a one-way door — the queue only dispatches 'active',
        # so anyone who ever sent an autoreply would leave outreach for good.
        # Hourly: return dates are day-granular, so an hour of lag costs nothing.
        "resume-out-of-office": {
            "task": "app.tasks.outreach.resume_out_of_office",
            "schedule": 3600.0,  # hourly
        },
        "check-company-news": {
            "task": "app.tasks.news.check_company_news_batch",
            "schedule": 21600.0,  # every 6 hours
        },
        # Feature 6 (signal-based timing): react to fresh funding/hire/expansion
        # signals by surfacing a company's un-enrolled prospects for priority
        # outreach. Runs every 2 hours so a signal is acted on well inside its
        # freshness window (default 48h), staggered off the 6-hourly news sweep.
        "enroll-on-company-signals": {
            "task": "app.tasks.news.enroll_on_company_signals",
            "schedule": 7200.0,  # every 2 hours
        },
        "auto-enrich-companies": {
            "task": "app.tasks.enrichment.auto_enrich_companies",
            "schedule": 3600.0,  # every hour
        },
        "auto-score-new-leads": {
            "task": "app.tasks.enrichment.auto_score_leads",
            "schedule": 1800.0,  # every 30 minutes
        },
        "generate-stale-lead-followups": {
            "task": "app.tasks.followups.generate_stale_lead_followups",
            "schedule": 14400.0,  # every 4 hours
        },
        "rescore-active-leads": {
            "task": "app.tasks.followups.rescore_active_leads",
            "schedule": 43200.0,  # twice a day
        },
        # Weekly lead-quality report (grade distribution, qualified vs disqualified).
        "weekly-lead-quality-report": {
            "task": "app.tasks.followups.generate_weekly_lead_quality_report",
            "schedule": crontab(hour=6, minute=15, day_of_week=1),  # Mondays 6:15 AM UTC
        },
        "sync-email-replies": {
            "task": "app.tasks.outreach.sync_email_replies",
            "schedule": 300.0,  # every 5 minutes
        },
        # --- Cold-call notification system ---
        "generate-morning-call-queue": {
            "task": "app.tasks.calls.generate_morning_call_queue",
            "schedule": crontab(hour=8, minute=0),  # daily at 8 AM UTC
        },
        "send-precall-reminders": {
            "task": "app.tasks.calls.send_precall_reminders",
            "schedule": 300.0,  # every 5 minutes (15-min look-ahead window)
        },
        "send-call-followup-reminders": {
            "task": "app.tasks.calls.send_followup_reminders",
            "schedule": 900.0,  # every 15 minutes
        },
        "detect-hot-leads": {
            "task": "app.tasks.calls.detect_hot_leads",
            "schedule": 1800.0,  # every 30 minutes
        },
        "flag-missed-calls": {
            "task": "app.tasks.calls.flag_missed_calls",
            "schedule": crontab(hour=18, minute=0),  # end of day, 6 PM UTC
        },
        # --- Phase 1 AI features ---
        # 1.4: backfill research briefs for active-pipeline companies missing one.
        "research-stale-companies": {
            "task": "app.tasks.research.research_stale_companies",
            "schedule": 21600.0,  # every 6 hours
        },
        # 1.5: analyze active email threads for momentum / next-best-message.
        "analyze-active-threads": {
            "task": "app.tasks.conversation.analyze_active_threads",
            "schedule": 3600.0,  # hourly
        },
        # 2.1: daily buyer-intent signal scan across active-pipeline companies.
        "scan-intent-signals": {
            "task": "app.tasks.intent.scan_intent_signals",
            "schedule": crontab(hour=7, minute=0),  # daily at 7 AM UTC
        },
        # 2.4: weekly per-rep performance + AI coaching snapshots.
        "generate-weekly-coaching": {
            "task": "app.tasks.coaching.generate_weekly_coaching",
            "schedule": crontab(hour=6, minute=0, day_of_week=1),  # Mondays 6 AM UTC
        },
        # 2.6: monitor competitor news for battle-card-worthy mentions.
        "monitor-competitor-news": {
            "task": "app.tasks.competitive.monitor_competitor_news",
            "schedule": 43200.0,  # twice a day
        },
        # 1.1: weekly retrain + rescore of deal close-probability predictions.
        "rescore-deal-predictions": {
            "task": "app.tasks.pipeline_predictions.rescore_deal_predictions",
            "schedule": crontab(hour=5, minute=0, day_of_week=1),  # Mondays 5 AM UTC
        },
        # 1.3: keep propensity / signal-velocity current for active leads.
        "refresh-propensity-scores": {
            "task": "app.tasks.lead_routing.refresh_propensity_scores",
            "schedule": 1800.0,  # every 30 minutes
        },
        # 1.3: refresh rep win-rates, velocity, and capacity daily.
        "recompute-rep-profiles": {
            "task": "app.tasks.lead_routing.recompute_rep_profiles",
            "schedule": crontab(hour=4, minute=0),  # daily 4 AM UTC
        },
        # 1.6: produce the prioritized daily AI action plan.
        "generate-daily-suggestions": {
            "task": "app.tasks.smart_suggestions_task.generate_daily_suggestions",
            "schedule": crontab(hour=7, minute=30),  # daily 7:30 AM UTC
        },
        # --- Phase 3 AI features ---
        # 3.1: autonomous prospecting run (daily) + weekly ICP relearn.
        "run-prospecting-agent": {
            "task": "app.tasks.prospecting.run_prospecting_agent",
            "schedule": crontab(hour=9, minute=0),  # daily 9 AM UTC
        },
        "relearn-icp": {
            "task": "app.tasks.prospecting.relearn_icp",
            "schedule": crontab(hour=3, minute=30, day_of_week=1),  # Mondays 3:30 AM UTC
        },
        # 3.2: weekly predictive churn + expansion pass across clients.
        "predict-client-health": {
            "task": "app.tasks.client_health.predict_client_health",
            "schedule": crontab(hour=6, minute=30, day_of_week=1),  # Mondays 6:30 AM UTC
        },
        # Client Success Agent: daily scan of client health → enqueue check-in /
        # escalation / celebration / review-call actions (per active-agent org).
        "client-success-scan": {
            "task": "app.tasks.client_success.scan_client_health_task",
            "schedule": crontab(hour=7, minute=45),  # daily 7:45 AM UTC
        },
        # Client Success Agent: hourly send of approved auto check-ins.
        "client-success-execute": {
            "task": "app.tasks.client_success.execute_success_actions",
            "schedule": 3600.0,  # hourly
        },
        # 3.3: hourly sender deliverability health assessment.
        "assess-sender-health": {
            "task": "app.tasks.deliverability.assess_sender_health",
            "schedule": 3600.0,  # hourly
        },
        # Feature 1: step every sender warm-up plan forward one day so the ramped
        # daily cap reflects today. Just after the UTC date rollover.
        "advance-warmup-progress": {
            "task": "app.tasks.warmup.advance_warmup_progress",
            # Hourly, not daily: the ramp steps on the *sending* day
            # (send_day.local_date), whose midnight isn't 00:00 UTC. advance_all
            # is idempotent — a pass inside the same local day changes nothing.
            "schedule": crontab(minute=10),
        },
        # Ingest delivery-failure notifications from the mailbox. Gmail is the
        # primary transport and has no bounce webhook — a DSN just lands in the
        # inbox — so this sweep is the only thing that reaches bounce_handler
        # automatically. Every 15 minutes: fast enough to stop mailing a dead
        # address early in a run, cheap because it dedups on message id.
        "ingest-bounces": {
            "task": "app.tasks.bounces.ingest_bounces",
            "schedule": 900.0,  # every 15 minutes
        },
        # Ingest complaint / feedback-loop (FBL) reports from the mailbox. Gmail
        # exposes no FBL, so a provider abuse report (Yahoo/Outlook/AOL) that is
        # forwarded to a monitored mailbox is the only per-recipient complaint
        # signal — this sweep suppresses the complainer, same path as a hard
        # bounce. Every 15 minutes, deduped on message id like the bounce sweep.
        "ingest-complaints": {
            "task": "app.tasks.complaints.ingest_complaints",
            "schedule": 900.0,  # every 15 minutes
        },
        # Feature 2: auto-pause on danger signals — 24h bounce/complaint monitor.
        # Every 5 minutes so an acute spike (bad list, burning domain) is stopped
        # fast and cheaply.
        "monitor-danger-signals": {
            "task": "app.tasks.danger_signals.monitor_danger_signals",
            "schedule": 300.0,  # every 5 minutes
        },
        # 3.4: index new content into the RAG knowledge base.
        "index-new-content": {
            "task": "app.tasks.rag.index_new_content",
            "schedule": 21600.0,  # every 6 hours
        },
        # 2.2: autonomous multi-channel orchestration — decide the next touch for
        # every autonomous-mode enrollment.
        "run-orchestrator": {
            "task": "app.tasks.orchestration.run_orchestrator",
            "schedule": 900.0,  # every 15 minutes
        },
        # Daily autonomous-activity digest — 6 PM IST (12:30 UTC).
        "generate-daily-summary": {
            "task": "app.tasks.reporting.generate_daily_summary",
            "schedule": crontab(hour=12, minute=30),
        },
        # Feature 1: classify newly-arrived replies and route them (hot leads →
        # priority inbox, unsubscribes → suppression, OOO → re-queue).
        "classify-new-replies": {
            "task": "app.tasks.replies.classify_new_replies",
            "schedule": 300.0,  # every 5 minutes
        },
        # Feature 5: daily SPF/DKIM/DMARC + DNSBL sweep per sending domain,
        # scored, persisted, and alerted on change. Daily is plenty (DNS moves
        # slowly) and polite to the DNSBL zones. 5 AM UTC.
        "check-domain-health": {
            "task": "app.tasks.domain_health.check_domain_health",
            "schedule": crontab(hour=5, minute=30),
        },
        # P2 audit: daily Google Postmaster Tools pull (reputation, spam rate,
        # SPF/DKIM/DMARC pass rates, delivery errors) per verified domain. Gmail
        # has no FBL, so this is the only complaint signal for the primary
        # transport. Daily is plenty — Postmaster reports trailing daily windows.
        # 5 AM UTC, alongside the other daily deliverability sweeps.
        "pull-postmaster-metrics": {
            "task": "app.tasks.postmaster.pull_postmaster_metrics",
            "schedule": crontab(hour=5, minute=45),
        },
        # Feature 3: auto-evaluate running A/B tests and declare/promote winners
        # once they reach significance (or the hard cap). Daily is plenty — copy
        # tests accrue sample slowly. 4:30 AM UTC.
        "evaluate-ab-tests": {
            "task": "app.tasks.ab_testing.evaluate_ab_tests",
            "schedule": crontab(hour=4, minute=30),
        },
        # Feature 6: nightly recompute of per-contact engagement profiles (local
        # (dow, hour) buckets, best day/hour) + timezone backfill, feeding the
        # per-recipient send-time scheduling done inside process_outreach_queue.
        # 3 AM UTC — quiet hours, and ahead of the morning sending window.
        "refresh-engagement-profiles": {
            "task": "app.tasks.send_timing.refresh_engagement_profiles",
            "schedule": crontab(hour=3, minute=0),
        },
        # P3: verify never-verified contacts (Hunter, behind the free syntax+MX
        # pre-filter). Every 15 min so an imported/prospected contact is verified
        # well before its sequence's first send, without making import block on a
        # paid API call. Idempotent via contacts.email_verified_at.
        "verify-pending-contacts": {
            "task": "app.tasks.verification.verify_pending_contacts",
            "schedule": 900.0,  # every 15 minutes
        },
        # P3: re-verify addresses that have gone stale (default >90 days) —
        # people leave and mailboxes are retired. Weekly, oldest-first, and
        # limited to contacts we could still email, so it can't burn credits on
        # a list we'd never send to. 2 AM Sunday UTC.
        "reverify-stale-contacts": {
            "task": "app.tasks.verification.reverify_stale_contacts",
            "schedule": crontab(hour=2, minute=0, day_of_week=0),
        },
        # P4: re-run saved Google Places searches. Weekly rather than daily on
        # purpose — local business data changes slowly, so a daily cadence pays
        # seven times over for nearly the same results. 4 AM Monday UTC, clear
        # of the Sunday-night verification sweep.
        "run-places-discoveries": {
            "task": "app.tasks.places_discovery.run_scheduled_discoveries",
            "schedule": crontab(hour=4, minute=0, day_of_week=1),
        },
        # P4: recover emails for discovered businesses that have a website but
        # no address (Hunter domain-search, then the free scraper). Daily and
        # separate from discovery: this is the hop where the funnel narrows, and
        # it shouldn't be gated on a Places search succeeding. 5 AM UTC.
        "enrich-discovered-emails": {
            "task": "app.tasks.places_discovery.enrich_discovered_emails",
            "schedule": crontab(hour=5, minute=0),
        },
        # Warm-lead phone enrichment: find prospects showing real interest (3+
        # opens or any click/reply) that have no phone on file and enrich them via
        # Apollo, so the operator can call while the lead is warm. Every 30 min —
        # fast enough to catch a warm lead the same day, and each contact is
        # attempted at most once (phone_enriched_at), so the cadence is cheap.
        "enrich-warm-lead-phones": {
            "task": "app.tasks.phone_enrichment.enrich_warm_lead_phones",
            "schedule": 1800.0,  # every 30 minutes
        },
        # --- Deal Progression Agent ---
        # Daily sweep: find stalled deals and record the right nudge (follow-up,
        # suggestion, at-risk flag, or manager escalation) per org.
        "scan-deal-pipeline": {
            "task": "app.tasks.deal_agent.scan_deal_pipeline",
            "schedule": crontab(hour=7, minute=45),  # daily 7:45 AM UTC
        },
        # Hourly: send approved follow-ups and expire actions on closed deals.
        "execute-deal-actions": {
            "task": "app.tasks.deal_agent.execute_pending_actions",
            "schedule": 3600.0,  # hourly
        },
        # --- Referral & Warm-Intro agent ---
        # Weekly scan for happy, long-tenured clients to ask for referrals.
        # Mondays 7 AM UTC, clear of the other Monday-morning report tasks.
        "scan-referral-candidates": {
            "task": "app.tasks.referral_agent.scan_referral_candidates",
            "schedule": crontab(hour=7, minute=0, day_of_week=1),
        },
        # Daily nudge on referral asks that have gone unanswered past threshold.
        "followup-pending-referrals": {
            "task": "app.tasks.referral_agent.followup_pending_referrals",
            "schedule": crontab(hour=9, minute=30),  # daily 9:30 AM UTC
        },
        # Daily check for referred contacts who have since closed a won deal.
        "check-referral-conversions": {
            "task": "app.tasks.referral_agent.check_referral_conversions",
            "schedule": crontab(hour=10, minute=0),  # daily 10 AM UTC
        },
        # Social Listening Agent: hourly per-org signal scan. Each org's own
        # scan_frequency_hours gates whether it's actually due this tick, so a
        # daily-cadence org isn't re-scanned every hour — the beat just wakes the
        # sweep often enough to honour the shortest configured cadence.
        "scan-social-signals": {
            "task": "app.tasks.social_listening.scan_signals",
            "schedule": 3600.0,  # hourly
        },
        # Escalate stored high-relevance signals to warm leads (hourly), covering
        # signals stored before an org flipped auto-create / changed thresholds.
        "process-new-social-signals": {
            "task": "app.tasks.social_listening.process_new_signals",
            "schedule": 3600.0,  # hourly
        },
        # Meeting Booking Agent: poll proposed meetings for the prospect's slot
        # pick and confirm/re-offer. Every 15 minutes — fast enough to reply while
        # the prospect is still engaged, and idempotent (acts only on replies
        # newer than the last logged turn).
        "check-booking-replies": {
            "task": "app.tasks.meeting_booking.check_booking_replies",
            "schedule": 900.0,  # every 15 minutes
        },
        # Meeting Booking Agent: send the ~1h pre-meeting reminder for confirmed
        # meetings and close out elapsed ones. Every 30 minutes with a 90-minute
        # look-ahead → exactly one reminder per meeting, ~1h out.
        "send-meeting-reminders": {
            "task": "app.tasks.meeting_booking.send_meeting_reminders",
            "schedule": 1800.0,  # every 30 minutes
        },
        # Inbox hygiene: null out elapsed thread snoozes so the column stays
        # clean. Resurfacing is already automatic (the query treats an elapsed
        # snooze as not-snoozed); this just tidies. Every 5 minutes.
        "clear-expired-snoozes": {
            "task": "app.tasks.inbox.clear_expired_snoozes",
            "schedule": 300.0,  # every 5 minutes
        },
        # Lead-gen health: poll the tracking tallies (the /leads/stats numbers)
        # and log a LEADGEN ALERT when today saw zero sends or zero AI-composed
        # copy. 21:30 UTC — after the US-business-hours send window has had its
        # full day, so "0 sends today" means the pipeline really was dark, not
        # that the day hasn't started.
        "check-lead-gen-health": {
            "task": "app.tasks.lead_monitor.check_lead_gen_health",
            "schedule": crontab(hour=21, minute=30),
        },
        # Uploaded discovery documents keep their bytes for 30 days; the
        # extracted candidates stay. Daily is plenty for a 30-day window.
        # Autopilot Outreach: reconcile, discover (daily), draft, and schedule
        # paced sends for every active campaign.
        "run-campaign-scheduler": {
            "task": "app.tasks.campaigns.run_campaign_scheduler",
            "schedule": 900.0,  # every 15 minutes
        },
        # Campaign sends whose worker died between claim and outcome.
        "reap-stuck-campaign-sends": {
            "task": "app.tasks.campaigns.reap_stuck_campaign_sends",
            "schedule": 600.0,  # every 10 minutes
        },
        "purge-expired-discovery-document-files": {
            "task": "app.tasks.discovery_documents.purge_expired_document_files",
            "schedule": crontab(hour=3, minute=40),
        },
    },
)

# Explicitly include task modules so autodiscover finds them
celery_app.autodiscover_tasks(["app.tasks"], force=True)

# Force-import all task modules to guarantee registration
import app.tasks.outreach  # noqa: F401
# Registers refresh_credentials — the "refresh-credentials" beat entry.
# Unregistered, dashboard credential rotations never reach this worker.
import app.tasks.credentials  # noqa: F401
# Registers send_due_scheduled_emails — the "send-due-scheduled-emails" beat
# entry above. Without the force-import the task is unregistered and every
# scheduled batch stays queued forever, exactly as it did before it existed.
import app.tasks.scheduled_sends  # noqa: F401
import app.tasks.warmup  # noqa: F401
import app.tasks.news  # noqa: F401
import app.tasks.scraping  # noqa: F401
import app.tasks.enrichment  # noqa: F401
import app.tasks.followups  # noqa: F401
import app.tasks.calls  # noqa: F401
import app.tasks.research  # noqa: F401
import app.tasks.conversation  # noqa: F401
import app.tasks.intent  # noqa: F401
import app.tasks.coaching  # noqa: F401
import app.tasks.competitive  # noqa: F401
import app.tasks.orchestration  # noqa: F401
import app.tasks.pipeline_predictions  # noqa: F401
import app.tasks.lead_routing  # noqa: F401
import app.tasks.smart_suggestions_task  # noqa: F401
import app.tasks.prospecting  # noqa: F401
import app.tasks.client_health  # noqa: F401
import app.tasks.deliverability  # noqa: F401
import app.tasks.bounces  # noqa: F401
import app.tasks.complaints  # noqa: F401
import app.tasks.danger_signals  # noqa: F401
import app.tasks.domain_health  # noqa: F401
import app.tasks.postmaster  # noqa: F401
import app.tasks.rag  # noqa: F401
import app.tasks.voice  # noqa: F401
import app.tasks.replies  # noqa: F401
import app.tasks.ab_testing  # noqa: F401
import app.tasks.send_timing  # noqa: F401
import app.tasks.verification  # noqa: F401
import app.tasks.places_discovery  # noqa: F401
import app.tasks.phone_enrichment  # noqa: F401
import app.tasks.deal_agent  # noqa: F401
import app.tasks.referral_agent  # noqa: F401
import app.tasks.social_listening  # noqa: F401
import app.tasks.meeting_booking  # noqa: F401
import app.tasks.lead_monitor  # noqa: F401
import app.tasks.client_success  # noqa: F401
# Registers generate_daily_summary — referenced by the "daily-summary" beat
# schedule above. Without this force-import the task is never registered and the
# beat-dispatched summary silently fails with an unregistered-task error.
import app.tasks.reporting  # noqa: F401
import app.tasks.inbox  # noqa: F401
# Registers the 30-day purge of uploaded discovery document files.
import app.tasks.discovery_documents  # noqa: F401
# Registers the Autopilot Outreach scheduler and per-prospect send task.
import app.tasks.campaigns  # noqa: F401
