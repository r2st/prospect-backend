"""SQLAlchemy models package.

Importing this module must register **every** mapped class on
``Base.metadata``. That is not a tidiness rule. ``company_group`` was missing
from the list below and from ``alembic/env.py``, so its table was absent from
the metadata that migration 078 diffed when it added ``org_id`` to 71 tables —
which is the whole reason ``company_groups`` spent two years as a namespace
shared by every tenant on the platform (see
``app.models.company_group`` and migration 111).

``tests/test_org_scoping_coverage.py`` now imports the package's modules
directly and fails if any mapped class is missing from here, so the blind spot
cannot silently reopen.
"""

from app.models.user import User
from app.models.company import Company
from app.models.company_group import CompanyGroup
from app.models.contact import Contact
from app.models.lead_score import LeadScore
from app.models.consent import ConsentRecord
from app.models.sequence import OutreachSequence, SequenceStep
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.models.contact_engagement import ContactEngagementProfile
from app.models.interaction import Interaction
from app.models.email_template import EmailTemplate, EmailTemplateAttachment, EmailSend
from app.models.follow_up import FollowUp
from app.models.gmail_account import GmailAccount
from app.models.smtp_account import SmtpAccount
from app.models.company_news import CompanyNews
from app.models.deal import Deal
from app.models.deal_activity import DealActivity
from app.models.deal_note import DealNote
from app.models.deal_progression_agent import DealProgressionAgent, DealAction
from app.models.task import Task
from app.models.client_health import ClientHealthSnapshot
from app.models.email_reply import EmailReply
from app.models.email_signature import EmailSignature
from app.models.feedback import Feedback
from app.models.knowledge import IndustryContact, KnowledgeDocument
from app.models.call import CallTask, CallScript, CallNotification
from app.models.investor import Investor, InvestorInteraction, InvestorDocument
from app.models.email_sentiment import EmailSentimentScore, ABTestExperiment, EmailTestExchange
from app.models.intent_signal import IntentSignal
from app.models.rep_performance import RepPerformance
from app.models.competitor import Competitor, CompetitorMention
from app.models.company_research import CompanyResearch
from app.models.conversation_analysis import ConversationAnalysis
from app.models.deal_prediction import DealPrediction
from app.models.rep_profile import RepProfile
from app.models.suggestion_outcome import SuggestionOutcome
from app.models.tenant import Organization, Tenant, TenantUsage
from app.models.integration import Integration, WebhookSubscription, WebhookEvent
from app.models.workflow import Workflow, WorkflowRun
from app.models.prospecting import IcpProfile, ProspectingRun, ProspectCandidate
from app.models.api_credit import ApiCreditLedger
from app.models.discovery_search import DiscoverySearch
from app.models.discovered_business import DiscoveredBusiness
from app.models.sender_health import SenderHealth, SenderHealthSnapshot
from app.models.sender_warmup import SenderWarmupPlan
from app.models.suppression import SuppressionEntry
from app.models.danger_signal import DangerSignalEvent
from app.models.domain_health import DomainHealth
from app.models.bounce_record import BounceRecord
from app.models.complaint_record import ComplaintRecord
from app.models.ai_email_draft import AiEmailDraft
from app.models.ab_test import AbTest, AbTestVariant, AbTestAssignment
from app.models.content_embedding import ContentEmbedding
from app.models.call_recording import CallRecording
from app.models.orchestration import OrchestrationPolicy
from app.models.channel_execution import ChannelStepExecution
from app.models.postmaster import PostmasterAccount, PostmasterMetric
from app.models.relationship import StakeholderMap, OrgRelationship
from app.models.meeting import MeetingBrief
from app.models.meeting_booking import BookedMeeting, MeetingBookingAgent
from app.models.agent_daily_summary import AgentDailySummary
from app.models.client_success_agent import ClientSuccessAgent, ClientSuccessAction
from app.models.system_setting import SystemSetting
from app.models.referral import ReferralAgent, Referral, ReferralAction
from app.models.social_listening_agent import (
    SocialListeningAgent,
    SignalEvent,
    MonitoredCompany,
)
from app.models.inbox_thread_state import InboxThreadState
from app.models.inbox_label import InboxLabel, InboxThreadLabel
from app.models.api_key import ApiKey
from app.models.app_credential import AppCredential
from app.models.org_llm_key import OrgLlmKey
from app.models.discovery_document import DiscoveryDocument
from app.models.campaign import Campaign, CampaignDailyLog, CampaignProspect

__all__ = [
    "SystemSetting",
    "User",
    "Company",
    "CompanyGroup",
    "Contact",
    "LeadScore",
    "ConsentRecord",
    "OutreachSequence",
    "SequenceStep",
    "OutreachEnrollment",
    "OutreachEvent",
    "ContactEngagementProfile",
    "Interaction",
    "EmailTemplate",
    "EmailTemplateAttachment",
    "EmailSend",
    "FollowUp",
    "GmailAccount",
    "SmtpAccount",
    "CompanyNews",
    "Deal",
    "DealActivity",
    "DealNote",
    "DealProgressionAgent",
    "DealAction",
    "Task",
    "ClientHealthSnapshot",
    "EmailSignature",
    "Feedback",
    "EmailReply",
    "IndustryContact",
    "KnowledgeDocument",
    "CallTask",
    "CallScript",
    "CallNotification",
    "Investor",
    "InvestorInteraction",
    "InvestorDocument",
    "EmailSentimentScore",
    "ABTestExperiment",
    "EmailTestExchange",
    "CompanyResearch",
    "ConversationAnalysis",
    "DealPrediction",
    "RepProfile",
    "SuggestionOutcome",
    "Organization",
    "Tenant",
    "TenantUsage",
    "Integration",
    "WebhookSubscription",
    "WebhookEvent",
    "Workflow",
    "WorkflowRun",
    "IcpProfile",
    "ProspectingRun",
    "ProspectCandidate",
    "ApiCreditLedger",
    "DiscoverySearch",
    "DiscoveredBusiness",
    "SenderHealth",
    "SenderHealthSnapshot",
    "SenderWarmupPlan",
    "SuppressionEntry",
    "DangerSignalEvent",
    "DomainHealth",
    "BounceRecord",
    "ComplaintRecord",
    "AiEmailDraft",
    "AbTest",
    "AbTestVariant",
    "AbTestAssignment",
    "ContentEmbedding",
    "CallRecording",
    "OrchestrationPolicy",
    "ChannelStepExecution",
    "PostmasterAccount",
    "PostmasterMetric",
    "StakeholderMap",
    "OrgRelationship",
    "MeetingBrief",
    "MeetingBookingAgent",
    "BookedMeeting",
    "IntentSignal",
    "RepPerformance",
    "Competitor",
    "CompetitorMention",
    "AgentDailySummary",
    "ReferralAgent",
    "Referral",
    "ReferralAction",
    "SocialListeningAgent",
    "SignalEvent",
    "MonitoredCompany",
    "ClientSuccessAgent",
    "ClientSuccessAction",
    "InboxThreadState",
    "InboxLabel",
    "InboxThreadLabel",
    "ApiKey",
    "AppCredential",
    "OrgLlmKey",
    "DiscoveryDocument",
    "Campaign",
    "CampaignProspect",
    "CampaignDailyLog",
]
