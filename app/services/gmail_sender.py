"""Send outreach email through the Gmail API as a connected user.

Sending via Gmail (rather than raw SES) means each message goes out through
Google's infrastructure as the connected user, so it inherits that domain's
SPF/DKIM/DMARC alignment — the practical reason these emails land in the inbox
instead of spam.

This module is synchronous (the Google client libraries are blocking); async
callers should invoke it via ``asyncio.to_thread``.
"""

import base64
import logging
from email.utils import formataddr
from typing import Optional, Sequence

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import get_settings
from app.services import crypto, google_oauth
from app.services.template_attachments import (
    OutgoingAttachment,
    apply_identity_headers,
    build_mime_body,
)

logger = logging.getLogger(__name__)

_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Minimal scopes needed for sending — avoids invalid_scope errors when
# the stored refresh token was authorized before gmail.readonly was added.
_SEND_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.send",
]


def _apply_unsubscribe_headers(
    message, list_unsubscribe_url: Optional[str], sender_email: Optional[str] = None
) -> None:
    """Add RFC 2369 List-Unsubscribe + RFC 8058 one-click headers.

    Offers both a one-click https URL and a mailto: address — Gmail/Yahoo
    bulk-sender guidelines since June 2024 recommend supporting both so mailbox
    providers can present a native "unsubscribe" affordance. The mailto defaults
    to the sending account's own address (a real, monitored, reply-synced
    mailbox) unless ``unsubscribe_mailto`` is explicitly configured.

    The header pair itself comes from :func:`opt_out.unsubscribe_headers` — one
    builder for all three transports, since the third one shipped without it.
    """
    from app.services.opt_out import unsubscribe_headers

    for name, value in unsubscribe_headers(list_unsubscribe_url, sender_email).items():
        message[name] = value


class GmailSender:
    """Sends email through the Gmail API for a single connected account.

    Construct with a :class:`~app.models.gmail_account.GmailAccount`. The stored
    refresh token (decrypted) is used to mint access tokens; google-auth
    refreshes transparently when needed.
    """

    def __init__(self, account):
        self.account = account
        self.sender_email = account.email
        self.display_name = account.display_name

    def _credentials(self) -> Credentials:
        settings = get_settings()
        refresh_token = crypto.decrypt(self.account.refresh_token_encrypted)
        access_token = None
        if self.account.access_token_encrypted:
            try:
                access_token = crypto.decrypt(self.account.access_token_encrypted)
            except Exception:
                access_token = None
        return Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri=_TOKEN_URI,
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=_SEND_SCOPES,
        )

    def _build_mime(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str],
        reply_to: Optional[str],
        list_unsubscribe_url: Optional[str] = None,
        attachments: Optional[Sequence[OutgoingAttachment]] = None,
    ) -> dict:
        message = build_mime_body(body_text, body_html, attachments)
        message["To"] = to_email
        message["From"] = formataddr((self.display_name or "", self.sender_email))
        message["Subject"] = subject
        apply_identity_headers(message, self.sender_email)
        if reply_to:
            message["Reply-To"] = reply_to
        _apply_unsubscribe_headers(message, list_unsubscribe_url, self.sender_email)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {"raw": raw}

    def send_email(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str] = None,
        reply_to: Optional[str] = None,
        thread_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        list_unsubscribe_url: Optional[str] = None,
        attachments: Optional[Sequence[OutgoingAttachment]] = None,
    ) -> dict:
        """Send a single email. Mirrors EmailSender.send_email's return shape.

        When ``thread_id``, ``in_reply_to``, and ``references`` are provided the
        message is sent as a reply within the existing Gmail thread.
        """
        try:
            creds = self._credentials()
            service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            body = self._build_mime(
                to_email, subject, body_html, body_text, reply_to,
                list_unsubscribe_url=list_unsubscribe_url,
                attachments=attachments,
            )

            # Add threading headers for replies
            if in_reply_to or references:
                # Re-build MIME with threading headers
                body = self._build_mime_threaded(
                    to_email, subject, body_html, body_text, reply_to,
                    in_reply_to=in_reply_to, references=references,
                    list_unsubscribe_url=list_unsubscribe_url,
                    attachments=attachments,
                )

            # Attach threadId so Gmail groups the message into the right thread
            if thread_id:
                body["threadId"] = thread_id

            sent = service.users().messages().send(userId="me", body=body).execute()
            gmail_message_id = sent.get("id")
            gmail_thread_id = sent.get("threadId")
            logger.info(
                "Gmail email sent to %s as %s — id: %s thread: %s",
                to_email, self.sender_email, gmail_message_id, gmail_thread_id,
            )
            return {
                "status": "sent",
                "message_id": gmail_message_id,
                "gmail_message_id": gmail_message_id,
                "gmail_thread_id": gmail_thread_id,
                "transport": "gmail",
                "sender": self.sender_email,
            }
        except HttpError as e:
            detail = getattr(e, "reason", None) or str(e)
            logger.error("Gmail API error sending to %s: %s", to_email, detail)
            return {"status": "error", "transport": "gmail", "detail": detail}
        except Exception as e:
            logger.error("Failed to send Gmail email to %s: %s", to_email, str(e))
            return {"status": "error", "transport": "gmail", "detail": str(e)}

    def _build_mime_threaded(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str],
        reply_to: Optional[str],
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        list_unsubscribe_url: Optional[str] = None,
        attachments: Optional[Sequence[OutgoingAttachment]] = None,
    ) -> dict:
        """Build MIME message with In-Reply-To and References headers for threading."""
        message = build_mime_body(body_text, body_html, attachments)
        message["To"] = to_email
        message["From"] = formataddr((self.display_name or "", self.sender_email))
        message["Subject"] = subject
        apply_identity_headers(message, self.sender_email)
        if reply_to:
            message["Reply-To"] = reply_to
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
        if references:
            message["References"] = references
        _apply_unsubscribe_headers(message, list_unsubscribe_url, self.sender_email)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {"raw": raw}
