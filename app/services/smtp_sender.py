"""Send outreach email through a configured SMTP server.

Used for sender accounts that aren't connected via Google OAuth — Outlook /
Office 365, custom mail hosts, or any provider exposing authenticated SMTP.

This module is synchronous (``smtplib`` is blocking); async callers should
invoke it via ``asyncio.to_thread`` (see ``email_dispatch``).
"""

import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, formatdate, make_msgid
from typing import Optional, Sequence

from app.services import crypto
from app.services.template_attachments import OutgoingAttachment, build_mime_body

logger = logging.getLogger(__name__)

# Connections to third-party SMTP hosts can be slow; cap so a wedged host can't
# hang a request/worker indefinitely.
_SMTP_TIMEOUT = 20


def _apply_unsubscribe_headers(
    message, list_unsubscribe_url: Optional[str], sender_email: Optional[str] = None
) -> None:
    """Add RFC 2369 List-Unsubscribe + RFC 8058 one-click headers.

    Mirrors ``gmail_sender._apply_unsubscribe_headers`` so SMTP-sent mail carries
    the same one-click unsubscribe affordance Gmail/Yahoo require of bulk senders
    (June 2024). Without these headers on the SMTP transport, sequences sent from
    Outlook/custom hosts were shipping with no List-Unsubscribe at all.

    The header pair itself comes from :func:`opt_out.unsubscribe_headers` — one
    builder for all three transports, since the third one shipped without it.
    """
    from app.services.opt_out import unsubscribe_headers

    for name, value in unsubscribe_headers(list_unsubscribe_url, sender_email).items():
        message[name] = value


class SmtpSender:
    """Sends email through an authenticated SMTP server for one account.

    Construct with a :class:`~app.models.smtp_account.SmtpAccount`. The stored
    password (decrypted) is used to authenticate on each send.
    """

    def __init__(self, account):
        self.account = account
        self.sender_email = account.email
        self.display_name = account.display_name

    def _connect(self) -> smtplib.SMTP:
        password = crypto.decrypt(self.account.password_encrypted)
        context = ssl.create_default_context()
        if self.account.use_ssl:
            server = smtplib.SMTP_SSL(
                self.account.host, self.account.port, timeout=_SMTP_TIMEOUT, context=context
            )
        else:
            server = smtplib.SMTP(self.account.host, self.account.port, timeout=_SMTP_TIMEOUT)
            server.ehlo()
            if self.account.use_tls:
                server.starttls(context=context)
                server.ehlo()
        server.login(self.account.username, password)
        return server

    def verify(self) -> dict:
        """Test that the host/credentials authenticate. Returns {ok, detail}."""
        try:
            server = self._connect()
            try:
                server.noop()
            finally:
                try:
                    server.quit()
                except Exception:
                    pass
            return {"ok": True}
        except smtplib.SMTPAuthenticationError as e:
            detail = e.smtp_error.decode(errors="replace") if isinstance(e.smtp_error, bytes) else str(e)
            return {"ok": False, "detail": f"Authentication failed: {detail}"}
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    def _build_message(
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
    ) -> MIMEMultipart:
        message = build_mime_body(body_text, body_html, attachments)
        message["To"] = to_email
        message["From"] = formataddr((self.display_name or "", self.sender_email))
        message["Subject"] = subject
        # Gmail's API and SES stamp these themselves; an SMTP relay may not, and
        # mail with no Message-ID or Date is a classic spam-filter strike. The
        # id uses the sender's own domain so it matches the From address.
        message["Date"] = formatdate(usegmt=True)
        message["Message-ID"] = make_msgid(domain=self.sender_email.rsplit("@", 1)[-1] or None)
        if reply_to:
            message["Reply-To"] = reply_to
        # Threading headers so replies thread correctly in the recipient's client.
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
        if references:
            message["References"] = references
        _apply_unsubscribe_headers(message, list_unsubscribe_url, self.sender_email)
        return message

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
        """Send a single email. Mirrors GmailSender.send_email's return shape.

        ``thread_id`` is accepted for interface parity with ``GmailSender`` (Gmail
        threads by its own thread id) but is not used over SMTP — RFC threading is
        carried entirely by the ``In-Reply-To``/``References`` headers.
        """
        try:
            message = self._build_message(
                to_email, subject, body_html, body_text, reply_to,
                in_reply_to=in_reply_to, references=references,
                list_unsubscribe_url=list_unsubscribe_url,
                attachments=attachments,
            )
            server = self._connect()
            try:
                server.sendmail(self.sender_email, [to_email], message.as_string())
            finally:
                try:
                    server.quit()
                except Exception:
                    pass
            logger.info("SMTP email sent to %s as %s", to_email, self.sender_email)
            return {
                "status": "sent",
                "transport": "smtp",
                "sender": self.sender_email,
                # Ours, so a reply's In-Reply-To can be matched to this send.
                "message_id": message["Message-ID"],
            }
        except Exception as e:
            logger.error("Failed to send SMTP email to %s as %s: %s", to_email, self.sender_email, e)
            return {"status": "error", "transport": "smtp", "detail": str(e)}
