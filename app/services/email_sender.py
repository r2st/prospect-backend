"""Amazon SES email sending service."""

import logging
from typing import Optional, Sequence

from jinja2.sandbox import SandboxedEnvironment


from app.config import get_settings
from app.services.template_attachments import OutgoingAttachment, build_mime_body

logger = logging.getLogger(__name__)
settings = get_settings()


class EmailSender:
    """Sends emails via Amazon SES with template rendering."""

    def __init__(self):
        self.sender = settings.ses_sender_email
        self._client = None

    @property
    def client(self):
        """Lazy-load boto3 SES client to avoid import-time failures in test environments."""
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "ses",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id or None,
                aws_secret_access_key=settings.aws_secret_access_key or None,
            )
        return self._client

    def render_template(self, template_str: str, context: dict) -> str:
        """Render a Jinja2 template string with the given context.

        Uses SandboxedEnvironment to prevent server-side template injection.
        """
        env = SandboxedEnvironment()
        tmpl = env.from_string(template_str)
        return tmpl.render(**context)

    def _build_unsubscribe_footer(self, contact_email: str) -> str:
        """CAN-SPAM compliant unsubscribe footer with signed token (no PII in URL).

        Both the token and the URL come from :mod:`app.services.opt_out`, which
        owns minting and verification. This was the second copy of both and it
        was wrong twice over: it signed with a hardcoded
        ``"landline-unsubscribe-secret"`` fallback the endpoint would never
        accept, and it pointed at ``landline.apprendtechnologies.com/unsubscribe``
        — a frontend route that does not exist. A recipient following this
        footer got a 404, and the token behind it would have failed to verify
        anyway. Only reachable today because both callers pass
        ``include_unsubscribe=False``; the default is ``True``.
        """
        from app.services.opt_out import make_unsubscribe_url

        unsubscribe_url = make_unsubscribe_url(contact_email)
        return (
            "\n\n—\n"
            f"Don't want these emails? Reply UNSUBSCRIBE or click: "
            f"{unsubscribe_url}\n"
            "Powered by DoAide Prospect — https://prospect.doaide.com"
        )

    def _send_raw(
        self,
        *,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str],
        reply_to: Optional[str],
        list_unsubscribe_url: Optional[str],
        attachments: Optional[Sequence[OutgoingAttachment]] = None,
    ) -> dict:
        """``send_raw_email`` with a MIME document, so headers can be set.

        Same body shape the structured call produces — ``multipart/alternative``
        with the plain-text part first, per RFC 2046 (clients pick the *last*
        part they can render, so text must precede HTML or a text-only client is
        the only one that sees the right thing).

        The mailto: half of ``List-Unsubscribe`` falls back to the SES sender
        identity, which is a real configured address here rather than a
        connected mailbox.
        """
        from app.services.opt_out import unsubscribe_headers

        message = build_mime_body(body_text, body_html, attachments)
        message["To"] = to_email
        message["From"] = self.sender
        message["Subject"] = subject
        if reply_to:
            message["Reply-To"] = reply_to
        for name, value in unsubscribe_headers(
            list_unsubscribe_url, self.sender
        ).items():
            message[name] = value

        return self.client.send_raw_email(
            Source=self.sender,
            Destinations=[to_email],
            RawMessage={"Data": message.as_string()},
        )

    def _send_structured(
        self,
        *,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str],
        reply_to: Optional[str],
    ) -> dict:
        """SES's ``send_email`` action — the shape this transport always used.

        No custom headers are possible here (see :meth:`_send_raw`); it is kept
        as its own method because it is now reached from two places: the ordinary
        no-URL call, and the fallback when the deployed IAM identity turns out
        not to be allowed to send raw.
        """
        message = {"Subject": {"Data": subject, "Charset": "UTF-8"}, "Body": {}}
        if body_html:
            message["Body"]["Html"] = {"Data": body_html, "Charset": "UTF-8"}
        if body_text:
            message["Body"]["Text"] = {"Data": body_text, "Charset": "UTF-8"}
        return self.client.send_email(
            Source=self.sender,
            Destination={"ToAddresses": [to_email]},
            Message=message,
            ReplyToAddresses=[reply_to] if reply_to else [],
        )

    @staticmethod
    def _is_authorization_error(exc: Exception) -> bool:
        """Is this SES refusing the *action*, rather than refusing the message?

        Narrow on purpose. A throttle, a rejected address or a verification
        failure will fail identically on either action, so retrying them on the
        structured call is a second wasted round trip and a second log line
        saying the same thing. An authorization failure is the one error whose
        cause is *which API call we made*.
        """
        error_response = getattr(exc, "response", None) or {}
        code = (error_response.get("Error") or {}).get("Code") or ""
        return code in {"AccessDenied", "AccessDeniedException", "AuthorizationError"}

    def send_email(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str] = None,
        reply_to: Optional[str] = None,
        include_unsubscribe: bool = True,
        list_unsubscribe_url: Optional[str] = None,
        attachments: Optional[Sequence[OutgoingAttachment]] = None,
    ) -> dict:
        """Send an email via SES.

        Returns dict with 'message_id' on success or 'error' on failure.

        ``list_unsubscribe_url`` adds the RFC 2369 / RFC 8058 header pair, which
        this transport could not carry at all: SES's ``send_email`` action takes
        a *structured* message (subject + body parts) and has nowhere to put a
        custom header, so the only way to set one is ``send_raw_email`` with a
        MIME document we build ourselves. That is why this path went without.

        Gmail and SMTP both set the headers, so the gap was invisible from every
        angle except the transport that fired **when neither of them was
        available** — the fallback that carries the volume on exactly the day an
        operator's mailboxes are down. Mail that reaches a bulk filter with no
        one-click affordance is rated as a bulk sender who doesn't offer one,
        and this fallback shares the sending domain's reputation with the rest.
        """
        if not settings.aws_access_key_id or not settings.aws_secret_access_key:
            logger.warning("AWS SES credentials not configured — email not sent")
            return {"status": "skipped", "reason": "no_ses_credentials"}

        if include_unsubscribe:
            footer = self._build_unsubscribe_footer(to_email)
            if body_text:
                body_text += footer
            if body_html:
                body_html += footer.replace("\n", "<br>")

        try:
            if attachments:
                # The structured action has no way to carry a file, so there is
                # no fallback here: a send without the attachment the template
                # promises is a different email, and it should fail as itself.
                response = self._send_raw(
                    to_email=to_email,
                    subject=subject,
                    body_html=body_html,
                    body_text=body_text,
                    reply_to=reply_to,
                    list_unsubscribe_url=list_unsubscribe_url,
                    attachments=attachments,
                )
            elif list_unsubscribe_url:
                try:
                    response = self._send_raw(
                        to_email=to_email,
                        subject=subject,
                        body_html=body_html,
                        body_text=body_text,
                        reply_to=reply_to,
                        list_unsubscribe_url=list_unsubscribe_url,
                    )
                except Exception as raw_exc:
                    if not self._is_authorization_error(raw_exc):
                        raise
                    # ``ses:SendRawEmail`` is a **separate IAM action** from
                    # ``ses:SendEmail``, and an identity provisioned before this
                    # transport learned to set headers holds only the latter.
                    # Setting the header pair is worth a great deal; it is not
                    # worth turning every fallback send into a hard failure on
                    # the day the operator's own mailboxes are down, which is
                    # the only day this transport runs. Send it without the
                    # headers and say exactly what to grant — the alternative is
                    # a silent full stop that reads as "SES is broken".
                    logger.error(
                        "SES rejected send_raw_email for lack of permission; "
                        "falling back to send_email WITHOUT the RFC 8058 "
                        "one-click unsubscribe headers. Grant ses:SendRawEmail "
                        "to the sending identity to restore them. (%s)",
                        raw_exc,
                    )
                    response = self._send_structured(
                        to_email=to_email,
                        subject=subject,
                        body_html=body_html,
                        body_text=body_text,
                        reply_to=reply_to,
                    )
            else:
                response = self._send_structured(
                    to_email=to_email,
                    subject=subject,
                    body_html=body_html,
                    body_text=body_text,
                    reply_to=reply_to,
                )
            message_id = response["MessageId"]
            logger.info("Email sent to %s — MessageId: %s", to_email, message_id)
            return {"status": "sent", "message_id": message_id}

        except Exception as e:
            # Handle boto3 ClientError specifically if boto3 is available, and
            # fall back to a generic error dict for any other failure. Both
            # paths must return (never re-raise) so callers get a consistent
            # result shape instead of an unhandled 500.
            error_response = getattr(e, "response", None)
            if error_response and "Error" in error_response:
                error_code = error_response["Error"]["Code"]
                error_msg = error_response["Error"]["Message"]
                logger.error("SES error sending to %s: [%s] %s", to_email, error_code, error_msg)
                return {"status": "error", "error_code": error_code, "detail": error_msg}
            logger.error("Failed to send email to %s: %s", to_email, str(e))
            return {"status": "error", "detail": str(e)}

    def send_sequence_email(
        self,
        to_email: str,
        subject_template: str,
        body_template: str,
        context: dict,
        reply_to: Optional[str] = None,
    ) -> dict:
        """Render templates and send a sequence email."""
        subject = self.render_template(subject_template, context)
        body_html = self.render_template(body_template, context)
        body_text = self.render_template(body_template, context)

        return self.send_email(
            to_email=to_email,
            subject=subject,
            body_html=body_html,
            body_text=body_text,
            reply_to=reply_to,
        )


def get_email_sender() -> EmailSender:
    return EmailSender()
