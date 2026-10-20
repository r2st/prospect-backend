"""Files attached to an email template, and the MIME body that carries them.

A template attachment is sent with **every** email rendered from that template —
a capabilities deck, a price sheet. That shapes the limits below: a file that is
fine to send once is not necessarily fine to send to a fifty-contact batch.

Three things are decided at upload rather than at send, because at send time the
only way to find out is a failed batch:

- **Size.** Gmail rejects a message over 25 MB, and attachments travel base64
  encoded, which inflates them by a third. ``MAX_TOTAL_BYTES`` of 15 MB stays
  under that with room for the body. A cap checked only at send would fail every
  recipient in the batch, one at a time.
- **File type.** Gmail refuses executables and scripts outright (the list below
  is its published one). The SMTP and SES transports would deliver them, but the
  receiving server usually would not, and a sender whose mail carries ``.exe``
  files is a sender whose mail stops arriving.
- **The filename.** It ends up in a MIME header, so it is reduced to a basename
  with no control characters — a ``\\r\\n`` in a filename is a header injection.

``build_mime_body`` is the one place a message body is assembled, for all three
transports. Without attachments it returns exactly the ``multipart/alternative``
each transport built before; with them it wraps that in ``multipart/mixed``,
which is the structure mail clients expect (body first, then files).
"""

from __future__ import annotations

import mimetypes
import re
import uuid
from dataclasses import dataclass
from email import encoders
from email.charset import QP, Charset
from email.utils import formatdate, make_msgid
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Iterable, Optional, Sequence

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.email_template import EmailTemplateAttachment as A

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 15 * 1024 * 1024
MAX_FILES = 10

#: Gmail's blocked attachment types
#: (https://support.google.com/mail/answer/6590). Checked on the final
#: extension, which is the one the receiving client acts on.
BLOCKED_EXTENSIONS = frozenset({
    ".ade", ".adp", ".apk", ".appx", ".appxbundle", ".bat", ".cab", ".chm",
    ".cmd", ".com", ".cpl", ".diagcab", ".diagcfg", ".diagpack", ".dll",
    ".dmg", ".ex", ".ex_", ".exe", ".hta", ".img", ".ins", ".iso", ".isp",
    ".jar", ".jnlp", ".js", ".jse", ".lib", ".lnk", ".mde", ".msc", ".msi",
    ".msix", ".msixbundle", ".msp", ".mst", ".nsh", ".pif", ".ps1", ".scr",
    ".sct", ".shb", ".sys", ".vb", ".vbe", ".vbs", ".vhd", ".vxd", ".wsc",
    ".wsf", ".wsh", ".xll",
})

_CONTENT_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class OutgoingAttachment:
    """One file as a transport needs it: name, MIME type and bytes."""

    filename: str
    content_type: str
    data: bytes


def sanitize_filename(raw: Optional[str]) -> str:
    """A filename safe to put in a ``Content-Disposition`` header.

    Browsers send a bare name, but nothing stops a client sending a path, so
    only the last segment of either separator is kept. Control characters are
    dropped rather than replaced: ``\\r\\n`` is the dangerous pair, and a
    visible stand-in would just make a stranger filename.
    """
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _CONTROL_CHARS_RE.sub("", name).strip().strip(".")
    if not name:
        return "attachment"
    if len(name) > 255:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) <= 16:
            name = stem[: 255 - len(ext) - 1] + "." + ext
        else:
            name = name[:255]
    return name


def resolve_content_type(declared: Optional[str], filename: str) -> str:
    """The declared type if it is well formed, else a guess from the name.

    The declared value comes from the browser and lands in a MIME header, so a
    malformed one is replaced rather than passed through.
    """
    candidate = (declared or "").split(";", 1)[0].strip().lower()
    if candidate and _CONTENT_TYPE_RE.match(candidate):
        return candidate
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def check_new_attachment(
    filename: str, size_bytes: int, existing_sizes: Sequence[int]
) -> None:
    """Raise if this file may not join a template already holding ``existing_sizes``."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in BLOCKED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{ext} files can't be attached — Gmail and most mail servers block them.",
        )
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="That file is empty."
        )
    if len(existing_sizes) >= MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A template can carry at most {MAX_FILES} attachments.",
        )
    if sum(existing_sizes) + size_bytes > MAX_TOTAL_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Attachments on one template can total at most "
                f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB — larger emails are "
                "rejected by Gmail once encoded."
            ),
        )


async def load_for_send(
    db: AsyncSession, template_id: Optional[uuid.UUID]
) -> list[OutgoingAttachment]:
    """Every attachment on a template, with its bytes, in upload order.

    Selects columns rather than entities: ``content`` is deferred on the model,
    and an attachment already in the identity map from a metadata load would
    otherwise come back without it.
    """
    if template_id is None:
        return []
    rows = await db.execute(
        select(A.filename, A.content_type, A.content)
        .where(A.template_id == template_id)
        .order_by(A.created_at.asc(), A.id.asc())
    )
    return [
        OutgoingAttachment(filename=r.filename, content_type=r.content_type, data=r.content)
        for r in rows
    ]


def _text_part(body: str, subtype: str) -> MIMEText:
    """A UTF-8 text part, quoted-printable rather than base64.

    ``MIMEText(body, subtype, "utf-8")`` base64-encodes the body. Ordinary mail
    clients send readable text as quoted-printable, and a base64-encoded
    text/plain or text/html part is a long-standing spam-filter strike (it is
    how content gets hidden from naive scanners).
    """
    charset = Charset("utf-8")
    charset.body_encoding = QP
    return MIMEText(body, subtype, charset)


def apply_identity_headers(message, sender_email: str) -> None:
    """Set ``Date`` and ``Message-ID``, as every well-formed message carries.

    Mail with no Message-ID or Date is a classic spam-filter strike. Gmail's
    API and SES fill them in on the way out, but only SMTP set them here, so
    whether a message was well-formed depended on which transport carried it.
    The Message-ID is minted on the sender's own domain so it aligns with From.
    """
    if "Date" not in message:
        message["Date"] = formatdate(usegmt=True)
    if "Message-ID" not in message:
        domain = (sender_email or "").rsplit("@", 1)[-1] or None
        message["Message-ID"] = make_msgid(domain=domain)


def build_mime_body(
    body_text: Optional[str],
    body_html: str,
    attachments: Optional[Iterable[OutgoingAttachment]] = None,
) -> MIMEMultipart:
    """The message container every transport sets its headers on.

    Plain text precedes HTML inside the alternative part (RFC 2046: clients
    render the *last* part they understand). With no ``body_html`` the body is a
    single ``text/plain`` part — a plain-text email, with no HTML for a client
    to render instead.
    """
    if body_html:
        alternative = MIMEMultipart("alternative")
        if body_text:
            alternative.attach(_text_part(body_text, "plain"))
        alternative.attach(_text_part(body_html, "html"))
    else:
        alternative = _text_part(body_text or "", "plain")

    files = list(attachments or ())
    if not files:
        return alternative

    mixed = MIMEMultipart("mixed")
    mixed.attach(alternative)
    for item in files:
        maintype, _, subtype = item.content_type.partition("/")
        part = MIMEBase(maintype or "application", subtype or "octet-stream")
        part.set_payload(item.data)
        encoders.encode_base64(part)
        # A non-ASCII name is RFC 2231-encoded by the email package itself.
        part.add_header("Content-Disposition", "attachment", filename=item.filename)
        mixed.attach(part)
    return mixed
