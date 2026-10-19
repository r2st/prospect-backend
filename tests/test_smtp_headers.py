"""SMTP mail carries its own Date and Message-ID (Gmail's API and SES add theirs)."""

from types import SimpleNamespace

from app.services.smtp_sender import SmtpSender


def _sender():
    return SmtpSender(SimpleNamespace(email="me@acme.io", display_name="Me"))


def test_message_has_date_and_a_message_id_on_the_sender_domain():
    msg = _sender()._build_message("to@x.com", "Hi", "", "Hello", None)
    assert msg["Date"]
    assert msg["Message-ID"].startswith("<") and msg["Message-ID"].endswith("@acme.io>")
    # Plain-text campaign mail: one text/plain part, no HTML alternative.
    assert msg.get_content_type() == "text/plain"


def test_each_message_gets_its_own_id():
    s = _sender()
    a = s._build_message("to@x.com", "Hi", "", "Hello", None)["Message-ID"]
    b = s._build_message("to@x.com", "Hi", "", "Hello", None)["Message-ID"]
    assert a != b
