"""
mailer.py — how a password-reset link reaches the person who asked for it.

There is deliberately no vendor here. Picking an email provider is a decision
about cost, deliverability and where your users' addresses are allowed to be
processed, and none of those are decisions this file should be making. What it
does instead is give the reset flow one function to call, with two backends:

  log   (default)  Writes the link to the application log. This is what runs in
                   development, and it is why a developer can complete a reset
                   without configuring anything at all.

  smtp             Sends through any SMTP server, which is the one protocol
                   every provider speaks. Configure it and it is used.

Configuration
-------------
  MAIL_BACKEND      "log" (default) or "smtp"
  SMTP_HOST         required for smtp
  SMTP_PORT         default 587
  SMTP_USER         optional
  SMTP_PASSWORD     optional
  SMTP_STARTTLS     "1" (default) — off only for a local relay you trust
  MAIL_FROM         default "no-reply@localhost"
  APP_BASE_URL      where reset links point, default http://127.0.0.1:8000

**The log backend prints a working credential into your logs.** That is fine on
a laptop and unacceptable anywhere else, so main.py refuses to start with
ENV=production unless MAIL_BACKEND is set to something else.
"""

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

logger = logging.getLogger("physio-backend.mail")

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def backend() -> str:
    return os.getenv("MAIL_BACKEND", "log").strip().lower()


def base_url() -> str:
    return os.getenv("APP_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def reset_url(token: str) -> str:
    return f"{base_url()}/reset-password.html?token={token}"


def send_password_reset(to_email: str, token: str, display_name: str | None = None) -> None:
    """
    Deliver a reset link. Never raises: a mail failure must not tell the caller
    whether the address existed, and must not turn into a 500 on an endpoint
    that is supposed to answer identically either way.
    """
    link = reset_url(token)
    greeting = f"Hello {display_name}," if display_name else "Hello,"
    body = (
        f"{greeting}\n\n"
        "Someone asked to reset the password on your Knee-Physiotherapy account.\n\n"
        f"Open this link to choose a new one:\n\n    {link}\n\n"
        "The link works once and expires in an hour.\n\n"
        "If this was not you, you do not need to do anything — your password has\n"
        "not changed, and the link above will expire on its own.\n"
    )

    try:
        if backend() == "smtp":
            _send_smtp(to_email, "Reset your Knee-Physiotherapy password", body)
            logger.info("Password reset email sent to %s", _redact(to_email))
        else:
            # Not logger.debug: the whole point is that a developer can see it.
            logger.warning(
                "MAIL_BACKEND=log — no email was sent. Reset link for %s:\n    %s",
                _redact(to_email), link,
            )
    except Exception:
        # Logged, not raised. See the docstring.
        logger.exception("Could not send the password reset email to %s", _redact(to_email))


def _send_smtp(to_email: str, subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST")
    if not host:
        raise RuntimeError("MAIL_BACKEND=smtp but SMTP_HOST is not set")
    port = int(os.getenv("SMTP_PORT", "587"))

    msg = EmailMessage()
    msg["From"] = os.getenv("MAIL_FROM", "no-reply@localhost")
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=10) as smtp:
        if os.getenv("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes"):
            smtp.starttls(context=ssl.create_default_context())
        user, password = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)


def _redact(email: str) -> str:
    """Enough to trace a delivery problem, not enough to harvest the log."""
    name, _, domain = email.partition("@")
    if not domain:
        return "***"
    return f"{name[:2]}***@{domain}"
