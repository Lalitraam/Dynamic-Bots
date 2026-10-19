"""Email delivery service with SMTP support and dev console fallback."""

from __future__ import annotations
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging

from app import config

logger = logging.getLogger(__name__)


def send_otp_email(to_email: str, otp_code: str) -> bool:
    """Send a 6-digit OTP verification email via SMTP if configured,

    or log to console with fallback notification.
    """
    to_email = to_email.strip().lower()
    subject = f"Your Doppelgambit Verification Code: {otp_code}"

    text_content = f"""Hello,

Your verification code for Doppelgambit is: {otp_code}

This code will expire in {config.OTP_EXPIRY_MINUTES} minutes.
If you did not request this code, please ignore this email.

— The Doppelgambit Team
"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0b0b0f; color: #f3f4f6; margin: 0; padding: 32px 16px; }}
    .card {{ max-width: 480px; margin: 0 auto; background: #13141c; border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 32px; box-shadow: 0 12px 30px rgba(0,0,0,0.5); }}
    .title {{ font-size: 20px; font-weight: 700; color: #ffffff; margin-top: 0; margin-bottom: 8px; letter-spacing: -0.02em; }}
    .desc {{ font-size: 14px; color: #9ca3af; line-height: 1.5; margin-bottom: 24px; }}
    .code-box {{ background: #1a1c29; border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 8px; padding: 18px; text-align: center; margin: 24px 0; }}
    .code {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 32px; font-weight: 700; letter-spacing: 6px; color: #a78bfa; margin: 0; }}
    .footer {{ font-size: 12px; color: #6b7280; margin-top: 24px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 16px; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="title">Verify Your Doppelgambit Account</div>
    <div class="desc">Enter the verification code below to complete your login or registration.</div>
    <div class="code-box">
      <div class="code">{otp_code}</div>
    </div>
    <div class="desc" style="font-size: 12px; color: #9ca3af;">Code expires in {config.OTP_EXPIRY_MINUTES} minutes. Do not share this code with anyone.</div>
    <div class="footer">If you did not request this email, you can safely ignore it.</div>
  </div>
</body>
</html>
"""

    # If SMTP is configured, attempt sending
    if config.SMTP_HOST:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = config.SMTP_FROM_EMAIL
            msg["To"] = to_email
            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            if config.SMTP_USE_SSL:
                server = smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=10)
            else:
                server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10)
                if config.SMTP_USE_TLS:
                    server.starttls()

            if config.SMTP_USER and config.SMTP_PASSWORD:
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)

            server.sendmail(config.SMTP_FROM_EMAIL, [to_email], msg.as_string())
            server.quit()
            logger.info("Successfully sent OTP email to %s via SMTP", to_email)
            print(f"\n[EMAIL SERVICE] Successfully sent OTP verification code to {to_email}\n", flush=True)
            return True
        except Exception as e:
            logger.error("Failed to send email via SMTP to %s: %s", to_email, e)
            print(f"\n[EMAIL SERVICE] SMTP delivery failed ({e}). Falling back to console output:\n", flush=True)

    # Fallback to console output (dev mode or unconfigured SMTP)
    print(
        f"\n{'='*64}\n"
        f"DOPPELGAMBIT OTP: Verification code for {to_email} is:\n"
        f"  >>> {otp_code} <<<\n"
        f"Valid for {config.OTP_EXPIRY_MINUTES} minutes.\n"
        f"{'='*64}\n",
        flush=True,
    )
    return False
