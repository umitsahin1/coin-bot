"""Telegram notifier. No-op if env vars are not set."""

from __future__ import annotations

import os

import requests


_TIMEOUT = 10


def _creds() -> tuple[str, str] | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return None
    return token, chat


def send(text: str) -> bool:
    """Send a Telegram message. Returns True on success, False on no creds / error."""
    creds = _creds()
    if not creds:
        return False
    token, chat = creds
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=_TIMEOUT,
        )
        return r.ok
    except requests.RequestException:
        return False


def is_configured() -> bool:
    return _creds() is not None
