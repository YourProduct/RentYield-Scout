"""Telegram vacancy forwarding bot.

This script reads updates for a bot, filters messages from configured source
chats by keywords, and republishes the matching posts to a target chat.
It is designed for manual, periodic execution (e.g., once per day).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests
from dotenv import load_dotenv

API_BASE_URL = "https://api.telegram.org/bot{token}/{method}"
DEFAULT_KEYWORDS = [
    "Product Manager",
    "Program Manager",
    "AI Product Manager",
    "Head of AI",
    "AI Tech Lead",
]
STATE_HISTORY_LIMIT = 500


@dataclass
class BotConfig:
    token: str
    target_chat_id: str
    source_identifiers: Sequence[str]
    keywords: Sequence[str]
    state_file: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forward filtered Telegram posts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process updates without forwarding messages to the target chat",
    )
    parser.add_argument(
        "--state-file",
        help="Override the path to the state JSON file",
    )
    return parser


def load_configuration(args: argparse.Namespace) -> BotConfig:
    load_dotenv()

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is not set. Provide it via environment or .env file.")

    target_chat = os.getenv("TARGET_CHAT_ID")
    if not target_chat:
        raise RuntimeError(
            "TARGET_CHAT_ID is not set. Provide the destination chat ID or @username."
        )

    raw_sources = os.getenv("SOURCE_CHAT_IDS")
    if not raw_sources:
        raise RuntimeError("SOURCE_CHAT_IDS is not set. Provide at least one source chat identifier.")

    source_identifiers = [value.strip() for value in raw_sources.split(",") if value.strip()]
    if not source_identifiers:
        raise RuntimeError("SOURCE_CHAT_IDS does not contain any valid identifiers.")

    raw_keywords = os.getenv("KEYWORDS")
    if raw_keywords:
        keywords = [value.strip() for value in raw_keywords.split(",") if value.strip()]
    else:
        keywords = list(DEFAULT_KEYWORDS)

    if not keywords:
        raise RuntimeError("Keyword list is empty. Provide at least one keyword to filter messages.")

    state_file = Path(
        args.state_file
        or os.getenv("STATE_FILE", "state/state.json")
    )

    return BotConfig(
        token=token,
        target_chat_id=target_chat,
        source_identifiers=source_identifiers,
        keywords=keywords,
        state_file=state_file,
    )


def load_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"last_update_id": None, "forwarded_messages": {}}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse state file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid state file format in {path}")

    data.setdefault("last_update_id", None)
    data.setdefault("forwarded_messages", {})
    return data


def save_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def request_telegram_api(token: str, method: str, payload: Optional[dict] = None) -> dict:
    url = API_BASE_URL.format(token=token, method=method)
    response = requests.post(url, json=payload or {})
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        description = result.get("description", "unknown error")
        raise RuntimeError(f"Telegram API call {method} failed: {description}")
    return result


def fetch_updates(token: str, offset: Optional[int]) -> List[dict]:
    payload = {
        "timeout": 0,
        "allowed_updates": ["message", "channel_post"],
    }
    if offset is not None:
        payload["offset"] = offset
    result = request_telegram_api(token, "getUpdates", payload)
    return result.get("result", [])


def copy_message(
    token: str,
    target_chat_id: str,
    from_chat_id: str,
    message_id: int,
) -> None:
    payload = {
        "chat_id": target_chat_id,
        "from_chat_id": from_chat_id,
        "message_id": message_id,
    }
    request_telegram_api(token, "copyMessage", payload)


def normalize_chat_identifier(chat: dict) -> Iterable[str]:
    identifiers = []
    chat_id = chat.get("id")
    if chat_id is not None:
        identifiers.append(str(chat_id))
    username = chat.get("username")
    if username:
        identifiers.append(f"@{username}")
    title = chat.get("title")
    if title:
        identifiers.append(title)
    return identifiers


def text_matches(text: Optional[str], keywords: Sequence[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def prune_history(history: List[int]) -> List[int]:
    if len(history) <= STATE_HISTORY_LIMIT:
        return history
    return history[-STATE_HISTORY_LIMIT:]


def process_updates(config: BotConfig, state: Dict[str, object], dry_run: bool) -> None:
    source_lookup = {identifier for identifier in config.source_identifiers}
    source_lookup_lower = {identifier.lower() for identifier in source_lookup}
    forwarded: Dict[str, List[int]] = state.setdefault("forwarded_messages", {})  # type: ignore
    last_update_id = state.get("last_update_id")

    updates = fetch_updates(config.token, None if last_update_id is None else last_update_id + 1)
    logging.info("Fetched %d updates", len(updates))

    forwarded_count = 0
    skipped_count = 0

    for update in updates:
        update_id = update.get("update_id")
        if update_id is None:
            skipped_count += 1
            continue

        message = update.get("message") or update.get("channel_post")
        if not message:
            skipped_count += 1
            continue

        chat = message.get("chat", {})
        chat_identifiers = set(normalize_chat_identifier(chat))

        chat_identifiers_lower = {identifier.lower() for identifier in chat_identifiers}

        if not (
            chat_identifiers & source_lookup
            or chat_identifiers_lower & source_lookup_lower
        ):
            skipped_count += 1
            last_update_id = update_id
            continue

        message_id = message.get("message_id")
        if message_id is None:
            skipped_count += 1
            last_update_id = update_id
            continue

        chat_id = chat.get("id")
        if chat_id is None:
            skipped_count += 1
            last_update_id = update_id
            continue

        text = message.get("text") or message.get("caption")
        if not text_matches(text, config.keywords):
            skipped_count += 1
            last_update_id = update_id
            continue

        chat_id_str = str(chat_id)
        history = forwarded.setdefault(chat_id_str, [])
        if message_id in history:
            logging.debug("Message %s from chat %s already forwarded", message_id, chat_id_str)
            skipped_count += 1
            last_update_id = update_id
            continue

        logging.info("Forwarding message %s from chat %s", message_id, chat_id_str)
        if not dry_run:
            try:
                copy_message(config.token, config.target_chat_id, chat_id_str, message_id)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Failed to forward message %s from chat %s: %s", message_id, chat_id_str, exc)
                last_update_id = update_id
                continue

        history.append(message_id)
        forwarded[chat_id_str] = prune_history(history)
        forwarded_count += 1
        last_update_id = update_id

    state["last_update_id"] = last_update_id
    logging.info("Forwarded %d messages, skipped %d", forwarded_count, skipped_count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_configuration(args)
    state = load_state(config.state_file)

    try:
        process_updates(config, state, dry_run=args.dry_run)
    finally:
        save_state(config.state_file, state)


if __name__ == "__main__":
    main()
