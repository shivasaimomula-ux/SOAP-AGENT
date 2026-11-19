"""Turn Gemini Live server messages into transcripts and PCM, without assuming one SDK shape."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any


@dataclass
class LiveEvent:
    input_transcript: str | None = None
    output_transcript: str | None = None
    audio: bytes | None = None
    turn_complete: bool = False
    interrupted: bool = False


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_bytes(data: Any) -> bytes | None:
    if isinstance(data, (bytes, bytearray)) and data:
        return bytes(data)
    if isinstance(data, str) and data:
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return None
        return decoded or None
    return None


def parse_live_message(message: Any) -> LiveEvent:
    """Read one Live server event. Audio and transcripts may share a single event."""
    event = LiveEvent()
    server_content = _get(message, "server_content")
    if server_content is None:
        event.audio = _as_bytes(_get(message, "data"))
        text = _get(message, "text")
        if isinstance(text, str) and text:
            event.output_transcript = text
        return event

    incoming = _get(server_content, "input_transcription")
    outgoing = _get(server_content, "output_transcription")
    if incoming is not None:
        text = _get(incoming, "text")
        if text:
            event.input_transcript = str(text)
    if outgoing is not None:
        text = _get(outgoing, "text")
        if text:
            event.output_transcript = str(text)
    event.turn_complete = bool(_get(server_content, "turn_complete"))
    event.interrupted = bool(_get(server_content, "interrupted"))

    model_turn = _get(server_content, "model_turn")
    parts = _get(model_turn, "parts") or []
    audio: list[bytes] = []
    for part in parts:
        inline = _get(part, "inline_data")
        chunk = _as_bytes(_get(inline, "data")) if inline is not None else None
        if chunk:
            audio.append(chunk)
        text = _get(part, "text")
        if text and not event.output_transcript:
            event.output_transcript = str(text)
    if audio:
        event.audio = b"".join(audio)
    return event


def merge_transcript(parts: list[str]) -> str:
    """Join Live transcription chunks.

    The API sends either a growing prefix ("Hello" then "Hello there") or deltas
    ("Hello" then " there"). Both collapse to one string.
    """
    merged = ""
    for part in parts:
        piece = part or ""
        if not piece:
            continue
        if not merged or piece.startswith(merged):
            merged = piece
        elif merged.endswith(piece):
            continue
        else:
            merged += piece
    return merged.strip()
