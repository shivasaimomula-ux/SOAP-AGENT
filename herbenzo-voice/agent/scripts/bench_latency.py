"""Measure intake latency against a running Herbenzo agent.

The server process needs GOOGLE_API_KEY (or Vertex settings). This script only talks
HTTP and WebSocket; it does not read or send the key.

Text turn: POST /v1/chat. Prints time-to-first-byte and time-to-full JSON for one
symptom message (the greeting is scripted and is printed separately).

Voice turn: WebSocket /v1/voice/live. Prints time from the end of the sent clip to the
assistant text and to the first 24 kHz PCM chunk.

From herbenzo-voice/agent, with the server already on port 8081:

    uv run python scripts/bench_latency.py
    uv run python scripts/bench_latency.py --base-url http://127.0.0.1:8081 --pcm speech.wav

--pcm is 16 kHz, 16-bit, mono WAV or raw little-endian PCM. Without it the voice turn
sends a short silence clip, so transcription may be empty; the timers still print.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path
from urllib.parse import quote, urlparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Herbenzo text and voice turn latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081", help="Agent origin, no trailing path.")
    parser.add_argument("--pcm", type=Path, default=None, help="16 kHz 16-bit mono WAV or raw PCM.")
    parser.add_argument(
        "--message",
        default="I have burning acidity after spicy meals for two weeks",
        help="User text for the chat turn.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request ceiling in seconds.")
    parser.add_argument("--token", default=None, help="CHAT_BOOTSTRAP_TOKEN when the server requires it.")
    return parser


def load_pcm(path: Path | None) -> bytes:
    """16 kHz 16-bit mono PCM. None builds a short silence clip so the socket path still runs."""
    if path is None:
        return b"\x00\x00" * int(16000 * 0.4)
    if path.suffix.lower() == ".wav":
        with path.open("rb") as raw, wave.open(raw, "rb") as handle:
            channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
            if channels != 1 or width != 2 or rate != 16000:
                raise SystemExit(
                    f"{path} must be 16 kHz, 16-bit, mono (got {rate} Hz, {width * 8}-bit, {channels} ch)"
                )
            return handle.readframes(handle.getnframes())
    return path.read_bytes()


def _ws_url(base: str, token: str | None) -> str:
    parsed = urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = f"?token={quote(token)}" if token else ""
    return f"{scheme}://{parsed.netloc}/v1/voice/live{query}"


def _headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Chat-Token"] = token
    return headers


async def _read_json(
    client, url: str, payload: dict, headers: dict[str, str], timeout: float
) -> tuple[float, float, dict]:
    started = time.perf_counter()
    first_byte = None
    body = bytearray()
    async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if first_byte is None:
                first_byte = time.perf_counter()
            body.extend(chunk)
    finished = time.perf_counter()
    ttfb = (first_byte or finished) - started
    return ttfb, finished - started, json.loads(body.decode() or "{}")


async def bench_text(base: str, message: str, token: str | None, timeout: float) -> None:
    import httpx

    headers = _headers(token)
    async with httpx.AsyncClient() as client:
        greet_ttfb, greet_total, greeting = await _read_json(client, f"{base}/v1/chat", {}, headers, timeout)
        session_id = greeting.get("session_id")
        if not session_id:
            raise SystemExit(f"text greeting returned no session_id: {greeting}")
        user_ttfb, user_total, turn = await _read_json(
            client,
            f"{base}/v1/chat",
            {"session_id": session_id, "message": message},
            headers,
            timeout,
        )
    print(f"text_greeting_ttfb_s={greet_ttfb:.3f} text_greeting_total_s={greet_total:.3f}")
    print(
        f"text_user_ttfb_s={user_ttfb:.3f} text_user_total_s={user_total:.3f} "
        f"phase={turn.get('phase')} reply_chars={len(turn.get('reply') or '')}"
    )


async def _next_event(ws, timeout: float) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout)
    event = json.loads(raw)
    if event.get("type") == "error":
        raise SystemExit(f"voice error: {event.get('detail')}")
    return event


async def bench_voice(base: str, pcm: bytes, token: str | None, timeout: float) -> None:
    import websockets

    started = time.perf_counter()
    async with websockets.connect(_ws_url(base, token), open_timeout=timeout, max_size=8_000_000) as ws:
        greeting_audio = None
        while True:
            event = await _next_event(ws, timeout)
            now = time.perf_counter()
            if event.get("type") == "audio" and greeting_audio is None:
                greeting_audio = now
            if event.get("type") == "turn_done":
                break
        if greeting_audio is None:
            raise SystemExit("voice greeting produced no audio")
        print(f"voice_greeting_first_audio_s={greeting_audio - started:.3f}")

        sent = time.perf_counter()
        await ws.send(json.dumps({"type": "utterance", "pcm_base64": base64.b64encode(pcm).decode("ascii")}))
        reply_text = None
        reply_audio = None
        transcript = ""
        while True:
            event = await _next_event(ws, timeout)
            now = time.perf_counter()
            if event.get("type") == "user":
                transcript = event.get("text") or ""
            if event.get("type") == "assistant" and reply_text is None:
                reply_text = now
            if event.get("type") == "audio" and reply_audio is None:
                reply_audio = now
            if event.get("type") == "turn_done":
                break
        text_s = f"{reply_text - sent:.3f}" if reply_text else "none"
        audio_s = f"{reply_audio - sent:.3f}" if reply_audio else "none"
        print(
            f"voice_reply_text_s={text_s} voice_reply_first_audio_s={audio_s} "
            f"transcript_chars={len(transcript)}"
        )


async def _main(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    pcm = load_pcm(args.pcm)
    await bench_text(base, args.message, args.token, args.timeout)
    await bench_voice(base, pcm, args.token, args.timeout)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.pcm:
        print("note: no --pcm file; voice turn uses silence and may not transcribe.", file=sys.stderr)
    try:
        asyncio.run(_main(args))
    except TimeoutError:
        raise SystemExit(f"timed out after {args.timeout:.0f}s") from None


if __name__ == "__main__":
    main()
