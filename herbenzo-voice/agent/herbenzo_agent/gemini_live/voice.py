"""One Live session per browser call: native audio in, scripted speech out.

The intake graph remains the source of the words. Caller audio is transcribed, then
`IntakeService` runs the usual slots, red flags, guardrails, and read-back. Only the
graph's reply is spoken, via a `SPEAK:` realtime text turn. Audio the model produces
while listening is discarded.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from herbenzo_agent.gemini_live.parsing import merge_transcript, parse_live_message

logger = logging.getLogger(__name__)

PCM_CHUNK_BYTES = 3200  # 100 ms at 16 kHz, 16-bit mono
SPEAK_PREFIX = "SPEAK:"


def pcm_chunks(pcm: bytes, size: int = PCM_CHUNK_BYTES):
    for index in range(0, len(pcm), size):
        yield pcm[index : index + size]


def _activity_start():
    from google.genai import types

    return types.ActivityStart()


def _activity_end():
    from google.genai import types

    return types.ActivityEnd()


def _pcm_blob(data: bytes, rate: int):
    from google.genai import types

    return types.Blob(data=data, mime_type=f"audio/pcm;rate={rate}")


class LiveReader:
    """Single `session.receive()` loop. The Live API allows one consumer per session."""

    def __init__(self, session: Any):
        self.session = session
        self.audio: asyncio.Queue[bytes] = asyncio.Queue()
        self.input_parts: list[str] = []
        self.turn_complete = asyncio.Event()
        self.accept_audio = False
        self.error: BaseException | None = None
        self._task = asyncio.create_task(self._run(), name="gemini-live-reader")

    async def _run(self) -> None:
        try:
            async for message in self.session.receive():
                event = parse_live_message(message)
                if event.input_transcript:
                    self.input_parts.append(event.input_transcript)
                if event.audio and self.accept_audio:
                    self.audio.put_nowait(event.audio)
                if event.turn_complete or event.interrupted:
                    self.turn_complete.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc
            logger.warning("gemini live reader stopped: %s", type(exc).__name__)
        finally:
            self.turn_complete.set()

    def begin_turn(self, *, accept_audio: bool) -> None:
        self.input_parts.clear()
        self.turn_complete.clear()
        self.accept_audio = accept_audio
        while True:
            try:
                self.audio.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def wait_turn(self, timeout_s: float) -> None:
        await asyncio.wait_for(self.turn_complete.wait(), timeout_s)
        if self.error:
            raise RuntimeError("gemini live session ended") from self.error

    async def transcribe(self, pcm: bytes, timeout_s: float) -> str:
        self.begin_turn(accept_audio=False)
        await self.session.send_realtime_input(activity_start=_activity_start())
        for chunk in pcm_chunks(pcm):
            await self.session.send_realtime_input(audio=_pcm_blob(chunk, 16000))
        await self.session.send_realtime_input(activity_end=_activity_end())
        await self.wait_turn(timeout_s)
        return merge_transcript(self.input_parts)

    async def speak(self, text: str, timeout_s: float) -> AsyncIterator[bytes]:
        spoken = " ".join(text.split())
        if not spoken:
            return
        self.begin_turn(accept_audio=True)
        await self.session.send_realtime_input(text=f"{SPEAK_PREFIX} {spoken}")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            if self.turn_complete.is_set() and self.audio.empty():
                try:
                    yield await asyncio.wait_for(self.audio.get(), 0.05)
                except TimeoutError:
                    break
                continue
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("gemini live speak timed out")
            try:
                chunk = await asyncio.wait_for(self.audio.get(), min(0.25, remaining))
            except TimeoutError:
                if self.turn_complete.is_set() or self.error:
                    break
                continue
            yield chunk
        if self.error:
            raise RuntimeError("gemini live session ended") from self.error

    async def aclose(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
