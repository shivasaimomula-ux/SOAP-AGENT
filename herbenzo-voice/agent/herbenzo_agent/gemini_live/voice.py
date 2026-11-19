"""One Live session per browser call: native audio in, scripted speech out.

The intake graph remains the source of the words. Caller audio is transcribed, then
`IntakeService` runs the usual slots, red flags, guardrails, and read-back. Only the
graph's reply is spoken, via a `SPEAK:` realtime text turn. Audio the model produces
while listening is discarded. PCM chunks are forwarded as they arrive.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from herbenzo_agent.gemini_live.parsing import merge_transcript, parse_live_message

logger = logging.getLogger(__name__)

INPUT_RATE = 16000
PCM_CHUNK_BYTES = 3200  # 100 ms at 16 kHz, 16-bit mono
SPEAK_PREFIX = "SPEAK:"
# If the model stays silent and never sends turn_complete, don't block the graph on it.
TRANSCRIPT_SETTLE_S = 0.35


def pcm_chunks(pcm: bytes, size: int = PCM_CHUNK_BYTES):
    for index in range(0, len(pcm), size):
        yield pcm[index : index + size]


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
        self._generation = 0
        self._task = asyncio.create_task(self._run(), name="gemini-live-reader")

    async def _run(self) -> None:
        try:
            async for message in self.session.receive():
                generation = self._generation
                event = parse_live_message(message)
                if generation != self._generation:
                    continue
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
        self._generation += 1
        self.input_parts.clear()
        self.turn_complete.clear()
        self.accept_audio = accept_audio
        while True:
            try:
                self.audio.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def transcribe(self, pcm: bytes, timeout_s: float) -> str:
        """Send one finished clip and return the input transcript.

        ``audio_stream_end`` is the end-of-speech signal when automatic activity
        detection is on. Without it the server holds the clip until its silence
        timer, and ``turn_complete`` often never arrives inside the listen timeout.
        """
        self.begin_turn(accept_audio=False)
        for chunk in pcm_chunks(pcm):
            await self.session.send_realtime_input(audio=_pcm_blob(chunk, INPUT_RATE))
        await self.session.send_realtime_input(audio_stream_end=True)
        return await self._wait_transcript(timeout_s)

    async def _wait_transcript(self, timeout_s: float) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_text = ""
        stable_since: float | None = None
        while True:
            if self.error:
                raise RuntimeError("gemini live session ended") from self.error
            text = merge_transcript(self.input_parts)
            now = loop.time()
            if self.turn_complete.is_set():
                return text
            if text != last_text:
                last_text = text
                stable_since = now if text else None
            elif text and stable_since is not None and now - stable_since >= TRANSCRIPT_SETTLE_S:
                # The listen turn may still be open. Give it a moment to close so its
                # audio cannot leak into the following SPEAK turn, then move on.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.turn_complete.wait(), 0.25)
                return text
            remaining = deadline - now
            if remaining <= 0:
                if text:
                    return text
                raise TimeoutError("gemini live transcription timed out")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.turn_complete.wait(), min(0.05, remaining))

    async def speak(self, text: str, timeout_s: float) -> AsyncIterator[bytes]:
        """Yield each output PCM chunk as soon as it arrives. Do not buffer the utterance."""
        spoken = " ".join(text.split())
        if not spoken:
            return
        self.begin_turn(accept_audio=True)
        await self.session.send_realtime_input(text=f"{SPEAK_PREFIX} {spoken}")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        yielded = False
        while True:
            if self.error and self.audio.empty():
                raise RuntimeError("gemini live session ended") from self.error
            try:
                chunk = self.audio.get_nowait()
            except asyncio.QueueEmpty:
                chunk = None
            if chunk is None:
                if yielded and self.turn_complete.is_set():
                    break
                remaining = deadline - loop.time()
                if remaining <= 0:
                    if yielded:
                        break
                    raise TimeoutError("gemini live speak timed out")
                try:
                    chunk = await asyncio.wait_for(self.audio.get(), min(0.05, remaining))
                except TimeoutError:
                    continue
            yielded = True
            yield chunk
        if self.error and not yielded:
            raise RuntimeError("gemini live session ended") from self.error

    async def aclose(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
