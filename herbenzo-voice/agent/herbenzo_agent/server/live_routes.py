"""Browser WebSocket that speaks the intake graph through Gemini Live native audio.

Protocol (JSON text frames):
  client -> {"type":"utterance","pcm_base64":"<16 kHz 16-bit little-endian mono>"}
  client -> {"type":"close"}
  server -> {"type":"session","session_id","model","sample_rate_in","sample_rate_out"}
  server -> {"type":"user","text"}
  server -> {"type":"assistant","text","phase","spec_ids"}
  server -> {"type":"audio","pcm_base64","sample_rate"}
  server -> {"type":"turn_done"}
  server -> {"type":"error","detail"}

Auth matches POST /v1/chat. Browsers cannot set headers on WebSocket, so the prod
bootstrap token is the `token` query parameter.
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid
from collections.abc import Callable

import bleach
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from herbenzo_agent.gemini_live.client import voice_session_config
from herbenzo_agent.gemini_live.voice import LiveReader
from herbenzo_agent.intake.graph import SESSION_START, IntakeService
from herbenzo_agent.llm_factory import LLMConfigError
from herbenzo_agent.server.protection import SlidingWindowLimiter
from herbenzo_agent.server.voice_routes import MAX_UTTERANCE_CHARS
from herbenzo_agent.settings import Settings

logger = logging.getLogger(__name__)

INPUT_RATE = 16000
OUTPUT_RATE = 24000
MAX_PCM_BYTES = 60 * INPUT_RATE * 2  # one minute of 16-bit mono


def register_live_routes(
    app: FastAPI,
    settings: Settings,
    locks,
    valid_thread: Callable[[str], str],
    metrics,
    chat_limiter: SlidingWindowLimiter,
    new_session_limiter: SlidingWindowLimiter,
) -> None:
    @app.websocket("/v1/voice/live")
    async def voice_live(websocket: WebSocket) -> None:
        caller = _caller(websocket, settings.trust_forwarded_for)
        if not chat_limiter.allow(caller) or not new_session_limiter.allow(caller):
            await websocket.close(code=1008, reason="too many requests")
            return
        if settings.requires_chat_auth():
            token = websocket.query_params.get("token")
            if not settings.chat_bootstrap_token or token != settings.chat_bootstrap_token:
                await websocket.close(code=1008, reason="chat bootstrap token required")
                return
        await websocket.accept()

        model = settings.gemini_live_model
        opener = websocket.app.state.gemini_opener
        if opener is None:
            try:
                from herbenzo_agent.gemini_live.client import make_session_opener

                opener = make_session_opener(
                    api_key=settings.google_api_key,
                    vertexai=settings.google_genai_use_vertexai,
                    project=settings.google_cloud_project,
                    location=settings.google_cloud_location,
                    api_version=settings.gemini_live_api_version,
                )
            except LLMConfigError as exc:
                await _error(websocket, str(exc))
                await websocket.close(code=1008)
                return

        thread_id = valid_thread(uuid.uuid4().hex)
        service: IntakeService = websocket.app.state.service
        speech = {"asr": f"gemini-live:{model}", "tts": f"gemini-live:{model}"}
        try:
            async with opener(model, voice_session_config(voice_name=settings.gemini_live_voice)) as session:
                reader = LiveReader(session)
                try:
                    await websocket.send_json(
                        {
                            "type": "session",
                            "session_id": thread_id,
                            "model": model,
                            "sample_rate_in": INPUT_RATE,
                            "sample_rate_out": OUTPUT_RATE,
                        }
                    )
                    await _speak_turn(
                        websocket,
                        reader,
                        settings,
                        locks,
                        metrics,
                        service,
                        thread_id,
                        SESSION_START,
                        speech,
                    )
                    while True:
                        incoming = await websocket.receive_json()
                        kind = incoming.get("type") if isinstance(incoming, dict) else None
                        if kind == "close":
                            break
                        if kind != "utterance":
                            await _error(websocket, "expected an utterance")
                            continue
                        pcm = _decode_pcm(incoming.get("pcm_base64", ""))
                        if pcm is None:
                            await _error(websocket, "audio was not valid base64 PCM")
                            continue
                        if len(pcm) > MAX_PCM_BYTES:
                            await _error(websocket, "audio is longer than one minute")
                            continue
                        if not pcm:
                            text = ""
                        else:
                            try:
                                text = await reader.transcribe(pcm, settings.gemini_live_listen_timeout_s)
                            except TimeoutError:
                                logger.warning("gemini live transcription timed out session=%s", thread_id)
                                text = ""
                        text = bleach.clean(text, strip=True)[:MAX_UTTERANCE_CHARS]
                        if text:
                            await websocket.send_json({"type": "user", "text": text})
                        await _speak_turn(
                            websocket,
                            reader,
                            settings,
                            locks,
                            metrics,
                            service,
                            thread_id,
                            text,
                            speech,
                        )
                finally:
                    await reader.aclose()
        except WebSocketDisconnect:
            logger.info("voice live disconnected session=%s", thread_id)
        except Exception:
            logger.exception("voice live failed session=%s", thread_id)
            await _error(websocket, "voice session failed")
        finally:
            try:
                async with locks[thread_id]:
                    await service.close(thread_id)
            except Exception:
                logger.exception("voice live close failed session=%s", thread_id)
            locks.pop(thread_id, None)


def _caller(websocket: WebSocket, trust_forwarded_for: bool) -> str:
    if trust_forwarded_for:
        forwarded = websocket.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return websocket.client.host if websocket.client else "unknown"


def _decode_pcm(payload: object) -> bytes | None:
    if not isinstance(payload, str):
        return None
    if payload == "":
        return b""
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None


async def _error(websocket: WebSocket, detail: str) -> None:
    try:
        await websocket.send_json({"type": "error", "detail": detail})
    except Exception:
        logger.debug("could not send voice error to client")


async def _speak_turn(
    websocket: WebSocket,
    reader: LiveReader,
    settings: Settings,
    locks,
    metrics,
    service: IntakeService,
    thread_id: str,
    text: str,
    speech: dict[str, str],
) -> None:
    async with locks[thread_id]:
        result = await service.turn(thread_id, text, channel="voice", speech_models=speech)
    metrics.record("voice", result.timings, result.counters)
    logger.info(
        "voice live turn session=%s phase=%s total=%.2fs",
        thread_id,
        result.phase,
        result.timings.get("turn_total", 0.0),
    )
    await websocket.send_json(
        {
            "type": "assistant",
            "text": result.reply,
            "phase": result.phase,
            "spec_ids": result.spec_ids,
        }
    )
    try:
        async for chunk in reader.speak(result.reply, settings.gemini_live_speak_timeout_s):
            await websocket.send_json(
                {
                    "type": "audio",
                    "pcm_base64": base64.b64encode(chunk).decode("ascii"),
                    "sample_rate": OUTPUT_RATE,
                }
            )
    except TimeoutError:
        logger.warning("gemini live speak timed out session=%s", thread_id)
        await _error(websocket, "voice audio timed out")
    await websocket.send_json({"type": "turn_done"})
