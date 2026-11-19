"""One-shot Live sessions for the extractor, responder, and red-flag classifier.

`gemini-3.8-live` is a native-audio model: the Live API returns audio, not a text
modality. These roles still need text (JSON or one question), so each call opens a
short session and reads `output_audio_transcription`. The intake graph, prompts,
lint, and slot contract are unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from herbenzo_agent.gemini_live.client import make_session_opener, text_session_config
from herbenzo_agent.gemini_live.parsing import merge_transcript, parse_live_message


def _message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


class GeminiLiveChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    api_key: str | None = Field(default=None, repr=False)
    vertexai: bool = False
    project: str | None = None
    location: str | None = None
    api_version: str = "v1beta"
    voice_name: str = "Kore"
    temperature: float = 0.0
    max_output_tokens: int = 256
    session_opener: Any = Field(default=None, exclude=True, repr=False)

    @property
    def _llm_type(self) -> str:
        return "gemini-live"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _opener(self):
        if self.session_opener is not None:
            return self.session_opener
        return make_session_opener(
            api_key=self.api_key,
            vertexai=self.vertexai,
            project=self.project,
            location=self.location,
            api_version=self.api_version,
        )

    async def _complete(self, system: str, user_text: str) -> str:
        config = text_session_config(
            system=system or "Follow the user message exactly.",
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            voice_name=self.voice_name,
        )
        parts: list[str] = []
        async with self._opener()(self.model_name, config) as session:
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": user_text}]},
                turn_complete=True,
            )
            async for message in session.receive():
                event = parse_live_message(message)
                if event.output_transcript:
                    parts.append(event.output_transcript)
                if event.turn_complete:
                    break
        return merge_transcript(parts)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("GeminiLiveChatModel only supports ainvoke")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in messages:
            text = _message_text(message)
            if getattr(message, "type", "") == "system":
                system_parts.append(text)
            else:
                user_parts.append(text)
        content = await self._complete("\n\n".join(system_parts), "\n\n".join(user_parts))
        if stop:
            for token in stop:
                index = content.find(token)
                if index != -1:
                    content = content[:index]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
