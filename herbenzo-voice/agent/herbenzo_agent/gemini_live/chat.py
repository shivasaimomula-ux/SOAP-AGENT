"""Text calls for the extractor, responder, and red-flag classifier.

Those roles need JSON or one short question. They call ``generateContent`` on a fast
Flash model with the same Google credentials as Live. They do not open a Live session:
native-audio Live spends the turn generating speech, then a transcript, which is why a
responder turn was 15–17 s. The voice socket keeps the only Live session.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from herbenzo_agent.gemini_live.client import build_genai_client


def _message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _response_text(response: Any) -> str:
    """Answer text only. Thought parts are not spoken and are not JSON."""
    direct = getattr(response, "text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


class GeminiTextChatModel(BaseChatModel):
    """``generateContent`` chat model. One shared client, no Live connect per call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    api_key: str | None = Field(default=None, repr=False)
    vertexai: bool = False
    project: str | None = None
    location: str | None = None
    api_version: str = "v1beta"
    max_output_tokens: int = 256
    thinking_level: str = "low"
    # Tests inject this. Production leaves it empty and calls generateContent.
    completer: Any = Field(default=None, exclude=True, repr=False)

    _client: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "gemini-text"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "thinking_level": self.thinking_level}

    def _request_config(self, system: str) -> dict[str, Any]:
        # Gemini 3 Flash rejects temperature / top_p / top_k. thinking_level is the latency knob.
        config: dict[str, Any] = {
            "system_instruction": system or "Follow the user message exactly.",
            "max_output_tokens": self.max_output_tokens,
        }
        if self.thinking_level:
            config["thinking_config"] = {"thinking_level": self.thinking_level}
        return config

    def _ensure_client(self):
        if self._client is None:
            self._client = build_genai_client(
                api_key=self.api_key,
                vertexai=self.vertexai,
                project=self.project,
                location=self.location,
                api_version=self.api_version,
            )
        return self._client

    async def _complete(self, system: str, user_text: str) -> str:
        config = self._request_config(system)
        if self.completer is not None:
            result = self.completer(system, user_text, config)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result or "").strip()
        response = await self._ensure_client().aio.models.generate_content(
            model=self.model_name,
            contents=user_text or " ",
            config=config,
        )
        return _response_text(response)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("GeminiTextChatModel only supports ainvoke")

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
