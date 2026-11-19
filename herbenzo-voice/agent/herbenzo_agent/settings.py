"""All runtime configuration, read from environment variables and the nearest `.env`.

Importing this module never fails on missing secrets; call `Settings.require()` at
the point a secret is actually needed (e.g. server startup) so tests stay hermetic.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["nvidia", "anthropic", "google", "gemini_live", "fake"]
GuardrailsBackend = Literal["auto", "nemo", "deterministic"]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
# Native-audio Gemini Live model (AI Studio and Vertex). Audio in, audio out.
DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.8-live"
_ROLE_MODELS = {
    "nvidia": DEFAULT_NVIDIA_MODEL,
    "gemini_live": DEFAULT_GEMINI_LIVE_MODEL,
    "google": DEFAULT_GEMINI_LIVE_MODEL,
}


def _find_env_file() -> Path | None:
    """Walk up from this file to the first directory containing a `.env`."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


class MissingConfigError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore", case_sensitive=False)

    env: Literal["dev", "test", "prod"] = "dev"

    # Provider credentials. Secrets come from the environment or a gitignored .env, never from code.
    nvidia_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    google_genai_use_vertexai: bool = False
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"

    # LLM_PROVIDER sets every role that does not set LLM_<ROLE>_PROVIDER itself.
    llm_provider: LLMProvider = "gemini_live"
    gemini_live_model: str = DEFAULT_GEMINI_LIVE_MODEL
    gemini_live_voice: str = "Kore"
    gemini_live_api_version: str = "v1beta"
    gemini_live_listen_timeout_s: float = 20.0
    gemini_live_speak_timeout_s: float = 30.0

    # LLM roles. Defaults follow Gemini Live; LLM_PROVIDER=nvidia restores Nemotron.
    llm_extractor_provider: LLMProvider = "gemini_live"
    llm_extractor_model: str = DEFAULT_GEMINI_LIVE_MODEL
    llm_extractor_base_url: str | None = None
    llm_responder_provider: LLMProvider = "gemini_live"
    llm_responder_model: str = DEFAULT_GEMINI_LIVE_MODEL
    llm_responder_base_url: str | None = None
    llm_red_flag_provider: LLMProvider = "gemini_live"
    llm_red_flag_model: str = DEFAULT_GEMINI_LIVE_MODEL
    llm_red_flag_base_url: str | None = None
    llm_disable_thinking: bool = True
    llm_extractor_timeout_s: float = 20.0

    # Guardrails. auto: NemoGuard when every role is nvidia, otherwise the local checks.
    guardrails_backend: GuardrailsBackend = "auto"
    guardrails_enabled: bool = True
    guardrails_config_path: str = "herbenzo_agent/guardrails/herbenzo-intake-nemoguard"
    guardrails_base_url: str = NVIDIA_BASE_URL
    # Hosted NemoGuard latency is bimodal (~0.3 s or 20 s+): a short timeout loses almost no verdicts
    # and keeps the rail off the critical path when the endpoint is stalling.
    guardrails_timeout_s: float = 1.5
    # NVIDIA's hosted topic-control NIM returned HTTP 500 on every call (2026-09-16); off-topic messages are
    # already handled by the extractor's intent. Re-enable when the endpoint (or a self-hosted NIM) works.
    guardrails_topic_control: bool = False
    # Output rail on phrased questions. Lint already blocks advice terms; this adds ~0.3-3 s per question turn.
    guardrails_output_enabled: bool = True

    # Storage
    database_url: str = "sqlite:///data/herbenzo_voice.db"

    # Service security
    internal_api_token: str | None = None
    admin_api_token: str | None = None
    # Shared secret for POST /v1/chat when chat auth is required (ENV=prod / prod profile).
    # Localhost demo (ENV=dev) leaves this unset and keeps /v1/chat open behind rate limits.
    chat_bootstrap_token: str | None = None
    # None = auto (required when ENV=prod). Set true/false to override.
    chat_auth_required: bool | None = None
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["https://localhost"])

    # Handoff to Stage A
    recommender_url: str = "http://localhost:8000"
    recommender_timeout_s: float = 600.0
    handoff_enabled: bool = True
    handoff_min_confidence: float = 0.5
    # When true (default), /ready requires Stage A at RECOMMENDER_URL. Set false for intake-only.
    # Also: unavailable NemoGuard input rails fail closed (block) when true; see guardrails_fail_closed().
    pipeline_mode: bool = True

    @model_validator(mode="after")
    def align_llm_roles(self):
        """Apply LLM_PROVIDER, and pick the model that matches a provider when one was not set."""
        shared = self.llm_provider if "llm_provider" in self.model_fields_set else None
        for role in ("extractor", "responder", "red_flag"):
            provider_field = f"llm_{role}_provider"
            model_field = f"llm_{role}_model"
            if shared and provider_field not in self.model_fields_set:
                setattr(self, provider_field, shared)
            provider = getattr(self, provider_field)
            if model_field not in self.model_fields_set and provider in _ROLE_MODELS:
                setattr(self, model_field, _ROLE_MODELS[provider])
        providers = {self.llm_extractor_provider, self.llm_responder_provider, self.llm_red_flag_provider}
        if providers & {"gemini_live", "google"}:
            # A Live turn speaks audio and then returns a transcript, so the old NVIDIA budgets are tight.
            if "responder_timeout_s" not in self.model_fields_set:
                self.responder_timeout_s = 15.0
            if "llm_extractor_timeout_s" not in self.model_fields_set:
                self.llm_extractor_timeout_s = 30.0
        return self

    def google_credentials_configured(self) -> bool:
        if self.google_genai_use_vertexai:
            return bool(self.google_cloud_project)
        return bool(self.google_api_key)

    def resolved_guardrails_backend(self) -> str:
        if not self.guardrails_enabled:
            return "off"
        if self.guardrails_backend != "auto":
            return self.guardrails_backend
        providers = {self.llm_extractor_provider, self.llm_responder_provider, self.llm_red_flag_provider}
        if providers and providers <= {"nvidia"}:
            return "nemo"
        return "deterministic"

    def requires_chat_auth(self) -> bool:
        """Prod profile (ENV=prod) requires a bootstrap token on /v1/chat; localhost demo does not.

        Explicit CHAT_AUTH_REQUIRED overrides the auto rule. PIPELINE_MODE alone does not —
        local pipeline demos keep ENV=dev with open chat + rate limits.
        """
        if self.chat_auth_required is not None:
            return self.chat_auth_required
        return self.env == "prod"

    def guardrails_fail_closed(self) -> bool:
        """Whether unavailable NemoGuard / content-safety must block input (not allow).

        Pipeline mode (default) and ENV=prod fail closed. Intake-only demo
        (`PIPELINE_MODE=false` with ENV≠prod) keeps the historical fail-open path so
        local chat still works when the hosted NIM is flaky.
        """
        return self.pipeline_mode or self.env == "prod"

    # Intake policy
    min_adult_age: int = 18
    max_user_turns: int = 30
    ayurvedic_context_enabled: bool = True
    responder_timeout_s: float = 6.0
    default_language: str = "en-IN"
    default_jurisdiction: str = "IN"
    transcript_retention_days: int = 90
    retention_purge_interval_s: float = 6 * 3600

    # Abuse protection (per client IP, per minute) for the public chat API
    rate_limit_chat_per_minute: int = 30
    rate_limit_new_sessions_per_minute: int = 10
    trust_forwarded_for: bool = False  # true only behind the Caddy proxy

    # Logging
    log_level: str = "INFO"
    log_content: bool = False

    def require(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise MissingConfigError(
                "Missing required configuration: "
                + ", ".join(n.upper() for n in missing)
                + ". Set them in your .env file (see herbenzo-voice/.env.example)."
            )
        if self.env == "prod" and self.log_content:
            raise MissingConfigError("LOG_CONTENT=true is not allowed when ENV=prod.")

    def validate_for_startup(self) -> list[str]:
        """Fail fast on unsafe production configuration. Returns warnings for non-production environments."""
        problems: list[str] = []
        if self.log_content and self.env == "prod":
            problems.append("LOG_CONTENT must be false in production")
        for name in ("internal_api_token", "admin_api_token"):
            value = getattr(self, name)
            if self.env == "prod" and (not value or len(value) < 24):
                problems.append(f"{name.upper()} must be set to a random value of at least 24 characters")
        if self.env == "prod":
            if any(o == "*" or "localhost" in o or o.startswith("http://") for o in self.cors_allow_origins):
                problems.append("CORS_ALLOW_ORIGINS must list only https production origins")
            providers = {self.llm_extractor_provider, self.llm_responder_provider, self.llm_red_flag_provider}
            self_hosted = bool(self.llm_extractor_base_url and self.llm_responder_base_url)
            if "nvidia" in providers and not self.nvidia_api_key and not self_hosted:
                problems.append("NVIDIA_API_KEY is required unless self-hosted base URLs are configured")
            if providers & {"gemini_live", "google"} and not self.google_credentials_configured():
                problems.append(
                    "GOOGLE_API_KEY is required for gemini_live "
                    "(or set GOOGLE_GENAI_USE_VERTEXAI with GOOGLE_CLOUD_PROJECT)"
                )
            if "anthropic" in providers and not self.anthropic_api_key:
                problems.append("ANTHROPIC_API_KEY is required for the anthropic provider")
            if (
                self.resolved_guardrails_backend() == "nemo"
                and not self.nvidia_api_key
                and "integrate.api.nvidia.com" in self.guardrails_base_url
            ):
                problems.append(
                    "NVIDIA_API_KEY is required for NemoGuard (or set GUARDRAILS_BACKEND=deterministic)"
                )
            if not self.guardrails_enabled:
                problems.append("GUARDRAILS_ENABLED must be true in production")
            if self.requires_chat_auth() and (
                not self.chat_bootstrap_token or len(self.chat_bootstrap_token) < 24
            ):
                problems.append(
                    "CHAT_BOOTSTRAP_TOKEN must be set to a random value of at least 24 characters "
                    "when chat auth is required (ENV=prod)"
                )
        if problems and self.env == "prod":
            raise MissingConfigError("Unsafe production configuration: " + "; ".join(problems))
        warnings = []
        if not self.internal_api_token:
            warnings.append("INTERNAL_API_TOKEN not set: the voice endpoint /generate is disabled")
        if not self.admin_api_token:
            warnings.append("ADMIN_API_TOKEN not set: review and audit endpoints are disabled")
        if self.requires_chat_auth() and not self.chat_bootstrap_token:
            warnings.append("CHAT_BOOTSTRAP_TOKEN not set: POST /v1/chat will reject all callers")
        elif not self.requires_chat_auth():
            warnings.append(
                "chat auth off (localhost demo profile): POST /v1/chat is open behind rate limits; "
                "set ENV=prod or CHAT_AUTH_REQUIRED=true for the prod profile"
            )
        return warnings


@lru_cache
def get_settings() -> Settings:
    return Settings()
