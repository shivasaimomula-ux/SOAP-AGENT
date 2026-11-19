import pytest

from herbenzo_agent.settings import MissingConfigError, Settings


def test_defaults_use_gemini_live():
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini_live"
    assert s.llm_extractor_provider == "gemini_live"
    assert s.gemini_live_model == "gemini-3.8-live"
    assert s.gemini_text_model == "gemini-3.8-flash"
    assert s.llm_extractor_model == "gemini-3.8-flash"
    assert s.llm_responder_model == "gemini-3.8-flash"
    assert s.llm_red_flag_model == "gemini-3.8-flash"
    assert s.resolved_guardrails_backend() == "deterministic"
    assert s.responder_timeout_s == 6.0
    assert s.llm_extractor_timeout_s == 20.0
    assert s.min_adult_age == 18
    assert s.pipeline_mode is True
    assert s.guardrails_fail_closed() is True
    assert s.recommender_url == "http://localhost:8000"
    assert s.requires_chat_auth() is False
    assert s.chat_bootstrap_token is None


def test_llm_provider_nvidia_restores_nemotron():
    s = Settings(_env_file=None, llm_provider="nvidia")
    assert s.llm_extractor_provider == "nvidia"
    assert s.llm_responder_provider == "nvidia"
    assert s.llm_red_flag_provider == "nvidia"
    assert s.llm_extractor_model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert s.resolved_guardrails_backend() == "nemo"


def test_responder_timeout_is_configurable_for_gemini_live():
    fast = Settings(_env_file=None, responder_timeout_s=2.5)
    assert fast.llm_provider == "gemini_live"
    assert fast.responder_timeout_s == 2.5


def test_gemini_text_model_sets_roles_unless_overridden():
    shared = Settings(_env_file=None, gemini_text_model="gemini-3.5-flash-lite")
    assert shared.llm_extractor_model == "gemini-3.5-flash-lite"
    assert shared.llm_responder_model == "gemini-3.5-flash-lite"
    pinned = Settings(_env_file=None, llm_extractor_model="gemini-3.1-flash-lite")
    assert pinned.llm_extractor_model == "gemini-3.1-flash-lite"
    assert pinned.llm_responder_model == "gemini-3.8-flash"


def test_explicit_role_model_is_kept_when_provider_changes():
    s = Settings(_env_file=None, llm_provider="nvidia", llm_extractor_model="custom/model")
    assert s.llm_extractor_provider == "nvidia"
    assert s.llm_extractor_model == "custom/model"
    assert s.llm_responder_model == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_guardrails_fail_closed_policy():
    assert Settings(_env_file=None, pipeline_mode=True, env="dev").guardrails_fail_closed() is True
    assert Settings(_env_file=None, pipeline_mode=False, env="dev").guardrails_fail_closed() is False
    # Prod stays fail-closed even if someone sets intake-only pipeline flag.
    assert Settings(_env_file=None, pipeline_mode=False, env="prod").guardrails_fail_closed() is True


def test_require_reports_missing_names():
    s = Settings(_env_file=None, internal_api_token=None)
    with pytest.raises(MissingConfigError, match="INTERNAL_API_TOKEN"):
        s.require("internal_api_token")


def test_prod_refuses_content_logging():
    s = Settings(_env_file=None, env="prod", log_content=True, internal_api_token="x")
    with pytest.raises(MissingConfigError, match="LOG_CONTENT"):
        s.require("internal_api_token")
