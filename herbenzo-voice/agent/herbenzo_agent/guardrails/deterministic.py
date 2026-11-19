"""Local input and output checks used when NemoGuard (NVIDIA) is not the guardrail backend.

Red-flag escalation stays in the intake screen, which runs before this rail, so an
emergency disclosure is never refused here. Output still refuses the blocked-term
list the responder lint uses. NemoGuard remains available when GUARDRAILS_BACKEND=nemo.
"""

from __future__ import annotations

import re
import time

from herbenzo_agent.guardrails.nemoguard import GuardResult, GuardStatus
from herbenzo_agent.intake.lint import blocked_patterns, blocked_terms_sha256

# Prompt-injection attempts. Symptom descriptions must not match these.
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions"
    r"|\bsystem prompt\b"
    r"|\bjailbreak\b"
    r"|\bdeveloper mode\b"
    r"|you are now\b",
    re.IGNORECASE,
)

# Harm aimed at someone else. Self-harm phrasing is left to the red-flag lexicon,
# which escalates instead of refusing.
_HARMFUL_OTHER = re.compile(
    r"\bhow (do|can|to) i (poison|kill|hurt)\b"
    r"|\b(poison|kill|hurt|attack)\b.{0,40}\b(him|her|them|someone|neighbour|neighbor|dog|person)\b",
    re.IGNORECASE,
)


class DeterministicGuardrails:
    def __init__(self):
        self.config_sha256 = "deterministic:" + blocked_terms_sha256()[:16]

    async def check_input(self, user_text: str) -> GuardResult:
        started = time.perf_counter()
        blocked = bool(_INJECTION.search(user_text) or _HARMFUL_OTHER.search(user_text))
        return GuardResult(
            GuardStatus.blocked if blocked else GuardStatus.passed,
            rail="deterministic_input",
            categories=["injection_or_harm"] if blocked else [],
            latency_s=time.perf_counter() - started,
        )

    async def check_output(self, user_text: str, bot_text: str) -> GuardResult:
        started = time.perf_counter()
        hit = next((pattern.pattern for pattern in blocked_patterns() if pattern.search(bot_text)), None)
        return GuardResult(
            GuardStatus.blocked if hit else GuardStatus.passed,
            rail="deterministic_output",
            categories=[hit] if hit else [],
            latency_s=time.perf_counter() - started,
        )
