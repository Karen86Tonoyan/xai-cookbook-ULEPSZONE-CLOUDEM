from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"


@dataclass
class FilterResult:
    name: str
    decision: Decision
    score: float
    reason: str
    transformed_text: str | None = None


@dataclass
class GatewayDecision:
    allowed: bool
    final_input: str | None
    results: list[FilterResult]
    audit_id: str
    created_at: str


@dataclass
class UserContext:
    user_id: str
    roles: set[str] = field(default_factory=set)
    verified_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    data_permissions: set[str] = field(default_factory=set)


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "api_key": re.compile(r"\b(?:sk|xai|ghp|AIza)[A-Za-z0-9_\-]{16,}\b"),
    "phone": re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),
    "polish_pesel": re.compile(r"\b\d{11}\b"),
}

JAILBREAK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all )?(previous|system|developer) instructions",
        r"developer mode",
        r"jailbreak",
        r"disable (all )?(safety|policy|filter)",
        r"you are not an ai",
        r"system override",
        r"bypass (your )?(rules|filters|safety)",
    )
]


def stable_audit_id(text: str, user_context: UserContext) -> str:
    payload = json.dumps(
        {
            "text": text,
            "user_id": user_context.user_id,
            "roles": sorted(user_context.roles),
            "created_at": datetime.now(timezone.utc).date().isoformat(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def context_gate(text: str, user_context: UserContext) -> FilterResult:
    if "according to the uploaded source" in text.lower() and not user_context.verified_sources:
        return FilterResult(
            name="context_gate",
            decision=Decision.BLOCK,
            score=0.0,
            reason="The request depends on uploaded or verified sources, but none are attached.",
        )
    return FilterResult("context_gate", Decision.ALLOW, 1.0, "Context requirement satisfied.")


def confidence_gate(text: str, user_context: UserContext, threshold: float = 0.55) -> FilterResult:
    if user_context.confidence < threshold:
        return FilterResult(
            name="confidence_gate",
            decision=Decision.BLOCK,
            score=user_context.confidence,
            reason=f"Confidence {user_context.confidence:.2f} is below required threshold {threshold:.2f}.",
        )
    return FilterResult("confidence_gate", Decision.ALLOW, user_context.confidence, "Confidence threshold met.")


def dlp_scanner(text: str, user_context: UserContext) -> FilterResult:
    findings = [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]
    if not findings:
        return FilterResult("dlp_scanner", Decision.ALLOW, 1.0, "No sensitive patterns found.")

    if "pii:read" not in user_context.data_permissions:
        redacted = text
        for name, pattern in PII_PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
        return FilterResult(
            name="dlp_scanner",
            decision=Decision.TRANSFORM,
            score=0.4,
            reason=f"Sensitive patterns redacted before model call: {', '.join(findings)}.",
            transformed_text=redacted,
        )

    return FilterResult("dlp_scanner", Decision.ALLOW, 0.8, "Sensitive data allowed by user permissions.")


def permission_gate(text: str, user_context: UserContext) -> FilterResult:
    lowered = text.lower()
    restricted_patterns = (
        r"\b(show|list|export|reveal|access|dump)\b.+\b(personal data|private profile|customer database)\b",
        r"\b(customer database|private profile)\b",
        r"\b(pokaż|pokaz|wypisz|wyeksportuj|ujawnij)\b.+\b(dane osobowe|profil prywatny)\b",
    )
    requests_restricted_data = any(re.search(pattern, lowered) for pattern in restricted_patterns)
    if requests_restricted_data and "personal_data:query" not in user_context.data_permissions:
        return FilterResult(
            name="permission_gate",
            decision=Decision.BLOCK,
            score=0.0,
            reason="User is allowed to ask questions, but not to access personal data.",
        )
    return FilterResult("permission_gate", Decision.ALLOW, 1.0, "Permission rule satisfied.")


def jailbreak_gate(text: str, user_context: UserContext) -> FilterResult:
    matches = [pattern.pattern for pattern in JAILBREAK_PATTERNS if pattern.search(text)]
    if matches:
        return FilterResult(
            name="jailbreak_gate",
            decision=Decision.BLOCK,
            score=0.0,
            reason="Prompt attempts to override system, developer, or safety controls.",
        )
    return FilterResult("jailbreak_gate", Decision.ALLOW, 1.0, "No override attempt detected.")


def normalize_input(text: str, user_context: UserContext) -> FilterResult:
    normalized = re.sub(r"\s+", " ", text).strip()
    if normalized != text:
        return FilterResult(
            name="normalize_input",
            decision=Decision.TRANSFORM,
            score=1.0,
            reason="Whitespace normalized before model call.",
            transformed_text=normalized,
        )
    return FilterResult("normalize_input", Decision.ALLOW, 1.0, "Input already normalized.")


PRE_MODEL_FILTERS: list[Callable[[str, UserContext], FilterResult]] = [
    context_gate,
    confidence_gate,
    dlp_scanner,
    permission_gate,
    jailbreak_gate,
    normalize_input,
]


def evaluate_pre_model_filters(text: str, user_context: UserContext) -> GatewayDecision:
    current = text
    results: list[FilterResult] = []

    for filter_fn in PRE_MODEL_FILTERS:
        result = filter_fn(current, user_context)
        results.append(result)

        if result.decision == Decision.BLOCK:
            return GatewayDecision(
                allowed=False,
                final_input=None,
                results=results,
                audit_id=stable_audit_id(text, user_context),
                created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            )

        if result.decision == Decision.TRANSFORM and result.transformed_text is not None:
            current = result.transformed_text

    return GatewayDecision(
        allowed=True,
        final_input=current,
        results=results,
        audit_id=stable_audit_id(text, user_context),
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )


def call_model_only_after_gateway(
    text: str,
    user_context: UserContext,
    model_call: Callable[[str], str],
) -> dict:
    decision = evaluate_pre_model_filters(text, user_context)
    if not decision.allowed:
        return {
            "sent_to_model": False,
            "gateway": asdict(decision),
            "model_response": None,
        }

    return {
        "sent_to_model": True,
        "gateway": asdict(decision),
        "model_response": model_call(decision.final_input or ""),
    }


def demo_model_call(text: str) -> str:
    return f"Model received sanitized input: {text[:120]}"


if __name__ == "__main__":
    context = UserContext(
        user_id="demo-user",
        roles={"analyst"},
        verified_sources=["customer-policy.md"],
        confidence=0.72,
        data_permissions={"question:ask"},
    )

    request = "Summarize this email: test@example.com. Do not expose personal data."
    result = call_model_only_after_gateway(request, context, demo_model_call)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
