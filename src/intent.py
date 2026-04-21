from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.state import IntentLabel

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None


GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good evening"}
PRICING_WORDS = {
    "price",
    "pricing",
    "plan",
    "plans",
    "feature",
    "features",
    "cost",
    "refund",
    "support",
}
HIGH_INTENT_WORDS = {
    "sign up",
    "signup",
    "start",
    "get started",
    "buy",
    "purchase",
    "subscribe",
    "trial",
    "i want",
    "i am interested",
    "sounds good",
}

HIGH_INTENT_PATTERNS = [
    r"\bi\s+want\s+to\s+(sign\s*up|get\s*started|buy|purchase|subscribe|try)\b",
    r"\bready\s+to\s+(sign\s*up|get\s*started|buy|purchase|subscribe)\b",
    r"\bsign\s*me\s*up\b",
    r"\bstart\s+my\s+(trial|subscription)\b",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _rule_based_intent(text: str) -> IntentLabel:
    normalized = _normalize(text)

    if any(word in normalized for word in HIGH_INTENT_WORDS):
        return "high_intent_lead"
    if any(word in normalized for word in PRICING_WORDS):
        return "product_or_pricing_inquiry"
    if normalized in GREETING_WORDS or any(normalized.startswith(w) for w in GREETING_WORDS):
        return "greeting"

    return "product_or_pricing_inquiry"


def _has_strong_high_intent_signal(text: str) -> bool:
    normalized = _normalize(text)
    if any(re.search(pattern, normalized) for pattern in HIGH_INTENT_PATTERNS):
        return True
    return any(word in normalized for word in HIGH_INTENT_WORDS)


def _llm_intent(text: str) -> Optional[IntentLabel]:
    if not settings.google_api_key or not settings.use_llm_intent_fallback or ChatGoogleGenerativeAI is None:
        return None

    candidate_models: list[str] = []
    for model_name in (settings.llm_model, *settings.llm_fallback_models):
        if model_name and model_name not in candidate_models:
            candidate_models.append(model_name)

    prompt = (
        "Classify user intent into one label only: greeting, product_or_pricing_inquiry, high_intent_lead. "
        "Return only the label text."
    )
    msg = [SystemMessage(content=prompt), HumanMessage(content=text)]

    out = None
    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.google_api_key,
                temperature=0,
            )
            out = llm.invoke(msg).content.strip().lower()
            break
        except Exception:
            continue

    if out is None:
        return None

    valid: set[str] = {"greeting", "product_or_pricing_inquiry", "high_intent_lead"}
    if out in valid:
        return out  # type: ignore[return-value]

    return None


def detect_intent(text: str) -> IntentLabel:
    if _has_strong_high_intent_signal(text):
        return "high_intent_lead"

    rule_intent = _rule_based_intent(text)
    if rule_intent == "product_or_pricing_inquiry":
        return rule_intent

    llm_intent = _llm_intent(text)
    if llm_intent:
        return llm_intent

    return rule_intent
