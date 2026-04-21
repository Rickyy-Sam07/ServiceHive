from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.intent import detect_intent
from src.rag import load_kb, retrieve_context
from src.state import AgentState, default_lead_details
from src.tools import mock_lead_capture
from src.config import settings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PLATFORM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\byoutube\b", re.IGNORECASE), "YouTube"),
    (re.compile(r"\binstagram\b", re.IGNORECASE), "Instagram"),
    (re.compile(r"\btiktok\b", re.IGNORECASE), "TikTok"),
    (re.compile(r"\blinkedin\b", re.IGNORECASE), "LinkedIn"),
    (re.compile(r"\b(?:x|twitter)\b", re.IGNORECASE), "X"),
    (re.compile(r"\bfacebook\b", re.IGNORECASE), "Facebook"),
]

PRICING_SWITCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(pricing|price|plan|plans|feature|features|cost|refund|support|basic|pro)\b", re.IGNORECASE),
    re.compile(r"\b(which|what|tell me|compare|difference|included|include|pick)\b", re.IGNORECASE),
]


def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _extract_name(text: str) -> str | None:
    m = re.search(r"(?:i am|i'm|my name is)\s+([A-Za-z][A-Za-z\s]{1,40})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().title()

    candidate = text.strip()
    if EMAIL_PATTERN.match(candidate):
        return None
    if _extract_platform(candidate):
        return None
    if re.fullmatch(r"[A-Za-z][A-Za-z\s]{1,40}", candidate):
        words = candidate.split()
        if 1 <= len(words) <= 3:
            return " ".join(w.capitalize() for w in words)

    return None


def _extract_email(text: str) -> str | None:
    for token in re.split(r"[\s,;]+", text):
        if EMAIL_PATTERN.match(token.strip()):
            return token.strip()
    return None


def _extract_platform(text: str) -> str | None:
    for pattern, label in PLATFORM_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _missing_fields(details: dict[str, str | None]) -> list[str]:
    out: list[str] = []
    if not details.get("name"):
        out.append("name")
    if not details.get("email"):
        out.append("email")
    if not details.get("platform"):
        out.append("platform")
    return out


def _is_pricing_switch_query(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return False
    if "?" in lower:
        return any(p.search(lower) for p in PRICING_SWITCH_PATTERNS)
    return bool(
        re.search(r"\b(tell me about|what are|which plan|compare plans|do you offer refunds|support)\b", lower)
    )


def _format_plan_summary(rows: list[tuple[str, str, str]]) -> str:
    lines = ["Pricing:"]
    for plan, price, includes in rows:
        lines.append(f"- {plan}: {price} | {includes}")
    return "\n".join(lines)


def _sanitize_llm_output(text: str) -> str:
    cleaned = text.replace("**", "").replace("__", "").replace("`", "")
    return cleaned.strip()


def _llm_grounded_reply(query: str, kb: dict, context: str) -> str | None:
    # Keep tests deterministic even if local env has keys.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    if not settings.google_api_key or ChatGoogleGenerativeAI is None:
        return None

    candidate_models: list[str] = []
    for model_name in (settings.llm_model, *settings.llm_fallback_models):
        if model_name and model_name not in candidate_models:
            candidate_models.append(model_name)

    facts = []
    for plan in kb.get("plans", []):
        facts.append(
            f"{plan['name']}: {plan['price']}; features: {', '.join(plan.get('features', []))}"
        )
    for policy in kb.get("policies", []):
        facts.append(f"Policy: {policy}")

    prompt = (
        "You are AutoStream's sales assistant.\n"
        "Rules:\n"
        "1) Use only the provided facts. Do not invent features, prices, or policies.\n"
        "2) If user asks recommendation, reason from their usage and compare plans briefly.\n"
        "3) Keep response concise and structured.\n"
        "4) For pricing/plan queries, include this format:\n"
        "Pricing:\n- Plan: price | features\n"
        "5) Mention policies when relevant.\n\n"
        "6) Output plain text only. Do not use markdown symbols such as **, __, or backticks.\n\n"
        f"User query: {query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Allowed facts:\n" + "\n".join(facts)
    )

    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.google_api_key,
                temperature=0.2,
            )
            return _sanitize_llm_output(str(llm.invoke(prompt).content))
        except Exception:
            continue

    return None


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    text = _last_user_message(state)

    # If qualification is in progress, allow users to switch back to pricing questions.
    if not state.get("lead_captured", False) and state.get("missing_fields"):
        if _is_pricing_switch_query(text):
            return {
                "intent": "product_or_pricing_inquiry",
                "missing_fields": [],
                "lead_details": default_lead_details(),
            }
        return {"intent": "high_intent_lead"}

    intent = detect_intent(text)

    if intent == "high_intent_lead":
        print("[DEMO] Agent detecting high-intent")

    # Start a fresh qualification cycle for a new signup request after prior capture.
    if intent == "high_intent_lead" and state.get("lead_captured", False):
        return {
            "intent": intent,
            "lead_captured": False,
            "lead_details": default_lead_details(),
            "missing_fields": [],
        }

    return {"intent": intent}


def product_pricing_node(state: AgentState) -> dict[str, Any]:
    text = _last_user_message(state)
    context = retrieve_context(text)
    kb = load_kb()

    llm_response = _llm_grounded_reply(text, kb, context)
    if llm_response:
        return {
            "retrieved_context": context,
            "response": llm_response,
            "messages": [AIMessage(content=llm_response)],
        }

    plans = {p["name"].lower(): p for p in kb.get("plans", [])}
    basic = plans.get("basic plan")
    pro = plans.get("pro plan")
    policies = kb.get("policies", [])
    query = text.lower()

    if "refund" in query:
        response = "No refunds are available after 7 days."
    elif "support" in query:
        response = "24/7 support is available only on the Pro plan."
    elif "pro" in query:
        response = _format_plan_summary(
            [
                (
                    "Pro Plan",
                    pro["price"],
                    ", ".join(pro["features"]),
                )
            ]
        )
        response += "\n\n24/7 support is included on Pro."
    elif "basic" in query:
        response = _format_plan_summary(
            [
                (
                    "Basic Plan",
                    basic["price"],
                    ", ".join(basic["features"]),
                )
            ]
        )
    else:
        response = _format_plan_summary(
            [
                (
                    "Basic Plan",
                    basic["price"],
                    ", ".join(basic["features"]),
                ),
                (
                    "Pro Plan",
                    pro["price"],
                    ", ".join(pro["features"]),
                ),
            ]
        )
        response += f"\n\nPolicies: {policies[0]}; {policies[1]}."

    return {
        "retrieved_context": context,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def greeting_node(state: AgentState) -> dict[str, Any]:
    response = "Hi! I can help with AutoStream pricing, features, and plan selection. What would you like to know?"
    return {"response": response, "messages": [AIMessage(content=response)]}


def lead_qualification_node(state: AgentState) -> dict[str, Any]:
    text = _last_user_message(state)
    details = dict(state.get("lead_details") or default_lead_details())

    maybe_name = _extract_name(text)
    maybe_email = _extract_email(text)
    maybe_platform = _extract_platform(text)

    if maybe_name and not details.get("name"):
        details["name"] = maybe_name
    if maybe_email and not details.get("email"):
        details["email"] = maybe_email
    if maybe_platform and not details.get("platform"):
        details["platform"] = maybe_platform

    missing = _missing_fields(details)

    if missing:
        print(
            "[DEMO] Agent collecting user details "
            f"(have: name={bool(details.get('name'))}, email={bool(details.get('email'))}, platform={bool(details.get('platform'))}; "
            f"missing: {', '.join(missing)})"
        )
        prompts = {
            "name": "Great choice. Could you share your name?",
            "email": "Please share your best email so we can set up your trial.",
            "platform": "Which creator platform do you primarily use (YouTube, Instagram, etc.)?",
        }
        ask = prompts[missing[0]]
        return {
            "lead_details": details,
            "missing_fields": missing,
            "lead_captured": False,
            "response": ask,
            "messages": [AIMessage(content=ask)],
        }

    return {
        "lead_details": details,
        "missing_fields": [],
        "lead_captured": state.get("lead_captured", False),
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    details = state.get("lead_details") or default_lead_details()
    missing = _missing_fields(details)

    if missing:
        response = "I still need a few details before I can register your lead."
        return {
            "missing_fields": missing,
            "lead_captured": False,
            "response": response,
            "messages": [AIMessage(content=response)],
        }

    if not state.get("lead_captured"):
        print("[DEMO] Successful lead capture using mock tool")
        mock_lead_capture(
            details["name"] or "",
            details["email"] or "",
            details["platform"] or "",
        )

    response = (
        "Perfect, you are all set. I have captured your details and our team will contact you shortly."
    )
    return {
        "lead_captured": True,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def intent_router(state: AgentState) -> str:
    intent = state.get("intent", "product_or_pricing_inquiry")
    if intent == "greeting":
        return "greeting"
    if intent == "high_intent_lead":
        return "lead_qualification"
    return "product_pricing"


def lead_router(state: AgentState) -> str:
    if state.get("missing_fields"):
        return END
    return "tool"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("product_pricing", product_pricing_node)
    graph.add_node("lead_qualification", lead_qualification_node)
    graph.add_node("tool", tool_node)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "greeting": "greeting",
            "product_pricing": "product_pricing",
            "lead_qualification": "lead_qualification",
        },
    )

    graph.add_edge("greeting", END)
    graph.add_edge("product_pricing", END)
    graph.add_conditional_edges(
        "lead_qualification",
        lead_router,
        {"tool": "tool", END: END},
    )
    graph.add_edge("tool", END)

    return graph.compile(checkpointer=MemorySaver())
