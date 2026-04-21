from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


IntentLabel = Literal["greeting", "product_or_pricing_inquiry", "high_intent_lead"]


class LeadDetails(TypedDict):
    name: str | None
    email: str | None
    platform: str | None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: IntentLabel
    missing_fields: list[str]
    lead_details: LeadDetails
    lead_captured: bool
    retrieved_context: str
    response: str


def default_lead_details() -> LeadDetails:
    return {"name": None, "email": None, "platform": None}
