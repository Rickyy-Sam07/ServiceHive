from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from src.graph import build_graph
from src.state import default_lead_details


@dataclass
class AgentResponse:
    text: str
    intent: str
    missing_fields: list[str]
    lead_captured: bool


class AutoStreamAgent:
    def __init__(self) -> None:
        self.graph = build_graph()

    def process(self, user_message: str, thread_id: str) -> AgentResponse:
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        return AgentResponse(
            text=result.get("response", ""),
            intent=result.get("intent", "product_or_pricing_inquiry"),
            missing_fields=result.get("missing_fields", []),
            lead_captured=bool(result.get("lead_captured", False)),
        )


agent = AutoStreamAgent()
