from __future__ import annotations

import json
import re
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "autostream_kb.json"


def load_kb() -> dict:
    with KB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def kb_to_chunks(kb: dict) -> list[str]:
    chunks: list[str] = []
    for plan in kb.get("plans", []):
        features = ", ".join(plan.get("features", []))
        chunks.append(f"{plan['name']}: {plan['price']}. Features: {features}.")

    for policy in kb.get("policies", []):
        chunks.append(f"Policy: {policy}.")

    return chunks


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def retrieve_context(query: str, top_k: int = 3) -> str:
    kb = load_kb()
    chunks = kb_to_chunks(kb)
    q = _tokenize(query)

    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        score = len(q.intersection(_tokenize(chunk)))
        scored.append((score, chunk))

    ranked = [c for s, c in sorted(scored, key=lambda x: x[0], reverse=True) if s > 0]
    if not ranked:
        ranked = chunks[:top_k]

    return "\n".join(ranked[:top_k])


def format_pricing_summary() -> str:
    kb = load_kb()
    basic, pro = kb["plans"]
    return (
        f"{basic['name']}: {basic['price']} ({', '.join(basic['features'])}).\n"
        f"{pro['name']}: {pro['price']} ({', '.join(pro['features'])}).\n"
        f"Policies: {kb['policies'][0]}; {kb['policies'][1]}."
    )
