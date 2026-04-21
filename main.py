from __future__ import annotations

import uuid

from src.agent_service import agent


def run_cli() -> None:
    thread_id = str(uuid.uuid4())
    print("AutoStream Agent CLI. Type 'exit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        out = agent.process(user_input, thread_id=thread_id)
        print(f"Agent: {out.text}")


if __name__ == "__main__":
    run_cli()
