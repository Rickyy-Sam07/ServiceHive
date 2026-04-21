from __future__ import annotations

import uuid
from collections import defaultdict

from flask import Flask, jsonify, render_template, request, session

from src.config import settings
from src.agent_service import agent


app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = settings.flask_secret_key

chat_logs: dict[str, list[dict[str, str]]] = defaultdict(list)


@app.get("/")
def index():
    if "chat_id" not in session:
        session["chat_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.get("/api/history")
def history():
    chat_id = session.get("chat_id")
    if not chat_id:
        return jsonify([])
    return jsonify(chat_logs[chat_id])


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    if "chat_id" not in session:
        session["chat_id"] = str(uuid.uuid4())
    chat_id = session["chat_id"]

    chat_logs[chat_id].append({"role": "user", "text": message})
    out = agent.process(message, thread_id=chat_id)
    chat_logs[chat_id].append({"role": "agent", "text": out.text})

    return jsonify(
        {
            "reply": out.text,
            "intent": out.intent,
            "missing_fields": out.missing_fields,
            "lead_captured": out.lead_captured,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
