# AutoStream Conversational Lead Agent

# DEMO :
https://drive.google.com/file/d/17SfthqK_ofeu_Zwg5wvm3iGMtTfGnE43/view?usp=sharing



This project implements a stateful conversational AI agent for the fictional SaaS product AutoStream. It supports intent classification, local-file RAG, lead qualification, and guarded tool execution.

## Features

- Intent identification across three labels:
  - greeting
  - product_or_pricing_inquiry
  - high_intent_lead
- RAG powered responses from local knowledge base file at data/autostream_kb.json
- Lead qualification flow that collects:
  - name
  - email
  - creator platform
- Tool calling safety:
  - mock_lead_capture is triggered only after all required fields are collected
- Memory across turns through LangGraph state + thread-based checkpointing
- Two interfaces:
  - CLI chat
  - minimal web chat UI

## Knowledge Base Included

- Basic Plan: $29/month, 10 videos/month, 720p resolution
- Pro Plan: $79/month, Unlimited videos, 4K resolution, AI captions
- Policies:
  - No refunds after 7 days
  - 24/7 support available only on Pro plan

## Tech Stack

- Python 3.9+
- LangGraph + LangChain
- Gemini 1.5 Flash via langchain-google-genai
- Flask for local demo web UI
- pytest for tests

## Local Setup

1. Create virtual environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Install dependencies

pip install -r requirements.txt

3. Configure environment

Copy .env.example to .env and provide a valid GOOGLE_API_KEY.

4. Run CLI

python main.py

5. Run web UI

python -m src.web_app

Then open http://127.0.0.1:5000

## Architecture Explanation

LangGraph was chosen because this use case requires explicit, stateful, multi-turn routing rather than a single prompt-response loop. The graph models the conversation lifecycle as deterministic nodes: classify intent, answer product/pricing questions with local retrieval context, run lead qualification, and finally execute the lead capture tool. This makes behavior transparent, testable, and safe for business workflows where a tool call must only happen under strict conditions.

State is managed with a typed AgentState that stores messages, intent label, missing qualification fields, captured lead details, retrieval context, and capture status. MemorySaver checkpointing is used with a thread_id so the same conversation can persist across 5 to 6 turns in both CLI and web sessions. This allows intent shift handling, for example from pricing inquiry to high-intent signup, without losing previously collected context. RAG is intentionally local-file based using data/autostream_kb.json, with lightweight retrieval over relevant chunks to keep answers grounded in known product and policy data.

## WhatsApp Integration via Webhooks

To integrate this agent with WhatsApp in production, use the WhatsApp Business Cloud API webhook flow:

1. Create a webhook endpoint (for example, FastAPI or Flask route) that receives inbound message events from Meta.
2. Verify webhook signatures and map each incoming WhatsApp user ID to a stable conversation thread_id.
3. Pass inbound text into the same agent service used in this project, preserving thread_id for memory continuity.
4. Return or send the agent response via WhatsApp send-message API.
5. For high-intent leads, on successful tool trigger, store captured data in CRM or backend service instead of only printing.
6. Add retry logic, idempotency keys, and queueing for reliability under burst traffic.
7. Add observability (structured logs, metrics, traces) and PII controls for compliance.

## Running Tests

.\venv\Scripts\python -m pytest -q

## Demo Checklist

Record a 2 to 3 minute video showing:

1. Pricing question answered from KB
2. Intent shift to high-intent lead
3. Agent collecting name, email, and platform
4. Successful mock lead capture only after all details are provided
