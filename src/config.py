from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    llm_fallback_models: tuple[str, ...] = tuple(
        m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "gemini-2.5-flash").split(",") if m.strip()
    )
    use_llm_intent_fallback: bool = os.getenv("USE_LLM_INTENT_FALLBACK", "true").lower() == "true"
    flask_secret_key: str = os.getenv("FLASK_SECRET_KEY", "dev-secret")


settings = Settings()
