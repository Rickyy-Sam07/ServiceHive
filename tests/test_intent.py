from src.intent import detect_intent


def test_detect_greeting() -> None:
    assert detect_intent("Hi there") == "greeting"


def test_detect_pricing_inquiry() -> None:
    assert detect_intent("Can you share pricing and feature details?") == "product_or_pricing_inquiry"


def test_pricing_question_not_high_intent() -> None:
    assert detect_intent("What is included in the Pro plan?") == "product_or_pricing_inquiry"


def test_detect_high_intent() -> None:
    assert detect_intent("Sounds good, I want to sign up for Pro") == "high_intent_lead"
