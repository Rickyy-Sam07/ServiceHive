from src.rag import format_pricing_summary, retrieve_context


def test_retrieve_context_contains_pro_plan() -> None:
    context = retrieve_context("What does Pro plan include?")
    assert "Pro Plan" in context
    assert "$79/month" in context


def test_format_summary_contains_policy() -> None:
    summary = format_pricing_summary()
    assert "No refunds after 7 days" in summary
