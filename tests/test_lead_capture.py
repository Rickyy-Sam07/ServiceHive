from src.agent_service import AutoStreamAgent
from src.tools import lead_capture_tracker


def test_no_premature_lead_capture() -> None:
    lead_capture_tracker.reset()
    agent = AutoStreamAgent()
    thread_id = "test-thread-1"

    out1 = agent.process("I want to sign up for Pro plan", thread_id)
    assert out1.lead_captured is False
    assert "name" in out1.missing_fields
    assert len(lead_capture_tracker.calls) == 0

    out2 = agent.process("My name is Sam", thread_id)
    assert out2.lead_captured is False
    assert "email" in out2.missing_fields
    assert len(lead_capture_tracker.calls) == 0

    out3 = agent.process("sam@example.com", thread_id)
    assert out3.lead_captured is False
    assert "platform" in out3.missing_fields
    assert len(lead_capture_tracker.calls) == 0

    out4 = agent.process("YouTube", thread_id)
    assert out4.lead_captured is True
    assert len(lead_capture_tracker.calls) == 1


def test_plain_name_is_accepted() -> None:
    lead_capture_tracker.reset()
    agent = AutoStreamAgent()
    thread_id = "test-thread-2"

    out1 = agent.process("I want to sign up for Pro plan", thread_id)
    assert "name" in out1.missing_fields

    out2 = agent.process("sambhranta", thread_id)
    assert out2.lead_captured is False
    assert "email" in out2.missing_fields
    assert len(lead_capture_tracker.calls) == 0


def test_second_signup_does_not_reuse_old_lead() -> None:
    lead_capture_tracker.reset()
    agent = AutoStreamAgent()
    thread_id = "test-thread-3"

    # First successful lead capture.
    agent.process("I want to sign up for Pro plan", thread_id)
    agent.process("My name is Sam", thread_id)
    agent.process("sam@example.com", thread_id)
    out1 = agent.process("YouTube", thread_id)
    assert out1.lead_captured is True
    assert len(lead_capture_tracker.calls) == 1

    # New high-intent should start fresh and ask for details again.
    out2 = agent.process("I want to get started", thread_id)
    assert out2.lead_captured is False
    assert "name" in out2.missing_fields
    assert len(lead_capture_tracker.calls) == 1


def test_incomplete_lead_can_switch_to_pricing() -> None:
    lead_capture_tracker.reset()
    agent = AutoStreamAgent()
    thread_id = "test-thread-4"

    out1 = agent.process("I want to sign up", thread_id)
    assert "name" in out1.missing_fields

    out2 = agent.process("Which plan should I pick?", thread_id)
    assert out2.lead_captured is False
    assert "Pricing:" in out2.text
    assert len(lead_capture_tracker.calls) == 0
