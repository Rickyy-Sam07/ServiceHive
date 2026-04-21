from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeadCaptureTracker:
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.calls.clear()


lead_capture_tracker = LeadCaptureTracker()


def mock_lead_capture(name: str, email: str, platform: str) -> None:
    lead_capture_tracker.calls.append((name, email, platform))
    print(f"Lead captured successfully: {name}, {email}, {platform}")
