from __future__ import annotations

from enum import Enum


class IssueStatus(str, Enum):
    NEW = "NEW"
    IDENTIFIED = "IDENTIFIED"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_USER_INFO = "WAITING_USER_INFO"
    SOLUTION_PROVIDED = "SOLUTION_PROVIDED"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


# Valid transitions: from_status -> {allowed to_status}
VALID_TRANSITIONS: dict[IssueStatus, set[IssueStatus]] = {
    IssueStatus.NEW: {IssueStatus.IDENTIFIED},
    IssueStatus.IDENTIFIED: {IssueStatus.IN_PROGRESS},
    IssueStatus.IN_PROGRESS: {
        IssueStatus.WAITING_USER_INFO,
        IssueStatus.SOLUTION_PROVIDED,
        IssueStatus.ESCALATED,
    },
    IssueStatus.WAITING_USER_INFO: {
        IssueStatus.IN_PROGRESS,
        IssueStatus.ESCALATED,
    },
    IssueStatus.SOLUTION_PROVIDED: {
        IssueStatus.RESOLVED,
        IssueStatus.IN_PROGRESS,  # user has follow-up questions
        IssueStatus.ESCALATED,
    },
    IssueStatus.RESOLVED: set(),
    IssueStatus.ESCALATED: set(),
}


# Transition triggers
TRANSITION_TRIGGERS = {
    (IssueStatus.NEW, IssueStatus.IDENTIFIED): "user_identified",
    (IssueStatus.IDENTIFIED, IssueStatus.IN_PROGRESS): "issue_raised",
    (IssueStatus.IN_PROGRESS, IssueStatus.WAITING_USER_INFO): "missing_key_info",
    (IssueStatus.IN_PROGRESS, IssueStatus.SOLUTION_PROVIDED): "solution_given",
    (IssueStatus.IN_PROGRESS, IssueStatus.ESCALATED): "escalation_triggered",
    (IssueStatus.WAITING_USER_INFO, IssueStatus.IN_PROGRESS): "info_received",
    (IssueStatus.WAITING_USER_INFO, IssueStatus.ESCALATED): "timeout_or_repeated",
    (IssueStatus.SOLUTION_PROVIDED, IssueStatus.RESOLVED): "user_confirmed",
    (IssueStatus.SOLUTION_PROVIDED, IssueStatus.IN_PROGRESS): "follow_up_issue",
    (IssueStatus.SOLUTION_PROVIDED, IssueStatus.ESCALATED): "user_dissatisfied",
}


class IssueStateMachine:
    def __init__(self, current_status: IssueStatus = IssueStatus.NEW):
        self.current_status = current_status

    @classmethod
    def can_transition(cls, from_status: IssueStatus, to_status: IssueStatus) -> bool:
        allowed = VALID_TRANSITIONS.get(from_status, set())
        return to_status in allowed

    @classmethod
    def get_trigger(cls, from_status: IssueStatus, to_status: IssueStatus) -> str | None:
        return TRANSITION_TRIGGERS.get((from_status, to_status))

    def transition(self, to_status: IssueStatus) -> tuple[bool, str | None]:
        if not self.can_transition(self.current_status, to_status):
            return False, None
        trigger = self.get_trigger(self.current_status, to_status)
        self.current_status = to_status
        return True, trigger

    def transition_to(self, to_status: IssueStatus) -> str | None:
        ok, trigger = self.transition(to_status)
        if not ok:
            raise ValueError(
                f"Invalid transition: {self.current_status.value} -> {to_status.value}"
            )
        return trigger


def should_request_clarification(confidence: float | None, risk_tags: list[str] | None = None) -> bool:
    if confidence is not None and confidence < 0.5:
        return True
    if risk_tags and "needs_clarification" in risk_tags:
        return True
    return False


def should_escalate(
    issue_status: IssueStatus,
    unanswered_rounds: int = 0,
    risk_tags: list[str] | None = None,
) -> bool:
    if risk_tags and "escalation_risk" in risk_tags:
        return True
    if risk_tags and "safety_concern" in risk_tags:
        return True
    if unanswered_rounds >= 3:
        return True
    if issue_status == IssueStatus.WAITING_USER_INFO and unanswered_rounds >= 2:
        return True
    return False
