import pytest
from agent.state import (
    IssueStateMachine,
    IssueStatus,
    should_escalate,
    should_request_clarification,
)


class TestIssueStateMachine:
    def test_initial_status_is_new(self):
        sm = IssueStateMachine()
        assert sm.current_status == IssueStatus.NEW

    def test_new_to_identified(self):
        sm = IssueStateMachine()
        ok, trigger = sm.transition(IssueStatus.IDENTIFIED)
        assert ok
        assert trigger == "user_identified"
        assert sm.current_status == IssueStatus.IDENTIFIED

    def test_identified_to_in_progress(self):
        sm = IssueStateMachine(IssueStatus.IDENTIFIED)
        ok, trigger = sm.transition(IssueStatus.IN_PROGRESS)
        assert ok
        assert trigger == "issue_raised"

    def test_in_progress_to_waiting(self):
        sm = IssueStateMachine(IssueStatus.IN_PROGRESS)
        ok, trigger = sm.transition(IssueStatus.WAITING_USER_INFO)
        assert ok
        assert trigger == "missing_key_info"

    def test_in_progress_to_solution_provided(self):
        sm = IssueStateMachine(IssueStatus.IN_PROGRESS)
        ok, trigger = sm.transition(IssueStatus.SOLUTION_PROVIDED)
        assert ok
        assert trigger == "solution_given"

    def test_in_progress_to_escalated(self):
        sm = IssueStateMachine(IssueStatus.IN_PROGRESS)
        ok, trigger = sm.transition(IssueStatus.ESCALATED)
        assert ok
        assert trigger == "escalation_triggered"

    def test_solution_to_resolved(self):
        sm = IssueStateMachine(IssueStatus.SOLUTION_PROVIDED)
        ok, trigger = sm.transition(IssueStatus.RESOLVED)
        assert ok
        assert trigger == "user_confirmed"

    def test_invalid_transition_returns_false(self):
        sm = IssueStateMachine(IssueStatus.NEW)
        ok, trigger = sm.transition(IssueStatus.RESOLVED)
        assert not ok
        assert trigger is None
        assert sm.current_status == IssueStatus.NEW  # unchanged

    def test_resolved_cannot_transition(self):
        sm = IssueStateMachine(IssueStatus.RESOLVED)
        assert not sm.transition(IssueStatus.IN_PROGRESS)[0]

    def test_escalated_cannot_transition(self):
        sm = IssueStateMachine(IssueStatus.ESCALATED)
        assert not sm.transition(IssueStatus.IN_PROGRESS)[0]

    def test_transition_to_raises_on_invalid(self):
        sm = IssueStateMachine(IssueStatus.RESOLVED)
        with pytest.raises(ValueError):
            sm.transition_to(IssueStatus.IN_PROGRESS)


class TestHelpers:
    def test_should_request_clarification_low_confidence(self):
        assert should_request_clarification(0.4) is True

    def test_should_request_clarification_needs_tag(self):
        assert should_request_clarification(0.8, ["needs_clarification"]) is True

    def test_should_request_clarification_high_confidence(self):
        assert should_request_clarification(0.8) is False

    def test_should_escalate_safety(self):
        assert should_escalate(IssueStatus.IN_PROGRESS, risk_tags=["safety_concern"]) is True

    def test_should_escalate_rounds(self):
        assert should_escalate(IssueStatus.IN_PROGRESS, unanswered_rounds=3) is True

    def test_should_not_escalate_normal(self):
        assert should_escalate(IssueStatus.IN_PROGRESS, unanswered_rounds=1) is False
