from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from storage.models import Issue, IssueEvent


class IssueRepo:
    def __init__(self, db: Session):
        self.db = db

    def create_issue(
        self,
        customer_id: str | None = None,
        session_id: str | None = None,
        title: str | None = None,
        status: str = "NEW",
    ) -> Issue:
        issue = Issue(customer_id=customer_id, session_id=session_id, title=title, status=status)
        self.db.add(issue)
        self.db.commit()
        self.db.refresh(issue)
        self._add_event(issue.id, "NEW", status, "issue_created")
        return issue

    def get_issue(self, issue_id: str) -> Issue | None:
        return self.db.query(Issue).filter(Issue.id == issue_id).first()

    def get_issues_by_session(self, session_id: str) -> list[Issue]:
        return self.db.query(Issue).filter(Issue.session_id == session_id).all()

    def update_status(self, issue_id: str, new_status: str, trigger: str | None = None) -> Issue:
        issue = self.db.query(Issue).filter(Issue.id == issue_id).first()
        if not issue:
            raise ValueError(f"Issue {issue_id} not found")
        old_status = issue.status
        issue.status = new_status
        issue.updated_at = datetime.now(timezone.utc)
        if new_status == "RESOLVED":
            issue.resolved_at = datetime.now(timezone.utc)
        elif new_status == "ESCALATED":
            issue.escalated_at = datetime.now(timezone.utc)
        self.db.commit()
        self._add_event(issue_id, old_status, new_status, trigger)
        return issue

    def _add_event(self, issue_id: str, from_status: str, to_status: str, trigger: str | None = None) -> IssueEvent:
        event = IssueEvent(
            issue_id=issue_id,
            from_status=from_status,
            to_status=to_status,
            trigger=trigger,
        )
        self.db.add(event)
        self.db.commit()
        return event

    def get_events(self, issue_id: str) -> list[IssueEvent]:
        return (
            self.db.query(IssueEvent)
            .filter(IssueEvent.issue_id == issue_id)
            .order_by(IssueEvent.created_at)
            .all()
        )
