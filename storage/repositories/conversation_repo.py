from __future__ import annotations

from sqlalchemy.orm import Session

from storage.models import Message, Session as SessionModel


class ConversationRepo:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_session(self, session_id: str | None, customer_id: str | None = None) -> SessionModel:
        if session_id:
            existing = self.db.query(SessionModel).filter(SessionModel.id == session_id).first()
            if existing:
                return existing
        new_session = SessionModel(id=session_id, customer_id=customer_id)
        self.db.add(new_session)
        self.db.commit()
        self.db.refresh(new_session)
        return new_session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        trace_id: str | None = None,
    ) -> Message:
        msg = Message(session_id=session_id, role=role, content=content, trace_id=trace_id)
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        return msg

    def get_session_messages(self, session_id: str) -> list[Message]:
        return (
            self.db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at)
            .all()
        )
