"""Feedback persistence shared by the HTTP and in-process adapters."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from core.schemas import FeedbackRequest
from db.models import Feedback, Message


def record_feedback(session_factory: sessionmaker[Session], request: FeedbackRequest) -> None:
    with session_factory() as db:
        message = db.get(Message, request.message_id)
        if message is None:
            raise LookupError("message not found")
        if message.role != "assistant":
            raise ValueError("feedback is only accepted for assistant messages")
        db.add(Feedback(message_id=request.message_id, rating=request.rating, comment=request.comment))
        db.commit()
