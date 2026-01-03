import os
from typing import List, Optional
from sqlmodel import SQLModel, create_engine, Session, select
from pydantic import BaseModel
from jugglecount.db.schema import User, JugglingSession, ThrowEvent, Run

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///jugglecount.db")
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)

def upsert_user(user_id: str, name: str, email: str) -> User:
    with Session(engine) as session:
        statement = select(User).where(User.user_id == user_id)
        user = session.exec(statement).first()
        if not user:
            user = User(user_id=user_id, name=name, email=email)
            session.add(user)
        else:
            user.name = name
            user.email = email
            session.add(user)
        session.commit()
        session.refresh(user)
        return user

def save_session_results(
    user_id: str, 
    session_uuid: str, 
    start_time: float, 
    end_time: float, 
    metrics_data: dict, # Kept for signature compatibility but ignored
    throws: List[ThrowEvent] = None,
    runs: List[Run] = None
) -> JugglingSession:
    with Session(engine, expire_on_commit=False) as session:
        # Create session
        db_session = JugglingSession(
            session_uuid=session_uuid,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        session.add(db_session)
        session.commit()
        session.refresh(db_session)
        
        # Save individual throws
        if throws:
            for t in throws:
                t.session_pk = db_session.id
                session.add(t)
                
        # Save individual runs
        if runs:
            for r in runs:
                r.session_pk = db_session.id
                session.add(r)
                
        session.commit()
        session.refresh(db_session)
        return db_session

class SessionStats(BaseModel):
    timestamp: float
    total_throws: int
    duration_seconds: float

def get_user_performance_history(user_id: str) -> List[SessionStats]:
    with Session(engine) as session:
        statement = select(JugglingSession).where(JugglingSession.user_id == user_id).order_by(JugglingSession.start_time)
        sessions = session.exec(statement).all()
        
        history = []
        for s in sessions:
            # Aggregate runs for total throws
            # Assuming s.runs is populated (lazy loading might require explicit join if sessions are detached)
            # But since we are in the session context, it should load.
            total_throws = sum(r.throw_count for r in s.runs)
            history.append(SessionStats(
                timestamp=s.end_time, # Using end_time as the specific timestamp
                total_throws=total_throws,
                duration_seconds=s.end_time - s.start_time if s.end_time > s.start_time else 0.0
            ))
        return history
def get_all_users() -> List[User]:
    with Session(engine, expire_on_commit=False) as session:
        return session.exec(select(User)).all()
