from jugglecount.db.db import engine
from jugglecount.db.schema import User, JugglingSession, ThrowEvent, Run
from sqlmodel import Session as DbSession, select

def verify():
    with DbSession(engine) as session:
        # Check users
        users = session.exec(select(User)).all()
        print(f"Users found: {len(users)}")
        for u in users:
            print(f"  - User: {u.user_id} ({u.name})")
        
        # Check sessions
        sessions = session.exec(select(JugglingSession)).all()
        print(f"Sessions found: {len(sessions)}")
        
        # Check throws
        throws = session.exec(select(ThrowEvent)).all()
        print(f"Throws found: {len(throws)}")
        
        # Check runs
        runs = session.exec(select(Run)).all()
        print(f"Runs found: {len(runs)}")
        
        if len(throws) > 0 and len(runs) > 0:
            print("SUCCESS: Throws and Runs are persisted.")
        else:
            print("FAILURE: Throws or Runs missing.")

if __name__ == "__main__":
    verify()
