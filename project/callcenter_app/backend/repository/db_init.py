from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import DateTime, func, create_engine
from datetime import datetime
from pathlib import Path

Path("db").mkdir(parents=True, exist_ok=True)

engine = create_engine("sqlite:///db/callcenter.db", echo=True)

# 세션 팩토리 생성
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# Base=DeclarativeBase()
class Base(DeclarativeBase):
    pass


# session을 다른 모듈에서 사용가능하도록
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
