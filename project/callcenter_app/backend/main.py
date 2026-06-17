from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.repository.db_init import Base, SessionLocal, engine
from backend.routers.call_router import router as call_router
from backend.repository.seed import seed_customers


# 앱 시작 시 자동 실핼
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작")

    # Base에 등록된 모든 모델의 테이블 자동 생성
    Base.metadata.create_all(bind=engine)
    print("[DB]테이블 생성 완료 (또는 이미 존재)")

    db = SessionLocal()
    try:
        seed_customers(db)
    finally:
        db.close()

    yield
    print("서버 종료")


app = FastAPI(title="상담 LLM", version="1.0", lifespan=lifespan)


# static 폴더 지정
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# # 라우터 등록

app.include_router(call_router)
# app.include_router()
