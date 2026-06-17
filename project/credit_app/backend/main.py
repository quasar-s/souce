from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from backend.services.db_service import init_db

from backend.routers.api_router import router as api_router
from backend.routers.page_router import router as page_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작")
    init_db()
    yield
    print("서버 종료")


app = FastAPI(lifespan=lifespan, title="Insights Advisor", version="1.0")

# static 폴더 지정
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# 라우터 등록
app.include_router(page_router)
app.include_router(api_router)
