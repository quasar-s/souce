from fastapi import APIRouter
from backend.schemas.summary_schema import CallSummary, SummaryRequest, CallRequest
from backend.services.call_service import summary_call, create_call_history
from fastapi import Depends
from sqlalchemy.orm import Session
from backend.repository.db_init import get_db

router = APIRouter(prefix="/api/calls", tags=["Summary"])


# 요약 라우터
@router.post("", response_model=CallSummary)
def generate_summary(req: SummaryRequest):
    return summary_call(req.transcript)


@router.post("/save", response_model=CallSummary)
def creat_call(req: CallRequest, db: Session = Depends(get_db)):
    return create_call_history(req, db)
