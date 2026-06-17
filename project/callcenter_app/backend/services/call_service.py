from langchain_core.prompts import ChatPromptTemplate

from sqlalchemy.orm import Session

from backend.repository.models import CallHistory
from backend.schemas.summary_schema import CallSummary, CallRequest, CallCreate
from backend.prompts.all_prompt import SUMMARY_SYSTEM_PROMPT
from backend.ai.llm import hugging_llm


# LLM transcript 요약시키기
def summary_call(transcript: str):
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human", "상담내용\n{transcript}"),
        ]
    )

    structured_llm = hugging_llm.with_structured_output(CallSummary)

    summary_chain = summary_prompt | structured_llm

    result = summary_chain.invoke({"transcript": transcript})
    return result


def save_call_history(db, data):
    """
    상담 저장
    """
    history = CallHistory(
        customer_id=data.customer_id,
        transcript=data.transcript,
        summary=data.summary,
        category=data.category,
        sentiment=data.sentiment,
        customer_issue=data.customer_issue,
        resolution=data.resolution,
    )

    db.add(history)  # 스테이징 영역에 추가
    db.commit()
    db.refresh(history)

    return history


def evaluate_call():
    """
    상담 평가
    """
    pass


def save_call_evaluation():
    """
    상담 평가 내용 저장
    """
    pass


def create_call_history(req: CallRequest, db: Session):
    # 요약
    summary = summary_call(req.transcript)

    # DataBase 저장
    call_data = CallCreate(
        customer_id=req.customer_id,
        transcript=req.transcript,
        summary=summary.summary,
        category=summary.category,
        sentiment=summary.sentiment,
        customer_issue=summary.customer_issue,
        resolution=summary.resolution,
    )

    return save_call_history(db=db, data=call_data)
