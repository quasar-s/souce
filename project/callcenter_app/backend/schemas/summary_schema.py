from pydantic import BaseModel


class SummaryRequest(BaseModel):
    transcript: str


class CallSummary(BaseModel):
    summary: str
    keywords: list[str]
    category: str
    sentiment: str
    action_items: list[str]
    customer_issue: str
    resolution: str


# 상담 요청시
class CallRequest(BaseModel):
    customer_id: int
    transcript: str


# 상담 요약 저장
class CallCreate(BaseModel):
    customer_id: int
    transcript: str
    summary: str
    category: str
    sentiment: str
    customer_issue: str
    resolution: str
