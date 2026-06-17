from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    question: str


class AnalysisResponse(BaseModel):
    message: str
