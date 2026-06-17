from fastapi import APIRouter, Request, UploadFile
from backend.services.card_service import (
    upload_csv,
    card_history,
    get_dashboard,
    card_analysis,
)
from fastapi.templating import Jinja2Templates
from backend.schemas.card_schema import AnalysisRequest
from langchain_core.prompts import ChatPromptTemplate
from backend.ai.llm import watson_llm
from langchain_core.output_parsers import StrOutputParser

router = APIRouter(prefix="/api/card")
templates = Jinja2Templates(directory="backend/templates")


@router.post("/upload")
async def upload_file(file: UploadFile):
    return await upload_csv(file)


@router.get("/history")
async def history(request: Request):

    card_infos = card_history()

    return templates.TemplateResponse(
        request=request, name="history.html", context={"history": card_infos}
    )


@router.get("/dashboard")
async def dashboard():
    return get_dashboard()


@router.post("/analysis")
async def sql_llm_analysis(request: AnalysisRequest):

    return card_analysis(request.question)
