from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")


@router.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/rag")
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="rag.html")
