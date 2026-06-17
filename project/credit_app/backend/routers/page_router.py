from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/card/upload")
async def rag(request: Request):
    return templates.TemplateResponse(request=request, name="card.html")


#  /card/dashboard  => dashboard.html
@router.get("/card/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@router.get("/card/analysis")
async def analysis(request: Request):
    return templates.TemplateResponse(request=request, name="analysis.html")
