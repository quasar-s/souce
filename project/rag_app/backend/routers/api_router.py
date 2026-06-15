from fastapi import APIRouter, Request, UploadFile
from backend.services.llm_service import question_and_answer
from backend.services.rag_service import upload_document
from backend.schemas.base_schema import QuestionRequest

router = APIRouter(prefix="/api")


# http://http://127.0.0.1:8000/api/question
@router.post("/question")
async def question(req: QuestionRequest):
    answer = question_and_answer(req.question)
    return {"message": answer}


# http://http://127.0.0.1:8000/api/rag/upload
@router.post("/rag/upload")
async def file_upload(file: UploadFile):
    # 서비스 호출
    return upload_document(file)


# http://http://127.0.0.1:8000/api/rag/question
