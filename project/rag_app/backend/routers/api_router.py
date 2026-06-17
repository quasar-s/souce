from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import StreamingResponse
from backend.services.llm_service import question_and_answer
from backend.services.rag_service import upload_document, rag_chat, rag_chat_stream
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
@router.post("/rag/question")
async def question(req: QuestionRequest):
    answer = rag_chat(req.question)
    return {"message": answer}


# http://http://127.0.0.1:8000/api/rag/question
@router.post("/rag/question/stream")
async def question(req: QuestionRequest):
    return StreamingResponse(rag_chat_stream(req.question), media_type="text")
