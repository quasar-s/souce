import gradio as gr
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import ChatWatsonx

from langchain_chroma import Chroma


from dotenv import load_dotenv

from pathlib import Path

import shutil

import os

load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")
hf_token = os.getenv("HF_TOKEN")
cohere_api_key = os.getenv("COHERE_API_KEY")

watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
    project_id=f"{project_id}",
    params={"max_tokens": 2000, "temperature": 0},
)

watsonx_enbedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
    project_id=f"{project_id}",
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".text": TextLoader,
}

CHROMA_DIR = "./data/chroma_db"
COLLECTION_NAME = "job_rag"
CHUNKS_PATH = "./db/chunks.pkl"

DOCUMETS = []
CHUNKS = []
VECTORSTORE = None


# =========================
# 1 단계 탭
# ==========================
def extract_metadata(file_path):
    # 2026 상 삼성E&A 직무기술서
    # {year:2026, recruitment_period:상반기,company:삼성E&A,file_name:2026 상 삼성E&A 직무기술서}

    # 확장자를 제외한 파일명
    name = file_path.stem
    datas = name.split()

    return {
        "year": int(datas[0]),
        "recruitment_period": datas[1] + "반기",
        "company": datas[2],
        "documet_type": datas[3],
        "file_name": name,
    }


def upload_files(files):
    """
    여러 개의 파일(pdf,csv,....)이 업로드 될 때 각 파일을 load 한 결과 DOCUMENTS에 추가
    몇 개의 문서가 업로드 되었는지 리턴
    확장자 분리
    """
    global DOCUMETS
    if not files:
        return "업로드된 파일이 없습니다."

    all_docs = []

    for file in files:
        path = Path(file.name)
        ext = path.suffix.lower()

        loader = LOADERS[ext](file.name)
        docs = loader.load()

        # metadata 정리
        meta_info = extract_metadata(path)
        # metadata 업데이트
        for doc in docs:
            doc.metadata.update(meta_info)

        all_docs.extend(docs)

    DOCUMETS = all_docs

    return f"문서 수 : {len(all_docs)}"


def preview_chunks():
    global DOCUMETS, CHUNKS

    if not DOCUMETS:
        return "문서 없음"

    # 전체 문서 = DOCUMENTS
    # 분리
    CHUNKS = splitter.split_documents(DOCUMETS)

    # 청크 10개까지만 내용 출력
    preview = []
    for i, chunk in enumerate(CHUNKS[:10]):
        preview.append(f"""
[CHUNK {i+1}]
{chunk.page_content[:100]}
""")
    return "\n\n".join(preview)


def build_vectorstore():
    global CHUNKS, VECTORSTORE

    if not CHUNKS:
        return "먼저 chunk를 생성하세요."

    # 기존의 vectorstore 제거
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    VECTORSTORE = Chroma.from_documents(
        documents=CHUNKS,
        embedding=watsonx_enbedding,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    return f"""
생성 완료

Chunk:
{len(CHUNKS)}

Vector:
{VECTORSTORE._collection.count()}
"""


# ======================
# Gradio UI
# ======================
with gr.Blocks() as app:
    gr.Markdown("# 사내 문서 RAG")
    # Tab 3개
    with gr.Tabs():
        with gr.Tab("문서관리"):
            # 파일 업로드 컴포넌트
            files = gr.File(file_count="multiple")
            upload_btn = gr.Button("문서 업로드")
            upload_status = gr.Textbox()
            upload_btn.click(upload_files, files, upload_status)
            # chunk 분리 컴포넌트
            chunk_btn = gr.Button("Chunk 확인")
            chunk_preview = gr.Textbox(lines=20)
            chunk_btn.click(preview_chunks, inputs=[], outputs=chunk_preview)
            # vectorDB 생성 및 저장 컴포넌트
            vector_db_btn = gr.Button("Vector DB 생성")
            vector_status = gr.Textbox()
            vector_db_btn.click(build_vectorstore, inputs=[], outputs=vector_status)
        with gr.Tab("검색 테스트"):
            search_input = gr.Textbox(label="검색어")
            search_btn = gr.Button("검색")
            bm25_output = gr.Textbox(label="BM25")
            dence_output = gr.Textbox(label="Dence")
            rerank_output = gr.Textbox(label="Rerank")
            search_btn.click(
                inputs=[search_input],
                outputs=[bm25_output, dence_output, rerank_output],
            )
        with gr.TabItem("RAG 채팅"):
            rag_btn = gr.Button("RAG 생성")
            rag_status = gr.Textbox(label="TextBox")
            rag_btn.click()
            # rag_chat = gr.ChatInterface()

if __name__ == "__main__":
    app.launch()
