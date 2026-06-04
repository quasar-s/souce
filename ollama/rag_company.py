import gradio as gr

from langchain_classic.chains.query_constructor.base import AttributeInfo

from langchain_classic.retrievers.self_query.chroma import ChromaTranslator

from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)

from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    BM25Retriever,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import ChatWatsonx

from langchain_chroma import Chroma

from langchain_cohere import CohereRerank

from dotenv import load_dotenv

from pathlib import Path

import pickle

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

BM25_RETRIEVER = None
DENSE_RETRIEVER = None
SELFQUERY_RETRIEVER = None
FINAL_RETRIEVER = None

META_FIELDS = [
    AttributeInfo(name="year", description="채용년도", type="integer"),
    AttributeInfo(
        name="recruitment_period", description="상반기 또는 하반기", type="string"
    ),
    AttributeInfo(name="company", description="회사명", type="string"),
    AttributeInfo(
        name="documet_type", description="직무기술서, 채용공고, 기업분석", type="string"
    ),
    AttributeInfo(name="file_name", description="파일명", type="string"),
]

# 대화 메모리
# ConversationBufferWindowMemory : 최근 k 개의 대화만 기억하는 창(window)
mmemory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer",
)
SYSTEM_PROMPT = """
당신은 회사 내부 문서를 기반으로 직원들의 질문에 답변을 하는 AI 어시스턴트입니다.

다음 규칙을 지키세요.
1.제공된 문서 내용에만 기반으로 답변하세요.
2.문서에 없는 내용은 "해당 문서에는 없는 내용입니다."라고 답하세요.
3. 답변 마지막에 참고 문서명을 명시하세요.
4. 답변은 한국어로 명확하게 하세요.
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        # MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """
[참고 문서]
{context}
[질문]
{question}
""",
        ),
    ]
)

QA_CHAIN = None


# =========================
# APP 시작 시
# =========================
def build_retriever(chunks, save_chunks=False):
    global BM25_RETRIEVER, DENSE_RETRIEVER, SELFQUERY_RETRIEVER, FINAL_RETRIEVER, VECTORSTORE, META_FIELDS

    # 검색 테스트 탭으로 바로 실행 시

    # BM25 indexing => 폴더에 저장
    if save_chunks:
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)  # Python에서만 사용하는 방식
    # Retriever 초기화
    # BM25 Index =\ chroma에 저장되지 않음
    BM25_RETRIEVER = BM25Retriever.from_documents(chunks, k=5)

    # 일반 검색
    DENSE_RETRIEVER = VECTORSTORE.as_retriever(k=20)

    # SelfQuery
    SELFQUERY_RETRIEVER = SelfQueryRetriever.from_llm(
        llm=watson_llm,
        vectorstore=VECTORSTORE,
        document_contents="계열사 직무기술서 문서",
        metadata_field_info=META_FIELDS,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={"k": 20},
    )

    # Final : BM25 + Danse + ReRank
    ensemble = EnsembleRetriever(
        retrievers=[BM25_RETRIEVER, DENSE_RETRIEVER, SELFQUERY_RETRIEVER],
        weights=[0.35, 0.45, 0.2],
    )

    reranker = CohereRerank(model="rerank-v4.0-pro", top_n=5)

    FINAL_RETRIEVER = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble
    )

    return "RETRIEVER 생성 완료"


def initialize():
    global VECTORSTORE, CHUNKS_PATH

    if not Path(CHUNKS_PATH).exists():
        print("기존 Vector 없음")
        return

    # BM25 제외한 retriever는 이 부분만 하면 가능
    # 기존 Vectorstoer 호출
    VECTORSTORE = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=watsonx_enbedding,
    )

    # BM25 파일
    if Path(CHUNKS_PATH).exists():
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        build_retriever(chunks=chunks, save_chunks=False)
        print("Retriever 로드")


# =========================
# 1 단계 탭
# =========================
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
        "document_type": datas[3],
        "file_name": name,
    }


def upload_files(files):
    """
    여러 개의 파일(pdf,csv,....)이 업로드 될 때 각 파일을 load 한 결과 DOCUMENTS에 추가
    몇 개의 문서가 업로드 되었는지 리턴
    확장자 분리
    """
    global DOCUMETS, CHUNKS, VECTORSTORE, BM25_RETRIEVER, DENSE_RETRIEVER, SELFQUERY_RETRIEVER, FINAL_RETRIEVER

    # 문서를 새롭게 업로드 할때 기존 내용이 있을 수 있으므로 제거
    DOCUMETS = []
    CHUNKS = []
    VECTORSTORE = None

    BM25_RETRIEVER = None
    DENSE_RETRIEVER = None
    SELFQUERY_RETRIEVER = None
    FINAL_RETRIEVER = None

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

    # # 기존의 vectorstore 제거
    # if Path(CHROMA_DIR).exists():
    #     shutil.rmtree(CHROMA_DIR)

    #  안전한 초기화 방식으로 대체
    if VECTORSTORE is not None:
        # 기존 컬렉션의 데이터만 비우기
        VECTORSTORE.delete_collection()

    VECTORSTORE = Chroma.from_documents(
        documents=CHUNKS,
        embedding=watsonx_enbedding,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # retrirver 생성
    build_retriever(CHUNKS, save_chunks=True)

    return f"""
생성 완료

Chunk:
{len(CHUNKS)}

Vector:
{VECTORSTORE._collection.count()}
"""


# =========================
# Tab 2 - 기능 구현
# 1. 임베딩 작업 완료
# 2. 문서 관리 => 검색 테스트
# =========================
def format_docs(docs):
    """Document 객체에서 page_content 추출"""

    if not docs:
        return "검색 결과 없음"

    result = []
    result.append(f"검색 결과 수 {len(docs)}건\n")
    for i, d in enumerate(docs[:3], 1):
        result.append(f"""
[문서 {i}]

회사 : {d.metadata.get("company","-")}
유형 : {d.metadata.get("document_type","-")}
년도 : {d.metadata.get("year","-")} {d.metadata.get("recruitment_period","-")}
출처 : {d.metadata.get("file_name","-")}

{d.page_content[:100]}
""")

    return "\n".join(result)


def search_form(query):
    global FINAL_RETRIEVER
    if FINAL_RETRIEVER is None:
        return (
            "BM25 retriever 미생성",
            "Dense retriever 미생성",
            "SelfQuery retriever 미생성",
            "Final retriever 미생성",
        )

    # 각각의 retriever 결과 추출(Document)한 후
    # format_docs() return

    # 1. 키워드 기반 BM25 결과 도출
    bm25_docs = format_docs(BM25_RETRIEVER.invoke(query))

    # 2. 의미 기반 Dense 유사도 결과 도출 (기본 설정된 Top K 수량 반환)
    dense_docs = format_docs(DENSE_RETRIEVER.invoke(query))

    # 3. LLM 메타데이터 필터 결합형 SelfQuery 결과 도출
    self_docs = format_docs(SELFQUERY_RETRIEVER.invoke(query))

    # 4. BM25 + Dense + SelfQuery 앙상블 조합 후 Cohere 재정렬을 거친 최종 산출물
    final_docs = format_docs(FINAL_RETRIEVER.invoke(query))

    return bm25_docs, dense_docs, self_docs, final_docs


# =========================
# Tab 3 - 기능 구현
# ChatInterFace
# - history : 대화 관리 이력
# RunnableWithMessageHistory
# =========================


def creat_chain():
    global QA_CHAIN

    if QA_CHAIN is None:
        QA_CHAIN = ConversationalRetrievalChain.from_llm(
            llm=watson_llm,
            retriever=FINAL_RETRIEVER,
            memory=mmemory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
        )

    return QA_CHAIN


def chat(message, history):
    global QA_CHAIN

    if FINAL_RETRIEVER is None:
        return ""

    QA_CHAIN = creat_chain()
    response = QA_CHAIN.invoke({"question": message})

    answer = response["answer"]

    sources = []
    for doc in response["source_documents"]:
        sources.append(
            f"{doc.metadata.get('company', '-')}"
            f"{doc.metadata.get('file_name', '-')}"
        )
    answer += "\n\n[]\n"
    answer += "\n".join(list(set(sources)))

    return answer


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
            query_input = gr.Textbox(label="검색어")
            search_btn = gr.Button("검색")
            bm25_output = gr.Textbox(label="BM25")  # 키워드
            dence_output = gr.Textbox(label="Dense")  # 일반 검색
            self_output = gr.Textbox(label="Self")  # SelfQuery
            rerank_output = gr.Textbox(label="Rerank")  # 다양성
            search_btn.click(
                search_form,
                inputs=[query_input],
                outputs=[bm25_output, dence_output, self_output, rerank_output],
            )
        with gr.TabItem("RAG 채팅"):
            # rag_btn = gr.Button("RAG 생성")
            # rag_status = gr.Textbox()
            # rag_btn.click()
            rag_chat = gr.ChatInterface(chat)

if __name__ == "__main__":
    initialize()
    app.launch()
