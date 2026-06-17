# 라이브러리 로드
from langchain_ollama import ChatOllama
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
import gradio as gr

# 모델(LLM, Embedding)
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ibm import WatsonxEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# .env 내용 가져오기
load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")
hf_token = os.getenv("HF_TOKEN")

watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
    project_id=f"{project_id}",
    params={"max_tokens": 2000, "temperature": 0},
)

# 1단계


def process_pdf(pdf_file):
    if pdf_file is None:
        return (
            "PDF파일을 업로드 해주시기 바랍니다.",
            "",
            "",
            "",
            "",
        )
    # STEP 1 : 문서 로드
    pdf_loader = PyPDFLoader(pdf_file)
    loaded_docs = pdf_loader.load()
    # STEP 2 : 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_chunks = text_splitter.split_documents(loaded_docs)

    total_page_count = len(loaded_docs)
    first_page_content = loaded_docs[0].page_content[:1000]

    total_chunk_count = len(split_chunks)

    first_chunk_content = (split_chunks[0].page_content,)
    first_chunk_metadata = split_chunks[0].metadata

    return (
        total_page_count,
        first_page_content,
        total_chunk_count,
        first_chunk_content,
        first_chunk_metadata,
    )


# 2단계
def qa_rag(pdf_file, user_question):
    if not pdf_file:
        return (
            "PDF파일을 업로드 해주시기 바랍니다.",
            "",
        )
    if not user_question.strip():
        return (
            "질문이 공백으로 등록 되었습니다. 다시 확인 후 실행하시기 바랍니다.",
            "",
        )
    # STEP 1 : 문서 로드
    pdf_loader = PyPDFLoader(pdf_file)
    loaded_docs = pdf_loader.load()
    # STEP 2 : 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_chunks = text_splitter.split_documents(loaded_docs)
    # STEP 3 : 인덱싱 - 임베딩
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    # STEP 4 : 벡터스토어(Chroma or FAISS)
    faiss_vector_store = FAISS.from_documents(
        split_chunks,
        embedding=ollama_embeddings,
        # persist_directory="./db/chroma_db",
        # collection_name="research",
    )
    # STEP 5 : as_retriever() : Vector Store => Retriever | connect LangChain
    document_retriever = faiss_vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    retrieved_docs = document_retriever.invoke(user_question)

    # STEP 6 : RAG 프롬프트 생성
    system_message = """\
    다음 컨텍스트를 참고하여 질문에 답하세요.
    컨텍스트에 없는 내용은 모른다고 답하세요.

    컨텍스트:
    {context}
    """

    rag_chat_prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", "{question}")]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_pipeline = (
        {"context": document_retriever | format_docs, "question": RunnablePassthrough()}
        | rag_chat_prompt
        | watson_llm
        | StrOutputParser()
    )

    final_answer = rag_chain_pipeline.invoke(user_question)

    searched_context = "\n\n=== 다음 Chunk ===\n\n".join(
        [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
    )

    return searched_context, final_answer


with gr.Blocks() as demo:
    gr.Markdown("# PDF RAG 학습 앱")

    with gr.Tabs():
        with gr.Tab("1단계 - PDF & chunk 확인"):
            # 파일 업로드 컴포넌트
            input_pdf1 = gr.File(label="PDF 업로드", file_types=[".pdf"])
            start_analysis_btn = gr.Button("분석 시작")
            # TextBox 5개
            page_count_output = gr.Textbox(label="총 페이지 수")
            first_page_output = gr.Textbox(label="첫 페이지 내용", lines=10)
            chunk_count_output = gr.Textbox(label="총 Chunk 수")
            first_chunk_output = gr.Textbox(label="첫 번째 Chunk", lines=10)
            first_metadata_output = gr.Textbox(label="첫 번째 Chunk Metadata", lines=5)
            start_analysis_btn.click(
                fn=process_pdf,
                inputs=input_pdf1,
                outputs=[
                    page_count_output,
                    first_page_output,
                    chunk_count_output,
                    first_chunk_output,
                    first_metadata_output,
                ],
            )
        with gr.Tab("2단계 - RAG QA"):
            input_pdf2 = gr.File(label="PDF 업로드", file_types=[".pdf"])
            question_input = gr.Textbox(label="질문 입력")
            ask_question_btn = gr.Button("질문하기")
            searched_chunk_output = gr.Textbox(label="검색된 Chunk")
            final_answer_output = gr.Textbox(label="최종 답변")
            ask_question_btn.click(
                fn=qa_rag,
                inputs=[input_pdf2, question_input],
                outputs=[searched_chunk_output, final_answer_output],
            )

demo.launch()
