from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from backend.ai.embedding import watsonx_embedding
from backend.ai.llm import watson_llm

import os

UPLOAD_PATH = "uploads"


def upload_document(file):
    # 파일 저장
    file_path = os.path.join(UPLOAD_PATH, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())
    # pdf 업로드 => 분할 => 인덱스 생성
    # PDF load
    pdf_loader = PyPDFLoader(file_path)
    loaded_docs = pdf_loader.load()
    # 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    chunks = text_splitter.split_documents(loaded_docs)

    faiss_store = FAISS.from_documents(chunks, embedding=watsonx_embedding)
    faiss_store.save_local("./db/vectorstore")

    return {"message": "업로드 성공"}


# 질문 => 유사도 검색 => 문서 => LLM
def rag_chat(question: str):
    faiss_store = FAISS.load_local(
        "./db/vectorstore", watsonx_embedding, allow_dangerous_deserialization=True
    )
    retriever = faiss_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    ### LLM
    # 1. prompt
    message = """\
    다음 컨텍스트를 참고하여 질문에 답하세요.
    컨텍스트에 없는 내용은 모른다고 답하세요.

    컨텍스트:
    {context}

    질문:
    {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(message)

    # 2. chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | watson_llm
        | StrOutputParser()
    )

    # 3.chain
    answer = chain.invoke(question)
    return answer


# 질문 => 유사도 검색 => 문서 => LLM
async def rag_chat_stream(question: str):
    faiss_store = FAISS.load_local(
        "./db/vectorstore", watsonx_embedding, allow_dangerous_deserialization=True
    )
    retriever = faiss_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    ### LLM
    # 1. prompt
    message = """\
    다음 컨텍스트를 참고하여 질문에 답하세요.
    컨텍스트에 없는 내용은 모른다고 답하세요.

    컨텍스트:
    {context}

    질문:
    {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(message)

    # 2. chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | watson_llm
        | StrOutputParser()
    )

    # 3.chain
    async for chunk in chain.astream(question):
        yield chunk
