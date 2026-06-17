from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from backend.ai.embedding import watsonx_embedding


def main():
    # data 폴더 안의 파일을 읽은 후
    data_dir = Path("data")

    documents = []
    # Document 객체 생성
    for file_path in data_dir.glob("*.text"):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(page_content=content, metadata={"source": file_path.name})
        )
    # 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    # 인덱스 설정(벡터db) ./ vectordb
    Chroma.from_documents(
        documents=split_docs,
        embedding=watsonx_embedding,
        persist_directory="./vectordb",
    )


if __name__ == "__main__":
    main()
