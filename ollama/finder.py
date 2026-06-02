# 사용자가 입력한 단어와 유사한 2개의 단어 추출
# csv 파일
import gradio as gr
from langchain_community.document_loaders import CSVLoader

from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import ChatWatsonx

from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

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

csv_loader = CSVLoader(
    file_path="./data/myData.csv",
    encoding="utf-8",
    csv_args={"delimiter": ",", "fieldnames": ["Words"]},
)

csv_docs = csv_loader.load()

csv_vectorstore = FAISS.from_documents(documents=csv_docs, embedding=watsonx_enbedding)


def find_simliar(query):
    if not query.strip():
        return "질문을 입력하여 주세요.", ""

    docs = csv_vectorstore.similarity_search(query, k=2)

    result1 = docs[0].page_content if len(docs) > 0 else ""
    result2 = docs[1].page_content if len(docs) > 0 else ""

    return result1, result2


with gr.Blocks() as app:
    gr.Markdown("## 🤖 Educate Kids")
    gr.Markdown("비슷한 단어 또는 문장을 찾아드립니다.")

    query = gr.Textbox(label="질문 입력", placeholder="질문을 입력하여 주십시오.")
    btn = gr.Button("Find Similar Things")

    output1 = gr.Textbox(label="Top Match 1")
    output2 = gr.Textbox(label="Top Match 2")

    btn.click(find_simliar, inputs=query, outputs=[output1, output2])

app.launch()
