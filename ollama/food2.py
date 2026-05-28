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
import gradio as gr
import os

# .env 내용 가죠오기
load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

# LLM 선언

watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
    project_id=f"{project_id}",
    params={"max_tokens": 2000},
)

qwen_llm = ChatOllama(model="qwen3.5:4b")

exaone_llm = ChatOllama(model="exaone3.5:2.4b")

system_prompt = """
당신은 요리 경력 20년의 전문셰프입니다.
입력되는 질문에 대해 요리 방법과 팁 등을 알려줍니다.
"""

template = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{question}")]
)

chain = template | watson_llm | StrOutputParser()


def chat(question, history):
    chat_history = []

    # history 직접 관리

    for msg in history:
        if msg["role"] == "user":
            chat_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(("ai", msg["content"]))

    response = chain.invoke({"history": chat_history, "question": question})

    return response


demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
    title="음식 레시피 AI",
    description="궁금한 요리를 입력하면 알려드립니다.",
)

demo.launch()
