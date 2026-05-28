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
import uuid

# .env 내용 가죠오기
load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")


# qwen_llm = ChatOllama(model="qwen3.5:4b")


# exaone_llm = ChatOllama(model="exaone3.5:2.4b")


def get_session_history(session_id):  # -> BaseChatMessageHistory()
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()  # 한 사람당 하나씩 부여
    return store[session_id]


def create_chain():
    # LLM 선언

    watson_llm = ChatWatsonx(
        model_id="ibm/granite-4-h-small",
        url=f"{watsonx_ai_url}",
        api_key=f"{apikey}",
        project_id=f"{project_id}",
        params={"max_tokens": 2000},
    )
    system_prompt = """
    당신은 요리 경력 20년의 전문셰프입니다.
    입력되는 질문에 대해 요리 방법과 팁 등을 알려줍니다.
    항상 한국어로 대답하세요.
    """

    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = template | watson_llm | StrOutputParser()

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


chain = create_chain()

store = {}


def chat(question, history, session_id):
    chat_history = []

    # history 직접 관리
    full_response = ""
    for chunk in chain.stream(
        {"question": question}, config={"configurable": {"session_id": session_id}}
    ):
        full_response += chunk
        yield full_response


with gr.Blocks() as demo:
    # 세션 ID uuid 사용
    # gr.State() : 사용자별 데이터를 서버 메모리에 저장하는 컴포넌트
    session_state = gr.State(str(uuid.uuid4()))

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[session_state],
        multimodal=True,
        title="음식 레시피 AI",
        description="궁금한 요리를 입력하면 알려드립니다.",
    )

demo.launch()
