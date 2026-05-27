# 라이브러리 로드
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

exaone_llm = ChatOllama(model="exaone3.5:2.4b")

system_prompt = """
당신은 요리 경력 20년의 전문셰프입니다.
입력되는 질문에 대해 요리 방법과 팁 등을 알려줍니다.
"""

template = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{question}")]
)

chain = template | exaone_llm | StrOutputParser()


def chat(question, history):

    response = chain.invoke({"question": question})
    return response


demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
    title="음식 레시피 AI",
    description="궁금한 요리를 입력하면 알려드립니다.",
)

demo.launch()
