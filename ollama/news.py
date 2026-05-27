# 라이브러리 로드
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
import gradio as gr

exaone_llm = ChatOllama(model="exaone3.5:2.4b")


system_prompt = """
당신은 전문 뉴스 라이터이자 데이터 분석가입니다.
제공된 뉴스 기사에서 정보를 정확히 추출하여 형식에 맞게 출력하세요.
반드시 아래 지침을 철저히 준수하여 JSON 형태로만 답변하세요.

{{"title":"기사 제목", "date":"작성일자, "keyword":["키워드 1","키워드 2","키워드 3"], "category":"카테고리"}}

"""

template = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{article}")]
)

chain = template | exaone_llm | JsonOutputParser()


def news_input(article):
    # === 를 기준으로 기사를 분리
    articles = [article.strip() for article in article.split("===") if article.strip()]
    response = chain.batch([{"article": article} for article in articles])

    return "\n\n".join(str(item) for item in response)


demo = gr.Interface(
    fn=news_input,
    inputs=[
        gr.Textbox(
            lines=10,
            placeholder="뉴스 기사를 입력하세요. 각 기사는 ==== 로 구분해 주세요.",
            label="뉴스 기사 입력",
        )
    ],
    outputs=[
        gr.Textbox(
            lines=20,
            label="요약",
        )
    ],
    title="뉴스 분석 AI",
    description="뉴스 기사에서 정보를 추출합니다.",
)

demo.launch()
