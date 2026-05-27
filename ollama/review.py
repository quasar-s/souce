# 라이브러리 로드
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
import gradio as gr
import time

exaone_llm = ChatOllama(model="exaone3.5:2.4b", temperature=0)


class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="리뷰의 전체적인 감정 (긍정, 부정, 중립)"
    )
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="감정의 강도 (0.0은 매우 부정, 1.0은 매우 긍정, 0미만은 0으로 계산한다.)",
    )
    pros: list[str] = Field(description="리뷰에서 언급된 장점 리스트")
    cons: list[str] = Field(description="리뷰에서 언급된 단점 혹은 아쉬운 점 리스트")
    recommend: bool = Field(description="제품 추천 여부 (추천하면 True, 아니면 False)")
    reply: str = Field(description="고객 리뷰에 대한 친절한 맞춤형 답글")


system_prompt = """
당신은 전문 리뷰어이자 데이터 분석가입니다.
제공된 리뷰에서 정보를 정확히 추출하여 형식에 맞게 출력하세요.
반드시 아래 지침을 철저히 준수하여 JSON 형태로만 답변하세요.

{format_instructions}
"""

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

template = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{reviews}")]
).partial(format_instructions=parser.get_format_instructions())

chain = template | exaone_llm | parser


def reviews_input(texts):
    # === 를 기준으로 리뷰를 분리
    reviews = [r.strip() for r in texts.split("===") if r.strip()]

    start = time.time()

    results = chain.batch([{"reviews": r} for r in reviews])

    elapsed = time.time() - start

    output = []
    for i, (review, result) in enumerate(zip(reviews, results), 1):
        emoji = {"positive": "🤩", "negative": "👿", "neutral": "🤔"}[result.sentiment]
        output.append(f"[리뷰 {i}]")
        output.append(f"고객 리뷰 {review[:40]}....")
        output.append(
            f"리뷰 감정 {emoji} {result.sentiment}(강도 : {result.score:.2f})"
        )
        output.append(f"장점 {', '.join(result.pros) if result.pros else "없음"}")
        output.append(f"단점 {', '.join(result.cons) if result.cons else "없음"}")
        output.append(f"추천 여부 {"✔️ 추천"if result.recommend else "❌비추천"}")
        output.append(f"판매자 답변 {result.reply}")
        output.append("-" * 40)
    output.append(f"소요 시간 : {elapsed:.2f}초 ({len(reviews)} 개 리뷰)")

    return "\n".join(output)


demo = gr.Interface(
    fn=reviews_input,
    inputs=[
        gr.Textbox(
            lines=10,
            placeholder="리뷰를 입력하세요. 각 리뷰는 ==== 로 구분해 주세요.",
            label="리뷰 입력",
        )
    ],
    outputs=[
        gr.Textbox(
            lines=20,
            label="요약",
        )
    ],
    title="리뷰 분석 AI",
    description="리뷰 입력 시 감정, 장단점, 추천여부를 분석합니다.",
)

demo.launch()


# "이 제품 정말 별로네요. 돈 낭비 였습니다."
# ===
# "그냥 폄범한 하루였습니다. 특별한 건 없었습니다."
# ===
# "와, 생각보다 훨씬 맛있었어요! 강력 추천드립니다."
# ===
# "서비스가 별로고 맛도 그닥이었습니다."
