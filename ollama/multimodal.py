from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import ChatWatsonx

from dotenv import load_dotenv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import base64
import shutil
import os

#####################
# 모델
#####################

# .env 내용 가죠오기
load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

# 유료 LLM 선언

watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
    project_id=f"{project_id}",
    params={"max_tokens": 2000, "temperature": 0},
)

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
    project_id=f"{project_id}",
)

vision_llm = ChatOllama(model="minimax-m3:cloud", temperature=0)

parser = StrOutputParser()

print("모델 초기화 완료")
print(f"LLM : ibm/granite-4-h-small")
print(f"Enbedding : ibm/granite-embedding-278m-multilingual")
print(f"Vision LLM : minimax-m3:cloud")


##################
# 이미지 유틸리티 함수
##################
def encode_image(image_path: str) -> str:
    """
    이미지 파일을 base64 문자열로 인코딩
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다.{image_path}")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


##################
# 샘플 테스트 데이터
##################


def create_sample_image(image_dir="./sample_images"):
    os.makedirs(image_dir, exist_ok=True)

    # 기본 이미지 생성
    img1 = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img1)

    draw.text((180, 20), "2024 분기별 매출 현황 (억원)", fill=(30, 30, 30))

    bars = [
        ("Q1", 120, (70, 130, 180)),
        ("Q2", 185, (60, 179, 113)),
        ("Q3", 160, (255, 105, 0)),
        ("Q4", 230, (220, 50, 50)),
    ]

    bar_w, base_y = 80, 330

    for i, (label, val, color) in enumerate(bars):
        x = 80 + i * 130
        h = val
        draw.rectangle([x, base_y - h, x + bar_w, base_y], fill=color)
        draw.text((x + 20, base_y - h - 20), f"{val}억", fill=(30, 30, 30))
        draw.text((x + 25, base_y + 5), label, fill=(30, 30, 30))

    draw.line([(60, 50), (60, base_y)], fill=(0, 0, 0), width=2)
    draw.line([(60, base_y), (500, base_y)], fill=(0, 0, 0), width=2)
    draw.text((10, 15), "상승 추세:Q4 최고치 달성", fill=(180, 0, 0))

    img1.save(f"{image_dir}/sales_cahrt_2024.jpg")
    print(f"   생성 : {image_dir}/sales_cahrt_2024.jpg")
    return image_dir


def create_sample_text_docs():
    """테스트용 샘플 텍스트 문서 생성"""
    docs = [
        Document(
            page_content="""
2024년 연간 매출 보고서 요약

당사의 2024년 전체 매출액은 695억원으로, 전년 대비 23% 성장을 달성했습니다.  
1분기(Q1) 120억, 2분기(Q2) 185억원, 3분기(Q3) 160억원, 4분기(Q4) 230억원을
기록했으며, Q4에 최고치를 달성했습니다

주요 성장 요인:
- 신제품 Pro-XL 출시로 인한 프리미엄 라인 매출 확대
- 해외 수출 비중 15% ~ 28% 증가
- 온라인 채널 전환으로 마진율 개선

4분기 실적은 연말 프로모션과 신규 피트너십 체결 효과로 큰 폭 상승했습니다.
""",
            metadata={
                "source": "annual_report_2024.text",
                "type": "text",
                "image_path": "재무",
            },
        )
    ]

    return docs


##################
# 실제 사용 함수
##################
def image_to_caption(image_path):
    """
    이미지를 텍스트로 변환
    Vision LLM이 이미지를 분석하여 검색 가능한 텍스트 설명 생성
    """
    img_b64 = encode_image(image_path)
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": """
                        이 이미지를 검색용 설명문으로 요약하세요.
                        200자 이낼 작성하세요.
                        핵심 객체와 텍스트만 포함하세요.
                        """,
            },
        ]
    )
    return vision_llm.invoke([message]).content


def build_multimodal_index(image_dir: str, text_docs: list[Document]):
    """
    이미지 폴더의 모든 이미지들을 캡셔닝하여 텍스트 문서와 함께 벡터 인덱스 구축
    """
    all_docs = list(text_docs)

    # image_dir 안 파일 가져오기
    # image_file = list(Path(image_dir).glob("*.{jpg,jpeg,png}"))
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(Path(image_dir).glob(ext))

    if not image_files:
        print(f"{image_dir}에서 이미지를 찾지 못했습니다.")
    else:
        print(f"이미지 캡셔닝: {image_dir}개")

    # caption 생성 => Document => 임베딩
    for img_path in image_files:
        caption = image_to_caption(image_path=img_path)

        doc = Document(
            page_content=caption,
            metadata={
                "source": str(img_path),
                "type": "image",
                "image_path": str(img_path),
            },
        )
        all_docs.append(doc)

    # 기존 DB 삭제
    db_path = "./db/multimodal_db"
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    return Chroma.from_documents(all_docs, watsonx_embedding, persist_directory=db_path)


# 검색 결과 이미지 포함 여부 확인
def search_with_images(vectorstore, query):
    """
    벡터 검색 결과를 텍스트/이미지 결과로 분리하여 반환
    """
    results = vectorstore.similarity_search(query, k=5)
    text_results = [r for r in results if r.metadata.get("type") != "image"]
    image_results = [r for r in results if r.metadata.get("type") == "image"]
    print(f"텍스트 결과: {len(text_results)}개, 이미지 결과: {len(image_results)}개")

    return text_results, image_results


def multimodal_answer(vectorstore, question):
    """
    텍스트 + 이미지 검색 결과를 통합하여 멀티 모달 최종 답변 생성

    args:
        vectorstore : Chroma 벡터 스토어
        question : 사용자 질문

    returns:
        {"answer" : str,"images" : list[str]}
    """
    text_results, image_results = search_with_images(
        vectorstore=vectorstore, query=question
    )

    # 텍스트 결과 하나의 컨텍스트로 생성
    text_context = "\n\n".join(r.page_content for r in text_results)

    # 이미지로 검색된 경우
    image_context = ""
    referenced_images = []
    for img_doc in image_results[:3]:
        img_path = img_doc.metadata.get("image_path")

        img_b64 = encode_image(img_path)
        analysis = vision_llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": "이 이미지에서 다음 질문과 관련된 내용을 설명하세요: {question}.",
                        },
                    ]
                )
            ]
        ).content

        image_context += f"[이미지 분석: {img_path}]\n{analysis}\n\n"
        referenced_images.append(img_path)

    # 최종 답변
    combined_context = text_context + "\n\n" + image_context

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "다음 텍스트와 이미지 분석 결과를 참고하여 질문에 답하세요\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    parser = StrOutputParser()

    chain = final_prompt | watson_llm | parser

    answer = chain.invoke({"context": combined_context, "question": question})

    return {"answer": answer, "images": referenced_images}


##################
# 기존 벡터 스토어 로드 함수
##################


def load_or_build_vectorstore(image_dir, text_docs, db_path, force_rebuild=False):
    """ """
    if not force_rebuild and Path(db_path).exists():
        print(f"기존 DB 로드 {db_path}")
        return Chroma(embedding_function=watsonx_embedding, persist_directory=db_path)

    return build_multimodal_index(image_dir, text_docs)


if __name__ == "__main__":
    IMAGE_DIR = "./sample_images"
    # 샘플 이미지 생성
    create_sample_image(IMAGE_DIR)
    # 샘플 텍스트 생성
    text_docs = create_sample_text_docs()
    # 벡터스토어(인덱스) 생성
    vectorstore = load_or_build_vectorstore(
        image_dir=IMAGE_DIR,
        text_docs=text_docs,
        db_path="./db/multimodal_db",
        force_rebuild=False,
    )

    print("\n")
    print("=" * 60)
    print("질의 응답 테스트")
    print("=" * 60)

    result = multimodal_answer(vectorstore, "매출 차트의 추세는?")
    print(result["answer"])

    if result["images"]:
        for img_path in result["images"]:
            print(img_path)
