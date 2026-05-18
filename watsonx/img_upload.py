from dotenv import load_dotenv
import os
import gradio as gr
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

import base64
import io

# .env 내용 가죠오기
load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

credentials = Credentials(
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
)
client = APIClient(credentials)


model = ModelInference(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    api_client=client,
    project_id=f"{project_id}",
    params={
        "max_tokens": 2000,  # 토큰 수
        # "temperature": 1.035,  # 온도
        # "frequency_penalty": 0.54,  # 빈도
        # "presence_penalty": -0.32,  # 존재
        # "top_p": 1,  # 상위 p
    },
)


def image_to_base64(image):

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image_base64


def image_text(image, user_prompt):
    if image is None:
        return "이미지를 입력하여 주세요."

    # 사용자 프롬포트
    if user_prompt == "":
        user_prompt = """
        이미지를 분석해줘
        """
    base64_image = image_to_base64(image)

    system_prompt = """
    당신은 이미지 분석 전문 AI입니다.
    사용자의 요청에 따라
    - 이미지 설명
    - 분위기 분석
    - 감정 분석
    - 객체 설명
    - 캡션 생성
    - 여행 추천
    - 스타일 분석
    등을 수행 하세요.

    항상:
    - 한국어로 답변
    - 이미지 분석이 최우선
    - 사용자의 요청 의도를 우선 반영하되 이미지와 연관시킬 것
    - 읽기 쉽게 작성
    """
    messages = [
        {"role": "system", "content": system_prompt},  # 필수 아님
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    generated_response = model.chat(messages=messages)

    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=image_text,
    title="이미지 분석 프로그렘",
    inputs=[gr.Image(type="pil"), gr.Text(label="명령어를 입력하여주세요(공란 허용)")],
    outputs=[gr.Markdown()],
    description="이미지 업로드 시 AI가 이미지를 분석해드립니다.",
)

demo.launch()
