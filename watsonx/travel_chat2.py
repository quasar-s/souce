from dotenv import load_dotenv
import os
import gradio as gr
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from PIL import Image

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
        "max_tokens": 5000,  # 토큰 수
        # "temperature": 1.035, #온도
        # "frequency_penalty": 0.54,#빈도
        # "presence_penalty": -0.32,#존재
        # "top_p": 1,   #상위 p
    },
)


def image_to_base64(image):

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image_base64


def recommend(message, history):

    system_prompt = f"""
    당신은 전문 여행 계획사 AI입니다.
    - 사용자가 업로드한 이미지의
    - 분위기
    - 감성
    - 색감
    - 스타일
    을 분석해서 여행지를 추천해줘

    반드시
    1. 이미지 분위기 분석
    2. 추천 여행지
    3. 추천 이유
    4. 추천 활동
    을 포함해줘
    항상 한국어로만 대답해줘
    """
    messages = [
        {"role": "system", "content": system_prompt},  # 필수 아님
    ]
    # 사용하는 모델 한가지 이미지만 해석가능
    # => 이전 답변은 텍스트만 보냄
    for item in history:
        role = item["role"]
        content = item["content"]

        # assistant 응답 저장
        texts = []

        if isinstance(content, list):
            for c in content:
                # 텍스트만 추출
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
        elif isinstance(content, str):
            texts.append(content)

        messages.append({"role": role, "content": "".join(texts)})

    # for item in history:
    #     if isinstance(item["content"], list):
    #         text_items = [c["text"] for c in item["content"] if c.get(type) == "text"]
    #         text_combind = "".join(text_items)
    #         messages.append({"role": item["role"], "content": text_combind})
    #     else:
    #         messages.append({"role": item["role"], "content": item["content"]})

    # message : text, files
    text = message.get("text", "")
    files = message.get("file", "")
    # files = message.get("file", [])

    if files:
        image = Image.open(files[0])

        # base64 인코딩 후
        base64_image = image_to_base64(image)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": text},
                ],
            },
        )
    else:
        messages = [
            {"role": "system", "content": text},  # 필수 아님
        ]

    messages.append({"role": "user", "content": text})
    # chat_stream()
    generate_response_stream = model.chat_stream(messages=messages)

    full_responce = ""

    for chunk in generate_response_stream:
        if chunk["choices"]:
            full_responce += chunk["choices"][0]["delta"].get("content", "")
            yield full_responce


demo = gr.ChatInterface(
    fn=recommend,
    multimodal=True,
    title="✈️ 감성 여행 플래너",
    description="가고싶은 여행지의 사진과 여행 스타일을 작성 시 AI가 맞춤 여행 일정을 작성해드립니다.",
)

demo.launch()
