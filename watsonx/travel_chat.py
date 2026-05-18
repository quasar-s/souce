from dotenv import load_dotenv
import os
import gradio as gr
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

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
    model_id="ibm/granite-4-h-small",
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


def recommend(message, history):
    if not message.strip():
        return "내용을 입력하여 주세요."

    system_prompt = f"""
    당신은 전문 여행 계획사 AI입니다.
    - 당신의 임무는 주어지는 정보를 분석하여 한국어로 여행일정을 생성하는 것입니다.
    - 반드시
    1.일정표
    2.추천장소
    3.맛집
    4.예상비용
    을 포함할 것
    """

    user_prompt = f"""
    
    """

    messages = [
        {"role": "system", "content": system_prompt},  # 필수 아님
    ]
    for item in history:
        content = item["content"][0]["text"]
        messages.append({"role": item["role"], "content": content})

    messages.append({"role": "user", "content": message})

    generate_response_stream = model.chat_stream(messages=messages)

    full_responce = ""

    for chunk in generate_response_stream:
        if chunk["choices"]:
            full_responce += chunk["choices"][0]["delta"].get("content", "")
            yield full_responce


demo = gr.ChatInterface(
    fn=recommend,
    title="✈️ 맞춤 여행 일정 생성 프로그렘",
    # inputs=[
    # ],
    # outputs=[gr.Markdown()],
    description="지역, 예산, 테마, 기간 작성 시 AI가 맞춤 여행 일정을 작성해드립니다.",
)

demo.launch()
