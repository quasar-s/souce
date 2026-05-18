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


def recommend(region, monney, keyword, days):
    if not region.strip():
        return "지역을 입력하여 주세요."
    if monney is None:
        return "예산을 입력하여 주세요."
    if not keyword.strip():
        return "테마를 입력하여 주세요."
    if not days.strip():
        return "일정을 입력하여 주세요."

    system_prompt = f"""
    당신은 전문 여행 계획사 AI입니다.
    - 당신의 임무는 주어지는 지역{region}, 예산{monney}, 테마{keyword}, 일정{days}을 분석하여 한국어로 여행일정을 생성하는 것입니다.
    - 반드시
    1.일정표
    2.추천장소
    3.맛집
    4.예상비용
    을 포함할 것
    """

    user_prompt = f"""
    아래의 내용을 참고하여 작성하시오.
    - 예산은 "{monney}*10000원"임
    - 여행일정 이외의 답변은 배제할 것
    - 주어진 값을 임의 변경하지 말 것
    - 현실적으로 여행 가능한 곳일것
    - 대한민국 현행법상 입국금지구역 및 출입 금지 지역은 제외할 것
    - 주어진 모든 것을 이용하여 분석할 것
    - 일반적인 20대 성인 여성 기준으로 소화 가능한 여행 일정일 것
    """

    messages = [
        {"role": "system", "content": system_prompt},  # 필수 아님
        {"role": "user", "content": user_prompt},
    ]

    # generated_response = model.chat(messages=messages)
    # return generated_response["choices"][0]["message"]["content"]
    generate_response_stream = model.chat_stream(messages=messages)

    full_responce = ""
    for chunk in generate_response_stream:
        if chunk["choices"]:
            full_responce += chunk["choices"][0]["delta"].get("content", "")
            yield full_responce


demo = gr.Interface(
    fn=recommend,
    title="✈️ 맞춤 여행 일정 생성 프로그렘",
    inputs=[
        gr.Text(label="여행 지역"),
        gr.Slider(10, 300, label="예산(만원)"),
        gr.Dropdown(["모험", "휴양", "자유", "관광", "쇼핑"], label="테마"),
        gr.Radio(["1일", "2~3일", "4~7일", "1주일이상"], label="여행 기간"),
    ],
    outputs=[gr.Markdown()],
    description="지역, 예산, 테마, 기간 작성 시 AI가 맞춤 여행 일정을 작성해드립니다.",
)

demo.launch()
