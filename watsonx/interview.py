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
        "max_tokens": 2000, #토큰 수
        "temperature": 1.035, #온도
        "frequency_penalty": 0.54,#빈도
        "presence_penalty": -0.32,#존재
        "top_p": 1,   #상위 p
    },
)


def interview_text(keyword):
    if not keyword.strip():
        return "텍스트를 입력하여 주세요."

    system_prompt = """
    당신은 인터뷰 질문을 한국어로 작성하는 전문가 입니다.
    - 당신의 임무는 주어지는 장르를 분석하여 한국어로 생성하는 것입니다.
    """

    user_prompt = f"""
    아래의 내용을 참고하여 작성하시오.
    - 장르 : {keyword}
    - 해당 장르의 특징을 5줄로 정리할 것
    - 해당 장르에 대한 인터뷰 질문 8가지를 작성할 것
    - 너무 딱딱한 어구를 사용하지 말 것
    - 기사나 칼럼작성 시 문제가 없을만한 질문을 작성할 것
    """

    messages = [
        {"role": "system", "content": system_prompt},  # 필수 아님
        {"role": "user", "content": user_prompt},
    ]

    generated_response = model.chat(messages=messages)

    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=interview_text,
    title="🎙️인터뷰 질문 생성 프로그렘",
    inputs=[gr.Text(label="장르")],
    outputs=[gr.Markdown()],
    description="장르 작성 시 AI가 장르의 특징 및 인터뷰 질문을 작성해드립니다.",
)

demo.launch()
