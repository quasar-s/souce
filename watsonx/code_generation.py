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
    model_id="ibm/granite-8b-code-instruct",
    api_client=client,
    project_id=f"{project_id}",
    params={
        "max_tokens": 2000, #토큰 수
        "temperature": 1, #온도
        "frequency_penalty": 0,#빈도
        "presence_penalty": 0,#존재
        "top_p": 1,   #상위 p
    },
)


def code_text(keyword):
    if not keyword.strip():
        return "텍스트를 입력하여 주세요."

    system_prompt = """
    당신은 전문 소프트웨어 개발 작성 AI입니다.
    사용자의 요구사항을 분석하여:
    - 정확한 코드
    - 실행 가능한 코드
    - 가독성이 좋은 코드
    - 초보자도 이해할 수 있는 설명
    을 제공하시오.

    [규칙]
    1. 반드시 코드 블록(````)형식으로 작성
    2. 코드에는 적절한 주석 포함
    3. 필요한 라이브러리가 있다면 함께 설명
    4. 코드 동작 원리를 간단히 설명
    5. 오류 가능성이 있는 부분은 주의사항 작성
    6. 사용자의 요청 언어(python, java,... 등)에 맞춰 작성
    7. 불필요하게 긴 설명은 피하고 핵심 위주로 작성
    8. 설명과 주석은 한국어로 작성

    [응답형식]
    1. 기능 설명
    2. 코드
    3. 코드 설명
    4. 실행 결과 또는 사용 예시
    """

    # user_prompt = f"""
    # 아래의 내용을 참고하여 작성하시오.
    # - 장르 : {keyword}
    # - 해당 장르의 특징을 5줄로 정리할 것
    # - 해당 장르에 대한 인터뷰 질문 8가지를 작성할 것
    # - 너무 딱딱한 어구를 사용하지 말 것
    # - 기사나 칼럼작성 시 문제가 없을만한 질문을 작성할 것
    # """

    messages = [
        {"role": "system", "content": system_prompt},  # 필수 아님
        {"role": "user", "content": keyword},
    ]

    generated_response = model.chat(messages=messages)

    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=code_text,
    title="⌨ 코드 생성 프로그렘",
    inputs=[gr.Textbox(lines=10, placeholder="여기에 코드를 입력하세요", label="코드입력")],
    outputs=[gr.Markdown()],
    description="코드 작성 시 AI가 코드를 작성해드립니다.",
)

demo.launch()
