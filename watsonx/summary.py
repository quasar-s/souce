from dotenv import load_dotenv
import os
import gradio as gr
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

#.env 내용 가죠오기
load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

credentials = Credentials(
      url = f"{watsonx_ai_url}",
      api_key = f"{apikey}",
)
client = APIClient(credentials)

model = ModelInference(
      model_id="ibm/granite-4-h-small",
      api_client=client,
      project_id=f"{project_id}",
      params = {
      "max_tokens": 1000
      }
)

def summarize_text(text):
    if not text.strip():
        return "텍스트를 입력하여 주세요."

    instrutios = """
    당신은 텍스트를 한국어로 요약하는 전문가 입니다.
    -당신의 임무는 아래 주어지는 텍스트 문장을 한국어로 요약하는 것입니다.
    -요약 시 다음 사항을 준수해야합니다.
    -중복된 내용은 생략하되, 반복 되는 내용은 요약해서 더 강조합니다.
    -사례 중심보다 개념과 주장 중심으로 요약합니다.
    -3줄 이내로 요약합니다.
    -블릿 기호 형식으로 작성합니다.
    """
    messages = [
        {"role":"system", "content":instrutios}, # 필수 아님
        {"role":"user", "content":text},
    ]

    generated_response = model.chat(messages=messages)

    return generated_response['choices'][0]['message']['content']

demo = gr.Interface(
    fn=summarize_text,
    inputs=[gr.TextArea(lines=10, placeholder="요약할 내용의 텍스트 입력..", label="입력")],
    outputs=[gr.Markdown()],
    title="watsonx기반의 요약 프로그램"
)

demo.launch()