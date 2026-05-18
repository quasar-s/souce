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
def ad_text(name, brand_name, strenghth, tone, keyword, value):
    if not name.strip():
        return "텍스트를 입력하여 주세요."

    system_prompt= """
    당신은 광고를 한국어로 작성하는 전문가 입니다.
    - 당신의 임무는 주어지는 내용들을 분석하여 한국어로 새로운 광고를 생성하는 것입니다.
    - 5개의 광고를 작성하시오.
    - 제품의 특징을 부각시키시오.
    - 브랜드명과 어울리도록 만드시오. 
    - 주어진 단어를 그대로 사용하지 않고 독창적인 광고를 작성하시오.
    - 소비자들에게 구매 욕구가 생길만한 느낌으로 부탁드립니다.
    """

    user_prompt = f"""
    아래의 내용을 참고하여 작성하시오.
    - 제품명 : {name}
    - 브랜드명 : {brand_name}
    - 제품 특징 : {strenghth}
    - 톤앤매너 : {tone}
    - 필수 포함 키워드 : {keyword}
    - 브랜드 핵심가치 : {value}
    """

    messages = [
        {"role":"system", "content":system_prompt}, # 필수 아님
        {"role":"user", "content":user_prompt}
    ]

    generated_response = model.chat(messages=messages)

    return generated_response['choices'][0]['message']['content']


demo = gr.Interface(
    fn=ad_text,    
    title="🍡광고문구 프로그램",
    inputs=[
        gr.Text(label="제품명"),
        gr.Text(label="브랜드명"),
        gr.Text(label="제품 특징"),
        gr.Text(label="톤앤매너"),
        gr.Text(label="필수 포함 키워드"),
        gr.Text(label="브랜드 핵심 가치")
    ],
    outputs=[gr.Markdown()],
    description="텍스트 작성 시 AI가 광고 문구를 작성해 드립니다."
)

demo.launch()