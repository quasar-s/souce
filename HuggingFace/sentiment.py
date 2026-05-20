import gradio as gr
from transformers import pipeline
import re

en_classifier = pipeline("sentiment-analysis")  # 감정 분석
ko_classifier = pipeline(
    "sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment"
)  # 감정 분석

g_result = ""


def is_korean(text):
    global g_result
    korean = re.search(r"[가-힣]", text)

    if korean != None:
        g_result = ko_classifier(text)
    else:
        g_result = en_classifier(text)


def analysis(text):
    global g_result

    is_korean(text)
    # 언어 구별

    label = g_result[0]["label"]
    score = g_result[0]["score"]

    if label == "LABEL_0" or label == "NEGATIVE":
        result = "부정"
    elif label == "LABEL_1" or label == "POSITIVE":
        result = "긍정"

    prompt = f"""
    감정 : {result}
    확률 : {score:.4f}
    """

    return prompt


demo = gr.Interface(
    fn=analysis,
    inputs=[
        gr.Textbox(lines=3, placeholder="여기에 텍스트를 입력하세요", label="text")
    ],
    outputs=[gr.Textbox(lines=3, label="greeting")],
    title="다국어 감정 분석 웹앱",
    description="영어는 Hugging Face 기본 모델, 한국어는 KoBERT 기반 감정분석 모델 사용.",
)

demo.launch()
