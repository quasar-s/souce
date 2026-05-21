import gradio as gr
from transformers import pipeline
import re

# 문장 여러개 = 파일화
# 리뷰 데이터 => 파일 분석

en_classifier = pipeline("sentiment-analysis", top_k=None)  # 감정 분석
ko_classifier = pipeline(
    "sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment", top_k=None
)  # 감정 분석


def is_korean(text):
    korean = re.search(r"[가-힣]", text)

    return korean is not None


def analysis(text):
    results_text = []

    # 엔터를 기준으로 문장 분리
    sentences = text.splitlines()
    sentences = [s.strip() for s in sentences if s.strip()]

    # 언어 구별
    if is_korean(text):
        results = ko_classifier(sentences)
    else:
        results = en_classifier(sentences)

    label_map = {
        "LABEL_0": "부정 👿",
        "NEGATIVE": "부정 👿",
        "LABEL_1": "긍정 😇",
        "POSITIVE": "긍정 😇",
    }

    for sentence, result in zip(sentences, results):
        best = max(result, key=lambda x: x["score"])
        label = best["label"]
        label = label_map.get(label, label)
        score = best["score"]

        results_text.append([sentence, label, score])

    return results_text


demo = gr.Interface(
    fn=analysis,
    inputs=[
        gr.Textbox(lines=3, placeholder="여기에 텍스트를 입력하세요", label="text")
    ],
    outputs=[gr.DataFrame(headers=["문장", "감정", "확률"])],
    title="다국어 감정 분석 웹앱",
    description="영어는 Hugging Face 기본 모델, 한국어는 KoBERT 기반 감정분석 모델 사용.",
)

demo.launch()
