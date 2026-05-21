import gradio as gr
from transformers import pipeline
import re
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 파일(csv)
# 리뷰 데이터 => 파일 분석

en_classifier = pipeline("sentiment-analysis", top_k=None)  # 감정 분석
ko_classifier = pipeline(
    "sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment", top_k=None
)  # 감정 분석


def is_korean(text):
    korean = re.search(r"[가-힣]", text)

    return korean is not None


def preditc_analysis(file):

    # 1. file => pandas
    df = pd.read_csv(file)
    reviews = df["review"].to_list()

    results_text = []

    label_map = {
        "LABEL_0": "부정 👿",
        "NEGATIVE": "부정 👿",
        "LABEL_1": "긍정 😇",
        "POSITIVE": "긍정 😇",
    }

    positive_count = 0
    negative_count = 0

    positive_score = 0
    negative_score = 0

    positive_len = 0
    negative_len = 0

    for sentence in reviews:
        # 언어 구별
        if is_korean(sentence):
            result = ko_classifier(sentence)[0]
        else:
            result = en_classifier(sentence)[0]

        best = max(result, key=lambda x: x["score"])
        label = best["label"]
        label = label_map.get(label, label)
        score = best["score"]

        if label == "긍정 😇":
            positive_count += 1
            positive_score += score
            positive_len += len(sentence)
        else:
            negative_count += 1
            negative_score += score
            negative_len += len(sentence)
        results_text.append([sentence, label, score])

    # 총 리뷰 수 :  개
    total = len(reviews)
    # 긍정 리뷰 :   개
    # 부정 리뷰 :   개

    # 긍정 비율 :   %
    positive_present = positive_count / total * 100
    # 부정 비율 :   %
    negative_present = negative_count / total * 100

    # 긍정 점수 평균
    mean_positive_score = positive_score / positive_count * 100
    # 부정 점수 평균
    mean_negative_score = negative_score / negative_count * 100

    mean_positive_len = positive_len / positive_count
    mean_negative_len = negative_len / negative_count

    # 가장 긍정적인 리뷰(best_positive_review)와 확률(best_positive_persent)
    # 가장 부정적인 리뷰(wast_negative_review)와 확률(wast_negative_persent)
    best_positive_review = max(
        (x for x in results_text if x[1] == "긍정 😇"), key=lambda x: x[2]
    )[0]
    wast_negative_review = max(
        (x for x in results_text if x[1] == "부정 👿"), key=lambda x: x[2]
    )[0]
    best_positive_persent = (
        max((x for x in results_text if x[1] == "긍정 😇"), key=lambda x: x[2])[2] * 100
    )
    wast_negative_persent = (
        max((x for x in results_text if x[1] == "부정 👿"), key=lambda x: x[2])[2] * 100
    )

    stats = f"""
    총 리뷰 수 : {total} 개

    긍정 리뷰 : {positive_count} 개
    부정 리뷰 : {negative_count} 개

    긍정 비율 : {positive_present:.2f}%
    부정 비율 : {negative_present:.2f}%

    가장 긍정적인 리뷰 🤩
    {best_positive_review}
    긍정 확률 : {best_positive_persent:.2f}%

    가장 부정적인 리뷰 😱
    {wast_negative_review}
    부정 확률 : {wast_negative_persent:.2f}%

    긍정 점수 평균 🤩
    {mean_positive_score:.2f}%
    부정 점수 평균 😱
    {mean_negative_score:.2f}%

    평균 긍정 리뷰 길이 🤩
    {mean_positive_len:.2f} 자
    평균 부정 리뷰 길이 😱
    {mean_negative_len:.2f} 자
    """
    fig, ax = plt.subplots()
    ax.pie(
        [positive_count, negative_present],
        labels=["긍정", "부정"],
        autopct="%.1f%%",
        startangle=90,
        counterclock=False,
    )
    # update
    return (
        gr.update(value=results_text, visible=True),
        gr.update(value=stats, visible=True),
        gr.update(value=fig, visible=True),
    )


with gr.Blocks() as demo:
    gr.Markdown("Hugging Face 기반 감정분석 모델 사용")

    inp = gr.File()
    btn = gr.Button("감정 분석")

    df = gr.DataFrame(headers=["문장", "감정", "확률"], visible=False)
    stats = gr.Textbox(label="분석 결과", visible=False)
    chartbox = gr.Plot(label="감정 비율 분석", visible=False)
    btn.click(fn=preditc_analysis, inputs=inp, outputs=[df, stats, chartbox])

demo.launch()
