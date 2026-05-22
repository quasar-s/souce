import gradio as gr
from transformers import pipeline
import re
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 파일(csv)
# 리뷰 데이터 파일 분석


english_classifier = pipeline("sentiment-analysis", top_k=None)
korean_classifier = pipeline(
    "sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment", top_k=None
)


def is_korean(text):
    korean = re.search(r"[가-힣]", text)
    return korean is not None


def predict_sentiment(file):

    # 1. file => pandas
    df = pd.read_csv(file)
    reviews = df["review"].to_list()

    results_text = []

    positive_count, negative_count = 0, 0
    # 가장 긍정,부정 리뷰와 점수 찾기
    best_positive_review, best_negative_review = "", ""
    best_positive_score, best_negative_score = 0, 0
    # 평균감정 점수
    positive_score, negative_score = 0, 0
    # 리뷰 길이 분석
    positive_length, negative_length = [], []

    label_map = {
        "LABEL_0": "부정 😡",
        "LABEL_1": "긍정 😉",
        "NEGATIVE": "부정 😡",
        "POSITIVE": "긍정 😉",
    }

    for sentence in reviews:

        # 한국말인지 확인하기
        if is_korean(sentence):
            result = korean_classifier(sentence)[0]
        else:
            result = english_classifier(sentence)[0]

        best = max(result, key=lambda x: x["score"])
        label = best["label"]
        label = label_map.get(label, label)
        score = best["score"]

        # 가장 긍정적인 리뷰(best_positive_review)와 확률(best_positive_score)
        # 가장 부정적인 리뷰(best_negative_review)와 확률(best_negative_score)
        # 긍정 점수 평균
        # 부정 점수 평균

        review_length = len(sentence)

        if label == "긍정 😉":
            positive_count += 1
            positive_score += score
            positive_length.append(review_length)

            # 가장 긍정적인 리뷰 변경
            if score > best_positive_score:
                best_positive_score = score
                best_positive_review = sentence
        else:
            negative_count += 1
            negative_score += score
            negative_length.append(review_length)
            # 가장 부정적인 리뷰 변경
            if score > best_negative_score:
                best_negative_score = score
                best_negative_review = sentence

        results_text.append([sentence, label, score])

    # 총 리뷰수 : 20 개
    total = len(reviews)
    # 긍정 리뷰 :  4 개
    # 부정 리뷰 : 16 개

    # 긍정 비율 : 21.05%
    positive_ratio = positive_count / total * 100
    # 부정 비율 : 78.95%
    negative_ratio = negative_count / total * 100

    # 점수 평균
    avg_positive = positive_score / positive_count if positive_count > 0 else 0
    avg_negative = negative_score / negative_count if negative_count > 0 else 0

    # 리뷰 길이 평균
    avg_positive_length = (
        sum(positive_length) / len(positive_length) if positive_length else 0
    )
    avg_negative_length = (
        sum(negative_length) / len(negative_length) if negative_length else 0
    )

    stats = f"""
    총 리뷰 수 : {total} 개

    긍정 리뷰 :  {positive_count} 개
    부정 리뷰 : {negative_count} 개

    긍정 비율 : {positive_ratio:.2f}%
    부정 비율 : {negative_ratio:.2f}%

    가장 긍정적인 리뷰 😊
    {best_positive_review}
    긍정 확률 : {best_positive_score:.2f}

    가장 부정적인 리뷰 🥵
    {best_negative_review}
    부정 확률 : {best_negative_score:.2f}

    긍정 점수 평균 : {avg_positive:.2f}
    부정 점수 평균 : {avg_negative:.2f}

    평균 긍정 리뷰 길이 😊
    {avg_positive_length:.2f} 자
    
    평균 부정 리뷰 길이 🥵
    {avg_negative_length:.2f} 자
    """

    # 차트
    fig, ax = plt.subplots()

    ax.pie(
        [positive_count, negative_count],
        labels=["긍정", "부정"],
        autopct="%.1f%%",
        startangle=90,
        counterclock=False,
    )

    return (
        gr.update(value=results_text, visible=True),
        gr.update(value=stats, visible=True),
        gr.update(value=fig, visible=True),
    )


with gr.Blocks() as demo:
    gr.Markdown("HuggingFace Transformer 기반 감정 분석 프로그램")

    inp = gr.File()
    btn = gr.Button("감정분석")

    df = gr.Dataframe(headers=["문장", "감정", "확률"], visible=False)
    stats = gr.Textbox(label="분석 결과", visible=False)
    chart_box = gr.Plot(label="감정 비율", visible=False)
    btn.click(fn=predict_sentiment, inputs=inp, outputs=[df, stats, chart_box])

demo.launch()
