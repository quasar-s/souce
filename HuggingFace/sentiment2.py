import gradio as gr
from transformers import pipeline

# 감정 분석
classifier = pipeline(
    "sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis"
)
# classifier = pipeline(
#     "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
# )


def analysis(text):
    # 언어 구별
    result = classifier(text)
    label = result[0]["label"]
    score = result[0]["score"]

    if "Negative" in label:
        sentiment = "부정"
    elif "Positive" in label:
        sentiment = "긍정"
    else:
        sentiment = label

    prompt = f"""
    감정 : {sentiment}
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
