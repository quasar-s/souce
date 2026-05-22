import gradio as gr
from transformers import pipeline

summarizer = pipeline("summarization")
classifier = pipeline("sentiment-analysis")
ner = pipeline("ner", grouped_entities=True)
translator = pipeline("translation_en_to_ko", model="facebook/nllb-200-distilled-600M")


def analyze_news(article):
    if not article:
        return "", "", "", ""

    summary_result = summarizer(article)
    summary = summary_result[0]["summary_text"]

    sentiment_result = classifier(article)

    sentiment = (
        f"감성 : {sentiment_result[0]['label']}"
        f"score : {sentiment_result[0]['score']:.4f}"
    )

    ner_result = ner(article)

    keywords = []

    for item in ner_result:
        word = item["word"]

        if word not in keywords:
            keywords.append(word)

    keyword_text = ", ".join(keywords)

    translation_result = translator(summary)
    translated_summary = translation_result[0]["translation_text"]

    return summary, sentiment, keyword_text, translated_summary


with gr.Blocks(title="AI 뉴스 분석 챗봇") as demo:
    gr.Markdown("## 📰AI 뉴스 분석기")
    with gr.Row():
        with gr.Column(scale=2):
            article_input = gr.Textbox(
                label="영어 기사 뉴스 기사 입력",
                lines=15,
                placeholder="영어 뉴스 기사를 입력 하세요",
            )
            analyze_btn = gr.Button("뉴스 분석 시작")
        with gr.Column(scale=2):
            summary_output = gr.Textbox(label="뉴스 요약", lines=3)
            sentiment_output = gr.Textbox(label="감성 분석")
            keyword_output = gr.Textbox(label="키워드 추출")
            translation_output = gr.Textbox(label="한국어 번역", lines=3)
    analyze_btn.click(
        fn=analyze_news,
        inputs=article_input,
        outputs=[summary_output, sentiment_output, keyword_output, translation_output],
    )
demo.launch()
