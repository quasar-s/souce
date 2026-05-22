import gradio as gr
from transformers import pipeline
import fitz
from fpdf import FPDF
import io

# import io

summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
classifier = pipeline("sentiment-analysis")
qa_model = pipeline(
    "question-answering", model="distilbert/distilbert-base-cased-distilled-squad"
)


def summarizater(file):
    if not file:
        return "pdf파일을 업로드 해주시기 바랍니다."

    doc = fitz.open(file)
    pdf_text = ""
    page = doc[0]
    for page in doc:
        pdf_text += page.get_text()

    summary_result = summarizer(pdf_text)
    summary = summary_result[0]["summary_text"]

    sentiment_result = classifier(summary)
    sentiment = f"""
    {sentiment_result[0]['label']}
    (확신도 {sentiment_result[0]['score']:.2f})
    """

    return summary, sentiment


def emotion_chat(question, summary, sentiment):

    if not summary.strip():
        return "관련된 요약이 없습니다."

    context = f"""
    요약: {summary}
    감정 분석 결과 : {sentiment}
    """
    try:
        result = qa_model(question=question, context=context)
        return result["answer"]
    except:
        return "질문에 답변할 수 없습니다."


def downloads_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for q, a in text:
        pdf.set_text_color(0, 0, 150)
        pdf.multi_cell(0, 8, f"질문: {q}")
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, f"질문: {a}")

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output


def chatbot_interaction(question, history, summary, sentiment):
    if not summary or not summary.strip():
        return "", history, "상담 내용을 먼저 업로드 해주세요.", sentiment

    ans = emotion_chat(question, summary, sentiment)
    history = history or []
    history.append((question, ans))
    return "", history, summary, sentiment


def handle_upload(file):
    summary, sentiment = summarizater(file)
    return summary, sentiment, []


with gr.Blocks() as demo:
    summary_state = gr.State("")
    sentiment_state = gr.State("")
    history_state = gr.State([])

    gr.Markdown("상담 보조 챗봇")
    with gr.Row():
        with gr.Column(scale=2):
            emotion_file = gr.File(label="상담 내용", file_types=[".pdf"])
            # upload_btn = gr.Button("업로드")
        with gr.Column(scale=2):
            emotion_text = gr.Textbox(label="전체 요약", lines=3)
            emotion_q = gr.Textbox(label="감정 분석", lines=3)
    with gr.Row():
        with gr.Column(scale=2):
            question_text = gr.Textbox(label="질문")
            question_btn = gr.Button("질문하기")
    with gr.Row():
        answer_text = gr.Chatbot(label="대화창", value=[])
    with gr.Row():
        d_btn = gr.Button("채팅 다운로드")
        pdf_file_output = gr.File(label="다운로드된 pdf")

    # 파일 업로드 & 상태 초기화
    emotion_file.upload(
        fn=handle_upload,
        inputs=emotion_file,
        outputs=[emotion_text, emotion_q, history_state],
    )

    # 질문 및 대화 상태 업데이트
    question_btn.click(
        fn=chatbot_interaction,
        inputs=[question_text, history_state, summary_state, sentiment_state],
        outputs=[question_text, answer_text, summary_state, sentiment_state],
    )

    # PDF 다운로드
    d_btn.click(fn=downloads_pdf, inputs=history_state, outputs=pdf_file_output)

demo.launch()
