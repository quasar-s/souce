import gradio as gr

# from multimodal import (encode_image,image_to_caption,build_multimodal_index,search_with_images,multimodal_answer,load_or_build_vectorstore,)
from image_analyzer import process_documents, build_rag_chain

vectorstore = None
rag_chain = None


def upload_and_process(files):
    global vectorstore, rag_chain

    if not files:
        return "파일을 업로드 해주시기 바랍니다."

    paths = [f.name for f in files]

    vectorstore = process_documents(paths)
    rag_chain = build_rag_chain(vectorstore)

    return f"{len(paths)}개 문서 처리 완료!! 질문하세요."


def answer_question(question):
    global rag_chain

    if not rag_chain:
        return "먼저 문서를 업로드 해주세요."

    result = rag_chain.invoke(question)
    return result


with gr.Blocks(title="이미지 기반 문서 분석 시스템") as app:
    gr.Markdown("# 이미지 기반 문서 분석 시스템")
    gr.Markdown("스캔 문서, 영수증, 계약서 이미지를 업로드하면 질문 할 수 있습니다.")

    with gr.Row():
        file_input = gr.Files(
            label="문서 이미지 업로드",
            file_types=[
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            ],
        )
        status_bar = gr.Textbox(label="처리상태", lines=6)
    upload_btn = gr.Button("문서 분석 시작", variant="primary")
    upload_btn.click(fn=upload_and_process, inputs=file_input, outputs=status_bar)

    gr.Markdown("---")
    question = gr.Textbox(
        label="질문 입력", placeholder="문서에서 찾고싶은 질문을 입력하세요"
    )
    answer_box = gr.Textbox(label="답변", lines=8)
    ask_btn = gr.Button("질문하기")
    ask_btn.click(fn=answer_question, inputs=question, outputs=answer_box)

app.launch()
