import gradio as gr
from transformers import pipeline
import edge_tts
import asyncio

whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")


change_text = ""
current_answer = ""


def audio_to_text(file):
    global change_text
    result = whisper(file, return_timestamps=True)
    change_text = result["text"]
    return change_text


def question_and_answer(question):
    global change_text, current_answer

    if not change_text.strip():
        return "오디오가 입력되지않았습니다."
    if not question.strip():
        return "질문을 입력하여 주십시오."

    # system_prompt = f"""
    # You are an AI that Question Answering.
    # You should follow:
    # 1. If there is a {change_text}, you should answer the question based on it.
    # 2. Answer in a kind way.
    # 3. Answer in 3 lines.
    # """
    prompt = f"""
    다음 음성 내용을 참고하여 질문에 답변하시오.

    음성 내용:
    {change_text}

    질문:
    {question}

    답변:

    """
    result = generator(
        prompt,
        max_new_tokens=250,
        return_full_text=False,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    current_answer = result[0]["generated_text"].strip()
    return current_answer


def make_voice():
    global current_answer

    if not current_answer:
        return "변환할 텍스트가 없습니다. 순서를 지켜 다시 시도해 주십시오."

    asyncio.run(text_to_audio(current_answer))

    return "answer.mp3"


async def text_to_audio(text):

    voice = "ko-KR-InjoonNeural"

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("answer.mp3")


with gr.Blocks(title="AI 음성 챗봇") as demo:
    gr.Markdown("## AI 음성 비서")
    with gr.Row():
        with gr.Column(scale=1):
            upload_file = gr.Audio(type="filepath")
            change_btn = gr.Button("텍스트 변환")
        with gr.Column(scale=1):
            change_otp = gr.Textbox(label=" 텍스트 변환", lines=3)
    with gr.Row():
        with gr.Column(scale=1):
            question_text = gr.Textbox(label="question")
            question_btn = gr.Button("질문하기")
        with gr.Column(scale=1):
            answer_text = gr.Textbox(label="answer", placeholder="답변이 표시됩니다.")
            answer_btn = gr.Button("답변 음성 변환")
    with gr.Row():
        answer_otp = gr.Audio(label="AI 음성 답변", autoplay=True)

    change_btn.click(fn=audio_to_text, inputs=upload_file, outputs=change_otp)
    question_btn.click(
        fn=question_and_answer, inputs=question_text, outputs=answer_text
    )
    answer_btn.click(fn=make_voice, outputs=answer_otp)

demo.launch()
