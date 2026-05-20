import gradio as gr
from transformers import pipeline
from PIL import Image

# 이미지 캡션
caption = pipeline("image-to-text")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

current_caption = ""


def chat(message, history):

    global current_caption

    # message에서 텍스트와 이미지 분리
    text = message["text"]

    # 이미지만
    if message.get("files"):
        image = message.get("files")[0]
        if image:
            result = caption(image)
            caption_result = result[0]["generated_text"]
            # 전역 변수에 저장
            current_caption = caption_result

            prompt = f"""
            이미지 설명:
            {caption_result}

            사용자 질문:
            {text}
            """
            return prompt
    elif text:
        if not current_caption:
            return f"""
            사용자질문:
            {text}
            """

        prompt = f"""
        당신은 이미지 분석 AI입니다.
        다음 이미지 설명을 참고하여 사용자의 질문에
        한 문장으로 대답하시오
        이미지 설명:
        {current_caption}

        사용자 질문:
        {text}

        """

        # caption_result값이 있다면 적절한 문장 생성하도록 만들기
        result = generator(
            prompt,
            max_new_tokens=50,
            return_full_text=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        print("text result", result)

        respons = result[0]["generated_text"]
        answer = respons.split("\n")[0].strip()
        return answer


demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
    title="🖼️멀티 모달 AI 챗봇",
    description="이미지를 업로드하면 이미지에 대한 설명을 생성하는 챗봇입니다. 텍스트로 질문도 가능합니다.",
)

demo.launch()
