import gradio as gr

def greet(name, grade):
    return name + "🌟" * int(grade)

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Text(label="음식명"), gr.Slider(1,5,step=1,label="별점")],
    outputs=[gr.Text(label="만족도 출력")],
    api_name="별점 리뷰 생성"
)

demo.launch()