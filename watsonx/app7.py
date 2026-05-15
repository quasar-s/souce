import gradio as gr

def cheer(name, level):
    return  name + "님 화이팅" + "👍" * int(level)

def greet(name, grade):
    return name + "🌟" * int(grade)

def bmi_calculator(height, weight):
    bmi = weight /((float(height) /100) ** 2)
    if bmi < 18.5:
        result = "저체중"
    elif bmi < 22.9:
        result = "정상체중" 
    elif bmi < 24.9:
        result = "과체중" 
    else:
        result = "비만"
    
    return f"당신의 BMI 지수는 키 : {height}, 몸무게 : {weight}, 판정은 {result}입니다."

with gr.Blocks() as demo:
    with gr.Tab("응원"):
        name = gr.Text(label="이름")
        chreer_strength = gr.Slider(1,5,step=1,label="응원강도")
        msg = gr.Textbox(label="응원 메세지")
        chreer_btn = gr.Button("응원")
        chreer_btn.click(fn=cheer, inputs=[name, chreer_strength], outputs=[msg])
        
    with gr.Tab("별점"):
        name = gr.Text(label="음식명")
        review_strength = gr.Slider(1,5,step=1,label="별점")
        msg = gr.Textbox(label="만족도 확인")
        review_btn = gr.Button("별점 등록")
        review_btn.click(fn=greet, inputs=[name, review_strength], outputs=[msg])
    with gr.Tab("BMI 판단기"):
        height = gr.Number(label="키")
        weight = gr.Number(label="몸무게")
        result = gr.Textbox(label="결과")
        gr.Button("BMI 판정").click(fn=bmi_calculator, inputs=[height, weight], outputs=[result])
        

demo.launch()