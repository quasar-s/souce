from backend.ai.llm import watson_llm


# LLM 모델 통신
# 데이터베이스 통신
def question_and_answer(question):
    response = watson_llm.invoke(question)
    return response.content
