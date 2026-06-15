from langchain_ibm import ChatWatsonx

from langchain_ollama import ChatOllama

from backend.config.settings import settings

watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=f"{settings.watsonx_url}",
    api_key=f"{settings.watsonx_api_key}",
    project_id=f"{settings.watsonx_project_id}",
    params={"max_tokens": 2000, "temperature": 0},
)

# 로컬 LLM 선언
qwen_llm = ChatOllama(model="qwen3.5:4b", temperature=0)

exaone_llm = ChatOllama(model="exaone3.5:2.4b", temperature=0)
