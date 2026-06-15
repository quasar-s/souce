from langchain_ibm import WatsonxEmbeddings
from backend.config.settings import settings

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=f"{settings.watsonx_url}",
    api_key=f"{settings.watsonx_api_key}",
    project_id=f"{settings.watsonx_project_id}",
)
