from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="backend/.env", extra="ignore")
    # 사용할 모델
    watsonx_api_key: str = Field(alias="WATSONX_API_KEY")
    watsonx_project_id: str = Field(alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(alias="WATSONX_URL")
    hf_token: str = Field(alias="HF_TOKEN")


settings = Settings()
