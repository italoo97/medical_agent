from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSortSettings(BaseModel):
    by: Literal['price', 'throughput', 'latency'] = 'throughput'
    partition: Literal['none'] = 'none'


class ProviderSettings(BaseModel):
    sort: ProviderSortSettings = Field(default_factory=ProviderSortSettings)


class Settings(BaseSettings):
    """Configuração da aplicação; segredos vêm do ambiente."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        extra='ignore',
    )

    openrouter_api_key: SecretStr = Field(
        validation_alias='OPENROUTER_API_KEY'
    )
    http_referer: str = 'http://pos-ia.com'
    x_title: str = 'MedicalAppointmentAgent'
    port: int = 3000

    models: list[str] = Field(
        default_factory=lambda: [
            'nvidia/nemotron-3-super-120b-a12b:free',
        ]
    )
    temperature: float = 0.2
    max_tokens: int = 1024
    system_prompt: str = 'You are a helpful assistant.'
    provider: ProviderSettings = Field(default_factory=ProviderSettings)

    google_service_account_json: SecretStr = Field(
        validation_alias='GOOGLE_SERVICE_ACCOUNT_JSON'
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
