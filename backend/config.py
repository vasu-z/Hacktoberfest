import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    saarthi_service_name: str = "saarthi-backend"
    saarthi_version: str = "0.1.0"
    saarthi_env: str = "development"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemma-4-26b-a4b-it"
    gemini_ocr_model: str = "gemini-3.5-flash-lite"

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __repr__(self) -> str:
        data = self.model_dump()
        if data.get("gemini_api_key"):
            data["gemini_api_key"] = "**********"
        return f"{self.__class__.__name__}({data})"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
