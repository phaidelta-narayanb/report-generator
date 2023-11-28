from pydantic import BaseModel
from pydantic_settings_yaml import YamlBaseSettings
from pydantic_settings import SettingsConfigDict

from typing import List, Optional


class UploadSettings(BaseModel):
    allowed_types: Optional[List[str]] = None


class CustomerSettings(BaseModel):
    logo_image_url: str


class Settings(YamlBaseSettings):
    upload: UploadSettings
    customer: CustomerSettings

    model_config = SettingsConfigDict(
        secrets_dir="secrets/",
        yaml_file="config.yaml"
    )

