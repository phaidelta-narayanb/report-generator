from pydantic import BaseModel
from pydantic_settings_yaml import YamlBaseSettings
from pydantic_settings import SettingsConfigDict

from typing import List, Dict, Literal, Union, Optional
from pathlib import Path


class UploadSettings(BaseModel):
    allowed_types: Optional[List[str]] = None


class ClientSettings(BaseModel):
    system_prompt: str
    user_prompts: List[Union[str, Dict[str, str]]]


class ReportSettings(BaseModel):
    templates_dir: Path = Path("templates/")
    report_template_file: Path
    export_method: Literal["pandoc", "weasyprint"] = "pandoc"

    logo_image_url: Optional[str] = None


class CustomerSettings(BaseModel):
    report: ReportSettings


class Settings(YamlBaseSettings):
    upload: UploadSettings
    client: ClientSettings
    customer: CustomerSettings

    model_config = SettingsConfigDict(
        secrets_dir="secrets/",
        yaml_file="config.yaml"
    )

