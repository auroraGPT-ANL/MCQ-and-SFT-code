# src/common/schema.py
"""
Pure *data-model* definitions.

Everything here is Pydantic models or enums—NO file-system access.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
#  Enums & small helpers
# --------------------------------------------------------------------------- #


class ModelProvider(str, Enum):
    """Known model providers."""

    OPENAI = "openai"
    ARGO = "argo"
    ARGO_DEV = "argo_dev"
    LOCAL = "local"
    HF = "hf"
    TEST = "test"
    CAFE = "cafe"
    ALCF = "alcf"


# --------------------------------------------------------------------------- #
#  Endpoint catalogue
# --------------------------------------------------------------------------- #


class Endpoint(BaseModel):
    """Configuration for a model endpoint."""

    shortname: str
    provider: ModelProvider
    base_url: str
    model: str
    cred_key: str


# --------------------------------------------------------------------------- #
#  Workflow-role config
# --------------------------------------------------------------------------- #


class WorkflowModels(BaseModel):
    """Which models play which role in the workflow."""

    extraction: str
    contestants: List[str]
    target: str

    @model_validator(mode="after")
    def _roles_consistent(cls, v: "WorkflowModels") -> "WorkflowModels":
        # ensure extraction appears in contestants
        if v.extraction not in v.contestants:
            v.contestants.append(v.extraction)
        if v.target not in v.contestants:
            raise ValueError("target model must appear in contestants list")
        if len(set(v.contestants)) != len(v.contestants):
            raise ValueError("duplicate models in contestants list")
        return v


# --------------------------------------------------------------------------- #
#  Directory & runtime-tuning sections
# --------------------------------------------------------------------------- #


class DirectoryConfig(BaseModel):
    papers: str = "_PAPERS"
    json_dir: str = Field("_JSON", alias="json")
    mcq: str = "_MCQ"
    results: str = "_RESULTS"

    @field_validator("*")
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Directory path cannot be empty")
        return v

    model_config = {"populate_by_name": True, "extra": "ignore"}


class QualityConfig(BaseModel):
    minScore: int = Field(7, ge=1, le=10)
    chunkSize: int = Field(1000, gt=0)
    save_interval: int = Field(50, gt=0)
    defaultThreads: int = Field(4, gt=0)


class HttpClientConfig(BaseModel):
    connect_timeout: float = Field(3.05, gt=0)
    read_timeout: int = Field(10, gt=0)
    max_retries: int = Field(1, ge=0)
    pool_connections: int = Field(1, gt=0)
    pool_maxsize: int = Field(1, gt=0)


# --------------------------------------------------------------------------- #
#  Top-level Settings (still *only* a dataclass)
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Validated configuration object produced by `common.loader.load_settings`."""

    workflow: WorkflowModels = WorkflowModels(
        extraction="test:all", contestants=["test:all"], target="test:all"
    )

    endpoints: Dict[str, Endpoint] = {}
    secrets: Dict[str, SecretStr] = {}

    directories: DirectoryConfig = DirectoryConfig()
    quality: QualityConfig = QualityConfig()
    http_client: HttpClientConfig = HttpClientConfig()
    prompts: Dict[str, Any] = {}

    timeout: int = Field(45, gt=0)
    default_temperature: float = Field(0.7, ge=0.0, le=2.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter=".",
        extra="allow",
        validate_default=True,
    )

    # --------------------------------------------------------------------- #
    #  Cross-field checks
    # --------------------------------------------------------------------- #

    @model_validator(mode="after")
    def _cross_checks(cls, v: "Settings") -> "Settings":
        secrets = v.secrets
        for key, value in list(secrets.items()):
            if isinstance(value, str):
                secrets[key] = SecretStr(value)

        # Skip deep validation for the default dummy config
        if (
            v.workflow.extraction == "test:all"
            and len(v.workflow.contestants) == 1
        ):
            return v

        providers_in_endpoints = {e.provider.value for e in v.endpoints.values()}

        for model in v.workflow.contestants:
            if model.startswith("test:"):
                continue
            if ":" in model:  # provider:model form
                provider = model.split(":")[0]
                if provider not in providers_in_endpoints:
                    raise ValueError(f"No endpoint configuration for provider '{provider}'")
            else:  # shortname
                if not any(e.shortname == model for e in v.endpoints.values()):
                    raise ValueError(f"No endpoint configuration for shortname '{model}'")

        for endpoint in v.endpoints.values():
            if endpoint.provider != ModelProvider.TEST and endpoint.cred_key not in secrets:
                raise ValueError(
                    f"Missing credential {endpoint.cred_key!r} for endpoint {endpoint.shortname}"
                )

        return v

