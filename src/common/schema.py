from __future__ import annotations

"""Common configuration **schema** (no I/O).

This file defines Pydantic models for loading and validating
configuration data. It was updated on 2025-05-24 to:
1. Make `target` optional.
2. Only validate credentials for endpoints in use.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Model provider types
# -----------------------------------------------------------------------------

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ARGO = "argo"
    ARGO_DEV = "argo_dev"
    LOCAL = "local"
    HF = "hf"
    TEST = "test"
    CAFE = "cafe"
    ALCF = "alcf"


# -----------------------------------------------------------------------------
# Endpoint record
# -----------------------------------------------------------------------------

class Endpoint(BaseModel):
    """Configuration for a model endpoint"""
    shortname: str
    provider: ModelProvider
    base_url: str
    model: str
    cred_key: str


# -----------------------------------------------------------------------------
# Workflow roles
# -----------------------------------------------------------------------------

class WorkflowModels(BaseModel):
    """Configuration for model roles in the workflow"""
    extraction: str = Field(..., description="Model used for MCQ/fact extraction")
    contestants: List[str] = Field(..., description="Models to be evaluated")
    target: Optional[str] = Field(
        None, description="Model to be fine-tuned (optional)"
    )

    @model_validator(mode="after")
    def validate_model_roles(cls, v: "WorkflowModels") -> "WorkflowModels":
        # ensure extraction is in contestants
        if v.extraction not in v.contestants:
            v.contestants.append(v.extraction)
        # target if provided must be in contestants
        if v.target:
            if v.target not in v.contestants:
                raise ValueError("target model must appear in contestants list")
        # no duplicates
        if len(set(v.contestants)) != len(v.contestants):
            raise ValueError("duplicate models in contestants list")
        return v


# -----------------------------------------------------------------------------
# Directory configuration
# -----------------------------------------------------------------------------

class DirectoryConfig(BaseModel):
    """Configuration for workflow directories"""
    papers: str = Field("_PAPERS", description="Dir for source PDFs")
    json_dir: str = Field(
        "_JSON", alias="json", description="Dir for parsed JSON"
    )
    mcq: str = Field("_MCQ", description="Dir for MCQ JSONL")
    results: str = Field("_RESULTS", description="Dir for outputs")

    @field_validator("*")
    def non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Directory path cannot be empty")
        return v

    class Config:
        populate_by_name = True
        extra = "ignore"


# -----------------------------------------------------------------------------
# Quality settings
# -----------------------------------------------------------------------------

class QualityConfig(BaseModel):
    minScore: int = Field(7, ge=1, le=10)
    chunkSize: int = Field(1000, gt=0)
    save_interval: int = Field(50, gt=0)
    defaultThreads: int = Field(4, gt=0)


# -----------------------------------------------------------------------------
# HTTP client settings
# -----------------------------------------------------------------------------

class HttpClientConfig(BaseModel):
    connect_timeout: float = Field(3.05, gt=0)
    read_timeout: int = Field(10, gt=0)
    max_retries: int = Field(1, ge=0)
    pool_connections: int = Field(1, gt=0)
    pool_maxsize: int = Field(1, gt=0)


# -----------------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    """Global settings configuration with hierarchical loading"""
    workflow: WorkflowModels = Field(
        default_factory=lambda: WorkflowModels(
            extraction="test:all", contestants=["test:all"], target=None
        )
    )
    endpoints: Dict[str, Endpoint] = Field(default_factory=dict)
    secrets: Dict[str, SecretStr] = Field(default_factory=dict)
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    http_client: HttpClientConfig = Field(default_factory=HttpClientConfig)
    prompts: Dict[str, Any] = Field(default_factory=dict)

    timeout: int = Field(45, gt=0)
    default_temperature: float = Field(
        0.7, ge=0.0, le=2.0
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter='.',
        extra='allow',
        validate_default=True,
    )

    # -------------------------------------------------------------------------
    # Cross-field validation
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def _cross_checks(cls, v: "Settings") -> "Settings":
        # Convert any raw string secrets to SecretStr
        for key, val in list(v.secrets.items()):
            if isinstance(val, str):
                v.secrets[key] = SecretStr(val)

        # Skip deep validation for default dummy
        if (
            v.workflow.extraction == "test:all"
            and len(v.workflow.contestants) == 1
        ):
            return v

        # Validate model->endpoint mapping
        providers = {e.provider.value for e in v.endpoints.values()}
        for m in v.workflow.contestants:
            if m.startswith("test:"):
                continue
            if ':' in m:
                prov = m.split(':')[0]
                if prov not in providers:
                    raise ValueError(f"No endpoint configuration for provider '{prov}'")
            else:
                if not any(e.shortname == m for e in v.endpoints.values()):
                    raise ValueError(f"No endpoint configuration for shortname '{m}'")

        # Check credentials only for used endpoints
        used_shortnames: Set[str] = set()
        used_providers: Set[str] = set()
        # include extraction and contestants
        for m in [v.workflow.extraction] + v.workflow.contestants:
            if ':' in m:
                used_providers.add(m.split(':')[0])
            else:
                used_shortnames.add(m)

        for ep in v.endpoints.values():
            if ep.shortname in used_shortnames or ep.provider.value in used_providers:
                if ep.provider != ModelProvider.TEST and ep.cred_key not in v.secrets:
                    raise ValueError(
                        f"Missing credential '{ep.cred_key}' required for endpoint '{ep.shortname}'"
                    )

        return v

