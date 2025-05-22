from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml, os
import contextlib

# ---------- endpoint record ----------
class Endpoint(BaseModel):
    shortname: str
    provider: str
    base_url: str
    model: str
    cred_key: Optional[str] = None

# ---------- workflow roles ----------
class WorkflowModels(BaseModel):
    extraction: str
    contestants: List[str]
    target: str

    @model_validator(mode="after")
    def checks(cls, v):
        if v.extraction not in v.contestants:
            v.contestants.append(v.extraction)
        if v.target not in v.contestants:
            raise ValueError("target model must appear in contestants list")
        return v

# ---------- directory configuration ----------
class DirectoryConfig(BaseModel):
    papers: str = "_PAPERS"
    json_dir: str = "_JSON"  # renamed from 'json' to avoid conflict with BaseModel attribute
    mcq: str = "_MCQ"
    results: str = "_RESULTS"

    class Config:
        # Allow field aliases to map config.yml's 'json' to our 'json_dir'
        populate_by_name = True
        # Fields that exist in the data but not in the model
        extra = "ignore"

# ---------- quality settings ----------
class QualityConfig(BaseModel):
    minScore: int = 7
    chunkSize: int = 1000
    save_interval: int = 50
    defaultThreads: int = 4

# ---------- HTTP client settings ----------
class HttpClientConfig(BaseModel):
    connect_timeout: float = 3.05
    read_timeout: int = 10
    max_retries: int = 1
    pool_connections: int = 1
    pool_maxsize: int = 1

# ---------- global settings ----------
class Settings(BaseSettings):
    # Core workflow configuration
    workflow: WorkflowModels = WorkflowModels(
        extraction="gpt4o", contestants=["gpt4o"], target="gpt4o"
    )

    # Endpoints and credentials
    endpoints: Dict[str, Endpoint] = {}
    secrets: Dict[str, str] = {}

    # Common configuration settings
    directories: DirectoryConfig = DirectoryConfig()
    quality: QualityConfig = QualityConfig()
    http_client: HttpClientConfig = HttpClientConfig()

    # Timeouts and model parameters
    timeout: int = 45
    default_temperature: float = 0.7

    # Allow additional fields for prompts and other configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",  # Allow extra fields from config.yml
        env_nested_delimiter="."  # Use dot notation for nested env vars
    )

# ---------- overlay merge ----------
OVERLAY = [
    Path("config.yml"),
    Path("servers.yml"),
    Path("servers.local.yml"),
    Path("config.local.yml"),
]

def _deep_merge(dst: dict, src: dict):
    """Recursively merge two dictionaries."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v

def _safe_load_yaml(path: Path) -> dict:
    """Safely load YAML file with proper resource handling."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def load_settings(extra_cfgs: Optional[List[Path]] = None) -> Settings:
    """Load settings from configuration files in priority order.

    The overlay order is:
    1. config.yml (base configuration)
    2. servers.yml (endpoint definitions)
    3. servers.local.yml (user-specific endpoints)
    4. config.local.yml (user-specific workflow configuration)
    5. secrets.yml (credentials)
    6. Environment variables (highest precedence)
    """
    data: dict = {}

    # Load and merge configuration files
    for p in OVERLAY + (extra_cfgs or []):
        if p.exists():
            _deep_merge(data, _safe_load_yaml(p))

    # Load secrets if available
    if Path("secrets.yml").exists():
        secrets_data = _safe_load_yaml(Path("secrets.yml"))
        data.setdefault("secrets", {}).update(secrets_data)

    # Flatten nested secrets structure
    secrets_flattened = {}

    # Process secrets.yml data
    for key, value in data.items():
        if key in ["argo", "openai"] and isinstance(value, dict):
            # Handle top-level argo and openai configurations that contain credentials
            for subkey, subvalue in value.items():
                if subkey in ["username", "access_token"]:
                    secrets_flattened[f"{key}_{subkey}"] = subvalue

    # Process explicit secrets section
    if "secrets" in data:
        for key, value in data["secrets"].items():
            if isinstance(value, dict):
                # For nested structures like argo: {username: "value"}, flatten to argo_username: "value"
                for subkey, subvalue in value.items():
                    secrets_flattened[f"{key}_{subkey}"] = subvalue
            else:
                secrets_flattened[key] = value

    # Update data with flattened secrets
    data["secrets"] = secrets_flattened

    # Environment variables get highest precedence
    # Convert dot notation env vars to nested dict structure
    env_vars = {}
    for key, value in os.environ.items():
        if '.' in key:
            parts = key.split('.')
            current = env_vars
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            # Add direct environment variables to secrets
            data.setdefault("secrets", {})[key] = value

    # Merge environment structure with data
    _deep_merge(data, env_vars)

    return Settings.model_validate(data)

