from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, model_validator, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml, os
import contextlib
from enum import Enum

# ---------- model provider types ----------
class ModelProvider(str, Enum):
    """Known model providers"""
    OPENAI = "openai"
    ARGO = "argo"
    ARGO_DEV = "argo_dev"
    LOCAL = "local"
    HF = "hf"
    TEST = "test"
    CAFE = "cafe"
    ALCF = "alcf"
    
# ---------- endpoint record ----------
class Endpoint(BaseModel):
    """Configuration for a model endpoint"""
    shortname: str = Field(..., description="Short identifier for this endpoint")
    provider: ModelProvider = Field(..., description="Provider type for this endpoint")
    base_url: str = Field(..., description="Base URL for API access")
    model: str = Field(..., description="Default model identifier")
    cred_key: str = Field(..., description="Key to lookup credential in secrets.yml")

# ---------- workflow roles ----------
class WorkflowModels(BaseModel):
    """Configuration for model roles in the workflow"""
    extraction: str = Field(..., description="Model used for MCQ/fact extraction")
    contestants: List[str] = Field(..., description="Models to be evaluated")
    target: str = Field(..., description="Model to be fine-tuned")

    @model_validator(mode="after")
    def validate_model_roles(cls, v):
        """Ensure model role assignments are valid"""
        if v.extraction not in v.contestants:
            v.contestants.append(v.extraction)
        if v.target not in v.contestants:
            raise ValueError("target model must appear in contestants list")
        if len(set(v.contestants)) != len(v.contestants):
            raise ValueError("duplicate models in contestants list")
        
        # Model format validation moved to Settings class
        return v

# ---------- directory configuration ----------
class DirectoryConfig(BaseModel):
    """Configuration for workflow directories"""
    papers: str = Field("_PAPERS", description="Directory for source papers")
    json_dir: str = Field("_JSON", description="Directory for parsed JSON")  # renamed from 'json' to avoid conflict with BaseModel attribute
    mcq: str = Field("_MCQ", description="Directory for MCQ processing")
    results: str = Field("_RESULTS", description="Directory for final results")

    @field_validator("*")
    def validate_directory(cls, v: str) -> str:
        """Ensure directory paths are valid"""
        if not v:
            raise ValueError("Directory path cannot be empty")
        return v

    class Config:
        # Allow field aliases to map config.yml's 'json' to our 'json_dir'
        populate_by_name = True
        # Fields that exist in the data but not in the model
        extra = "ignore"

# ---------- quality settings ----------
class QualityConfig(BaseModel):
    """Configuration for quality control parameters"""
    minScore: int = Field(7, ge=1, le=10, description="Minimum acceptable score")
    chunkSize: int = Field(1000, gt=0, description="Size of processing chunks")
    save_interval: int = Field(50, gt=0, description="Interval for saving progress")
    defaultThreads: int = Field(4, gt=0, description="Default number of worker threads")

# ---------- HTTP client settings ----------
class HttpClientConfig(BaseModel):
    """Configuration for HTTP client behavior"""
    connect_timeout: float = Field(3.05, gt=0, description="Connection timeout in seconds")
    read_timeout: int = Field(10, gt=0, description="Read timeout in seconds")
    max_retries: int = Field(1, ge=0, description="Maximum number of retries")
    pool_connections: int = Field(1, gt=0, description="Number of connection pools")
    pool_maxsize: int = Field(1, gt=0, description="Maximum pool size")

# ---------- global settings ----------
class Settings(BaseSettings):
    """Global settings configuration with hierarchical loading"""
    # Core workflow configuration from config.local.yml
    workflow: WorkflowModels = Field(
        default_factory=lambda: WorkflowModels(
            extraction="test:all", 
            contestants=["test:all"], 
            target="test:all"
        ),
        description="Model workflow configuration"
    )
    
    # Endpoints catalog from servers.yml
    endpoints: Dict[str, Endpoint] = Field(
        default_factory=dict,
        description="Model endpoint configurations"
    )
    
    # Credentials from secrets.yml
    secrets: Dict[str, SecretStr] = Field(
        default_factory=dict,
        description="Sensitive credentials"
    )
    
    # Static configuration from config.yml
    directories: DirectoryConfig = Field(
        default_factory=DirectoryConfig,
        description="Workflow directory configuration"
    )
    quality: QualityConfig = Field(
        default_factory=QualityConfig,
        description="Quality control settings"
    )
    http_client: HttpClientConfig = Field(
        default_factory=HttpClientConfig,
        description="HTTP client configuration"
    )
    prompts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Prompt templates and system messages"
    )
    
    # Model parameters (can be overridden in config.local.yml)
    timeout: int = Field(45, gt=0, description="Global timeout in seconds")
    default_temperature: float = Field(
        0.7, ge=0.0, le=2.0,
        description="Default model temperature"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",  # Allow extra fields from config.yml
        env_nested_delimiter=".",  # Use dot notation for nested env vars
        validate_default=True
    )
    
    @model_validator(mode="after")
    def validate_settings(cls, values):
        """Perform cross-field validation"""
        # Convert string secrets to SecretStr for credential fields
        secrets = values.secrets
        for key, value in list(secrets.items()):
            if isinstance(value, str):
                secrets[key] = SecretStr(value)
    
        # Skip additional validation for test mode
        if values.workflow.extraction == "test:all" and len(values.workflow.contestants) == 1:
            return values
        
        # Validate all contestant models have endpoint configurations
        workflow = values.workflow
        endpoints = values.endpoints
    
        for model in workflow.contestants:
            if model.startswith("test:"):  # Skip validation for test models
                continue
            if ':' in model:  # provider:model format
                provider = model.split(':')[0]
                if provider not in [e.provider.value for e in endpoints.values()]:
                    raise ValueError(f"No endpoint configuration found for provider: {provider}")
            else:  # shortname format
                if not any(e.shortname == model for e in endpoints.values()):
                    raise ValueError(f"No endpoint configuration found for shortname: {model}")
    
        # Validate all required credentials exist
        for endpoint in endpoints.values():
            if endpoint.cred_key not in secrets and endpoint.provider != "test":
                raise ValueError(f"Missing credential for {endpoint.shortname}: {endpoint.cred_key}")
    
        return values

# ---------- file loading logic ----------
REPO_CONFIG_FILES = [
    Path("config.yml"),      # Static configuration
    Path("servers.yml"),     # Endpoint catalog
]

LOCAL_CONFIG_FILES = [
    Path("config.local.yml"),  # User's model choices
    Path("secrets.yml"),       # User's credentials
]

def _deep_merge(dst: dict, src: dict):
    """Recursively merge two dictionaries, preserving secrets."""
    for k, v in src.items():
        if k == "secrets":
            # Special handling for secrets: preserve existing secrets
            dst.setdefault(k, {})
            dst[k].update(v)  # Only add new secrets, don't overwrite
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
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
    1. config.yml (static configuration)
    2. servers.yml (endpoint catalog)
    3. config.local.yml (user's model choices)
    4. secrets.yml (credentials)
    5. Environment variables (highest precedence)
    
    Parameters:
        extra_cfgs: Optional list of additional configuration file paths to load
        
    Returns:
        Settings: Validated settings object with all configurations merged
    
    Note:
        - Configuration files are merged in the order listed above
        - Later files override earlier ones when keys conflict
        - Environment variables have the highest precedence
        - Sensitive credentials are automatically wrapped in SecretStr
        - Files in LOCAL_CONFIG_FILES are git-ignored and must be created by each user
    """
    data: dict = {}
    data["secrets"] = {}  # Initialize secrets dict
    
    # Load repository-tracked configuration
    for p in REPO_CONFIG_FILES:
        if p.exists():
            _deep_merge(data, _safe_load_yaml(p))
    
    # Load user-specific configuration
    for p in LOCAL_CONFIG_FILES + (extra_cfgs or []):
        if p.exists():
            config_data = _safe_load_yaml(p)
            if p.name == 'secrets.yml':
                # Direct update of secrets
                data["secrets"].update(config_data)
            else:
                _deep_merge(data, config_data)
    
    # Process secrets
    secrets_flattened = {}
    
    # Process any nested credentials
    for key, value in data.items():
        if key in ["argo", "openai"] and isinstance(value, dict):
            # Handle top-level argo and openai configurations that contain credentials
            for subkey, subvalue in value.items():
                if subkey in ["username", "access_token", "api_key", "token"]:
                    secrets_flattened[f"{key}_{subkey}"] = subvalue
    
    # Process explicit secrets section
    if "secrets" in data:
        for key, value in data["secrets"].items():
            if isinstance(value, dict):
                # Flatten nested structures
                for subkey, subvalue in value.items():
                    secrets_flattened[f"{key}_{subkey}"] = subvalue
            else:
                secrets_flattened[key] = value
    
    # Update data with flattened secrets
    data["secrets"].update(secrets_flattened)
    
    # Environment variables get highest precedence for non-secret fields
    # or for secrets that don't exist
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
            # Only add environment variables if they don't exist in secrets
            if key not in data["secrets"]:
                data["secrets"][key] = value
    
    # Merge environment structure with data
    _deep_merge(data, env_vars)
    
        # ---- validate (with friendly missing-endpoint message) --------------
    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            msg = err.get("msg", "")
            if "No endpoint configuration found for shortname" in msg:
                missing = msg.split("shortname:")[-1].strip()
                print(
                    f"❌  Model '{missing}' not found in servers.yml — "
                    "add an endpoint or use a different model name."
                )
                sys.exit(1)
        # Any other validation problems: show the full traceback
        raise

