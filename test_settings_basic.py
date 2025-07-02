#!/usr/bin/env python3
"""
Basic verification of the new settings system.
Tests hierarchical loading and credential handling.
"""
import os
import sys
from pathlib import Path
from settings import load_settings, Settings
from pydantic import SecretStr

def test_basic_settings():
    """
    Test the basic functionality of the settings system.
    Creates minimal test config files and verifies loading behavior.
    """
    # Create test configuration files
    with open("test_config.yml", "w") as f:
        f.write("""
directories:
    papers: base_papers
    json: base_json
workflow:
    extraction: test:all
    contestants: [test:all]
    target: test:all
""")

    with open("test_servers.yml", "w") as f:
        f.write("""
endpoints:
    test_endpoint:
        shortname: test
        provider: test
        base_url: http://test
        model: test-model
        cred_key: test_api_key
""")

    with open("test_config.local.yml", "w") as f:
        f.write("""
directories:
    papers: local_papers
workflow:
    extraction: test:local
    contestants: [test:local]
    target: test:local
""")

    with open("test_secrets.yml", "w") as f:
        f.write("""
test_key: secret123
test_api_key: secret123
openai_api_key: sk-test
""")

    try:
        # Set an environment variable to test highest precedence
        os.environ["directories.papers"] = "env_papers"

        # Load settings with our test files
        settings = load_settings([
            Path("test_config.yml"),
            Path("test_servers.yml"),
            Path("test_config.local.yml"),
            Path("test_secrets.yml")
        ])

        # 1. Test hierarchical loading
        print("\nTesting hierarchical loading:")
        print(f"directories.papers = {settings.directories.papers}")  # Should be env_papers
        print(f"workflow.extraction = {settings.workflow.extraction}")  # Should be test:local
        print(f"endpoints = {list(settings.endpoints.keys())}")  # Should include test_endpoint

        # 2. Test credential handling
        print("\nTesting credential handling:")
        print("Secrets keys:", list(settings.secrets.keys()))
        # Check if openai_api_key is wrapped as SecretStr
        is_secret = isinstance(settings.secrets.get("openai_api_key"), SecretStr)
        print("API key is SecretStr:", is_secret)
        if is_secret:
            print("Secret value is hidden:", settings.secrets["openai_api_key"])
            # We can still access the value when needed
            print("Can access value:", settings.secrets["openai_api_key"].get_secret_value())

        # 3. Test validation
        print("\nTesting validation:")
        print("Workflow contestants:", settings.workflow.contestants)
        print("Target in contestants:", settings.workflow.target in settings.workflow.contestants)
        
        # 4. Test defaults and constraints
        print("\nTesting defaults and constraints:")
        print(f"Default min score: {settings.quality.minScore}")
        print(f"Default temperature: {settings.default_temperature}")
        
        # All tests passed
        return all([
            settings.directories.papers == "env_papers",  # Environment override worked
            settings.workflow.extraction == "test:local",  # Local config override worked
            "test_endpoint" in settings.endpoints,  # Endpoint from servers.yml loaded
            isinstance(settings.secrets.get("openai_api_key"), SecretStr),  # Credential handling
            settings.workflow.target in settings.workflow.contestants,  # Validation rule enforced
        ])

    except Exception as e:
        print(f"Error during testing: {e}")
        return False

    finally:
        # Clean up test files
        for file in ["test_config.yml", "test_servers.yml", "test_config.local.yml", "test_secrets.yml"]:
            try:
                os.remove(file)
            except:
                pass
        # Clean up environment variable
        if "directories.papers" in os.environ:
            del os.environ["directories.papers"]

if __name__ == "__main__":
    success = test_basic_settings()
    print(f"\nTest {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)

