#!/usr/bin/env python3
"""
Test validation of settings system, particularly the relationship
between servers.yml endpoint definitions and secrets.yml credentials.
"""

import os
import sys
from pathlib import Path
from settings import load_settings, Settings, WorkflowModels, ModelProvider

def test_endpoint_credential_validation():
    """Test that settings properly validates endpoint/credential relationships."""
    
    # Handle existing config files
    original_servers = None
    if Path("servers.yml").exists():
        original_servers = Path("servers.yml")
        os.rename("servers.yml", "servers.yml.bak")
    
    try:
        # Create test configuration files
        with open("servers.yml", "w") as f:
            f.write("""
endpoints:
    openai:
        shortname: openai
        provider: openai
        base_url: https://api.openai.com/v1
        model: gpt-4
        cred_key: openai_api_key

    argo:
        shortname: argo
        provider: argo
        base_url: https://apps.inside.anl.gov/argoapi/api/v1/resource/chat
        model: llama2-70b
        cred_key: argo_token
""")

        with open("test_config.local.yml", "w") as f:
            f.write("""
workflow:
    extraction: openai:gpt-4
    contestants: [openai:gpt-4, argo:llama2-70b]
    target: openai:gpt-4
""")

        success = True
        
        # Test 1: Missing credential should raise error
        print("\nTest 1: Testing missing credential detection...")
        with open("test_secrets.yml", "w") as f:
            f.write("""
openai_api_key: sk-test-key
# Missing argo_token intentionally
""")

        try:
            settings = load_settings([
                Path("test_config.local.yml"),
                Path("test_secrets.yml")
            ])
            print("ERROR: Should have raised ValueError for missing argo_token")
            success = False
        except ValueError as e:
            if "Missing credential" in str(e) and "argo_token" in str(e):
                print("PASS: Correctly detected missing credential")
            else:
                print(f"ERROR: Unexpected error message: {e}")
                success = False

        # Test 2: All credentials present should work
        print("\nTest 2: Testing valid configuration...")
        with open("test_secrets.yml", "w") as f:
            f.write("""
openai_api_key: sk-test-key
argo_token: test-token
""")

        try:
            settings = load_settings([
                Path("test_config.local.yml"),
                Path("test_secrets.yml")
            ])
            if (settings.secrets["openai_api_key"].get_secret_value() == "sk-test-key" and
                settings.secrets["argo_token"].get_secret_value() == "test-token"):
                print("PASS: Validation passed with all credentials present")
            else:
                print("ERROR: Credential values don't match expected values")
                success = False
        except Exception as e:
            print(f"ERROR: Unexpected exception: {e}")
            success = False

        # Test 3: Unknown provider should raise error
        print("\nTest 3: Testing unknown provider detection...")
        with open("test_config.local.yml", "w") as f:
            f.write("""
workflow:
    extraction: unknown:model
    contestants: [unknown:model]
    target: unknown:model
""")

        try:
            settings = load_settings([
                Path("test_config.local.yml"),
                Path("test_secrets.yml")
            ])
            print("ERROR: Should have raised ValueError for unknown provider")
            success = False
        except ValueError as e:
            if "No endpoint configuration found" in str(e):
                print("PASS: Correctly detected unknown provider")
            else:
                print(f"ERROR: Unexpected error message: {e}")
                success = False

    finally:
        # Cleanup test files
        for file in ["test_config.local.yml", "test_secrets.yml"]:
            try:
                os.remove(file)
            except:
                pass
        
        # Restore original servers.yml if it existed
        if original_servers:
            os.remove("servers.yml")
            os.rename("servers.yml.bak", "servers.yml")

    return success

if __name__ == "__main__":
    success = test_endpoint_credential_validation()
    print(f"\nOverall test result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)

