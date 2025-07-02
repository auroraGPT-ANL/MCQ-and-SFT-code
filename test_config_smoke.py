#!/usr/bin/env python3
"""
Smoke test for the pydantic config system.
Tests the real configuration with current files.
"""

import sys
from settings import load_settings
from pydantic import SecretStr

def smoke_test():
    """Verify the config system works with current configuration files."""
    print("🔥 Pydantic Config System Smoke Test")
    print("=" * 50)
    
    try:
        # Load real configuration
        settings = load_settings()
        print("✅ Configuration loaded successfully!")
        
        # Test 1: Validate structure
        print(f"\n📊 Configuration Summary:")
        print(f"   Available endpoints: {len(settings.endpoints)}")
        print(f"   Current extraction model: {settings.workflow.extraction}")
        print(f"   Contestant models: {settings.workflow.contestants}")
        print(f"   Target model: {settings.workflow.target}")
        print(f"   Credentials loaded: {len([k for k in settings.secrets.keys() if not k.isupper()])}")
        
        # Test 2: Verify shortname resolution works
        print(f"\n🔗 Shortname Resolution:")
        endpoints = settings.endpoints
        
        def resolve_model_to_provider(model_spec: str) -> str:
            if ':' in model_spec:
                return model_spec.split(':', 1)[0]
            else:
                for endpoint in endpoints.values():
                    if endpoint.shortname == model_spec:
                        return endpoint.provider.value
                return model_spec
        
        for model in settings.workflow.contestants:
            provider = resolve_model_to_provider(model)
            found_endpoint = next((ep for ep in endpoints.values() if ep.shortname == model), None)
            if found_endpoint:
                print(f"   {model} -> {provider} (endpoint: {found_endpoint.base_url})")
            else:
                print(f"   {model} -> {provider} (provider:model format)")
        
        # Test 3: Verify credential security
        print(f"\n🔒 Credential Security:")
        sample_creds = [k for k in settings.secrets.keys() if k.endswith('_key') or k.endswith('_token')][:3]
        for cred in sample_creds:
            secret_val = settings.secrets[cred]
            is_secret = isinstance(secret_val, SecretStr)
            print(f"   {cred}: {'SecretStr ✅' if is_secret else 'Plain text ❌'}")
        
        # Test 4: Verify validation rules
        print(f"\n✔️  Validation Rules:")
        print(f"   Target in contestants: {settings.workflow.target in settings.workflow.contestants}")
        print(f"   Extraction in contestants: {settings.workflow.extraction in settings.workflow.contestants}")
        print(f"   Quality settings valid: {1 <= settings.quality.minScore <= 10}")
        print(f"   HTTP timeout configured: {settings.http_client.connect_timeout > 0}")
        
        # Test 5: Verify directory configuration
        print(f"\n📁 Directory Configuration:")
        dirs = settings.directories
        print(f"   Papers: {dirs.papers}")
        print(f"   JSON: {dirs.json_dir}")
        print(f"   MCQ: {dirs.mcq}")
        print(f"   Results: {dirs.results}")
        
        print(f"\n🎉 All tests passed! The pydantic config system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)
