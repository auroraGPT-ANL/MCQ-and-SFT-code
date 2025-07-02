#!/usr/bin/env python

"""Test suite for the new settings.py configuration system."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import yaml

# Add src to path if not already there
if "common" not in sys.modules:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Now we can import our module
from common.settings import Endpoint, WorkflowModels, Settings, load_settings

class TestSettings(unittest.TestCase):
    """Test cases for the settings module."""

    def test_basic_model_relationship(self):
        """Test the basic workflow model relationship validation from the blueprint."""
        # Verify the simple assertion from the blueprint
        s = load_settings()
        self.assertIn(s.workflow.target, s.workflow.contestants)

    def test_workflow_model_validation(self):
        """Test that WorkflowModels enforces the required relationships."""
        # Valid case (extraction and target both in contestants)
        valid = WorkflowModels(
            extraction="model_a",
            contestants=["model_a", "model_b", "model_c"],
            target="model_b"
        )
        self.assertIn(valid.extraction, valid.contestants)
        self.assertIn(valid.target, valid.contestants)

        # Auto-add extraction if missing from contestants
        auto_fix = WorkflowModels(
            extraction="model_x", 
            contestants=["model_y", "model_z"], 
            target="model_z"
        )
        self.assertIn(auto_fix.extraction, auto_fix.contestants)
        
        # Should raise error if target not in contestants
        with self.assertRaises(ValueError):
            WorkflowModels(
                extraction="model_a",
                contestants=["model_a", "model_b"],
                target="model_c"  # Not in contestants
            )

    def test_configuration_overlay(self):
        """Test that configuration files are loaded in the correct order."""
        # Create temporary config files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config files with different values for the same keys
            with open(temp_path / "config.yml", "w") as f:
                yaml.dump({"workflow": {"extraction": "base_model"}}, f)
            
            with open(temp_path / "servers.yml", "w") as f:
                yaml.dump({"endpoints": {"test": {"shortname": "test", "provider": "test", 
                                                 "base_url": "http://test", "model": "test"}}}, f)
            
            with open(temp_path / "config.local.yml", "w") as f:
                yaml.dump({"workflow": {"extraction": "local_model", 
                                       "contestants": ["local_model", "other_model"],
                                       "target": "local_model"}}, f)
            
            with open(temp_path / "secrets.yml", "w") as f:
                yaml.dump({"TEST_KEY": "test_value"}, f)
            
            # Set environment variable to test highest precedence
            os.environ["workflow.extraction"] = "env_model"
            
            # Test with temporary directory
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                settings = load_settings()
                
                # Verify overlay order: env vars should have highest precedence
                self.assertEqual(settings.workflow.extraction, "env_model")
                
                # Test endpoint from servers.yml
                self.assertIn("test", settings.endpoints)
                self.assertEqual(settings.endpoints["test"].provider, "test")
                
                # Test secret from secrets.yml
                self.assertEqual(settings.secrets["TEST_KEY"], "test_value")
                
            finally:
                os.chdir(old_cwd)
                os.environ.pop("workflow.extraction", None)

    def test_endpoint_credential_resolution(self):
        """Test that endpoint credential keys resolve correctly to secrets."""
        # Create a test settings object with endpoints and secrets
        settings = Settings(
            endpoints={
                "test_model": Endpoint(
                    shortname="test_model",
                    provider="test",
                    base_url="http://test",
                    model="test-model",
                    cred_key="TEST_API_KEY"
                )
            },
            secrets={
                "TEST_API_KEY": "test_secret_value"
            }
        )
        
        # Test credential resolution
        endpoint = settings.endpoints["test_model"]
        credential = settings.secrets.get(endpoint.cred_key)
        self.assertEqual(credential, "test_secret_value")


if __name__ == "__main__":
    unittest.main()

