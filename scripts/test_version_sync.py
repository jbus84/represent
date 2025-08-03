#!/usr/bin/env python3
"""
Test script to verify version synchronization across files.

This script validates that all version locations are properly synchronized
and that the semantic-release configuration is correctly set up.
"""

import re
from pathlib import Path
from typing import Dict, List


def extract_pyproject_version() -> str:
    """Extract project version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    
    return match.group(1)


def extract_commitizen_version() -> str:
    """Extract commitizen version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Find the commitizen section
    cz_section = re.search(r'\[tool\.commitizen\](.*?)(?=\[|$)', content, re.DOTALL)
    if not cz_section:
        raise ValueError("Could not find commitizen config")
    
    version_match = re.search(r'version = "([^"]+)"', cz_section.group(1))
    if not version_match:
        raise ValueError("Could not find version in commitizen config")
    
    return version_match.group(1)


def extract_init_version() -> str:
    """Extract __version__ from represent/__init__.py."""
    init_path = Path("represent/__init__.py")
    content = init_path.read_text()
    
    match = re.search(r'__version__ = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find __version__ in __init__.py")
    
    return match.group(1)


def check_semantic_release_config() -> Dict[str, List[str]]:
    """Check that semantic release is configured to update both locations."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    result: Dict[str, List[str]] = {}
    
    # Find version_toml configuration
    toml_match = re.search(r'version_toml\s*=\s*\[(.*?)\]', content)
    if toml_match:
        toml_str = toml_match.group(1)
        toml_vars = [var.strip().strip('"') for var in toml_str.split(',') if var.strip()]
        result['version_toml'] = toml_vars
    else:
        result['version_toml'] = []
    
    # Find version_variables configuration
    var_match = re.search(r'version_variables\s*=\s*\[(.*?)\]', content)
    if var_match:
        variables_str = var_match.group(1)
        variables = [var.strip().strip('"') for var in variables_str.split(',') if var.strip()]
        result['version_variables'] = variables
    else:
        result['version_variables'] = []
    
    return result


def main():
    """Test version synchronization."""
    print("🔍 Testing Version Synchronization")
    print("=" * 40)
    
    try:
        # Extract versions from all locations
        pyproject_version = extract_pyproject_version()
        commitizen_version = extract_commitizen_version()
        init_version = extract_init_version()
        
        print(f"📄 pyproject.toml version: {pyproject_version}")
        print(f"⚙️  commitizen version: {commitizen_version}")
        print(f"🐍 __init__.py version: {init_version}")
        
        # Check synchronization
        if pyproject_version == commitizen_version == init_version:
            print("\n✅ All versions are synchronized!")
        else:
            print("\n❌ Version mismatch detected!")
            return 1
        
        # Check semantic release configuration
        print("\n🔧 Checking semantic-release configuration...")
        config = check_semantic_release_config()
        print(f"📝 Version TOML: {config['version_toml']}")
        print(f"📝 Version variables: {config['version_variables']}")
        
        expected_toml = ["pyproject.toml:project.version"]
        expected_variables = ["represent/__init__.py:__version__"]
        
        toml_ok = set(config['version_toml']) == set(expected_toml)
        vars_ok = set(config['version_variables']) == set(expected_variables)
        
        if toml_ok and vars_ok:
            print("✅ Semantic-release is configured to update both locations!")
            print("   📄 pyproject.toml via version_toml")
            print("   🐍 __init__.py via version_variables")
        else:
            print("❌ Semantic-release configuration issue!")
            if not toml_ok:
                print(f"   version_toml - Expected: {expected_toml}, Found: {config['version_toml']}")
            if not vars_ok:
                print(f"   version_variables - Expected: {expected_variables}, Found: {config['version_variables']}")
            return 1
        
        print("\n🎉 Version bumping system is properly configured!")
        print("\nFor future version bumps:")
        print("1. Use conventional commits (feat:, fix:, BREAKING CHANGE:)")
        print("2. Run 'semantic-release' to automatically bump versions")
        print("3. Both pyproject.toml and __init__.py will be updated automatically")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())