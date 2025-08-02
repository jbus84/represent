#!/usr/bin/env python3
"""
Test script to verify version synchronization across files.

This script validates that all version locations are properly synchronized
and that the semantic-release configuration is correctly set up.
"""

import re
from pathlib import Path


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
    
    match = re.search(r'__version__: str = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find __version__ in __init__.py")
    
    return match.group(1)


def check_semantic_release_config() -> list[str]:
    """Check that semantic release is configured to update both locations."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Find version_variables directly in the full content
    var_match = re.search(r'version_variables\s*=\s*\[(.*?)\]', content)
    if not var_match:
        raise ValueError("Could not find version_variables in semantic_release config")
    
    # Parse the list 
    variables_str = var_match.group(1)
    variables = [var.strip().strip('"') for var in variables_str.split(',') if var.strip()]
    
    return variables


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
        version_variables = check_semantic_release_config()
        print(f"📝 Version variables: {version_variables}")
        
        expected_variables = ["pyproject.toml:version", "represent/__init__.py:__version__"]
        if set(version_variables) == set(expected_variables):
            print("✅ Semantic-release is configured to update both locations!")
        else:
            print("❌ Semantic-release configuration issue!")
            print(f"   Expected: {expected_variables}")
            print(f"   Found: {version_variables}")
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