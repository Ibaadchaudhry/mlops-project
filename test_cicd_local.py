#!/usr/bin/env python3
"""
Local CI/CD Testing Script
==========================

This script tests key components of our CI/CD pipeline locally 
to help debug and validate before GitHub Actions execution.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description, allow_failure=False):
    """Run a command and report results."""
    print(f"\n🔍 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print(f"Output:\n{result.stdout[:500]}")
        else:
            print(f"❌ FAILED: {description}")
            if result.stderr:
                print(f"Error:\n{result.stderr[:500]}")
            if not allow_failure:
                return False
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False
    
    return True

def main():
    """Run local CI/CD tests."""
    print("🚀 MLOps CI/CD Local Testing")
    print("=" * 50)
    
    # Test 1: Python environment
    print("\n📦 Testing Python Environment")
    if not run_command("python --version", "Python version check"):
        print("❌ Python not available")
        return False
    
    # Test 2: Dependencies
    print("\n📚 Testing Dependencies")
    run_command("pip list | grep -E '(pytest|torch|pandas|numpy)'", "Core dependencies", allow_failure=True)
    
    # Test 3: Code quality tools (allow failures)
    print("\n🔍 Testing Code Quality Tools")
    run_command("python -m black --version", "Black formatter", allow_failure=True)
    run_command("python -m flake8 --version", "Flake8 linter", allow_failure=True)
    run_command("python -m pytest --version", "Pytest", allow_failure=True)
    
    # Test 4: Run basic tests
    print("\n🧪 Running Basic Tests")
    if Path("tests").exists():
        if not run_command("python -m pytest tests/test_basic.py -v", "Basic test suite"):
            print("⚠️ Some tests failed - check test output above")
    else:
        print("⚠️ Tests directory not found")
    
    # Test 5: Module imports
    print("\n📦 Testing Module Imports")
    modules_to_test = [
        ("import torch; print('✅ PyTorch:', torch.__version__)", "PyTorch"),
        ("import pandas; print('✅ Pandas:', pandas.__version__)", "Pandas"),
        ("import numpy; print('✅ NumPy:', numpy.__version__)", "NumPy"),
        ("from model import TabularMLP; print('✅ Model module')", "Model module"),
        ("from drift_detector import psi; print('✅ Drift detector')", "Drift detector"),
    ]
    
    for cmd, desc in modules_to_test:
        run_command(f'python -c "{cmd}"', desc, allow_failure=True)
    
    # Test 6: Docker (if available)
    print("\n🐳 Testing Docker")
    if run_command("docker --version", "Docker availability", allow_failure=True):
        run_command("docker-compose --version", "Docker Compose", allow_failure=True)
        
        # Test Docker build (just syntax check)
        if Path("Dockerfile.api").exists():
            run_command("docker build -f Dockerfile.api -t test-api . --dry-run 2>/dev/null || echo 'Docker build syntax test'", "Docker build test", allow_failure=True)
    
    # Test 7: Git status
    print("\n📝 Git Status Check")
    run_command("git status --porcelain", "Git status", allow_failure=True)
    run_command("git log --oneline -n 3", "Recent commits", allow_failure=True)
    
    print("\n" + "=" * 50)
    print("🎯 Local CI/CD Testing Complete!")
    print("🌐 Check GitHub Actions for live results:")
    print("   https://github.com/Ibaadchaudhry/mlops-project/actions")
    print("=" * 50)

if __name__ == "__main__":
    main()