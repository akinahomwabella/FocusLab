#!/usr/bin/env python3
"""
Initialize FocusLab project directory structure
"""
import os
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories for the project"""
    
    base_dirs = [
        # Source code
        "src/models",
        "src/api",
        "src/data",
        "src/inference",
        "src/utils",
        "src/streaming",
        "src/llm",
        
        # Data directories
        "data/raw",
        "data/processed",
        "data/synthetic",
        
        # Notebooks
        "notebooks/exploration",
        "notebooks/evaluation",
        
        # Tests
        "tests/unit",
        "tests/integration",
        
        # Configuration
        "configs",
        
        # Docker
        "docker",
        
        # Scripts
        "scripts",
        
        # Documentation
        "docs/api",
        "docs/guides",
        
        # MLflow artifacts
        "mlruns",
        
        # Logs
        "logs",
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py for Python packages
        if dir_path.startswith("src/"):
            init_file = Path(dir_path) / "__init__.py"
            init_file.touch(exist_ok=True)
        print(f"✓ Created {dir_path}")
    
    print("\n✅ Directory structure created successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -e .")
    print("2. Set up environment: cp .env.example .env")
    print("3. Start building!")

if __name__ == "__main__":
    create_directory_structure()
