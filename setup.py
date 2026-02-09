from setuptools import setup, find_packages

setup(
    name="focuslab",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        # Core ML
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "hmmlearn>=0.3.0",
        
        # API & Web
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        
        # Data Processing
        "pandas>=2.0.0",
        "pyarrow>=13.0.0",
        
        # Caching & Queuing
        "redis>=5.0.0",
        "aioredis>=2.0.1",
        
        # Monitoring & Logging
        "mlflow>=2.8.0",
        "prometheus-client>=0.18.0",
        "loguru>=0.7.0",
        
        # LLM Integration
        "openai>=1.3.0",
        "anthropic>=0.7.0",
        "tiktoken>=0.5.0",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
        "spark": [
            "pyspark>=3.5.0",
            "delta-spark>=2.4.0",
        ],
        "cloud": [
            "azure-ai-ml>=1.11.0",
            "boto3>=1.28.0",
        ],
    },
)
