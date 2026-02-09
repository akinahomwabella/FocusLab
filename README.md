# FocusLab: Probabilistic Behavioral State Inference

A production-grade system for inferring latent cognitive states (Focused, Fatigued, Distracted) from noisy behavioral signals using Hidden Markov Models with online Bayesian updating.

## Features
- Real-time cognitive state inference using HMMs
- High-performance FastAPI backend with async support
- Streaming data pipeline with Redis caching
- LLM-enhanced state interpretation
- Calibrated uncertainty quantification
- Production-ready with Docker + K8s support
- MLflow experiment tracking

## Tech Stack
- **ML**: HMM, Bayesian inference, scikit-learn, PyTorch
- **Backend**: FastAPI, gRPC, Redis
- **Data**: Apache Kafka, PySpark, Delta Lake
- **LLM**: OpenAI/Anthropic API
- **Deploy**: Docker, Kubernetes, Azure ML
- **Monitoring**: MLflow, Prometheus, Grafana

## Quick Start
```bash
# Clone and setup
git clone <your-repo>
cd focuslab
pip install -e .

# Run development server
python src/api/main.py

# Or with Docker
docker-compose up
```

## Project Structure
```
focuslab/
├── src/              # Source code
├── notebooks/        # Analysis notebooks
├── data/            # Datasets
├── configs/         # Configuration files
├── tests/           # Unit tests
└── docs/            # Documentation
```


