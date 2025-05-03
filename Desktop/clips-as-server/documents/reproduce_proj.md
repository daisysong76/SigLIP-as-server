# Create project root
mkdir clip-service && cd clip-service
# Set up Python virtual environment
python3 -m venv .venv
# Activate it
source .venv/bin/activate  # for Mac/Linux
# .venv\Scripts\activate    # for Windows (CMD)
# .venv\Scripts\Activate.ps1  # for Windows (PowerShell)
# Upgrade pip
pip install --upgrade pip


# Basic libraries you'll definitely need
pip install fastapi uvicorn torch torchvision faiss-cpu transformers pyyaml numpy
# Extras for code quality & monitoring
pip install black isort flake8 pre-commit matplotlib tensorboard
# Save dependencies
pip freeze > requirements.txt




# Create folders inside the existing clip-service
mkdir -p config api/routers api/middleware models inference optimization indexing monitoring batching docker k8s
touch pyproject.toml README.md Makefile .pre-commit-config.yaml main.py
touch config/config.yaml config/logging.yaml config/models.yaml
touch api/__init__.py api/app.py
touch api/routers/__init__.py api/routers/embeddings.py api/routers/search.py api/routers/health.py
touch api/middleware/__init__.py api/middleware/auth.py
touch models/__init__.py models/registry.py models/loaders.py
touch inference/__init__.py inference/pipeline.py inference/text_encoder.py inference/image_encoder.py inference/multimodal.py
touch optimization/__init__.py optimization/compile.py optimization/quantization.py optimization/profiler.py
touch indexing/__init__.py indexing/vector_store.py indexing/faiss_index.py indexing/scann_index.py
touch monitoring/__init__.py monitoring/metrics.py monitoring/tracing.py monitoring/dashboard.py
touch batching/__init__.py batching/dynamic_batcher.py batching/predictor.py
touch docker/Dockerfile docker/Dockerfile.dev docker/docker-compose.yaml
touch k8s/deployment.yaml k8s/service.yaml k8s/ingress.yaml



# Root project directory
mkdir -p clip-service && cd clip-service

# Top-level files
touch pyproject.toml README.md Makefile .pre-commit-config.yaml main.py

# Config
mkdir -p config && touch config/config.yaml config/logging.yaml config/models.yaml

# API
mkdir -p api/routers api/middleware
touch api/__init__.py api/app.py
touch api/routers/__init__.py api/routers/embeddings.py api/routers/search.py api/routers/health.py
touch api/middleware/__init__.py api/middleware/auth.py

# Models
mkdir -p models && touch models/__init__.py models/registry.py models/loaders.py

# Inference
mkdir -p inference
touch inference/__init__.py inference/pipeline.py inference/text_encoder.py inference/image_encoder.py inference/multimodal.py

# Optimization
mkdir -p optimization
touch optimization/__init__.py optimization/compile.py optimization/quantization.py optimization/profiler.py

# Indexing
mkdir -p indexing
touch indexing/__init__.py indexing/vector_store.py indexing/faiss_index.py indexing/scann_index.py

# Monitoring
mkdir -p monitoring
touch monitoring/__init__.py monitoring/metrics.py monitoring/tracing.py monitoring/dashboard.py

# Batching
mkdir -p batching
touch batching/__init__.py batching/dynamic_batcher.py batching/predictor.py

# Docker
mkdir -p docker
touch docker/Dockerfile docker/Dockerfile.dev docker/docker-compose.yaml

# Kubernetes
mkdir -p k8s
touch k8s/deployment.yaml k8s/service.yaml k8s/ingress.yaml
