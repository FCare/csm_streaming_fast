#!/bin/bash
set -e

echo "Starting CSM Streaming Inference API..."

# Vérifier la disponibilité du GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "Warning: No NVIDIA GPU detected."
fi

# Login Hugging Face si token fourni
if [ -n "$HF_TOKEN" ]; then
    echo "Setting up Hugging Face token..."
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token configured via environment variable"
else
    echo "Warning: No HF_TOKEN provided. Model download may fail."
fi

# Test des imports Python critiques
echo "Testing Python imports..."
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')" || echo "Warning: PyTorch import failed"
python -c "import transformers; print(f'Transformers {transformers.__version__}')" || echo "Warning: Transformers import failed"

echo "Starting CSM Streaming Inference API on port 8000..."
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo "Generate Audio: POST http://localhost:8000/generate"
exec "$@"