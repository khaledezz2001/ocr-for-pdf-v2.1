FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------
# HF CACHE PATHS
# -------------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV TOKENIZERS_PARALLELISM=false

# -------------------------------
# RTX 5090 CUDA OPTIMIZATIONS
# -------------------------------
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_MODULE_LOADING=LAZY

# -------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# -------------------------------
# PYTHON DEPENDENCIES
# -------------------------------
COPY requirements.txt .

# Install PyTorch with CUDA 12.6 support
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Try to install Flash Attention (optional - won't fail build if it doesn't work)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention installation failed - continuing without it"

# -------------------------------
# MODEL DOWNLOAD (BUILD TIME)
# -------------------------------
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="reducto/RolmOCR",
    local_dir="/models/hf/reducto/RolmOCR",
    local_dir_use_symlinks=False
)

print("✓ RolmOCR model downloaded successfully")
EOF

# -------------------------------
# LOCK OFFLINE MODE (RUNTIME)
# -------------------------------
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# -------------------------------
# APP
# -------------------------------
COPY handler.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "-u", "handler.py"]
