FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------
# HF CACHE PATH S
# -------------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV TOKENIZERS_PARALLELISM=false

# -------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# -------------------------------
# PYTHON DEPENDENCIES
# -------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r requirements.txt

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

print("RolmOCR downloaded")
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

CMD ["python", "-u", "handler.py"]