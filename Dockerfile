# Utiliser la dernière image CUDA 12.9 officielle avec Ubuntu
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Éviter les interactions pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Variables d'environnement pour optimiser PyTorch
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_DISABLE_CUDA_GRAPHS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Mettre à jour et installer les dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer un lien symbolique pour python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Créer un utilisateur non-root
RUN useradd -m -s /bin/bash csm
USER csm
WORKDIR /home/csm

RUN mkdir -p /home/csm/.cache/huggingface/hub && \
    chown -R csm:csm /home/csm/.cache

# Créer l'environnement virtuel
RUN python -m venv .venv
ENV PATH="/home/csm/.venv/bin:$PATH"

# Mettre à jour pip
RUN pip install --upgrade pip setuptools wheel

# Installer les dépendances Python essentielles d'abord
RUN pip install requests urllib3 certifi

# Installer PyTorch pre-release avec CUDA 12.8 pour RTX 5060 Ti
# RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Installer les autres dépendances depuis requirements.txt
COPY --chown=csm:csm requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-deps moshi

# Installer flash-attention pour les optimisations (optionnel)
# RUN pip install flash-attn==1.0.5 --no-build-isolation || echo "Flash attention installation failed, continuing without it"
#RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp310-cp310-linux_x86_64.whl
#RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp311-cp311-linux_x86_64.whl
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp312-cp312-linux_x86_64.whl
#RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp313-cp313-linux_x86_64.whl


# Copier les fichiers nécessaires pour l'API d'inférence
COPY --chown=csm:csm generator.py models.py main_inference.py ./
COPY --chown=csm:csm docker/entrypoint.sh /home/csm/entrypoint.sh
RUN chmod +x /home/csm/entrypoint.sh

# Exposer le port de l'API
EXPOSE 8000

ENTRYPOINT ["/home/csm/entrypoint.sh"]
CMD ["python", "main_inference.py"]