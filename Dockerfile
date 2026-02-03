FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
        torchaudio==2.6.0 \
    && pip install \
        lmdeploy \
        librosa \
        "fastaudiosr @ git+https://github.com/ysharma3501/FlashSR.git" \
        "ncodec @ git+https://github.com/ysharma3501/FastBiCodec.git" \
        einops \
        onnxruntime-gpu \
        gradio \
        soundfile \
        soe-vinorm \
        huggingface_hub \
        omegaconf

EXPOSE 7860

CMD ["python", "app.py"]
