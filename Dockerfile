FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip \
    && pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
       --index-url https://download.pytorch.org/whl/cu128
RUN git clone --depth 1 --branch v0.34.0 https://github.com/Comfy-Org/ComfyUI.git /comfyui
RUN pip install -r /comfyui/requirements.txt 'transformers>=4.50.3,<5' 'huggingface-hub<1' \
    runpod requests
# Test core imports during the build; model inference still needs a GPU test.
RUN cd /comfyui && python main.py --quick-test-for-ci --cpu
WORKDIR /worker
COPY handler.py workflow.py ./
CMD ["python", "-u", "handler.py"]
