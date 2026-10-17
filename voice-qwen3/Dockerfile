# Qwen3-TTS 1.7B CustomVoice without ComfyUI (see handler.py). Weights come from the Azure Files share
# (/runpod-volume/qwen3-tts/1.7b-customvoice), staged by the GPU fleet's fetch job; the image holds none.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_DIR="/runpod-volume/qwen3-tts/1.7b-customvoice" \
    DTYPE="float32" \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin the base image's torch as constraints (CUDA 13 wheels need a newer driver; see vid-wan-a14b).
RUN python -c "import torch, torchvision, torchaudio; \
open('/tmp/constraints.txt','w').write(''.join(f'{n}=={v}\n' for n,v in \
[('torch',torch.__version__),('torchvision',torchvision.__version__),('torchaudio',torchaudio.__version__)]))" \
    && cat /tmp/constraints.txt

# qwen-tts 0.1.1 pins transformers==4.57.3 itself.
RUN pip install --no-cache-dir -c /tmp/constraints.txt "qwen-tts==0.1.1" "soundfile" "runpod>=1.10.1"

COPY build_check.py /build_check.py
RUN python /build_check.py
COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
