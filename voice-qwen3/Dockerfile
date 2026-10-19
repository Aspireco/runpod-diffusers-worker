# Qwen3-TTS 1.7B without ComfyUI (see handler.py): CustomVoice, VoiceDesign and Base (cloning)
# share this one image, selected at runtime by QWEN_MODE (custom | design | clone). Weights come
# from the Azure Files share, staged by each mode's own fetch job (gpu/manifests/voice-qwen3.tsv,
# voice-design.tsv, voice-clone-qwen.tsv); the image holds none. MODEL_DIR below is the default for
# QWEN_MODE=custom (the original, still-default lane) -- design and clone lanes set their own
# MODEL_DIR, or handler.py falls back to each mode's own default under /runpod-volume/qwen3-tts/.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    QWEN_MODE="custom" \
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
