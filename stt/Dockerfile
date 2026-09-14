# faster-whisper transcription. large-v3 at int8 is ~1.5GB, so it downloads on first boot
# in seconds rather than being baked in.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# The torch280 base ships HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package,
# so the first runtime download from HuggingFace dies with "Fast download using
# 'hf_transfer' is enabled ... but 'hf_transfer' package is not available". Found live on
# the music lane 2026-09-14; every lane on this base inherits it. Turning the fast path
# off costs nothing -- it was never installed to begin with.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    WHISPER_MODEL=large-v3 \
    WHISPER_COMPUTE=int8_float16 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed cryptography
RUN pip install --no-cache-dir \
      "faster-whisper" \
      "huggingface-hub<1.0" \
      "runpod>=1.10.1"

COPY handler.py /handler.py
RUN python -c "import ast; ast.parse(open('/handler.py').read()); print('handler parses OK')"

CMD ["python", "-u", "/handler.py"]
